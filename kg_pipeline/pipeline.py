"""Orchestrate stages and orderly shutdown; no extraction rules live here."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from .config import Config
from .events import Events
from .fetch import SourceReader, fetch_types
from .llm import VLLMClient
from .models import STOP, SourceWork, now
from .postprocess import finish_source, process_result
from .preprocess import Preprocessor
from .prompts import Vocabulary
from .storage import Store
from .upload import submit_record_run

LOG = logging.getLogger(__name__)


def describe_exception(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe_exception(child) for child in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


class Pipeline:
    def __init__(self, config: Config, source_client: httpx.AsyncClient,
                 llm_client: httpx.AsyncClient, upload_client: httpx.AsyncClient,
                 store: Store | None = None):
        self.config = config
        self.source_client, self.upload_client = source_client, upload_client
        self.store = store or Store(config.output)
        self.reader = SourceReader(source_client, config)
        self.llm = VLLMClient(llm_client, config)
        self.events = Events(self.store, config.event_queue_size)
        self.queues = {name: asyncio.Queue(maxsize=config.queue_size) for name in
                       ("source", "llm", "postprocess", "save", "upload", "receipt")}
        self.source_slots = asyncio.BoundedSemaphore(config.inflight_sources)
        self.monitor_stop = asyncio.Event()
        self.stats = {"fetched": 0, "prepared": 0, "llm_jobs": 0, "llm_completed": 0,
                      "llm_inflight": 0, "chunks_failed": 0, "saved": 0,
                      "uploaded": 0, "upload_errors": 0, "completion_tokens": 0}
        self.fetch_error: str | None = None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.batch_id = f"local_vllm_{stamp}_{uuid4().hex[:12]}"
        self.metadata = {"batch_id": self.batch_id, "started_at": now(), "config": config.public_dict()}

    async def run(self) -> dict:
        self.config.validate()
        await self.store.initialize(self.metadata)
        started = time.monotonic()
        status, fatal = "completed", None
        cancelled = False
        try:
            # All startup reads can overlap. There is one shared vocabulary for the whole run.
            async with asyncio.TaskGroup() as group:
                entity_task = group.create_task(fetch_types(self.source_client, self.config.entity_url, self.config))
                relationship_task = group.create_task(fetch_types(self.source_client, self.config.relationship_url, self.config))
                group.create_task(self.llm.initialize())
            self.vocabulary = Vocabulary.build(entity_task.result(), relationship_task.result())
            self.preprocessor = Preprocessor(self.config, self.llm, self.vocabulary)
            await self.preprocessor.preflight()
            self.metadata.update(model=self.llm.model, context_length=self.llm.context_length,
                                 entity_types=self.vocabulary.entity_rows,
                                 relationship_types=self.vocabulary.relationship_rows,
                                 system_prompt=self.vocabulary.system_prompt, schema=self.vocabulary.schema)
            await self.store.save_metadata(self.metadata)
            LOG.info("Model=%s context=%d LLM workers=%d dry_run=%s test=%s",
                     self.llm.model, self.llm.context_length, self.config.llm_workers,
                     self.config.dry_run, self.config.test)
            async with asyncio.TaskGroup() as group:
                group.create_task(self.events.run(), name="run-step-writer")
                group.create_task(self.monitor(), name="progress-monitor")
                group.create_task(self.workflow(), name="pipeline-stages")
            if self.fetch_error:
                status, fatal = "failed", self.fetch_error
        except asyncio.CancelledError:
            status, fatal, cancelled = "interrupted", "Pipeline was cancelled", True
        except Exception as exc:
            status, fatal = "failed", describe_exception(exc)
            LOG.error("Pipeline failed: %s", fatal)
        summary = await self.store.finalize({
            "batch_id": self.batch_id, "status": status, "error": fatal,
            "started_at": self.metadata["started_at"], "finished_at": now(),
            "seconds": time.monotonic() - started, "dry_run": self.config.dry_run, "test": self.config.test,
            "high_water_id": self.reader.high_water_id, "source_pages": self.reader.pages, **self.stats,
        })
        if cancelled:
            raise asyncio.CancelledError
        return summary

    async def close_queue(self, name: str, workers: int) -> None:
        for _ in range(workers):
            await self.queues[name].put(STOP)

    async def worker_pool(self, count: int, worker: Callable[[], Awaitable[None]],
                          next_queue: str | None = None, next_workers: int = 1) -> None:
        # Only this supervisor closes the downstream queue, after ALL workers finish.
        async with asyncio.TaskGroup() as group:
            for index in range(count):
                group.create_task(worker(), name=f"{worker.__name__}-{index}")
        if next_queue:
            await self.close_queue(next_queue, next_workers)

    async def workflow(self) -> None:
        cfg = self.config
        await self.events.emit("pipeline", "started", batch_id=self.batch_id)
        async with asyncio.TaskGroup() as group:
            group.create_task(self.produce(), name="fetch")
            group.create_task(self.worker_pool(cfg.preprocess_workers, self.preprocess_worker, "llm", cfg.llm_workers))
            group.create_task(self.worker_pool(cfg.llm_workers, self.llm_worker, "postprocess", cfg.postprocess_workers))
            group.create_task(self.worker_pool(cfg.postprocess_workers, self.postprocess_worker, "save"))
            group.create_task(self.save_worker(), name="local-save")
            if not cfg.dry_run:
                group.create_task(self.worker_pool(cfg.upload_workers, self.upload_worker, "receipt"))
                group.create_task(self.receipt_worker(), name="upload-receipts")
        await self.events.emit("pipeline", "finished", **self.stats)
        await self.events.queue.put(STOP)
        self.monitor_stop.set()

    async def produce(self) -> None:
        try:
            async for batch in self.reader.batches():
                await self.events.emit("fetch_page", "success", rows=len(batch),
                                       first_id=batch[0]["id"], last_id=batch[-1]["id"])
                for source in batch:
                    await self.source_slots.acquire()
                    work = SourceWork(source, f"{self.batch_id}_s{source['id']}")
                    await self.queues["source"].put(work)
                    self.stats["fetched"] += 1
            if self.stats["fetched"] == 0 and (self.config.test or self.config.source_id is not None):
                raise ValueError("No source matched the requested single-source run")
        except Exception as exc:
            # Already-fetched work drains normally; a pagination failure is never reported as success.
            self.fetch_error = describe_exception(exc)
            await self.events.emit("fetch", "failed", error=self.fetch_error)
            LOG.error("Fetching stopped: %s", self.fetch_error)
        await self.close_queue("source", self.config.preprocess_workers)

    async def preprocess_worker(self) -> None:
        queue = self.queues["source"]
        while True:
            work = await queue.get()
            try:
                if work is STOP:
                    return
                started = time.monotonic()
                await self.events.emit("preprocess", "started", source_id=work.source_id)
                try:
                    jobs = await self.preprocessor.prepare(work)
                except (ValueError, RuntimeError, httpx.HTTPError) as exc:
                    work.preprocess_seconds = time.monotonic() - started
                    record = await asyncio.to_thread(finish_source, work, self.llm.model,
                                                     self.config, describe_exception(exc))
                    await self.events.emit("preprocess", "failed", source_id=work.source_id,
                                           error=describe_exception(exc))
                    await self.queues["save"].put(record)
                    continue
                work.expected_chunks = len(jobs)
                work.prepared_at = now()
                work.preprocess_seconds = time.monotonic() - started
                self.stats["prepared"] += 1
                self.stats["llm_jobs"] += len(jobs)
                await self.events.emit("preprocess", "success", source_id=work.source_id,
                                       chunks=len(jobs), seconds=work.preprocess_seconds)
                # Feed jobs continuously, without waiting for a whole source/page to finish inference.
                for job in jobs:
                    await self.queues["llm"].put(job)
            finally:
                queue.task_done()

    async def llm_worker(self) -> None:
        queue = self.queues["llm"]
        while True:
            chunk = await queue.get()
            try:
                if chunk is STOP:
                    return
                details = {"source_id": chunk.work.source_id, "chunk_index": chunk.index}
                await self.events.emit("llm", "started", **details)

                async def retry(**info) -> None:
                    await self.events.emit("llm", "retry", **details, **info)

                self.stats["llm_inflight"] += 1
                try:
                    result = await self.llm.generate(chunk, retry)
                finally:
                    self.stats["llm_inflight"] -= 1
                self.stats["llm_completed"] += 1
                await self.queues["postprocess"].put(result)
                await self.events.emit("llm", "failed" if result.error else "response_received",
                                       **details, seconds=result.seconds, attempts=result.attempts,
                                       error=result.error)
                # No parsing, disk write, upload, or source-level aggregation in this worker.
            finally:
                queue.task_done()

    async def postprocess_worker(self) -> None:
        queue = self.queues["postprocess"]
        while True:
            result = await queue.get()
            try:
                if result is STOP:
                    return
                entry = await asyncio.to_thread(process_result, result, self.vocabulary)
                work = result.chunk.work
                # No await in this state transition: one worker claims source completion.
                work.chunks[result.chunk.index] = entry
                complete = len(work.chunks) == work.expected_chunks
                if entry["status"] != "success":
                    self.stats["chunks_failed"] += 1
                tokens = entry.get("usage", {}).get("completion_tokens", 0)
                if type(tokens) is int:
                    self.stats["completion_tokens"] += tokens
                await self.events.emit("postprocess", entry["status"], source_id=work.source_id,
                                       chunk_index=result.chunk.index, error=entry.get("error"))
                if complete:
                    record = await asyncio.to_thread(finish_source, work, self.llm.model, self.config)
                    await self.queues["save"].put(record)
            finally:
                queue.task_done()

    async def save_worker(self) -> None:
        queue = self.queues["save"]
        while True:
            record = await queue.get()
            try:
                if record is STOP:
                    break
                await self.store.save_record(record)
                self.stats["saved"] += 1
                self.source_slots.release()
                await self.events.emit("local_save", "success", source_id=record["source_id"],
                                       extraction_status=record["status"], upload_status=record["upload_status"])
                if not self.config.dry_run and record["status"] == "success":
                    # Only IDs are queued for upload; the payload already exists on disk.
                    await self.queues["upload"].put(record["source_id"])
            finally:
                queue.task_done()
        if not self.config.dry_run:
            await self.close_queue("upload", self.config.upload_workers)

    async def upload_worker(self) -> None:
        queue = self.queues["upload"]
        while True:
            source_id = await queue.get()
            try:
                if source_id is STOP:
                    return
                record = await self.store.read_record(source_id)
                # A persisted 'started' receipt signals an uncertain outcome after a hard crash.
                await self.store.save_receipt({"source_id": source_id, "run_id": record["run_id"],
                                                "status": "started", "started_at": now()})
                await self.events.emit("upload", "started", source_id=source_id, run_id=record["run_id"])
                receipt = await submit_record_run(self.upload_client, record["record_run_payload"], self.config)
                receipt["source_id"] = source_id
                await self.queues["receipt"].put(receipt)
            finally:
                queue.task_done()

    async def receipt_worker(self) -> None:
        queue = self.queues["receipt"]
        while True:
            receipt = await queue.get()
            try:
                if receipt is STOP:
                    return
                await self.store.save_receipt(receipt)
                self.stats["uploaded" if receipt["status"] == "success" else "upload_errors"] += 1
                await self.events.emit("upload", receipt["status"], source_id=receipt["source_id"],
                                       run_id=receipt["run_id"], seconds=receipt["seconds"])
            finally:
                queue.task_done()

    async def monitor(self) -> None:
        interval = self.config.progress_seconds
        if not interval:
            await self.monitor_stop.wait()
            return
        while not self.monitor_stop.is_set():
            try:
                await asyncio.wait_for(self.monitor_stop.wait(), timeout=interval)
            except TimeoutError:
                LOG.info("fetched=%d saved=%d LLM completed=%d active=%d queued=%d uploaded=%d errors=%d",
                         self.stats["fetched"], self.stats["saved"], self.stats["llm_completed"],
                         self.stats["llm_inflight"], self.queues["llm"].qsize(),
                         self.stats["uploaded"], self.stats["chunks_failed"] + self.stats["upload_errors"])


async def run_pipeline(config: Config, *, source_transport=None, llm_transport=None,
                       upload_transport=None, store: Store | None = None) -> dict:
    """Optional transports/store support deterministic, network-free integration tests."""
    def headers(token: str) -> dict:
        return {"Authorization": f"Bearer {token}"} if token else {}

    limits = httpx.Limits(max_connections=max(16, config.preprocess_workers + config.llm_workers + 4),
                         max_keepalive_connections=max(8, config.llm_workers + 2))
    # Separate pools prevent slow uploads from taking the LLM's connections.
    async with (
        httpx.AsyncClient(timeout=config.http_timeout, headers=headers(config.source_api_token),
                          transport=source_transport, follow_redirects=True) as source_client,
        httpx.AsyncClient(timeout=config.llm_timeout, headers=headers(config.llm_api_key),
                          transport=llm_transport, limits=limits, trust_env=False) as llm_client,
        httpx.AsyncClient(timeout=config.http_timeout, headers=headers(config.record_api_token),
                          transport=upload_transport, follow_redirects=False) as upload_client,
    ):
        return await Pipeline(config, source_client, llm_client, upload_client, store).run()
