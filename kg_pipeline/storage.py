"""Incremental local persistence and a final JSON assembled with bounded memory."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


async def disk_call(function: Callable, *args: Any) -> Any:
    """Finish an in-progress write before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root: Path):
        self.root = root

    async def initialize(self, metadata: dict) -> None:
        def create() -> None:
            # Exclusive creation avoids accidental reuse and duplicate/mixed run records.
            self.root.mkdir(parents=True, exist_ok=False)
            (self.root / "records").mkdir()
            (self.root / "receipts").mkdir()
            (self.root / "run_steps.jsonl").touch(exist_ok=False)
            atomic_json(self.root / "run.json", metadata)
        await disk_call(create)

    def record_path(self, source_id: int) -> Path:
        return self.root / "records" / f"source_{source_id}.json"

    def receipt_path(self, source_id: int) -> Path:
        return self.root / "receipts" / f"source_{source_id}.json"

    async def save_record(self, record: dict) -> None:
        await disk_call(atomic_json, self.record_path(record["source_id"]), record)

    async def read_record(self, source_id: int) -> dict:
        return await disk_call(read_json, self.record_path(source_id))

    async def save_receipt(self, receipt: dict) -> None:
        await disk_call(atomic_json, self.receipt_path(receipt["source_id"]), receipt)

    async def append_events(self, events: list[dict]) -> None:
        def append() -> None:
            with (self.root / "run_steps.jsonl").open("a", encoding="utf-8") as stream:
                for event in events:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
        await disk_call(append)

    async def save_metadata(self, metadata: dict) -> None:
        await disk_call(atomic_json, self.root / "run.json", metadata)

    async def finalize(self, summary: dict) -> dict:
        """Retain individual records even if final JSON assembly is interrupted."""
        def assemble() -> dict:
            counts = {"saved": 0, "successful_extractions": 0, "failed_sources": 0,
                      "uploaded": 0, "upload_failed": 0, "upload_unknown": 0,
                      "upload_pending": 0, "dry_run_saved": 0}
            temporary = self.root / "results.json.tmp"
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write('{"run":')
                json.dump(read_json(self.root / "run.json"), stream, ensure_ascii=False)
                stream.write(',"results":[')
                # Iterate files, not all result data. Output order is unspecified.
                first = True
                for path in (self.root / "records").glob("source_*.json"):
                    record = read_json(path)
                    receipt_path = self.receipt_path(record["source_id"])
                    if receipt_path.exists():
                        receipt = read_json(receipt_path)
                        record["record_run_response"] = receipt
                        record["upload_status"] = receipt["status"]
                    status = record["upload_status"]
                    counts["saved"] += 1
                    counts["successful_extractions" if record["status"] == "success" else "failed_sources"] += 1
                    counter = {"success": "uploaded", "failed": "upload_failed", "unknown": "upload_unknown",
                               "started": "upload_unknown", "pending": "upload_pending",
                               "skipped_dry_run": "dry_run_saved"}.get(status)
                    if counter:
                        counts[counter] += 1
                    if not first:
                        stream.write(",")
                    json.dump(record, stream, ensure_ascii=False)
                    first = False
                final = {**summary, **counts}
                final["sources_without_saved_result"] = max(0, final.get("fetched", 0) - counts["saved"])
                if final["status"] == "completed" and any(final[k] for k in (
                    "failed_sources", "upload_failed", "upload_unknown", "upload_pending", "sources_without_saved_result"
                )):
                    final["status"] = "completed_with_errors"
                stream.write('],"summary":')
                json.dump(final, stream, ensure_ascii=False)
                stream.write("}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / "results.json")
            atomic_json(self.root / "summary.json", final)
            return final
        return await disk_call(assemble)
