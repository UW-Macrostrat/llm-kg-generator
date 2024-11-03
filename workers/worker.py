import asyncio
import argparse
import logging
import sys
import os
from workers.wrapper_classes.worker_wrapper import Worker, Metadata
import workers.pb.job_manager_pb2 as pb
from typing import Callable, Awaitable
import workers.handlers.weaviate.weaviate_worker as weaviate_worker
import workers.handlers.map_descriptions.map_worker as map_worker
import workers.handlers.test.test_worker as test_worker


class Handler:
    ctx: dict[str, any]
    startup: Callable[[dict], Awaitable[None]]
    shutdown: Callable[[dict], Awaitable[None]]
    process_job: Callable[[dict, any, Metadata, bool], Awaitable[dict | None]]
    has_initialized: bool

    def __init__(
        self,
        startup: Callable[[dict], Awaitable[None]],
        shutdown: Callable[[dict], Awaitable[None]],
        process_job: Callable[[dict, any, Metadata, bool], Awaitable[dict | None]],
    ):
        self.startup = startup
        self.shutdown = shutdown
        self.process_job = process_job
        self.has_initialized = False

    async def initialize_ctx(self) -> None:
        self.ctx = {}
        await self.startup(self.ctx)

    async def ensure_intialized(self) -> None:
        if not self.has_initialized:
            await self.initialize_ctx()
            self.has_initialized = True

    async def shutdown_if_initialized(self) -> None:
        if self.has_initialized:
            await self.shutdown(self.ctx)


async def route_handlers(ctx: dict, job: pb.GetJobResponse, metadata: Metadata, return_results: bool = False) -> None:
    job_type = job.WhichOneof("job_data")
    if job_type not in ctx["handlers"]:
        logging.fatal("invalid job type: %s", job)
        raise Exception("invalid job")
    job_data = getattr(job, job_type)
    handler = ctx["handlers"][job_type]
    await handler.ensure_intialized()
    await handler.process_job(handler.ctx, job_data, metadata, return_results)


async def shutdown_handlers(ctx: dict) -> None:
    for handler in ctx["handlers"].values():
        await handler.shutdown_if_initialized()


async def run_workers(worker_count: int) -> None:
    ctx = {
        "handlers": await create_handlers(),
    }
    worker = Worker(os.getenv("MANAGER_HOST"), ctx=ctx)
    try:
        await worker.run_pool(route_handlers, worker_count)
    except Exception as e:
        logging.exception("shutting down after error occurred: %s", e)
        await shutdown_handlers(ctx)


async def create_handlers() -> dict[str, Handler]:
    handlers = {
        "weaviate_data": Handler(startup=weaviate_worker.startup, shutdown=weaviate_worker.shutdown, process_job=weaviate_worker.process_paragraphs),
        "map_description_data": Handler(startup=map_worker.startup, shutdown=map_worker.shutdown, process_job=map_worker.process_descriptions),
        "test_data": Handler(startup=test_worker.startup, shutdown=test_worker.shutdown, process_job=test_worker.process_text),
    }
    return handlers


async def main(worker_count: int):
    logging.basicConfig(format="[%(asctime)s] %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    await run_workers(worker_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-count", type=int, default=4, help="Number of workers to run")

    args = parser.parse_args()
    asyncio.run(main(worker_count=args.worker_count))
