import logging
import asyncio
import workers.pb.job_manager_pb2 as pb


async def startup(ctx: dict):
    logging.info("starting up")


async def shutdown(ctx: dict):
    logging.info("shutting down")


async def process_text(
    ctx: dict,
    job_data: pb.TestJob,
    run_metadata: dict,
    return_results: bool = False,
) -> dict | None:
    logging.info("processing job %s", job_data)
    await asyncio.sleep(3)
