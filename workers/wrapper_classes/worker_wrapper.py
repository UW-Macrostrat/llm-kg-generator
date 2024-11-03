import asyncio
import httpx
import grpc
from typing import Callable, Awaitable
import logging
import time
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import workers.pb.job_manager_pb2 as pb
import workers.pb.job_manager_pb2_grpc as pb_grpc


@contextmanager
def deferred(func):
    try:
        yield
    finally:
        func()


@dataclass
class Metadata:
    run_id: str
    pipeline_id: str


class Worker:
    def __init__(
        self,
        manager: str,
        ctx: dict = {},
        retry_after: int = 5,
    ) -> None:
        self.manager = manager
        self.ctx = ctx
        self.retry_after = retry_after
        
        options = [
            ('grpc.keepalive_time_ms', 10_000),
            ('grpc.keepalive_timeout_ms', 1_000)  
        ]
        self.channel = grpc.insecure_channel(manager, options=options)
        self.stub = pb_grpc.JobManagerStub(self.channel)

    async def run_pool(
        self,
        worker_func: Callable[[dict, pb.GetJobResponse, Metadata, bool], Awaitable[dict | None]],
        pool_count: int,
    ):
        workers = []
        for _ in range(pool_count):
            workers.append(asyncio.create_task(self.run(worker_func)))
        await asyncio.gather(*workers)

    async def run(self, worker_func: Callable[[dict, pb.GetJobResponse, Metadata, bool], Awaitable[dict | None]]):
        await self.get_metadata()

        timeout_count = 0
        while timeout_count < 5:
            start_time = time.time()
            # request job from manager
            try:
                response = self.stub.GetJob(pb.GetJobRequest())
                timeout_count = 0
            except grpc.RpcError:
                timeout_count += 1
                logging.error(
                    "Job request failed. Retrying in %s seconds... [%s/5]",
                    self.retry_after,
                    timeout_count,
                )
                await asyncio.sleep(self.retry_after)
                continue

            job_data = response
            if job_data.type == pb.JobType.wait:
                await asyncio.sleep(self.retry_after)
                continue

            # start sending health checks to manager
            stop_event = {"stop": False}
            health_task = threading.Thread(
                target=self.health_update,
                args=(job_data.id, stop_event),
            )
            health_task.start()

            def stop():
                stop_event["stop"] = True

            # start worker function and stop health task once it is done
            # need_return = job_data.type == job_manager_pb2.JobType.on_demand TODO: implement
            need_return = False
            with deferred(stop):
                logging.info("Starting job id %s", job_data.id)
                try:
                    result = await worker_func(self.ctx, job_data, self.metadata, need_return)
                except Exception as e:
                    logging.error("Job id %s failed", job_data.id)
                    raise e

            # tell manager that job has finished and send return values if needed
            elapsed_time = time.time() - start_time
            try:
                if need_return:
                    # await self.httpx_client.post(
                    #     f"{self.manager}/finish_job",
                    #     headers={"Content-Type": "application/json"},
                    #     json={"ID": job_data["ID"], "Result": result},
                    # )
                    pass
                else:
                    self.stub.FinishJob(
                        pb.FinishJobRequest(
                            id=job_data.id,
                        )
                    )
            except grpc.RpcError:
                logging.error("Finish message failed.")
                continue

            logging.info(
                "Finished job id %s after %s seconds",
                job_data.id,
                round(elapsed_time, 2),
            )

        logging.info("Worker done.")

    async def get_metadata(self):
        retries = 0
        while retries < 5:
            try:
                response = self.stub.GetMetadata(pb.GetMetadataRequest())
                self.health_timeout = response.health_timeout
                self.metadata = Metadata(run_id=response.run_id, pipeline_id=response.pipeline_id)
                return
            except grpc.RpcError:
                retries += 1
                logging.error(
                    "Job request failed. Retrying in %s seconds... [%s/5]",
                    self.retry_after,
                    retries,
                )
                await asyncio.sleep(self.retry_after)
        logging.fatal("failed to retrieve run metadata")
        raise Exception("failed to retrieve run metadata")

    def health_update(self, job_id: int, stop_event: dict):
        timeout_count = 0
        while not stop_event["stop"]:
            try:
                self.stub.UpdateHealth(
                    pb.UpdateHealthRequest(
                        id=job_id,
                    )
                )
                timeout_count = 0
            except grpc.RpcError:
                timeout_count += 1
                logging.error("health check timed out. [%s/5]", timeout_count)
                if timeout_count == 5:
                    return

            time.sleep(self.health_timeout / 4)


async def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    async def worker_func(ctx: dict, job_data: pb.GetJobResponse, metadata: Metadata, need_return: bool) -> dict:
        await asyncio.sleep(5)
        print("job metadata:", metadata)
        print("job data:", job_data)

        if need_return:
            return {"test": job_data}

    worker = Worker(os.getenv("MANAGER_HOST"))
    await worker.run_pool(worker_func, 1)


if __name__ == "__main__":
    asyncio.run(main())
