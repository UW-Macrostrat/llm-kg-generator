import asyncio
import httpx
from typing import Callable, Awaitable
import logging
import time
import os
import threading
from contextlib import contextmanager


@contextmanager
def deferred(func):
    try:
        yield
    finally:
        func()


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

    async def run_pool(
        self,
        worker_func: Callable[[dict, list[str], dict, bool], Awaitable[dict | None]],
        pool_count: int,
    ):
        workers = []
        for _ in range(pool_count):
            workers.append(asyncio.create_task(self.run(worker_func)))
        await asyncio.gather(*workers)

    async def run(self, worker_func: Callable[[dict, list[str], dict, bool], Awaitable[dict | None]]):
        timeout_count = 0
        async with httpx.AsyncClient() as self.httpx_client:
            while timeout_count < 5:
                start_time = time.time()
                # request job from manager
                try:
                    response = await self.httpx_client.post(f"{self.manager}/request_job")
                    timeout_count = 0
                except httpx.HTTPError:
                    timeout_count += 1
                    logging.error(
                        "Job request failed. Retrying in %s seconds... [%s/5]",
                        self.retry_after,
                        timeout_count,
                    )
                    await asyncio.sleep(self.retry_after)
                    continue

                job_data = response.json()
                if job_data["JobType"] == "wait":
                    await asyncio.sleep(self.retry_after)
                    continue

                # start sending health checks to manager
                stop_event = {"stop": False}
                health_task = threading.Thread(
                    target=self.health_check,
                    args=(job_data["ID"], job_data["HealthTimeout"], stop_event),
                )
                health_task.start()

                def stop():
                    stop_event["stop"] = True

                # start worker function and stop health task once it is done
                with deferred(stop):
                    logging.info("Starting job id %s", job_data["ID"])
                    need_return = job_data["JobType"] == "on_demand"
                    try:
                        result = await worker_func(self.ctx, job_data["JobData"], job_data["Metadata"], need_return)
                    except Exception as e:
                        logging.error("Job id %s failed", job_data["ID"])
                        raise e

                # tell manager that job has finished and send return values if needed
                elapsed_time = time.time() - start_time
                try:
                    if need_return:
                        await self.httpx_client.post(
                            f"{self.manager}/finish_job",
                            headers={"Content-Type": "application/json"},
                            json={"ID": job_data["ID"], "Result": result},
                        )
                    else:
                        await self.httpx_client.post(
                            f"{self.manager}/finish_job",
                            headers={"Content-Type": "application/json"},
                            json={"ID": job_data["ID"]},
                        )
                except httpx.HTTPError:
                    logging.error("Finish message failed.")
                    continue

                logging.info(
                    "Finished job id %s after %s seconds",
                    job_data["ID"],
                    round(elapsed_time, 2),
                )

            logging.info("Worker done.")

    def health_check(self, job_id: int, health_timeout: int, stop_event: dict):
        timeout = httpx.Timeout(10.0, read=None)
        timeout_count = 0
        with httpx.Client() as client:
            while not stop_event["stop"]:
                try:
                    client.post(
                        f"{self.manager}/health_check",
                        headers={"Content-Type": "application/json"},
                        json={"ID": job_id},
                        timeout=timeout,
                    )
                    timeout_count = 0
                except httpx.HTTPError:
                    timeout_count += 1
                    logging.error("Health check timed out. [%s/5]", timeout_count)
                    if timeout_count == 5:
                        return

                time.sleep(health_timeout)


async def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    async def worker_func(ctx: dict, id_list: list[str], metadata: dict, need_return: bool) -> dict:
        await asyncio.sleep(5)
        print("job metadata:", metadata)
        print("job data:", id_list)

        if need_return:
            return {"test": id_list}

    worker = Worker(os.getenv("MANAGER_HOST"))
    await worker.run_pool(worker_func, 2)


if __name__ == "__main__":
    asyncio.run(main())
