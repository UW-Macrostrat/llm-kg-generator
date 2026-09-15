"""Run-step recording has its own queue and batches disk writes."""
from __future__ import annotations

import asyncio

from .models import STOP, now
from .storage import Store


class Events:
    def __init__(self, store: Store, maxsize: int):
        self.store = store
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def emit(self, stage: str, status: str, **details) -> None:
        await self.queue.put({"timestamp": now(), "stage": stage, "status": status, **details})

    async def run(self) -> None:
        while True:
            item = await self.queue.get()
            if item is STOP:
                self.queue.task_done()
                return
            batch = [item]
            stopping = False
            while len(batch) < 128:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is STOP:
                    self.queue.task_done()
                    stopping = True
                    break
                batch.append(item)
            await self.store.append_events(batch)
            for _ in batch:
                self.queue.task_done()
            if stopping:
                return
