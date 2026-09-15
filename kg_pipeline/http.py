"""Shared retry policy for reads and repeatable inference requests only."""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

RETRYABLE = {408, 429, 500, 502, 503, 504}


class RequestFailed(RuntimeError):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


async def request(
    client: httpx.AsyncClient, method: str, url: str, *, attempts: int,
    retry_delay: float, on_retry: Callable[..., Awaitable[None]] | None = None,
    **kwargs: Any,
) -> tuple[httpx.Response, int]:
    for attempt in range(1, attempts + 1):
        wait = min(15, retry_delay * 2 ** (attempt - 1))
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            error = f"{type(exc).__name__}: {exc}"
            retryable = True
        else:
            if response.is_success:
                return response, attempt
            error = f"HTTP {response.status_code}: {response.text[:4000]}"
            retryable = response.status_code in RETRYABLE
            try:
                wait = max(wait, min(60, float(response.headers.get("Retry-After", "0"))))
            except ValueError:
                pass
        if not retryable or attempt == attempts:
            raise RequestFailed(error, attempt)
        if on_retry:
            await on_retry(attempt=attempt, error=error)
        await asyncio.sleep(wait + random.uniform(0, min(0.25, wait / 4)))
    raise AssertionError("attempts must be positive")


async def get_json(client: httpx.AsyncClient, url: str, *, attempts: int,
                   retry_delay: float, **kwargs: Any) -> Any:
    response, _ = await request(client, "GET", url, attempts=attempts,
                                retry_delay=retry_delay, **kwargs)
    # Source pages can be large. Decode outside the LLM event loop.
    return await asyncio.to_thread(response.json)
