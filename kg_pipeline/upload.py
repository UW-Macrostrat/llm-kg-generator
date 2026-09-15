"""Exactly one application-level POST attempt per successfully extracted source."""
from __future__ import annotations

import asyncio
import time

import httpx

from .config import Config
from .models import now


async def submit_record_run(client: httpx.AsyncClient, payload: dict, config: Config) -> dict:
    if config.dry_run:
        # Defense in depth: dry runs never reach the remote write call.
        raise RuntimeError("Upload is disabled by --dry-run")
    started_at, started = now(), time.monotonic()
    result = {"run_id": payload["run_id"], "started_at": started_at, "attempts": 1}
    try:
        response = await client.post(config.record_run_url, json=payload, follow_redirects=False)
    except httpx.TransportError as exc:
        # A timeout may occur after the server committed. Retrying could duplicate data.
        result.update(status="unknown", error=f"{type(exc).__name__}: {exc}")
    else:
        try:
            body = await asyncio.to_thread(response.json)
        except ValueError:
            body = response.text
        # A server failure can also happen after commit; its outcome needs reconciliation.
        status = "success" if response.is_success else ("unknown" if response.status_code >= 500 else "failed")
        result.update(status=status, status_code=response.status_code,
                      content_type=response.headers.get("content-type"), body=body)
    result.update(finished_at=now(), seconds=time.monotonic() - started)
    return result
