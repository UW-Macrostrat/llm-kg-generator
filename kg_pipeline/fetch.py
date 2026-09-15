"""Read-only PostgREST pagination with an ascending integer ID cursor."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Config
from .http import get_json


def require_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} must return an array of objects")
    return value


class SourceReader:
    def __init__(self, client: httpx.AsyncClient, config: Config):
        self.client, self.config = client, config
        self.high_water_id: int | None = None
        self.pages = 0
        url = httpx.URL(config.source_url)
        # A leftover id=eq.19744 from the original script must not silently limit a full run.
        forbidden = {"id", "order", "limit", "offset", "select", "and"}
        if forbidden.intersection(url.params.keys()):
            raise ValueError("SOURCE_TEXT_URL must not contain id/order/limit/offset/select/and; "
                             "use --source-id or --after-id for ID selection")
        self.url = url

    async def _rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = self.url.copy_merge_params(params)
        result = await get_json(self.client, str(url), attempts=self.config.read_attempts,
                                retry_delay=self.config.retry_delay)
        return require_rows(result, "source_text")

    async def batches(self) -> AsyncIterator[list[dict[str, Any]]]:
        cfg = self.config
        if cfg.source_id is not None:
            requested_ids = sorted(set(cfg.source_id))

            params: dict[str, Any] = {
                "order": "id.asc",
                "limit": len(requested_ids),
                "id": f"in.({','.join(map(str, requested_ids))})",
            }

            rows = await self._rows(params)

            if len(rows) > len(requested_ids):
                raise ValueError("source_text returned more rows than requested")

            self._validate_ids(rows, None)

            returned_ids = [row["id"] for row in rows]
            missing_ids = sorted(set(requested_ids) - set(returned_ids))

            if missing_ids:
                raise ValueError(
                    f"source_text did not return requested IDs: {missing_ids}"
                )

            self.pages += 1
            yield rows
            return

        if cfg.test:
            params: dict[str, Any] = {
                "order": "id.asc",
                "limit": 1,
            }

            if cfg.after_id is not None:
                params["id"] = f"gt.{cfg.after_id}"

            rows = await self._rows(params)

            if len(rows) > 1:
                raise ValueError("source_text ignored limit=1")

            self._validate_ids(rows, cfg.after_id)

            if rows:
                self.pages += 1
                yield rows

            return

        bounds = await self._rows({"select": "id", "order": "id.desc", "limit": 1})
        if not bounds:
            return
        self._validate_ids(bounds, None)
        self.high_water_id = bounds[0]["id"]
        cursor = cfg.after_id
        remaining = cfg.max_sources

        while cursor is None or cursor < self.high_water_id:
            if remaining is not None and remaining <= 0:
                break

            page_size = cfg.batch_size
            if remaining is not None:
                page_size = min(page_size, remaining)

            condition = f"id.lte.{self.high_water_id}"
            if cursor is not None:
                condition += f",id.gt.{cursor}"

            rows = await self._rows({
                "order": "id.asc",
                "limit": page_size,
                "and": f"({condition})",
            })

            if not rows:
                break

            if len(rows) > page_size:
                raise ValueError("source_text ignored the requested page size")

            self._validate_ids(rows, cursor)

            if rows[-1]["id"] > self.high_water_id:
                raise ValueError("source_text ignored the high-water ID filter")

            cursor = rows[-1]["id"]
            self.pages += 1

            if remaining is not None:
                remaining -= len(rows)

            yield rows

    # Continue past short pages: the server may enforce a smaller limit.

    @staticmethod
    def _validate_ids(rows: list[dict], previous: int | None) -> None:
        for row in rows:
            value = row.get("id")
            if type(value) is not int or value < 0:
                raise ValueError("source_text.id must be a non-negative integer")
            if previous is not None and value <= previous:
                raise ValueError("Source IDs did not advance; refusing an infinite/duplicate scan")
            previous = value


async def fetch_types(client: httpx.AsyncClient, url: str, config: Config) -> list[dict]:
    # Vocabularies can also be capped by PostgREST; page them instead of assuming one response.
    rows: list[dict] = []
    cursor: int | None = None
    endpoint = httpx.URL(url)
    while True:
        params: dict[str, Any] = {"order": "id.asc", "limit": 100}
        if cursor is not None:
            params["id"] = f"gt.{cursor}"
        page = require_rows(await get_json(
            client, str(endpoint.copy_merge_params(params)), attempts=config.read_attempts,
            retry_delay=config.retry_delay), "type definitions")
        if not page:
            break
        SourceReader._validate_ids(page, cursor)
        rows.extend(page)
        cursor = page[-1]["id"]
    if not rows:
        raise ValueError("The type-definition endpoint returned no records")
    return rows
