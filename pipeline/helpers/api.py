"""Read sources/types, retry repeatable requests, and POST record_run exactly once."""
import time

import httpx

from .logging import now

RETRYABLE = {408, 429, 500, 502, 503, 504}


class RequestFailed(RuntimeError):
    def __init__(self, message, attempts, status_code=None, response_excerpt=None):
        super().__init__(message)
        self.attempts, self.status_code, self.response_excerpt = attempts, status_code, response_excerpt


def auth(token) -> dict:
    value = token.get_secret_value()
    return {"Authorization": f"Bearer {value}"} if value else {}


def request(context, method: str, url: str, *, stage: str, source_id=None, **kwargs) -> tuple[httpx.Response, int]:
    """For GETs and repeatable model inference only. record_run does not use this."""
    for attempt in range(1, context.config.attempts + 1):
        try:
            response = context.http.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            failure = RequestFailed(f"{method} request failed: {type(exc).__name__}: {exc}", attempt)
            retryable = True
            delay = context.config.retry_delay * 2 ** (attempt - 1)
        else:
            if response.is_success:
                return response, attempt
            failure = RequestFailed(f"{method} returned HTTP {response.status_code}", attempt,
                                    response.status_code, response.text[:4000])
            retryable = response.status_code in RETRYABLE
            delay = context.config.retry_delay * 2 ** (attempt - 1)
            try:
                delay = max(delay, float(response.headers.get("Retry-After", "0")))
            except ValueError:
                pass
        if not retryable or attempt == context.config.attempts:
            raise failure
        context.log.event("WARNING", "request_retry", stage=stage, source_id=source_id,
                          attempt=attempt, message=str(failure), status_code=failure.status_code)
        time.sleep(min(30, delay))
    raise AssertionError("attempts must be positive")


def get_rows(context, url: str, params: dict, label: str) -> list[dict]:
    response, _ = request(context, "GET", str(httpx.URL(url).copy_merge_params(params)),
                          stage="pull_resources", headers=auth(context.config.source_api_token), follow_redirects=True)
    try:
        rows = response.json()
    except ValueError as exc:
        raise ValueError(f"{label} did not return JSON: {response.text[:1000]}") from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must return an array of objects")
    return rows


def validate_ids(rows: list[dict], previous: int | None):
    for row in rows:
        value = row.get("id")
        if type(value) is not int or value < 0 or (previous is not None and value <= previous):
            raise ValueError("API IDs must be nonnegative integers in strictly increasing order")
        previous = value


def fetch_types(context, url: str, label: str) -> list[dict]:
    rows, cursor = [], None
    while True:
        params = {"order": "id.asc", "limit": 100}
        if cursor is not None:
            params["id"] = f"gt.{cursor}"
        page = get_rows(context, url, params, label)
        if not page:
            return rows
        validate_ids(page, cursor)
        rows.extend(page)
        cursor = page[-1]["id"]


def fetch_sources(context) -> list[dict]:
    config = context.config
    forbidden = {"id", "and", "or", "order", "limit", "offset", "select"}
    if forbidden.intersection(httpx.URL(config.source_url).params.keys()):
        raise ValueError("source_url contains selection parameters; use --source-id, --after-id, or --max")
    rows = []
    if config.source_ids:
        for source_id in sorted(set(config.source_ids)):
            page = get_rows(context, config.source_url, {"id": f"eq.{source_id}", "limit": 1}, "source_text")
            if len(page) != 1 or type(page[0].get("id")) is not int or page[0]["id"] != source_id:
                raise ValueError(f"source_text did not return requested ID {source_id}")
            rows.extend(page)
        return rows
    cursor = config.after_id
    while len(rows) < config.max_sources:
        limit = min(100, config.max_sources - len(rows))
        params = {"order": "id.asc", "limit": limit}
        if cursor is not None:
            params["id"] = f"gt.{cursor}"
        page = get_rows(context, config.source_url, params, "source_text")
        if not page:
            break
        if len(page) > limit:
            raise ValueError("source_text ignored the requested limit")
        validate_ids(page, cursor)
        rows.extend(page)
        cursor = page[-1]["id"]
    return rows


def submit_record_run(context, payload: dict) -> dict:
    if context.config.dry_run:
        raise RuntimeError("Uploads are disabled in dry-run mode")
    started = time.monotonic()
    receipt = {"run_id": payload["run_id"], "started_at": now(), "attempts": 1}
    try:
        response = context.http.post(context.config.record_run_url, json=payload,
                                     headers=auth(context.config.record_api_token), follow_redirects=False)
    except httpx.TransportError as exc:
        # The server may have committed before a timeout. Do not automatically retry.
        receipt.update(status="unknown", error=f"{type(exc).__name__}: {exc}")
    else:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        status = "success" if response.is_success else ("unknown" if response.status_code >= 500 else "failed")
        receipt.update(status=status, status_code=response.status_code, body=body)
    receipt.update(finished_at=now(), seconds=time.monotonic() - started)
    return receipt
