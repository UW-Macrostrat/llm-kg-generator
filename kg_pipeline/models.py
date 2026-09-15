"""Messages passed between stages; a source is one source_text row."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

STOP = object()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceWork:
    source: dict[str, Any]
    run_id: str
    fetched_at: str = field(default_factory=now)
    prepared_at: str | None = None
    preprocess_seconds: float = 0
    expected_chunks: int = 0
    # Only the event-loop thread mutates this dictionary, without awaiting.
    chunks: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def source_id(self) -> int:
        return self.source["id"]


@dataclass
class PreparedChunk:
    work: SourceWork
    index: int
    start: int
    end: int
    prompt_tokens: int
    body: dict[str, Any]


@dataclass
class LLMResult:
    chunk: PreparedChunk
    response: bytes | None
    started_at: str
    finished_at: str
    seconds: float
    attempts: int
    error: str | None = None
