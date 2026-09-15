"""Validate source metadata and create context-sized, offset-preserving jobs."""
from __future__ import annotations

import asyncio
import re

from .config import Config
from .llm import VLLMClient
from .models import PreparedChunk, SourceWork
from .prompts import Vocabulary

REQUIRED_SOURCE_FIELDS = (
    "id",
    "source_text_type",
    "paragraph_text",
)

def validate_source(source: dict) -> str:
    missing = [
        field
        for field in REQUIRED_SOURCE_FIELDS
        if field not in source
    ]

    if missing:
        raise ValueError(
            "Missing source fields: " + ", ".join(missing)
        )

    text = source["paragraph_text"]

    if not isinstance(text, str) or not text.strip():
        raise ValueError("paragraph_text must be a nonempty string")

    return text


class Preprocessor:
    def __init__(self, config: Config, llm: VLLMClient, vocabulary: Vocabulary):
        self.config, self.llm, self.vocabulary = config, llm, vocabulary
        self.input_budget = llm.context_length - config.max_tokens - config.token_margin

    async def preflight(self) -> None:
        base_tokens = await self.llm.count_tokens(self.vocabulary.messages(""))
        if base_tokens >= self.input_budget:
            raise ValueError(
                f"System prompt alone uses {base_tokens} tokens; only {self.input_budget} input "
                "tokens remain. Reduce --max-tokens, shorten type descriptions, or increase "
                "the actual vLLM context length."
            )

    async def prepare(self, work: SourceWork) -> list[PreparedChunk]:
        text = await asyncio.to_thread(validate_source, work.source)
        spans = await self.split(text)
        jobs = []
        for index, (start, end, count) in enumerate(spans):
            body = {
                "model": self.llm.model,
                "messages": self.vocabulary.messages(text[start:end]),
                "temperature": 0.7, "top_p": 0.8,
                "max_tokens": self.config.max_tokens,
                "top_k": 20, "min_p": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
                "structured_outputs": {"json": self.vocabulary.schema},
            }
            jobs.append(PreparedChunk(work, index, start, end, count, body))
        return jobs

    async def split(self, text: str) -> list[tuple[int, int, int]]:
        """Measure the exact chat prompt; never truncate the source to make it fit.

        The tokenizer may be imperfectly monotonic, so every emitted span has
        an observed fitting count. Binary search need not find the largest fit.
        """
        spans = []
        start = 0
        while start < len(text):
            async def count(end: int) -> int:
                return await self.llm.count_tokens(self.vocabulary.messages(text[start:end]))

            end = len(text)
            tokens = await count(end)
            if tokens > self.input_budget:
                lo, hi = start + 1, end - 1
                best: tuple[int, int] | None = None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    candidate_tokens = await count(mid)
                    if candidate_tokens <= self.input_budget:
                        best = mid, candidate_tokens
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best is None:
                    raise ValueError("Cannot fit even one source character in the context budget")
                end, tokens = best
                # Prefer a word boundary near the fitted end, while retaining exact characters.
                boundary_start = start + int((end - start) * 0.85)
                boundaries = list(re.finditer(r"\s+", text[boundary_start:end]))
                if boundaries:
                    candidate = boundary_start + boundaries[-1].end()
                    candidate_tokens = await count(candidate)
                    if candidate > start and candidate_tokens <= self.input_budget:
                        end, tokens = candidate, candidate_tokens
            spans.append((start, end, tokens))
            if end == len(text):
                break
            overlap = min(self.config.chunk_overlap_chars, (end - start) // 4)
            next_start = end - overlap
            if overlap:
                boundary = re.search(r"\s+", text[next_start:end])
                if boundary:
                    next_start += boundary.end()
            start = max(start + 1, next_start)
        return spans
