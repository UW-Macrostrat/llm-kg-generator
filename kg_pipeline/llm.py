"""Only network inference here. Parsing and validation belong downstream."""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import httpx

from .config import Config
from .http import RequestFailed, get_json, request
from .models import LLMResult, PreparedChunk, now


class VLLMClient:
    def __init__(self, client: httpx.AsyncClient, config: Config):
        self.client, self.config = client, config
        self.base_url = config.base_url.rstrip("/")
        self.model = ""
        self.context_length = 0

    async def initialize(self) -> None:
        models = await get_json(
            self.client,
            f"{self.base_url}/models",
            attempts=self.config.read_attempts,
            retry_delay=self.config.retry_delay,
        )

        data = models.get("data") if isinstance(models, dict) else None

        if not isinstance(data, list) or not data:
            raise ValueError("LLM server returned no served models from /models")

        selected = next(
            (item for item in data if item.get("id") == self.config.model),
            None,
        )

        if selected is None:
            if self.config.model:
                raise ValueError(
                    f"Model {self.config.model!r} is not served; available: "
                    f"{[item.get('id') for item in data]}"
                )
            selected = data[0]

        self.model = selected["id"]

        actual_limit = selected.get("max_model_len")
        if type(actual_limit) is not int or actual_limit <= 0:
            actual_limit = None

        if (
            actual_limit
            and self.config.context_length
            and self.config.context_length > actual_limit
        ):
            raise ValueError(
                f"Requested context {self.config.context_length} "
                f"exceeds server limit {actual_limit}"
            )

        self.context_length = (
            self.config.context_length
            or actual_limit
            or 2048
        )

    async def count_tokens(self, messages: list[dict]) -> int:
        # Open WebUI does not expose vLLM's /tokenize endpoint.
        # Temporary approximation.
        text = "\n".join(
            str(message.get("content", ""))
            for message in messages
        )

        return max(1, len(text) // 4)

    async def generate(
        self,
        chunk: PreparedChunk,
        on_retry: Callable[..., Awaitable[None]],
    ) -> LLMResult:

        started_at, started = now(), time.monotonic()

        body = dict(chunk.body)

        body["model"] = self.model

        messages = [
            dict(message)
            for message in body.get("messages", [])
        ]

        # Disable Qwen3 thinking.
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                content = str(messages[i].get("content", ""))

                if not content.startswith("/no_think"):
                    messages[i]["content"] = f"/no_think\n{content}"

                break

        body["messages"] = messages
        body["reasoning_effort"] = "low"

        try:
            response, attempts = await request(
                self.client,
                "POST",
                f"{self.base_url}/chat/completions",
                attempts=self.config.llm_attempts,
                retry_delay=self.config.retry_delay,
                on_retry=on_retry,
                json=body,
            )

            return LLMResult(
                chunk,
                response.content,
                started_at,
                now(),
                time.monotonic() - started,
                attempts,
            )

        except RequestFailed as exc:
            return LLMResult(
                chunk,
                None,
                started_at,
                now(),
                time.monotonic() - started,
                exc.attempts,
                str(exc),
            )