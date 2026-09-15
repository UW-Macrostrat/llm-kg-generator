"""CLI configuration. No imports of CUDA, torch, or vLLM are required."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    output: Path = Path("pipeline_output")
    base_url: str = "https://llm.chtc.wisc.edu/api"
    model: str | None = None
    llm_api_key: str = "EMPTY"
    source_url: str = "https://dev.macrostrat.org/api/pg/source_text"
    entity_url: str = "https://dev.macrostrat.org/api/pg/kg_entity_type"
    relationship_url: str = "https://dev.macrostrat.org/api/pg/kg_relationship_type"
    record_run_url: str = "https://macrostrat-xdd.dev.svc.macrostrat.org/record_run"
    source_api_token: str = ""
    record_api_token: str = ""
    batch_size: int = 100
    preprocess_workers: int = 2
    llm_workers: int = 2
    postprocess_workers: int = 2
    upload_workers: int = 2
    queue_size: int = 200
    inflight_sources: int = 200
    event_queue_size: int = 4096
    max_tokens: int = 1024
    context_length: int | None = None
    token_margin: int = 32
    chunk_overlap_chars: int = 100
    http_timeout: float = 120
    llm_timeout: float = 300
    read_attempts: int = 3
    llm_attempts: int = 3
    retry_delay: float = 0.5
    progress_seconds: float = 10
    source_id: list[int] | None = None
    after_id: int | None = None
    extraction_pipeline_id: str = "0"
    model_version: int = 0
    dry_run: bool = False
    test: bool = False
    max_sources: int | None = None

    def validate(self) -> None:
        positive = ("batch_size", "preprocess_workers", "llm_workers",
                    "postprocess_workers", "upload_workers", "queue_size",
                    "inflight_sources", "event_queue_size", "max_tokens",
                    "http_timeout", "llm_timeout", "read_attempts", "llm_attempts")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("token_margin", "chunk_overlap_chars", "retry_delay", "progress_seconds"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.context_length is not None and self.context_length <= self.max_tokens + self.token_margin:
            raise ValueError("context-length must leave space beyond max-tokens and token-margin")
        if self.source_id is not None and self.after_id is not None:
            raise ValueError("Use either --source-id or --after-id, not both")
        if self.output.suffix.lower() in {".json", ".jsonl"}:
            raise ValueError("--output is a directory; results.json is created inside it")
        if self.max_sources is not None and self.max_sources <= 0:
            raise ValueError("--max must be greater than zero")

    def public_dict(self) -> dict:
        result = asdict(self)
        result["output"] = str(self.output)
        for key in ("llm_api_key", "source_api_token", "record_api_token"):
            result.pop(key)
        return result


def parse_args(argv: list[str] | None = None) -> Config:
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    preliminary, _ = env_parser.parse_known_args(argv)
    load_dotenv(preliminary.env_file, override=False)
    p = argparse.ArgumentParser(
        parents=[env_parser],
        description="Extract all source_text rows through independent async queues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    defaults = Config()
    p.add_argument("--output", type=Path, default=Path(f"runs/{stamp}_{uuid4().hex[:8]}"),
                   help="New output directory (must not already exist)")
    p.add_argument(
        "--base-url",
        default=os.getenv("CHTC_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or defaults.base_url,
    )
    p.add_argument(
        "--model",
        default=os.getenv("CHTC_MODEL_NAME") or os.getenv("MODEL_NAME") or None,
        help="Served model ID; discovered from /models when omitted",
    )
    for arg, env in (("source-url", "SOURCE_TEXT_URL"), ("entity-url", "ENTITY_TYPE_URL"),
                     ("relationship-url", "RELATIONSHIP_TYPE_URL"), ("record-run-url", "RECORD_RUN_URL")):
        p.add_argument(f"--{arg}", default=os.getenv(env, getattr(defaults, arg.replace("-", "_"))))
    for arg in ("batch-size", "preprocess-workers", "llm-workers", "postprocess-workers",
                "upload-workers", "queue-size", "inflight-sources", "event-queue-size",
                "max-tokens", "token-margin", "chunk-overlap-chars", "read-attempts", "llm-attempts"):
        p.add_argument(f"--{arg}", type=int, default=getattr(defaults, arg.replace("-", "_")))
    p.add_argument("--context-length", type=int, default=None,
                   help="Server context limit; auto-detect from /v1/models, otherwise 2048")
    for arg in ("http-timeout", "llm-timeout", "retry-delay", "progress-seconds"):
        p.add_argument(f"--{arg}", type=float, default=getattr(defaults, arg.replace("-", "_")))
    p.add_argument(
        "--source-id",
        type=int,
        nargs="+",
        help="Process only these source_text IDs",
    )
    p.add_argument("--after-id", type=int, help="Start after this source_text ID; not automatic resume")
    p.add_argument("--extraction-pipeline-id", default=os.getenv("EXTRACTION_PIPELINE_ID", "0"))
    p.add_argument("--model-version", type=int, default=int(os.getenv("MODEL_VERSION", "0")))
    p.add_argument("--dry-run", action="store_true", help="Save all results locally; never POST /record_run")
    p.add_argument("--test", action="store_true", help="Fetch and process only one source_text row")
    p.add_argument(
        "--max",
        dest="max_sources",
        type=int,
        default=None,
        help="Maximum source-text rows to process; omitted means all",
    )
    values = vars(p.parse_args(argv))
    values.pop("env_file")
    values.update(
        llm_api_key=os.getenv("CHTC_API_KEY", os.getenv("VLLM_API_KEY", "EMPTY")),
        source_api_token=os.getenv("SOURCE_API_TOKEN", ""),
        record_api_token=os.getenv("RECORD_API_TOKEN", ""),
    )
    config = Config(**values)
    try:
        config.validate()
    except ValueError as exc:
        p.error(str(exc))
    return config
