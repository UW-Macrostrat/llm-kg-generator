"""Runtime settings; extraction/data rules live in models.Constraints."""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output: Path = Path("pipeline_output")
    constraints_file: Path | None = None
    base_url: str = "https://llm.chtc.wisc.edu/api"
    model: str | None = None
    ner_model: str | None = None
    re_model: str | None = None
    source_url: str = "https://dev.macrostrat.org/api/pg/source_text"
    entity_url: str = "https://dev.macrostrat.org/api/pg/kg_entity_type"
    relationship_url: str = "https://dev.macrostrat.org/api/pg/kg_relationship_type"
    record_run_url: str = "https://macrostrat-xdd.dev.svc.macrostrat.org/record_run"
    llm_api_key: SecretStr = SecretStr("EMPTY")
    source_api_token: SecretStr = SecretStr("")
    record_api_token: SecretStr = SecretStr("")
    source_ids: list[int] | None = None
    after_id: int | None = Field(default=None, ge=0)
    max_sources: int = Field(default=10, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    context_length: int | None = Field(default=None, ge=1)
    timeout: float = Field(default=300, gt=0)
    attempts: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=1, ge=0)
    structured_output: str = "vllm"
    disable_thinking: bool = True
    extraction_pipeline_id: str = "0"
    model_version: int = 0
    dry_run: bool = True
    draw_graph: Path | None = None
    write_constraints: Path | None = None

    @model_validator(mode="after")
    def check_settings(self):
        if self.source_ids is not None and (not self.source_ids or any(i < 0 for i in self.source_ids)):
            raise ValueError("source_ids must contain nonnegative IDs")
        if self.source_ids and self.after_id is not None:
            raise ValueError("Use --source-id or --after-id, not both")
        if self.structured_output not in {"vllm", "json_schema", "none"}:
            raise ValueError("Invalid structured_output mode")
        if self.context_length and self.context_length <= self.max_tokens:
            raise ValueError("context_length must leave room beyond max_tokens")
        return self

    def public_dict(self):
        return self.model_dump(mode="json", exclude={"llm_api_key", "source_api_token", "record_api_token"})


def parse_args(argv=None) -> Config:
    env = argparse.ArgumentParser(add_help=False)
    env.add_argument("--env-file", type=Path, default=Path(".env"))
    preliminary, _ = env.parse_known_args(argv)
    load_dotenv(preliminary.env_file, override=False)
    parser = argparse.ArgumentParser(parents=[env], description="Sequential four-node NER/RE pipeline")
    defaults = Config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", type=Path, default=Path(f"runs/{stamp}_{uuid4().hex[:8]}"))
    parser.add_argument("--constraints", dest="constraints_file", type=Path)
    parser.add_argument("--base-url", default=os.getenv("CHTC_BASE_URL") or os.getenv("VLLM_BASE_URL") or defaults.base_url)
    parser.add_argument("--model", default=os.getenv("CHTC_MODEL_NAME") or os.getenv("MODEL_NAME"))
    parser.add_argument("--ner-model", default=os.getenv("NER_MODEL"))
    parser.add_argument("--re-model", default=os.getenv("RE_MODEL"))
    for arg, name in (("source-url", "SOURCE_TEXT_URL"), ("entity-url", "ENTITY_TYPE_URL"),
                      ("relationship-url", "RELATIONSHIP_TYPE_URL"), ("record-run-url", "RECORD_RUN_URL")):
        parser.add_argument(f"--{arg}", default=os.getenv(name, getattr(defaults, arg.replace("-", "_"))))
    parser.add_argument("--source-id", dest="source_ids", nargs="+", type=int)
    parser.add_argument("--after-id", type=int)
    parser.add_argument("--max", dest="max_sources", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--structured-output", choices=["vllm", "json_schema", "none"], default="vllm")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extraction-pipeline-id", default=os.getenv("EXTRACTION_PIPELINE_ID", "0"))
    parser.add_argument("--model-version", type=int, default=int(os.getenv("MODEL_VERSION", "0")))
    uploads = parser.add_mutually_exclusive_group()
    uploads.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    uploads.add_argument("--upload", dest="dry_run", action="store_false", help="Enable record_run POSTs")
    parser.add_argument("--draw-graph", type=Path, help="Export Mermaid Markdown and exit without network calls")
    parser.add_argument("--write-constraints", type=Path, help="Write the default constraints JSON and exit")
    values = vars(parser.parse_args(argv))
    values.pop("env_file")
    values.update(llm_api_key=os.getenv("CHTC_API_KEY") or os.getenv("VLLM_API_KEY", "EMPTY"),
                  source_api_token=os.getenv("SOURCE_API_TOKEN", ""), record_api_token=os.getenv("RECORD_API_TOKEN", ""))
    try:
        return Config(**values)
    except ValueError as exc:
        parser.error(str(exc))
