"""Human-readable tracebacks, structured events, and per-source JSON artifacts."""
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from ..models import Issue, SourceResult


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    def __init__(self, root: Path, secrets: list[str] | None = None):
        self.root = root
        self.secrets = [s for s in secrets or [] if s and s != "EMPTY"]
        root.mkdir(parents=True, exist_ok=False)
        (root / "sources").mkdir()
        self.logger = logging.getLogger(f"kg_pipeline.{uuid4().hex}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in (logging.FileHandler(root / "pipeline.log", encoding="utf-8"), logging.StreamHandler()):
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler)

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    def event(self, level: str, event: str, *, stage: str, source_id=None, **details):
        value = {"timestamp": now(), "level": level, "event": event,
                 "stage": stage, "source_id": source_id, **details}
        serialized = self.redact(json.dumps(value, ensure_ascii=False, default=str))
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
            stream.flush()
        message = f"{stage} source={source_id} {event}"
        if details.get("message"):
            message += f": {details['message']}"
        self.logger.log(getattr(logging, level), self.redact(message))

    def exception(self, stage: str, exc: Exception, source_id=None) -> Issue:
        details = {"exception_type": type(exc).__name__, "traceback": traceback.format_exc()}
        if isinstance(exc, ValidationError):
            details["validation_errors"] = exc.errors(include_url=False, include_context=False)
        for key in ("status_code", "attempts", "response_excerpt"):
            if hasattr(exc, key):
                details[key] = getattr(exc, key)
        issue = Issue(stage=stage, code=type(exc).__name__, message=self.redact(str(exc)), details=details)
        self.event("ERROR", "failed", stage=stage, source_id=source_id,
                   message=issue.message, **details)
        self.logger.error(self.redact(details["traceback"]))
        return issue

    def rejections(self, source_id: int, issues: list[Issue]):
        for issue in issues:
            self.event("WARNING", "prediction_rejected", stage=issue.stage, source_id=source_id,
                       code=issue.code, message=issue.message, details=issue.details)

    def write(self, relative_path: str, value):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(self.redact(json.dumps(value, ensure_ascii=False, indent=2, default=str)) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save_source(self, result: SourceResult):
        self.write(f"sources/{result.source_id}/result.json", result.model_dump(mode="json"))

    def close(self):
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
