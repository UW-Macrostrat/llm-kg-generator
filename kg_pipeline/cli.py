"""Command-line entry point and exit codes."""
from __future__ import annotations

import asyncio
import logging
import sys

from .config import parse_args
from .pipeline import describe_exception, run_pipeline


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        summary = asyncio.run(run_pipeline(config))
    except KeyboardInterrupt:
        print(f"Interrupted. Saved records are in {config.output.resolve()}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {describe_exception(exc)}", file=sys.stderr)
        return 1
    print(f"{summary['status']}: fetched={summary['fetched']} saved={summary['saved']} "
          f"uploaded={summary['uploaded']} failed_sources={summary['failed_sources']}")
    print(f"Results: {(config.output / 'results.json').resolve()}")
    if summary.get("error"):
        print(summary["error"], file=sys.stderr)
    return 0 if summary["status"] == "completed" else (2 if summary["status"] == "completed_with_errors" else 1)
