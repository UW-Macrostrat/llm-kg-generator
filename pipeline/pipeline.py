"""Four-node graph wiring and a sequential command-line runner."""
import json
import sys
import time
from uuid import uuid4

import httpx
from langgraph.graph import END, START, StateGraph

from .config import Config, parse_args
from .helpers.logging import RunLog, now
from .models import Constraints
from .nodes.ner import ner
from .nodes.pull_resources import pull_resources
from .nodes.re import re
from .nodes.record_run import record_run
from .state import Context, PipelineState


def build_graph():
    graph = StateGraph(PipelineState, context_schema=Context)
    graph.add_node("pull_resources", pull_resources)
    graph.add_node("ner", ner)
    graph.add_node("re", re)
    graph.add_node("record_run", record_run)

    graph.add_edge(START, "pull_resources")
    graph.add_edge("pull_resources", "ner")
    graph.add_edge("ner", "re")
    graph.add_edge("re", "record_run")
    graph.add_edge("record_run", END)
    return graph.compile()


def run_pipeline(config: Config, *, transport=None):
    """Optional HTTP transport enables complete offline tests without changing the graph."""
    secrets = [getattr(config, key).get_secret_value() for key in ("llm_api_key", "source_api_token", "record_api_token")]
    log = RunLog(config.output, secrets)
    run_id, started, started_at = f"kg_{uuid4().hex}", time.monotonic(), now()
    try:
        rules = Constraints.model_validate_json(config.constraints_file.read_text()) if config.constraints_file else Constraints()
        graph = build_graph()
        log.write("run.json", {"run_id": run_id, "started_at": started_at, "config": config.public_dict(),
                               "constraints": rules.model_dump(), "graph_mermaid": graph.get_graph().draw_mermaid()})
        with httpx.Client(timeout=config.timeout, transport=transport, follow_redirects=False, trust_env=False) as client:
            context = Context(config, rules, client, log, run_id)
            state = graph.invoke({}, context=context)
        summary = {**state["summary"], "started_at": started_at, "finished_at": now(), "seconds": time.monotonic() - started}
        log.write("results.json", {"summary": summary, "results": [r.model_dump(mode="json") for r in state["results"]]})
    except KeyboardInterrupt:
        summary = {"run_id": run_id, "status": "interrupted", "finished_at": now()}
        log.event("ERROR", "interrupted", stage="pipeline", message="Partial source results remain saved")
    except Exception as exc:
        issue = log.exception("pipeline", exc)
        summary = {"run_id": run_id, "status": "failed", "finished_at": now(), "error": issue.model_dump(mode="json")}
    finally:
        # A persistence error is intentionally fatal; do not report unsaved results as successful.
        try:
            if "summary" in locals():
                log.write("summary.json", summary)
        finally:
            log.close()
    return summary


def main(argv=None):
    config = parse_args(argv)
    try:
        if config.write_constraints:
            config.write_constraints.parent.mkdir(parents=True, exist_ok=True)
            config.write_constraints.write_text(Constraints().model_dump_json(indent=2) + "\n", encoding="utf-8")
            print(config.write_constraints.resolve())
            return 0
        if config.draw_graph:
            config.draw_graph.parent.mkdir(parents=True, exist_ok=True)
            diagram = build_graph().get_graph().draw_mermaid()
            config.draw_graph.write_text("# Extraction pipeline\n\n```mermaid\n" + diagram + "\n```\n", encoding="utf-8")
            print(config.draw_graph.resolve())
            return 0
        summary = run_pipeline(config)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    print(f"Logs and results: {config.output.resolve()}")
    return {"completed": 0, "completed_with_errors": 2, "interrupted": 130}.get(summary["status"], 1)
