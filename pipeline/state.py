"""Graph data and shared dependencies. Each node returns a partial state update."""
from dataclasses import dataclass
from typing import TypedDict

import httpx

from .config import Config
from .helpers.logging import RunLog
from .models import Constraints, Ontology, SourceResult


class PipelineState(TypedDict, total=False):
    ontology: Ontology
    results: list[SourceResult]
    summary: dict
    macrostrat_terms: list[dict]


@dataclass
class Context:
    config: Config
    constraints: Constraints
    http: httpx.Client
    log: RunLog
    run_id: str
    # Cache /models per run; no clients or secrets are placed in graph state.
    served_models: list[dict] | None = None
