"""Node 1: fetch source paragraphs, ontology, and terms for direct matching."""

import os

from langgraph.runtime import Runtime

from ..helpers.api import fetch_sources, fetch_types, get_rows
from ..models import Ontology, SourceResult, SourceText
from ..state import Context, PipelineState


TERMS_URL = "https://dev.macrostrat.org/api/pg/kg_macrostrat_terms"


def fetch_terms(ctx):
    # This view uses macrostrat_terms_id, while fetch_types expects id.
    url = os.getenv("MACROSTRAT_TERMS_URL", TERMS_URL)
    terms = []
    cursor = None

    while True:
        params = {"order": "macrostrat_terms_id.asc", "limit": 1000}
        if cursor is not None:
            params["macrostrat_terms_id"] = f"gt.{cursor}"

        page = get_rows(ctx, url, params, "Macrostrat terms")
        if not page:
            return terms

        for term in page:
            term_id = term.get("macrostrat_terms_id")
            if (
                type(term_id) is not int
                or term_id <= 0
                or (cursor is not None and term_id <= cursor)
            ):
                raise ValueError("Macrostrat term IDs must be positive and strictly increasing")
            if not isinstance(term.get("name"), str) or not term["name"].strip():
                raise ValueError(f"Macrostrat term {term_id} has a missing or blank name")
            terms.append(term)
            cursor = term_id


def pull_resources(state: PipelineState, runtime: Runtime[Context]):
    ctx = runtime.context
    ctx.log.event("INFO", "started", stage="pull_resources")

    try:
        entity_rows = fetch_types(ctx, ctx.config.entity_url, "entity types")
        relation_rows = fetch_types(ctx, ctx.config.relationship_url, "relationship types")
        ontology = Ontology.from_rows(entity_rows, relation_rows)

        macrostrat_terms = fetch_terms(ctx)
        if not macrostrat_terms:
            raise ValueError("No Macrostrat terms returned for entity matching")

        sources = fetch_sources(ctx)
        if not sources:
            raise ValueError("No source paragraphs matched this run")
    except Exception as exc:
        ctx.log.exception("pull_resources", exc)
        raise

    ctx.log.write("ontology.json", {
        "raw_entity_rows": entity_rows,
        "raw_relationship_rows": relation_rows,
        "formatted": ontology.model_dump(mode="json"),
    })
    ctx.log.write("macrostrat_terms.json", macrostrat_terms)

    results = []
    for raw in sources:
        result = SourceResult(source_id=raw["id"], source=raw)
        try:
            result.source = SourceText.model_validate(raw).model_dump(mode="json")
        except Exception as exc:
            result.status = "failed"
            result.errors.append(ctx.log.exception("pull_resources", exc, result.source_id))
        ctx.log.save_source(result)
        results.append(result)

    ctx.log.event(
        "INFO", "completed", stage="pull_resources",
        sources=len(results),
        entity_types=len(ontology.entity_types),
        relation_types=len(ontology.relationship_types),
        macrostrat_terms=len(macrostrat_terms),
    )

    return {
        "ontology": ontology,
        "macrostrat_terms": macrostrat_terms,
        "results": results,
    }
