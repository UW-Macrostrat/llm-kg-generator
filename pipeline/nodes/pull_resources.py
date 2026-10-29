"""Node 1: fetch source paragraphs, ontology, and the scoped term dictionary."""

import os

from langgraph.runtime import Runtime

from ..helpers.api import fetch_sources, fetch_types, get_rows
from ..helpers.dictionary_ner import TERM_TYPES
from ..models import Ontology, SourceResult, SourceText
from ..state import Context, PipelineState


TERMS_URL = "https://dev.macrostrat.org/api/pg/kg_macrostrat_terms"


def fetch_terms(ctx):
    # This view uses macrostrat_terms_id, while fetch_types expects id.
    url = os.getenv("MACROSTRAT_TERMS_URL", TERMS_URL)
    terms = []
    cursor = None

    while True:
        params = {
            "entity_type": "in.(" + ",".join(TERM_TYPES) + ")",
            "order": "macrostrat_terms_id.asc",
            "limit": 1000,
        }
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
            if term.get("entity_type") not in TERM_TYPES:
                raise ValueError("Macrostrat terms endpoint ignored the entity_type filter")
            terms.append(term)
            cursor = term_id


def pull_resources(state: PipelineState, runtime: Runtime[Context]):
    ctx = runtime.context
    ctx.log.event("INFO", "started", stage="pull_resources")

    try:
        entity_rows = fetch_types(ctx, ctx.config.entity_url, "entity types")
        relation_rows = fetch_types(ctx, ctx.config.relationship_url, "relationship types")
        ontology = Ontology.from_rows(entity_rows, relation_rows)
        # NER and downstream relation candidates use only these three types.
        allowed = set(TERM_TYPES.values())
        missing = allowed - set(ontology.entity_types)
        if missing:
            raise ValueError(f"Ontology is missing required entity types: {sorted(missing)}")
        entity_types = {
            name: kind for name, kind in ontology.entity_types.items() if name in allowed
        }
        allowed_ids = {kind.id for kind in entity_types.values()}
        ontology = Ontology(
            entity_types=entity_types,
            relationship_types={
                name: kind for name, kind in ontology.relationship_types.items()
                if kind.src_entity_type_id in allowed_ids and kind.dst_entity_type_id in allowed_ids
            },
        )

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
        result = SourceResult(
            source_id=raw["id"], source=raw,
            strat_name=ctx.config.strat_name.strip() if ctx.config.strat_name else None,
        )
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
