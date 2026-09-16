"""Node 4: confirm the final result, build the API payload, persist, and optionally upload."""

from langgraph.runtime import Runtime

from ..helpers.api import submit_record_run
from ..helpers.logging import now
from ..helpers.validation import confirm_extraction
from ..models import Issue, RecordRunPayload, SourceText
from ..state import Context, PipelineState


def build_payload(result, ctx):
    source = SourceText.model_validate(result.source)
    by_id = {entity.id: entity for entity in result.entities}

    relationships = []
    for relation in result.relations:
        src = by_id[relation.src_id]
        dst = by_id[relation.dst_id]

        relationships.append({
            "src": src.text,
            "relationship_type": relation.relationship_type,
            "dst": dst.text,
            "reasoning": " ".join(relation.evidence.split()[:19])[:160],
            "src_start_idx": src.start,
            "src_end_idx": src.end,
            "dst_start_idx": dst.start,
            "dst_end_idx": dst.end,
        })

    # Include every entity so relationship endpoints also upload their matches.
    entities = [
        {
            "entity": entity.text,
            "entity_type": entity.entity_type,
            "start_idx": entity.start,
            "end_idx": entity.end,
            "macrostrat_terms_id": entity.macrostrat_terms_id,
        }
        for entity in result.entities
    ]

    names = list(dict.fromkeys(result.models.values()))
    model_label = (
        names[0]
        if len(names) == 1
        else "; ".join(
            f"{stage.upper()}={model}"
            for stage, model in result.models.items()
        )
    )

    return RecordRunPayload.model_validate({
        "run_id": f"{ctx.run_id}_s{source.id}",
        "extraction_pipeline_id": ctx.config.extraction_pipeline_id,
        "model_name": model_label,
        "model_version": ctx.config.model_version,
        "results": [
            {
                "text": {
                    "preprocessor_id": source.preprocessor_id,
                    "paper_id": source.paper_id,
                    "hashed_text": source.hashed_text,
                    "weaviate_id": source.weaviate_id,
                    "paragraph_text": source.paragraph_text,
                    "text_type": source.source_text_type,
                    "legend_id": source.map_legend_id,
                },
                "relationships": relationships,
                "just_entities": entities,
            }
        ],
    })


def record_run(state: PipelineState, runtime: Runtime[Context]):
    ctx = runtime.context
    results = [
        result.model_copy(deep=True)
        for result in state["results"]
    ]

    for result in results:
        if result.status == "failed":
            result.upload_status = "skipped_failed"
            ctx.log.event(
                "WARNING",
                "skipped_failed_source",
                stage="record_run",
                source_id=result.source_id,
            )
            ctx.log.save_source(result)
            continue

        ctx.log.event(
            "INFO",
            "started",
            stage="record_run",
            source_id=result.source_id,
        )

        try:
            if result.status != "re_complete":
                raise ValueError("Source did not complete NER and RE")

            confirm_extraction(
                result.entities,
                result.relations,
                result.source["paragraph_text"],
                state["ontology"],
                ctx.constraints,
            )

            result.payload = build_payload(result, ctx)
            result.status = "success"
            result.upload_status = (
                "dry_run" if ctx.config.dry_run else "started"
            )

            if not ctx.config.dry_run:
                result.receipt = {
                    "status": "started",
                    "run_id": result.payload.run_id,
                    "started_at": now(),
                    "message": (
                        "If interrupted, reconcile this run_id before retrying."
                    ),
                }

            # Persist the exact payload and run ID before making the request.
            ctx.log.save_source(result)

            if not ctx.config.dry_run:
                result.receipt = submit_record_run(
                    ctx,
                    result.payload.model_dump(mode="json"),
                )
                result.upload_status = result.receipt["status"]

                if result.upload_status != "success":
                    issue = Issue(
                        stage="record_run",
                        code=f"upload_{result.upload_status}",
                        message=(
                            f"Upload outcome: {result.upload_status}; "
                            "no automatic retry"
                        ),
                        details=result.receipt,
                    )
                    result.errors.append(issue)

                    ctx.log.event(
                        "ERROR",
                        issue.code,
                        stage="record_run",
                        source_id=result.source_id,
                        message=issue.message,
                        receipt=result.receipt,
                    )

            ctx.log.event(
                "INFO",
                "completed",
                stage="record_run",
                source_id=result.source_id,
                upload_status=result.upload_status,
                entities=len(result.entities),
                relationships=len(result.relations),
                matched=sum(
                    entity.macrostrat_terms_id is not None
                    for entity in result.entities
                ),
            )

        except Exception as exc:
            result.errors.append(
                ctx.log.exception("record_run", exc, result.source_id)
            )

            if result.upload_status == "started":
                result.upload_status = "unknown"
            else:
                result.status = "failed"
                result.upload_status = "skipped_failed"

        finally:
            ctx.log.save_source(result)

    errors = sum(bool(result.errors) for result in results)

    summary = {
        "run_id": ctx.run_id,
        "status": "completed_with_errors" if errors else "completed",
        "sources": len(results),
        "successful_extractions": sum(
            result.status == "success" for result in results
        ),
        "failed_sources": sum(
            result.status == "failed" for result in results
        ),
        "uploaded": sum(
            result.upload_status == "success" for result in results
        ),
        "upload_failed": sum(
            result.upload_status == "failed" for result in results
        ),
        "upload_unknown": sum(
            result.upload_status == "unknown" for result in results
        ),
        "sources_with_rejections": sum(
            bool(result.rejected) for result in results
        ),
        "rejected_predictions": sum(
            len(result.rejected) for result in results
        ),
        "llm_calls": sum(
            len(result.llm_calls) for result in results
        ),
        "dry_run": ctx.config.dry_run,
    }

    return {"results": results, "summary": summary}