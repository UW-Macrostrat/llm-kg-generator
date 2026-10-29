"""Node 2: dictionary suggestions, LLM review, and occurrence-level grounding."""

from langgraph.runtime import Runtime

from ..helpers.llm import call_llm
from ..helpers.validation import decode_response
from ..helpers.dictionary_ner import (
    prepare_terms,
    find_dictionary_mentions,
    dictionary_ner_prompt,
    validate_dictionary_entities,
)
from ..state import Context, PipelineState


def ner(state: PipelineState, runtime: Runtime[Context]):
    ctx = runtime.context

    results = [
        result.model_copy(deep=True)
        for result in state["results"]
    ]

    for result in results:
        if result.status == "failed":
            continue

        ctx.log.event(
            "INFO",
            "started",
            stage="ner",
            source_id=result.source_id,
        )

        try:
            text = result.source["paragraph_text"]

            terms = prepare_terms(state["macrostrat_terms"], state["ontology"])
            suggestions = find_dictionary_mentions(
                text, terms, fuzzy_min_score=ctx.config.fuzzy_threshold,
            )
            messages, schema = dictionary_ner_prompt(
                text, terms, suggestions, ctx.constraints, strat_name=result.strat_name,
            )
            ctx.log.write(
                f"sources/{result.source_id}/ner_suggestions.json",
                {"allowed_term_count": len(terms), "suggestions": suggestions},
            )

            raw, call = call_llm(
                ctx,
                "ner",
                result.source_id,
                messages,
                schema,
            )

            result.llm_calls.append(call)
            result.models["ner"] = call["model"]

            output = decode_response(raw, call)

            # Validate each occurrence and selected ID without rematching by name.
            result.entities, rejected, match_stats, grounding = validate_dictionary_entities(
                output,
                text,
                state["ontology"],
                ctx.constraints,
                terms,
            )
            ctx.log.write(
                f"sources/{result.source_id}/ner_grounding.json",
                {"mentions": grounding, "statistics": match_stats},
            )

            result.rejected.extend(rejected)
            ctx.log.rejections(result.source_id, rejected)

            # Do NOT fail the entire source just because all proposed
            # entities were rejected. Preserve diagnostics and continue.
            if output["entities"] and not result.entities:
                ctx.log.event(
                    "WARNING",
                    "all_entities_rejected",
                    stage="ner",
                    source_id=result.source_id,
                    message=(
                        "NER proposed entities but none survived validation"
                    ),
                )

            if len(output["entities"]) == ctx.constraints.max_entities:
                ctx.log.event(
                    "WARNING",
                    "entity_limit_reached",
                    stage="ner",
                    source_id=result.source_id,
                    message=(
                        "Output reached max_entities; "
                        "additional mentions may have been omitted"
                    ),
                )

            result.status = "ner_complete"

            ctx.log.event(
                "INFO",
                "completed",
                stage="ner",
                source_id=result.source_id,
                proposed=len(output["entities"]),
                validated=len(result.entities),
                rejected=len(rejected),
                matched=match_stats["matched"],
                unmatched=match_stats["unmatched"],
                ambiguous=match_stats["ambiguous"],
            )

        except Exception as exc:
            if hasattr(exc, "call_info"):
                result.llm_calls.append(exc.call_info)

            result.status = "failed"

            result.errors.append(
                ctx.log.exception(
                    "ner",
                    exc,
                    result.source_id,
                )
            )

        finally:
            for call in result.llm_calls:
                if call["stage"] == "ner":
                    ctx.log.write(
                        f"sources/{result.source_id}/ner_call.json",
                        call,
                    )

            ctx.log.save_source(result)

    return {"results": results}
