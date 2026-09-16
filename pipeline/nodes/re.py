"""Node 3: extract relations between fixed mentions; validate, rank, and select in code."""
from langgraph.runtime import Runtime

from ..helpers.llm import call_llm
from ..helpers.prompts import re_prompt
from ..helpers.validation import build_candidates, decode_response, rank_and_select_relations, validate_relations
from ..state import Context, PipelineState


def re(state: PipelineState, runtime: Runtime[Context]):
    ctx = runtime.context
    results = [result.model_copy(deep=True) for result in state["results"]]
    for result in results:
        if result.status == "failed":
            continue
        ctx.log.event("INFO", "started", stage="re", source_id=result.source_id)
        try:
            text = result.source["paragraph_text"]
            result.candidates = build_candidates(result.entities, state["ontology"], ctx.constraints)
            ctx.log.write(f"sources/{result.source_id}/re_candidates.json", [c.model_dump() for c in result.candidates])
            if not result.candidates:
                result.status = "re_complete"
                ctx.log.event("INFO", "skipped_no_candidates", stage="re", source_id=result.source_id)
                continue
            messages, schema = re_prompt(text, result.entities, result.candidates, state["ontology"], ctx.constraints)
            raw, call = call_llm(ctx, "re", result.source_id, messages, schema)
            result.llm_calls.append(call)
            result.models["re"] = call["model"]
            output = decode_response(raw, call)
            relations, rejected = validate_relations(output, text, result.entities, result.candidates,
                                                      state["ontology"], ctx.constraints)
            result.rejected.extend(rejected)
            ctx.log.rejections(result.source_id, rejected)
            if output["relations"] and not relations:
                raise ValueError("RE proposed relations but none survived validation; see rejection details")
            if len(output["relations"]) == ctx.constraints.max_relations:
                ctx.log.event("WARNING", "relation_limit_reached", stage="re", source_id=result.source_id,
                              message="Output reached max_relations; additional relations may have been omitted")
            result.relations, result.ranked_relations, removed = rank_and_select_relations(
                relations, text, result.entities, state["ontology"], ctx.constraints)
            result.rejected.extend(removed)
            ctx.log.rejections(result.source_id, removed)
            result.status = "re_complete"
            ctx.log.event("INFO", "completed", stage="re", source_id=result.source_id,
                          candidates=len(result.candidates), selected=len(result.relations),
                          rejected=len(rejected) + len(removed))
        except Exception as exc:
            if hasattr(exc, "call_info"):
                result.llm_calls.append(exc.call_info)
            result.status = "failed"
            result.errors.append(ctx.log.exception("re", exc, result.source_id))
        finally:
            for call in result.llm_calls:
                if call["stage"] == "re":
                    ctx.log.write(f"sources/{result.source_id}/re_call.json", call)
            ctx.log.save_source(result)
    return {"results": results}
