"""Node 2: extract entities, then validate and ground them deterministically."""
from langgraph.runtime import Runtime

from ..helpers.llm import call_llm
from ..helpers.prompts import ner_prompt
from ..helpers.validation import decode_response, validate_entities
from ..helpers.matching import match_entities
from ..state import Context, PipelineState


def ner(state: PipelineState, runtime: Runtime[Context]):
    ctx = runtime.context
    results = [result.model_copy(deep=True) for result in state["results"]]
    for result in results:
        if result.status == "failed":
            continue
        ctx.log.event("INFO", "started", stage="ner", source_id=result.source_id)
        try:
            text = result.source["paragraph_text"]
            messages, schema = ner_prompt(text, state["ontology"], ctx.constraints)
            raw, call = call_llm(ctx, "ner", result.source_id, messages, schema)
            result.llm_calls.append(call)
            result.models["ner"] = call["model"]
            output = decode_response(raw, call)
            result.entities, rejected = validate_entities(output, text, state["ontology"], ctx.constraints)
            match_entities(result.entities, state["macrostrat_terms"])
            result.rejected.extend(rejected)
            ctx.log.rejections(result.source_id, rejected)
            if output["entities"] and not result.entities:
                raise ValueError("NER proposed entities but none survived validation; see rejection details")
            if len(output["entities"]) == ctx.constraints.max_entities:
                ctx.log.event("WARNING", "entity_limit_reached", stage="ner", source_id=result.source_id,
                              message="Output reached max_entities; additional mentions may have been omitted")
            result.status = "ner_complete"
            ctx.log.event("INFO", "completed", stage="ner", source_id=result.source_id,
                          entities=len(result.entities), rejected=len(rejected))
        except Exception as exc:
            if hasattr(exc, "call_info"):
                result.llm_calls.append(exc.call_info)
            result.status = "failed"
            result.errors.append(ctx.log.exception("ner", exc, result.source_id))
        finally:
            for call in result.llm_calls:
                if call["stage"] == "ner":
                    ctx.log.write(f"sources/{result.source_id}/ner_call.json", call)
            ctx.log.save_source(result)
    return {"results": results}
