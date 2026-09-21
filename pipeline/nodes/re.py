"""Node 3: select relations between local fixed candidates; validate, rank, and select in code."""

import re

from langgraph.runtime import Runtime

from ..helpers.llm import call_llm
from ..helpers.prompts import re_prompt
from ..helpers.validation import (
    build_candidates,
    decode_response,
    rank_and_select_relations,
    validate_relations,
)
from ..state import Context, PipelineState


_SENTENCE_BREAK = re.compile(r"[.!?]\s+|\n+")


def _local_candidates(
    text,
    entities,
    candidates,
    constraints,
):
    """
    Remove structurally legal but textually implausible candidates before
    sending them to the LLM.

    A relation candidate is retained only when:
      1. both endpoints exist,
      2. the deterministic evidence span is short enough,
      3. the endpoints do not cross a sentence boundary.

    This keeps candidate generation deterministic while preventing the
    O(N^2) global-pair explosion on long paragraphs.
    """

    by_id = {
        entity.id: entity
        for entity in entities
    }

    max_span = constraints.local_candidates_max_span_chars

    if max_span is None:
        max_span = getattr(
            constraints,
            "max_evidence_chars",
            None,
        )

    kept = []

    for candidate in candidates:
        src = by_id.get(candidate.src_id)
        dst = by_id.get(candidate.dst_id)

        if src is None or dst is None:
            continue

        # Deterministic evidence span used later by validation.
        evidence_start = min(src.start, dst.start)
        evidence_end = max(src.end, dst.end)
        evidence_length = evidence_end - evidence_start

        # Don't send something to the LLM that validation will
        # deterministically reject anyway.
        if max_span is not None and evidence_length > max_span:
            continue

        # Find the text strictly between the two entity mentions.
        if src.start <= dst.start:
            between = text[src.end:dst.start]
        else:
            between = text[dst.end:src.start]

        # Relationships across sentences are optional and configurable.
        if (
            constraints.local_candidates_same_sentence_only
            and _SENTENCE_BREAK.search(between)
        ):
            continue

        kept.append(candidate)

    return kept


def re(state: PipelineState, runtime: Runtime[Context]):
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
            stage="re",
            source_id=result.source_id,
        )

        try:
            text = result.source["paragraph_text"]

            # ---------------------------------------------------------
            # 1. Generate all structurally legal relation candidates.
            # ---------------------------------------------------------

            raw_candidates = build_candidates(
                result.entities,
                state["ontology"],
                ctx.constraints,
            )

            # ---------------------------------------------------------
            # 2. Remove non-local candidates BEFORE the LLM sees them.
            # ---------------------------------------------------------

            result.candidates = _local_candidates(
                text,
                result.entities,
                raw_candidates,
                ctx.constraints,
            )

            ctx.log.event(
                "INFO",
                "candidates_pruned",
                stage="re",
                source_id=result.source_id,
                raw_candidates=len(raw_candidates),
                local_candidates=len(result.candidates),
                removed=len(raw_candidates) - len(result.candidates),
            )

            ctx.log.write(
                f"sources/{result.source_id}/re_candidates.json",
                [
                    candidate.model_dump()
                    for candidate in result.candidates
                ],
            )

            if not result.candidates:
                result.relations = []
                result.ranked_relations = []
                result.status = "re_complete"

                ctx.log.event(
                    "INFO",
                    "skipped_no_candidates",
                    stage="re",
                    source_id=result.source_id,
                )

                continue

            # ---------------------------------------------------------
            # 3. Model chooses only from the local fixed candidates.
            # ---------------------------------------------------------

            messages, schema = re_prompt(
                text,
                result.entities,
                result.candidates,
                state["ontology"],
                ctx.constraints,
            )

            raw, call = call_llm(
                ctx,
                "re",
                result.source_id,
                messages,
                schema,
            )

            result.llm_calls.append(call)
            result.models["re"] = call["model"]

            output = decode_response(
                raw,
                call,
            )

            # ---------------------------------------------------------
            # 4. Deterministically resolve selected candidate IDs.
            # ---------------------------------------------------------

            relations, rejected = validate_relations(
                output,
                text,
                result.entities,
                result.candidates,
                state["ontology"],
                ctx.constraints,
            )

            result.rejected.extend(rejected)
            ctx.log.rejections(
                result.source_id,
                rejected,
            )

            proposed = len(
                output.get("relations", [])
            )

            if proposed and not relations:
                ctx.log.event(
                    "WARNING",
                    "all_relations_rejected",
                    stage="re",
                    source_id=result.source_id,
                    message=(
                        "Continuing with validated entities "
                        "and matches only."
                    ),
                    rejected=len(rejected),
                )

            if proposed == ctx.constraints.max_relations:
                ctx.log.event(
                    "WARNING",
                    "relation_limit_reached",
                    stage="re",
                    source_id=result.source_id,
                    message=(
                        "Output reached max_relations; "
                        "additional relations may have been omitted"
                    ),
                )

            # ---------------------------------------------------------
            # 5. Rank/select validated relations.
            # ---------------------------------------------------------

            (
                result.relations,
                result.ranked_relations,
                removed,
            ) = rank_and_select_relations(
                relations,
                text,
                result.entities,
                state["ontology"],
                ctx.constraints,
            )

            result.rejected.extend(removed)

            ctx.log.rejections(
                result.source_id,
                removed,
            )

            result.status = "re_complete"

            ctx.log.event(
                "INFO",
                "completed",
                stage="re",
                source_id=result.source_id,
                raw_candidates=len(raw_candidates),
                candidates=len(result.candidates),
                proposed=proposed,
                validated=len(relations),
                selected=len(result.relations),
                rejected=len(rejected) + len(removed),
            )

        except Exception as exc:
            if hasattr(exc, "call_info"):
                result.llm_calls.append(
                    exc.call_info
                )

            result.status = "failed"

            result.errors.append(
                ctx.log.exception(
                    "re",
                    exc,
                    result.source_id,
                )
            )

        finally:
            for call in result.llm_calls:
                if call["stage"] == "re":
                    ctx.log.write(
                        f"sources/{result.source_id}/re_call.json",
                        call,
                    )

            ctx.log.save_source(result)

    return {"results": results}