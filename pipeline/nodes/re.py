"""Node 3: select relations between local fixed candidates."""

import re
from collections import defaultdict

from langgraph.runtime import Runtime

from ..helpers.kg_image import save_kg_image
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


def _sentence_spans(text: str):
    """Return (start, end) spans for each sentence-like region."""
    spans = []
    start = 0

    for match in _SENTENCE_BREAK.finditer(text):
        end = match.start()

        if end > start:
            spans.append((start, end))

        start = match.end()

    if start < len(text):
        spans.append((start, len(text)))

    return spans


def _local_candidates(
    text,
    entities,
    candidates,
    constraints,
):
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

        evidence_start = min(src.start, dst.start)
        evidence_end = max(src.end, dst.end)
        evidence_length = evidence_end - evidence_start

        if (
            max_span is not None
            and evidence_length > max_span
        ):
            continue

        if src.start <= dst.start:
            between = text[src.end:dst.start]
        else:
            between = text[dst.end:src.start]

        if (
            constraints.local_candidates_same_sentence_only
            and _SENTENCE_BREAK.search(between)
        ):
            continue

        kept.append(candidate)

    return kept


def _group_candidates_by_sentence(
    text,
    entities,
    candidates,
):
    """
    Group each candidate into the sentence containing both endpoints.
    """

    entity_by_id = {
        entity.id: entity
        for entity in entities
    }

    spans = _sentence_spans(text)

    groups = defaultdict(list)

    for candidate in candidates:
        src = entity_by_id[candidate.src_id]
        dst = entity_by_id[candidate.dst_id]

        candidate_start = min(src.start, dst.start)
        candidate_end = max(src.end, dst.end)

        for index, (sentence_start, sentence_end) in enumerate(spans):
            if (
                candidate_start >= sentence_start
                and candidate_end <= sentence_end
            ):
                groups[index].append(candidate)
                break

    return [
        (
            spans[index][0],
            spans[index][1],
            groups[index],
        )
        for index in sorted(groups)
    ]

def _group_relations_by_sentence(
    text,
    relations,
):
    """
    Group final selected relations by the sentence containing
    their deterministic evidence span.
    """

    spans = _sentence_spans(text)
    groups = defaultdict(list)

    for relation in relations:
        for index, (sentence_start, sentence_end) in enumerate(spans):
            if (
                relation.evidence_start >= sentence_start
                and relation.evidence_end <= sentence_end
            ):
                groups[index].append(relation)
                break

    return [
        (
            index,
            spans[index][0],
            spans[index][1],
            groups[index],
        )
        for index in sorted(groups)
        if groups[index]
    ]

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
            # 1. All structurally legal candidates
            # ---------------------------------------------------------

            raw_candidates = build_candidates(
                result.entities,
                state["ontology"],
                ctx.constraints,
            )

            # ---------------------------------------------------------
            # 2. Remove impossible/non-local candidates
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
                removed=(
                    len(raw_candidates)
                    - len(result.candidates)
                ),
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
                continue

            # ---------------------------------------------------------
            # 3. Split candidates by sentence
            # ---------------------------------------------------------

            groups = _group_candidates_by_sentence(
                text,
                result.entities,
                result.candidates,
            )

            selected_candidate_ids = []

            for (
                sentence_start,
                sentence_end,
                sentence_candidates,
            ) in groups:

                sentence_text = text[
                    sentence_start:sentence_end
                ]

                messages, schema = re_prompt(
                    sentence_text,
                    result.entities,
                    sentence_candidates,
                    state["ontology"],
                    ctx.constraints,
                    text_offset=sentence_start,
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

                for item in output.get("relations", []):
                    candidate_id = item.get("candidate_id")

                    if (
                        candidate_id
                        and candidate_id
                        not in selected_candidate_ids
                    ):
                        selected_candidate_ids.append(
                            candidate_id
                        )

            # One combined model output for validation.
            output = {
                "relations": [
                    {
                        "candidate_id": candidate_id
                    }
                    for candidate_id
                    in selected_candidate_ids
                ]
            }

            # ---------------------------------------------------------
            # 4. Deterministic validation
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
                output["relations"]
            )

            # ---------------------------------------------------------
            # 5. Rank/select
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

            # ---------------------------------------------------------
            # 6. Render one KG per sentence containing relationships
            # ---------------------------------------------------------

            relation_groups = _group_relations_by_sentence(
                text,
                result.relations,
            )

            for (
                sentence_index,
                sentence_start,
                sentence_end,
                sentence_relations,
            ) in relation_groups:

                # Skip sentences without final relationships.
                if not sentence_relations:
                    continue

                # Only include entities that actually participate
                # in relationships in this sentence.
                entity_ids = {
                    entity_id
                    for relation in sentence_relations
                    for entity_id in (
                        relation.src_id,
                        relation.dst_id,
                    )
                }

                sentence_entities = [
                    entity
                    for entity in result.entities
                    if entity.id in entity_ids
                ]

                sentence_text = text[
                    sentence_start:sentence_end
                ].strip()

                try:
                    save_kg_image(
                        ctx,
                        result.source_id,
                        sentence_entities,
                        sentence_relations,
                        filename=(
                            f"kg_sentence_{sentence_index + 1:02d}.png"
                        ),
                        title=sentence_text,
                    )

                except Exception as exc:
                    ctx.log.event(
                        "WARNING",
                        "kg_image_failed",
                        stage="re",
                        source_id=result.source_id,
                        sentence=sentence_index + 1,
                        message=str(exc),
                    )
            ctx.log.event(
                "INFO",
                "completed",
                stage="re",
                source_id=result.source_id,
                raw_candidates=len(raw_candidates),
                candidates=len(result.candidates),
                sentence_groups=len(groups),
                proposed=proposed,
                validated=len(relations),
                selected=len(result.relations),
                rejected=(
                    len(rejected)
                    + len(removed)
                ),
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

    return {
        "results": results
    }