"""Pure deterministic parsing, grounding, ontology checks, ranking, and graph selection."""

import json
import re

from ..models import (
    Candidate,
    Entity,
    Issue,
    NEROutput,
    RankedRelation,
    Relation,
    REOutput,
)


def unique_keys(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")

        result[key] = value

    return result


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def decode_response(raw: str, call_info: dict):
    def decode(value):
        return json.loads(
            value,
            object_pairs_hook=unique_keys,
            parse_constant=reject_constant,
        )

    envelope = decode(raw)

    if (
        not isinstance(envelope, dict)
        or not isinstance(envelope.get("choices"), list)
        or not envelope["choices"]
    ):
        raise ValueError("Invalid chat-completion response envelope")

    if isinstance(envelope.get("usage"), dict):
        call_info["usage"] = envelope["usage"]

    choice = envelope["choices"][0]

    if not isinstance(choice, dict):
        raise ValueError("Invalid chat-completion choice")

    call_info["finish_reason"] = choice.get("finish_reason")

    if choice.get("finish_reason") != "stop":
        raise ValueError(
            f"Incomplete model output: "
            f"finish_reason={choice.get('finish_reason')!r}. "
            "For 'length', increase --max-tokens or reduce extraction limits."
        )

    message = choice.get("message")

    if (
        not isinstance(message, dict)
        or not isinstance(message.get("content"), str)
    ):
        raise ValueError("Model response has no text content")

    return decode(message["content"])


# ---------------------------------------------------------------------------
# Exact span helpers
# ---------------------------------------------------------------------------

def exact_span(
    text: str,
    phrase: str,
    occurrence: int,
    *,
    entity=False,
):
    """
    Resolve one exact occurrence.

    Currently retained for relationship evidence validation.
    NER no longer uses this function.
    """

    matches = list(
        re.finditer(
            f"(?={re.escape(phrase)})",
            text,
        )
    )

    if (
        not phrase
        or occurrence < 0
        or occurrence >= len(matches)
    ):
        raise ValueError(
            "Exact phrase occurrence is absent from source text"
        )

    start = matches[occurrence].start()
    end = start + len(phrase)

    if entity and (
        (
            start
            and phrase[0].isalnum()
            and text[start - 1].isalnum()
        )
        or (
            end < len(text)
            and phrase[-1].isalnum()
            and text[end].isalnum()
        )
    ):
        raise ValueError(
            "Entity is a substring inside a larger word"
        )

    return start, end


def exact_entity_spans(text: str, phrase: str):
    """
    Find every exact, case-sensitive occurrence of an entity phrase.

    The LLM does NOT determine occurrence indexes. Python finds the
    occurrences deterministically.

    Matches that occur only as substrings inside larger words are ignored.
    """

    if not phrase:
        raise ValueError("Entity text is empty")

    spans = []

    # Lookahead allows overlapping matches if they ever occur.
    matches = re.finditer(
        f"(?={re.escape(phrase)})",
        text,
    )

    for match in matches:
        start = match.start()
        end = start + len(phrase)

        # Don't allow an entity to start in the middle of a larger word.
        if (
            start
            and phrase[0].isalnum()
            and text[start - 1].isalnum()
        ):
            continue

        # Don't allow an entity to end in the middle of a larger word.
        if (
            end < len(text)
            and phrase[-1].isalnum()
            and text[end].isalnum()
        ):
            continue

        spans.append((start, end))

    if not spans:
        raise ValueError(
            "Exact phrase is absent from source text"
        )

    return spans


def rejection(stage, code, message, **details):
    return Issue(
        stage=stage,
        code=code,
        message=message,
        details=details,
    )


# ---------------------------------------------------------------------------
# Entity validation
# ---------------------------------------------------------------------------

def validate_entities(
    raw,
    text,
    ontology,
    constraints,
):
    context = {
        "text": text,
        "ontology": ontology,
        "constraints": constraints,
    }

    output = NEROutput.model_validate(
        raw,
        context=context,
    )

    found = {}
    conflicts = set()
    rejected = []

    # Prevent the same LLM prediction from being processed twice.
    #
    # IMPORTANT:
    # occurrence is intentionally NOT part of this key.
    # Python determines occurrences from the paragraph.
    seen_drafts = set()

    for draft in output.entities:
        draft_key = (
            draft.text,
            draft.entity_type,
        )

        if draft_key in seen_drafts:
            rejected.append(
                rejection(
                    "ner",
                    "duplicate_entity_draft",
                    "Duplicate entity draft",
                    item=draft.model_dump(),
                )
            )
            continue

        seen_drafts.add(draft_key)

        try:
            # Do NOT use draft.occurrence.
            #
            # If "sandstone" occurs twice, one LLM prediction creates
            # both source mentions deterministically.
            spans = exact_entity_spans(
                text,
                draft.text,
            )

            for start, end in spans:
                entity = Entity.model_validate(
                    {
                        "id": f"e{start}_{end}",
                        "text": draft.text,
                        "entity_type": draft.entity_type,
                        "start": start,
                        "end": end,
                    },
                    context=context,
                )

                previous = found.get(entity.id)

                if (
                    previous
                    and previous.entity_type
                    != entity.entity_type
                ):
                    conflicts.add(entity.id)

                    rejected.append(
                        rejection(
                            "ner",
                            "conflicting_types",
                            (
                                "Multiple types for the same mention; "
                                "all discarded"
                            ),
                            first=previous.model_dump(),
                            second=entity.model_dump(),
                        )
                    )

                elif entity.id in conflicts:
                    rejected.append(
                        rejection(
                            "ner",
                            "conflicting_types",
                            (
                                "Mention already rejected for "
                                "conflicting types"
                            ),
                            item=entity.model_dump(),
                        )
                    )

                elif previous:
                    rejected.append(
                        rejection(
                            "ner",
                            "duplicate_entity",
                            "Duplicate entity mention",
                            item=entity.model_dump(),
                        )
                    )

                else:
                    found[entity.id] = entity

        except ValueError as exc:
            rejected.append(
                rejection(
                    "ner",
                    "invalid_entity",
                    str(exc),
                    item=draft.model_dump(),
                )
            )

    entities = sorted(
        (
            entity
            for entity_id, entity in found.items()
            if entity_id not in conflicts
        ),
        key=lambda entity: (
            entity.start,
            entity.end,
        ),
    )

    # The LLM schema limits drafts, but one draft can now expand into
    # multiple repeated mentions. Enforce the final mention limit too.
    if len(entities) > constraints.max_entities:
        overflow = entities[constraints.max_entities:]

        for entity in overflow:
            rejected.append(
                rejection(
                    "ner",
                    "entity_limit_exceeded",
                    (
                        "Entity removed because deterministic occurrence "
                        "expansion exceeded max_entities"
                    ),
                    item=entity.model_dump(),
                )
            )

        entities = entities[:constraints.max_entities]

    return entities, rejected


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def build_candidates(
    entities,
    ontology,
    constraints,
):
    candidates = []

    for src in entities:
        for dst in entities:
            if (
                src.id == dst.id
                and not constraints.allow_self_relations
            ):
                continue

            types = (
                ontology.entity_types[
                    src.entity_type
                ].id,
                ontology.entity_types[
                    dst.entity_type
                ].id,
            )

            for name, kind in sorted(
                ontology.relationship_types.items()
            ):
                if types != (
                    kind.src_entity_type_id,
                    kind.dst_entity_type_id,
                ):
                    continue

                if (
                    len(candidates)
                    >= constraints.max_candidates
                ):
                    raise ValueError(
                        "max_candidates exceeded. "
                        "Increase it in constraints.json "
                        "or use a smaller source."
                    )

                candidates.append(
                    Candidate(
                        id=f"c{len(candidates)}",
                        src_id=src.id,
                        relationship_type=name,
                        dst_id=dst.id,
                    )
                )

    return candidates


# ---------------------------------------------------------------------------
# Relation validation
# ---------------------------------------------------------------------------

def validate_relations(
    raw,
    text,
    entities,
    candidates,
    ontology,
    constraints,
):
    by_entity_id = {
        entity.id: entity
        for entity in entities
    }

    context = {
        "text": text,
        "entities": by_entity_id,
        "ontology": ontology,
        "constraints": constraints,
    }

    output = REOutput.model_validate(
        raw,
        context=context,
    )

    if len(output.relations) > len(candidates):
        raise ValueError(
            "RE returned more relations than supplied candidates"
        )

    by_candidate_id = {
        candidate.id: candidate
        for candidate in candidates
    }

    relations = []
    rejected = []
    seen = set()

    for draft in output.relations:
        try:
            candidate_id = draft.candidate_id

            if candidate_id not in by_candidate_id:
                raise ValueError(
                    "Candidate ID was not supplied to the model"
                )

            if candidate_id in seen:
                rejected.append(
                    rejection(
                        "re",
                        "duplicate_candidate_selection",
                        "Candidate selected more than once",
                        candidate_id=candidate_id,
                    )
                )
                continue

            seen.add(candidate_id)

            candidate = by_candidate_id[candidate_id]

            src = by_entity_id.get(candidate.src_id)
            dst = by_entity_id.get(candidate.dst_id)

            if src is None or dst is None:
                raise ValueError(
                    "Candidate references an entity that does not exist"
                )

            # Shortest contiguous span containing both fixed endpoint mentions.
            evidence_start = min(
                src.start,
                dst.start,
            )

            evidence_end = max(
                src.end,
                dst.end,
            )

            evidence = text[
                evidence_start:evidence_end
            ]

            if (
                constraints.max_evidence_chars
                and len(evidence) > constraints.max_evidence_chars
            ):
                raise ValueError(
                    "Deterministic evidence span exceeds "
                    "max_evidence_chars"
                )

            relation = Relation.model_validate(
                {
                    "src_id": candidate.src_id,
                    "dst_id": candidate.dst_id,
                    "relationship_type": (
                        candidate.relationship_type
                    ),
                    "evidence": evidence,
                    "evidence_start": evidence_start,
                    "evidence_end": evidence_end,
                },
                context=context,
            )

            relations.append(relation)

        except ValueError as exc:
            rejected.append(
                rejection(
                    "re",
                    "invalid_relation",
                    str(exc),
                    item=draft.model_dump(),
                )
            )

    return relations, rejected

# ---------------------------------------------------------------------------
# Relation ranking / graph selection
# ---------------------------------------------------------------------------

def has_path(children, start, target):
    pending = [start]
    visited = set()

    while pending:
        node = pending.pop()

        if node == target:
            return True

        if node not in visited:
            visited.add(node)
            pending.extend(
                children.get(node, [])
            )

    return False


def rank_and_select_relations(
    relations,
    text,
    entities,
    ontology,
    constraints,
):
    by_id = {
        entity.id: entity
        for entity in entities
    }

    context = {
        "text": text,
        "entities": by_id,
        "ontology": ontology,
        "constraints": constraints,
    }

    weights = constraints.ranking.model_dump()
    total = sum(weights.values())

    ranked = []

    for relation in relations:
        src = by_id[relation.src_id]
        dst = by_id[relation.dst_id]

        left, right = sorted(
            (src, dst),
            key=lambda entity: (
                entity.start,
                entity.end,
            ),
        )

        gap = max(
            0,
            right.start - left.end,
        )

        between = text[
            left.end:right.start
        ]

        width = (
            max(src.end, dst.end)
            - min(src.start, dst.start)
        )

        evidence_width = (
            relation.evidence_end
            - relation.evidence_start
        )

        features = {
            "same_sentence": float(
                not re.search(
                    r"[.!?](?:\s|$)|\n\s*\n",
                    between,
                )
            ),
            "proximity": (
                1 / (1 + gap / 80)
            ),
            "evidence_compactness": (
                min(
                    1.0,
                    width / evidence_width,
                )
                if evidence_width > 0
                else 0.0
            ),
        }

        score = round(
            sum(
                weights[name] * value
                for name, value
                in features.items()
            )
            / total,
            6,
        )

        ranked.append(
            RankedRelation.model_validate(
                {
                    "relation": (
                        relation.model_dump()
                    ),
                    "score": score,
                    "features": features,
                },
                context=context,
            )
        )

    ranked.sort(
        key=lambda ranked_relation: (
            -ranked_relation.score,
            ranked_relation.relation.src_id,
            ranked_relation.relation.dst_id,
            ranked_relation.relation.relationship_type,
            ranked_relation.relation.evidence_start,
            ranked_relation.relation.evidence_end,
        )
    )

    selected = []
    rejected = []
    seen = set()
    children = {}
    incoming = {}

    for ranked_relation in ranked:
        relation = ranked_relation.relation

        src = relation.src_id
        dst = relation.dst_id

        key = (
            src,
            relation.relationship_type,
            dst,
        )

        reason = None

        if key in seen:
            reason = "duplicate_relation"

        elif (
            ranked_relation.score
            < constraints.min_relation_score
        ):
            reason = "below_minimum_score"

        elif (
            constraints.max_incoming_relations
            and incoming.get(dst, 0)
            >= constraints.max_incoming_relations
        ):
            reason = "incoming_relation_limit"

        elif (
            not constraints.allow_cycles
            and has_path(
                children,
                dst,
                src,
            )
        ):
            reason = "would_create_cycle"

        seen.add(key)

        if reason:
            rejected.append(
                rejection(
                    "re",
                    reason,
                    (
                        "Relation removed during "
                        "deterministic selection"
                    ),
                    item=ranked_relation.model_dump(),
                )
            )

        else:
            selected.append(relation)

            incoming[dst] = (
                incoming.get(dst, 0) + 1
            )

            children.setdefault(
                src,
                [],
            ).append(dst)

    return selected, ranked, rejected


# ---------------------------------------------------------------------------
# Final invariant checks
# ---------------------------------------------------------------------------

def confirm_extraction(
    entities,
    relations,
    text,
    ontology,
    constraints,
):
    """
    Recheck every invariant before serialization/upload,
    even if earlier helpers are edited.
    """

    context = {
        "text": text,
        "ontology": ontology,
        "constraints": constraints,
    }

    if (
        len(entities) > constraints.max_entities
        or len(relations) > constraints.max_relations
    ):
        raise ValueError(
            "Final extraction exceeds configured limits"
        )

    if (
        len({entity.id for entity in entities})
        != len(entities)
    ):
        raise ValueError(
            "Final extraction contains duplicate mention IDs"
        )

    for entity in entities:
        Entity.model_validate(
            entity.model_dump(),
            context=context,
        )

    context["entities"] = {
        entity.id: entity
        for entity in entities
    }

    seen = set()
    children = {}
    incoming = {}

    for relation in relations:
        Relation.model_validate(
            relation.model_dump(),
            context=context,
        )

        key = (
            relation.src_id,
            relation.relationship_type,
            relation.dst_id,
        )

        if key in seen:
            raise ValueError(
                "Final extraction contains duplicate relations"
            )

        seen.add(key)

        dst = relation.dst_id
        src = relation.src_id

        incoming[dst] = (
            incoming.get(dst, 0) + 1
        )

        if (
            constraints.max_incoming_relations
            and incoming[dst]
            > constraints.max_incoming_relations
        ):
            raise ValueError(
                "Final extraction violates "
                "incoming-relation limit"
            )

        if (
            not constraints.allow_cycles
            and has_path(
                children,
                dst,
                src,
            )
        ):
            raise ValueError(
                "Final extraction contains a cycle"
            )

        children.setdefault(
            src,
            [],
        ).append(dst)