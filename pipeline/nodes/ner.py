"""Node 2: contextual NER + LLM semantic grounding."""

import json
import re

from langgraph.runtime import Runtime

from ..helpers.llm import call_llm
from ..helpers.validation import decode_response
from ..helpers.dictionary_ner import (
    prepare_terms,
    validate_dictionary_entities,
)
from ..state import Context, PipelineState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Only these entity types are extracted.
ENTITY_TYPES = (
    "lith",
    "lith_att",
    "environ",
)

# Number of Macrostrat terms Qwen sees at one time during grounding.
TERM_BATCH_SIZE = 50

# Each dictionary batch can nominate at most this many terms per mention.
CANDIDATES_PER_BATCH = 2

# Context around each source mention.
CONTEXT_CHARS = 250


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _get(obj, key, default=None):
    """Read a value from either a dict or an object."""

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _chunks(items, size):
    """Split a list into fixed-size chunks."""

    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------------------------------------------------------------------
# Stage 1: Contextual entity discovery
# ---------------------------------------------------------------------------

def discovery_prompt(
    text,
    ontology,
    constraints,
):
    """
    Discover geological entities using only:

    - paragraph context
    - ontology entity-type definitions

    No Macrostrat term dictionary is supplied at this stage.
    """

    definitions = {
        name: ontology.entity_types[name].description or ""
        for name in ENTITY_TYPES
        if name in ontology.entity_types
    }

    instructions = """
Extract every explicit geological entity mention from the paragraph that
belongs to one of the supplied entity types.

Use the supplied entity-type definitions to determine what qualifies.

IMPORTANT:

- Find entities from geological meaning and context.
- There is NO dictionary of allowed geological terms in this step.
- A valid entity does NOT need to match a known database term.
- Do NOT assign Macrostrat IDs.
- Copy the exact source text.
- Do NOT normalize spelling, plurality, capitalization, or hyphenation.
- Do NOT paraphrase.
- Do NOT extract citations or author names.
- Do NOT extract locations.
- Do NOT extract stratigraphic names.
- Do NOT infer entities that are not explicitly expressed in the paragraph.
- Return each distinct (text, entity_type) pair once.
- Repeated occurrences will be located by code afterward.

ENTITY-TYPE GUIDANCE:

lith:
A rock, sediment, or lithologic material.

Examples include sandstone, conglomerate, argillite, sand, limestone,
mudstone, diamictite, basalt, shale, etc.

lith_att:
A property or feature describing a rock, sediment, bed, or rock body.

This includes:

- composition
- color
- grain size
- sorting
- texture
- bedding
- stratification
- lamination
- sedimentary structures
- deformation structures
- other descriptive lithologic features

A lith_att does NOT need to be an adjective.

Noun phrases describing geological structures can be lith_att.

Examples include:

- cross bedding
- ripple cross lamination
- load casts
- ball and pillow structure
- soft sediment deformation
- convolute bedding
- graded bedding
- laminated
- poorly sorted

environ:
An explicitly stated depositional or geological environment.

Examples include:

- marine
- fluvial
- lacustrine
- deltaic
- tidal flat
- offshore
- alluvial fan

Do NOT classify a sedimentary process, current, mechanism, or structure
as an environment merely because it may imply one.

For example:

"turbulent currents" is NOT itself a depositional environment.

Before returning the answer, scan the entire paragraph again for:

1. rocks and sediments
2. lithologic modifiers
3. composition
4. color
5. grain size
6. sorting and texture
7. bedding and stratification
8. lamination
9. sedimentary structures
10. deformation structures
11. explicitly stated depositional environments

Return only JSON matching the supplied schema.
"""

    messages = [
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "paragraph": text,
                    "entity_type_definitions": definitions,
                },
                ensure_ascii=False,
            ),
        },
    ]

    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "maxItems": constraints.max_entities,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": (
                                "Exact contiguous text copied "
                                "from the paragraph."
                            ),
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": list(definitions),
                        },
                    },
                    "required": [
                        "text",
                        "entity_type",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["entities"],
        "additionalProperties": False,
    }

    return messages, schema


# ---------------------------------------------------------------------------
# Turn discovered entities into source occurrences
# ---------------------------------------------------------------------------

def expand_occurrences(
    discovery,
    text,
):
    """
    Convert each discovered phrase into occurrence-level mentions.

    Exact matching here is ONLY used to recover source offsets.

    It is NOT being used to determine whether something is an entity
    or which Macrostrat term it maps to.
    """

    mentions = []
    seen = set()

    for entity in discovery.get("entities", []):
        mention_text = entity["text"]
        entity_type = entity["entity_type"]

        if entity_type not in ENTITY_TYPES:
            continue

        # Exact whole-word source occurrence matching.
        pattern = re.compile(
            rf"(?<!\w){re.escape(mention_text)}(?!\w)"
        )

        matches = list(
            pattern.finditer(text)
        )

        for occurrence, match in enumerate(matches):
            start = match.start()
            end = match.end()

            key = (
                start,
                end,
                entity_type,
            )

            if key in seen:
                continue

            seen.add(key)

            context_start = max(
                0,
                start - CONTEXT_CHARS,
            )

            context_end = min(
                len(text),
                end + CONTEXT_CHARS,
            )

            mentions.append(
                {
                    "mention_id": None,
                    "text": mention_text,
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                    "occurrence": occurrence,
                    "context": text[
                        context_start:context_end
                    ],
                }
            )

    mentions.sort(
        key=lambda mention: (
            mention["start"],
            mention["end"],
        )
    )

    for i, mention in enumerate(mentions):
        mention["mention_id"] = f"m{i}"

    return mentions


# ---------------------------------------------------------------------------
# Stage 2A: Candidate generation
# ---------------------------------------------------------------------------

def candidate_prompt(
    mentions,
    terms,
    entity_type,
):
    """
    Ask Qwen whether any terms in a SMALL dictionary chunk are
    semantically equivalent to the discovered mentions.
    """

    mention_views = [
        {
            "mention_id": mention["mention_id"],
            "text": mention["text"],
            "context": mention["context"],
        }
        for mention in mentions
    ]

    term_views = []

    for term in terms:
        aliases = (
            _get(
                term,
                "aliases",
                [],
            )
            or []
        )

        term_views.append(
            {
                "id": int(
                    _get(term, "id")
                ),
                "name": _get(
                    term,
                    "name",
                    "",
                ),
                "aliases": aliases,
            }
        )

    instructions = f"""
You are performing semantic ontology matching for geological entities.

All discovered mentions supplied in this request have entity type:

{entity_type}

Compare each discovered mention against the supplied Macrostrat terms.

This is a SEMANTIC matching task.

DO NOT simply compare character strings.

Use:

- geological meaning
- paragraph context
- grammatical variants
- noun/adjective variants
- singular/plural variants
- spelling variants
- spacing variants
- hyphenation variants
- geological synonyms

Examples of valid semantic equivalence:

"ball and pillow structure"
    -> "ball-and-pillow"

"ball and pillow structures"
    -> "ball-and-pillow"

"loadcasts"
    -> "load casts"

"cross bedding"
    -> "cross-bedded"

"cross bedding"
    -> "cross stratification"
when the geological meaning is equivalent.

Do NOT select a term merely because its spelling looks similar.

For example:

"basal"
must NOT match
"basalt"

"unit"
must NOT match
"dunite"

Only nominate a term when it could genuinely represent the same
geological concept.

The supplied term IDs are arbitrary database identifiers.
Their numerical values have no semantic meaning.

You do NOT have to choose anything from this batch.

If this batch contains no plausible canonical equivalent for a mention,
return an empty candidate_term_ids list for that mention.

Return at most {CANDIDATES_PER_BATCH} plausible term IDs per mention.

Do not invent IDs.
Only use IDs present in the supplied terms.
"""

    messages = [
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "mentions": mention_views,
                    "terms": term_views,
                },
                ensure_ascii=False,
            ),
        },
    ]

    schema = {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mention_id": {
                            "type": "string",
                        },
                        "candidate_term_ids": {
                            "type": "array",
                            "maxItems": CANDIDATES_PER_BATCH,
                            "items": {
                                "type": "integer",
                            },
                        },
                    },
                    "required": [
                        "mention_id",
                        "candidate_term_ids",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["selections"],
        "additionalProperties": False,
    }

    return messages, schema


def collect_candidates(
    ctx,
    source_id,
    mentions,
    terms,
):
    """
    Search the Macrostrat dictionary semantically using Qwen.

    IMPORTANT:

    `terms` remains a dict because validate_dictionary_entities()
    expects that structure later.

    Returns:

        candidates_by_mention, calls
    """

    candidates_by_mention = {
        mention["mention_id"]: []
        for mention in mentions
    }

    calls = []

    # prepare_terms() returns a dict.
    # Use values ONLY when iterating over actual term objects.
    term_values = list(
        terms.values()
    )

    terms_by_id = {
        int(_get(term, "id")): term
        for term in term_values
    }

    for entity_type in ENTITY_TYPES:
        type_mentions = [
            mention
            for mention in mentions
            if mention["entity_type"] == entity_type
        ]

        if not type_mentions:
            continue

        type_terms = [
            term
            for term in term_values
            if _get(
                term,
                "entity_type",
            ) == entity_type
        ]

        print(
            f"\nNER grounding {entity_type}: "
            f"{len(type_mentions)} mentions, "
            f"{len(type_terms)} Macrostrat terms"
        )

        for batch_index, term_batch in enumerate(
            _chunks(
                type_terms,
                TERM_BATCH_SIZE,
            )
        ):
            print(
                f"  batch {batch_index + 1}: "
                f"{len(term_batch)} terms"
            )

            messages, schema = candidate_prompt(
                type_mentions,
                term_batch,
                entity_type,
            )

            raw, call = call_llm(
                ctx,
                "ner",
                source_id,
                messages,
                schema,
            )

            calls.append(call)

            output = decode_response(
                raw,
                call,
            )

            # IDs Qwen is actually allowed to return
            # for this specific batch.
            batch_valid_ids = {
                int(_get(term, "id"))
                for term in term_batch
            }

            for selection in output.get(
                "selections",
                [],
            ):
                mention_id = selection.get(
                    "mention_id"
                )

                if (
                    mention_id
                    not in candidates_by_mention
                ):
                    continue

                for term_id in selection.get(
                    "candidate_term_ids",
                    [],
                ):
                    term_id = int(term_id)

                    # Ignore hallucinated IDs.
                    if term_id not in batch_valid_ids:
                        continue

                    if (
                        term_id
                        not in candidates_by_mention[
                            mention_id
                        ]
                    ):
                        candidates_by_mention[
                            mention_id
                        ].append(term_id)

    # Turn IDs back into useful candidate objects
    # for the final grounding call.
    final_candidates = {}

    for mention in mentions:
        mention_id = mention["mention_id"]

        ids = candidates_by_mention[
            mention_id
        ]

        final_candidates[mention_id] = [
            {
                "id": term_id,
                "name": _get(
                    terms_by_id[term_id],
                    "name",
                    "",
                ),
                "entity_type": _get(
                    terms_by_id[term_id],
                    "entity_type",
                    "",
                ),
                "aliases": (
                    _get(
                        terms_by_id[term_id],
                        "aliases",
                        [],
                    )
                    or []
                ),
            }
            for term_id in ids
            if term_id in terms_by_id
        ]

    return (
        final_candidates,
        calls,
    )


# ---------------------------------------------------------------------------
# Stage 2B: Final candidate decision
# ---------------------------------------------------------------------------

def grounding_prompt(
    mentions,
    candidates_by_mention,
):
    """
    Final Qwen pass.

    Qwen now sees only the small candidate set that survived
    semantic candidate generation.
    """

    items = []

    for mention in mentions:
        mention_id = mention[
            "mention_id"
        ]

        items.append(
            {
                "mention_id": mention_id,
                "text": mention["text"],
                "entity_type": mention[
                    "entity_type"
                ],
                "context": mention["context"],
                "candidates":
                    candidates_by_mention.get(
                        mention_id,
                        [],
                    ),
            }
        )

    instructions = """
Choose the best Macrostrat canonical term for each discovered geological
mention.

The mentions themselves have already been identified from the source
paragraph.

Your job is only to determine whether one of the supplied candidate
terms represents the SAME geological concept.

Use:

- geological meaning
- paragraph context
- candidate term names
- candidate aliases
- entity type

Do NOT choose based merely on similar spelling.

Different source and canonical wording is expected.

Examples:

"ball and pillow structure"
can map to
"ball-and-pillow"

"ball and pillow structures"
can map to
"ball-and-pillow"

"loadcasts"
can map to
"load casts"

"cross bedding"
can map to
"cross-bedded"
or another canonical cross-stratification term when semantically
appropriate.

If NONE of the candidates represents the same geological concept,
return null.

If the candidate list is empty, return null.

Do not invent term IDs.

Do not change entity types.

Return one decision for every supplied mention.
"""

    messages = [
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "mentions": items,
                },
                ensure_ascii=False,
            ),
        },
    ]

    schema = {
        "type": "object",
        "properties": {
            "groundings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mention_id": {
                            "type": "string",
                        },
                        "term_id": {
                            "anyOf": [
                                {
                                    "type": "integer",
                                },
                                {
                                    "type": "null",
                                },
                            ],
                        },
                    },
                    "required": [
                        "mention_id",
                        "term_id",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["groundings"],
        "additionalProperties": False,
    }

    return messages, schema


# ---------------------------------------------------------------------------
# Convert grounding decisions to existing validator format
# ---------------------------------------------------------------------------

def build_validator_output(
    mentions,
    candidates_by_mention,
    grounding_output,
):
    """
    Convert semantic-grounding decisions into the format already expected
    by validate_dictionary_entities().
    """

    grounding_by_id = {
        item["mention_id"]:
            item["term_id"]
        for item in grounding_output.get(
            "groundings",
            [],
        )
    }

    entities = []

    for mention in mentions:
        mention_id = mention[
            "mention_id"
        ]

        term_id = grounding_by_id.get(
            mention_id
        )

        if term_id is None:
            continue

        term_id = int(term_id)

        # Only allow an ID that appeared in the candidate
        # list for this specific mention.
        valid_ids = {
            int(candidate["id"])
            for candidate
            in candidates_by_mention.get(
                mention_id,
                [],
            )
        }

        if term_id not in valid_ids:
            continue

        entities.append(
            {
                "text": mention["text"],
                "entity_type": mention[
                    "entity_type"
                ],
                "occurrence": mention[
                    "occurrence"
                ],
                "status": "linked",
                "macrostrat_terms_id":
                    term_id,
                "candidate_term_ids": [
                    term_id
                ],
            }
        )

    return {
        "entities": entities,
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def ner(
    state: PipelineState,
    runtime: Runtime[Context],
):
    ctx = runtime.context

    results = [
        result.model_copy(
            deep=True
        )
        for result
        in state["results"]
    ]

    # -----------------------------------------------------------------------
    # Prepare dictionary once.
    #
    # IMPORTANT:
    #
    # prepare_terms() returns a DICT.
    #
    # Keep it as a dict because validate_dictionary_entities()
    # calls terms.values().
    # -----------------------------------------------------------------------

    all_terms = prepare_terms(
        state["macrostrat_terms"],
        state["ontology"],
    )

    # Keep only the entity types that this NER node handles,
    # but preserve the dictionary structure.
    terms = {
        term_id: term
        for term_id, term
        in all_terms.items()
        if _get(
            term,
            "entity_type",
        ) in ENTITY_TYPES
    }

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
            text = result.source[
                "paragraph_text"
            ]

            # ==============================================================
            # STEP 1
            #
            # Context-only NER.
            #
            # Qwen sees:
            #
            # paragraph
            # +
            # entity type definitions
            #
            # It does NOT see the Macrostrat dictionary.
            # ==============================================================

            messages, schema = discovery_prompt(
                text,
                state["ontology"],
                ctx.constraints,
            )

            raw, discovery_call = call_llm(
                ctx,
                "ner",
                result.source_id,
                messages,
                schema,
            )

            result.llm_calls.append(
                discovery_call
            )

            result.models["ner"] = (
                discovery_call["model"]
            )

            discovery = decode_response(
                raw,
                discovery_call,
            )

            ctx.log.write(
                (
                    f"sources/"
                    f"{result.source_id}/"
                    f"ner_discovery.json"
                ),
                discovery,
            )

            print(
                "\n========== NER DISCOVERY =========="
            )

            print(
                json.dumps(
                    discovery,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            print(
                "===================================\n"
            )

            # ==============================================================
            # STEP 2
            #
            # Recover source occurrences.
            # ==============================================================

            mentions = expand_occurrences(
                discovery,
                text,
            )

            ctx.log.write(
                (
                    f"sources/"
                    f"{result.source_id}/"
                    f"ner_mentions.json"
                ),
                {
                    "mentions": mentions,
                },
            )

            # ==============================================================
            # STEP 3
            #
            # Semantic candidate generation.
            #
            # Qwen sees only small batches of SAME-TYPE
            # Macrostrat terms.
            # ==============================================================

            (
                candidates_by_mention,
                candidate_calls,
            ) = collect_candidates(
                ctx,
                result.source_id,
                mentions,
                terms,
            )

            result.llm_calls.extend(
                candidate_calls
            )

            ctx.log.write(
                (
                    f"sources/"
                    f"{result.source_id}/"
                    f"ner_candidates.json"
                ),
                {
                    "mentions": [
                        {
                            **mention,
                            "candidates":
                                candidates_by_mention.get(
                                    mention[
                                        "mention_id"
                                    ],
                                    [],
                                ),
                        }
                        for mention
                        in mentions
                    ],
                },
            )

            print(
                "\n========== NER CANDIDATES =========="
            )

            for mention in mentions:
                mention_id = mention[
                    "mention_id"
                ]

                print(
                    f"\n{mention_id}: "
                    f"{mention['text']} "
                    f"({mention['entity_type']})"
                )

                for candidate in (
                    candidates_by_mention.get(
                        mention_id,
                        [],
                    )
                ):
                    print(
                        f"  {candidate['id']} "
                        f"{candidate['name']}"
                    )

            print(
                "\n====================================\n"
            )

            # ==============================================================
            # STEP 4
            #
            # Final semantic grounding.
            # ==============================================================

            if mentions:
                (
                    messages,
                    schema,
                ) = grounding_prompt(
                    mentions,
                    candidates_by_mention,
                )

                raw, grounding_call = call_llm(
                    ctx,
                    "ner",
                    result.source_id,
                    messages,
                    schema,
                )

                result.llm_calls.append(
                    grounding_call
                )

                grounding_output = (
                    decode_response(
                        raw,
                        grounding_call,
                    )
                )

            else:
                grounding_output = {
                    "groundings": [],
                }

            ctx.log.write(
                (
                    f"sources/"
                    f"{result.source_id}/"
                    f"ner_grounding_decisions.json"
                ),
                grounding_output,
            )

            # ==============================================================
            # STEP 5
            #
            # Convert output into format expected by your
            # existing dictionary validator.
            # ==============================================================

            output = build_validator_output(
                mentions,
                candidates_by_mention,
                grounding_output,
            )

            ctx.log.write(
                (
                    f"sources/"
                    f"{result.source_id}/"
                    f"ner_pre_validation.json"
                ),
                output,
            )

            # ==============================================================
            # STEP 6
            #
            # Existing occurrence / ID validation.
            #
            # `terms` MUST remain a dict here.
            # ==============================================================

            (
                result.entities,
                rejected,
                match_stats,
                grounding,
            ) = validate_dictionary_entities(
                output,
                text,
                state["ontology"],
                ctx.constraints,
                terms,
            )

            ctx.log.write(
                (
                    f"sources/"
                    f"{result.source_id}/"
                    f"ner_grounding.json"
                ),
                {
                    "mentions": grounding,
                    "statistics":
                        match_stats,
                },
            )

            result.rejected.extend(
                rejected
            )

            ctx.log.rejections(
                result.source_id,
                rejected,
            )

            # ==============================================================
            # Diagnostics
            # ==============================================================

            if (
                discovery["entities"]
                and not result.entities
            ):
                ctx.log.event(
                    "WARNING",
                    "all_entities_rejected",
                    stage="ner",
                    source_id=result.source_id,
                    message=(
                        "NER discovered entities "
                        "but none survived grounding "
                        "and validation"
                    ),
                )

            if (
                len(discovery["entities"])
                == ctx.constraints.max_entities
            ):
                ctx.log.event(
                    "WARNING",
                    "entity_limit_reached",
                    stage="ner",
                    source_id=result.source_id,
                    message=(
                        "NER discovery reached "
                        "max_entities; additional "
                        "mentions may have been omitted"
                    ),
                )

            grounded_count = len(
                output["entities"]
            )

            result.status = (
                "ner_complete"
            )

            ctx.log.event(
                "INFO",
                "completed",
                stage="ner",
                source_id=result.source_id,
                discovered=len(
                    discovery["entities"]
                ),
                occurrences=len(
                    mentions
                ),
                grounded=grounded_count,
                validated=len(
                    result.entities
                ),
                rejected=len(
                    rejected
                ),
                matched=match_stats[
                    "matched"
                ],
                unmatched=match_stats[
                    "unmatched"
                ],
                ambiguous=match_stats[
                    "ambiguous"
                ],
            )

        except Exception as exc:
            if hasattr(
                exc,
                "call_info",
            ):
                result.llm_calls.append(
                    exc.call_info
                )

            result.status = "failed"

            result.errors.append(
                ctx.log.exception(
                    "ner",
                    exc,
                    result.source_id,
                )
            )

        finally:
            # There are now multiple NER LLM calls.
            # Give each one a unique log filename.
            ner_calls = [
                call
                for call
                in result.llm_calls
                if call.get(
                    "stage"
                ) == "ner"
            ]

            for i, call in enumerate(
                ner_calls
            ):
                ctx.log.write(
                    (
                        f"sources/"
                        f"{result.source_id}/"
                        f"ner_call_"
                        f"{i + 1}.json"
                    ),
                    call,
                )

            ctx.log.save_source(
                result
            )

    return {
        "results": results,
    }