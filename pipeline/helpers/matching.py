import re
import unicodedata
from collections import defaultdict


def normalize_term(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold().strip()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_type(entity_type: str) -> str:
    """Normalize singular/plural category names."""
    return entity_type.casefold().strip().removesuffix("s")


def match_entities(entities, terms):
    """
    Dynamically discover supported vocabulary categories from terms.

    Match entities by normalized category and normalized name.
    Assign an ID only when exactly one distinct ID matches.
    """

    term_index = defaultdict(set)
    supported_types = set()

    # Build the vocabulary index dynamically.
    for term in terms:
        term_type = term.get("entity_type")
        term_name = term.get("name")
        term_id = term.get("macrostrat_terms_id")

        if not isinstance(term_type, str):
            continue

        if not isinstance(term_name, str):
            continue

        if type(term_id) is not int or term_id <= 0:
            continue

        category = normalize_type(term_type)
        name = normalize_term(term_name)

        if not category or not name:
            continue

        supported_types.add(category)
        term_index[(category, name)].add(term_id)

    stats = {
        "matched": 0,
        "unmatched": 0,
        "ambiguous": 0,
        "unsupported": 0,
    }

    for entity in entities:
        entity.macrostrat_terms_id = None

        category = normalize_type(entity.entity_type)

        if category not in supported_types:
            stats["unsupported"] += 1
            continue

        name = normalize_term(entity.text)
        matching_ids = term_index.get((category, name), set())

        if len(matching_ids) == 1:
            entity.macrostrat_terms_id = next(iter(matching_ids))
            stats["matched"] += 1

        elif not matching_ids:
            stats["unmatched"] += 1
            print(
                f"UNMATCHED: {entity.text!r} "
                f"type={entity.entity_type}"
            )

        else:
            stats["ambiguous"] += 1
            print(
                f"AMBIGUOUS: {entity.text!r} "
                f"type={entity.entity_type} "
                f"ids={sorted(matching_ids)}"
            )

    return stats