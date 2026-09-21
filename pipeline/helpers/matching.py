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


def match_entities(entities, terms):
    term_index = defaultdict(list)

    for term in terms:
        name = normalize_term(term["name"])

        if not name:
            continue

        term_index[name].append(term)

    stats = {
        "matched": 0,
        "unmatched": 0,
        "ambiguous": 0,
    }

    for entity in entities:
        normalized = normalize_term(entity.text)

        candidates = term_index.get(
            normalized,
            [],
        )

        matching_ids = {
            candidate["macrostrat_terms_id"]
            for candidate in candidates
            if candidate.get("macrostrat_terms_id") is not None
        }

        if len(matching_ids) == 1:
            entity.macrostrat_terms_id = next(
                iter(matching_ids)
            )

            stats["matched"] += 1

        elif len(matching_ids) == 0:
            entity.macrostrat_terms_id = None

            stats["unmatched"] += 1

            print(
                f"UNMATCHED: "
                f"{entity.text!r} "
                f"type={entity.entity_type} "
                f"normalized={normalized!r}"
            )

        else:
            entity.macrostrat_terms_id = None

            stats["ambiguous"] += 1

            print(
                f"AMBIGUOUS: "
                f"{entity.text!r} "
                f"type={entity.entity_type} "
                f"ids={sorted(matching_ids)} "
                f"candidates={candidates}"
            )

    return stats