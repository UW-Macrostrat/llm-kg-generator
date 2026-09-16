def match_entities(entities, terms):
    for entity in entities:
        matching_ids = {
            term["macrostrat_terms_id"]
            for term in terms
            if term["name"].casefold() == entity.text.casefold()
        }

        # Leave unmatched or ambiguous names unlinked.
        entity.macrostrat_terms_id = (
            next(iter(matching_ids))
            if len(matching_ids) == 1
            else None
        )