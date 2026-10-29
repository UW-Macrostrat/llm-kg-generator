"""Offline contract and full-graph tests for scoped dictionary NER."""

import json

import httpx
import pytest

from pipeline.config import Config, parse_args
from pipeline.helpers.dictionary_ner import (
    dictionary_ner_prompt,
    find_dictionary_mentions,
    prepare_terms,
    validate_dictionary_entities,
)
from pipeline.models import Constraints, Ontology
from pipeline.pipeline import run_pipeline


ENTITY_TYPES = [
    {"id": 1, "name": "strat_name"},
    {"id": 2, "name": "lith"},
    {"id": 3, "name": "lith_att"},
    {"id": 12, "name": "mineral"},
    {"id": 19, "name": "environ"},
]
RELATION_TYPES = [
    {"id": 11, "name": "lith_to_lith_att", "src_entity_type_id": 2, "dst_entity_type_id": 3},
    {"id": 34, "name": "strat_name_to_lith", "src_entity_type_id": 1, "dst_entity_type_id": 2},
    {"id": 43, "name": "lith_to_environ", "src_entity_type_id": 2, "dst_entity_type_id": 19},
]
TERMS = [
    {"macrostrat_terms_id": 8080, "name": "marine", "entity_type": "environs"},
    {"macrostrat_terms_id": 8199, "name": "sandstone", "entity_type": "liths", "aliases": ["sand rock"]},
    {"macrostrat_terms_id": 9000, "name": "thin", "entity_type": "lith_atts"},
]
ONTOLOGY = Ontology.from_rows(ENTITY_TYPES, RELATION_TYPES)
RULES = Constraints()


def draft(text="sandstone", term_id=8199, entity_type="lith", occurrence=0, **changes):
    result = {
        "text": text, "entity_type": entity_type, "occurrence": occurrence,
        "status": "linked", "macrostrat_terms_id": term_id,
        "candidate_term_ids": [term_id],
    }
    result.update(changes)
    return result


def validate(text, *drafts, rows=TERMS):
    return validate_dictionary_entities(
        {"entities": list(drafts)}, text, ONTOLOGY, RULES, prepare_terms(rows, ONTOLOGY),
    )


def test_cli_model_source_and_context():
    cfg = parse_args([
        "--env-file", "/nonexistent.env", "--model", "qwen3", "--source-id", "32749",
        "--strat-name", "  Example Formation  ", "--fuzzy-threshold", "0.9",
    ])
    assert (cfg.model, cfg.source_ids, cfg.strat_name) == ("qwen3", [32749], "Example Formation")
    assert cfg.fuzzy_threshold == 0.9
    assert cfg.dry_run is True


@pytest.mark.parametrize("kwargs", [
    {"strat_name": " "}, {"strat_name": "Example Formation"},
    {"strat_name": "Example Formation", "source_ids": [1, 2]},
    {"fuzzy_threshold": 0}, {"fuzzy_threshold": 1.1},
])
def test_invalid_cli_combinations(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_full_vocabulary_and_context_in_prompt():
    terms = prepare_terms(TERMS, ONTOLOGY)
    messages, schema = dictionary_ner_prompt(
        "sandstone", terms, [], RULES, strat_name="Example Formation",
    )
    payload = json.loads(messages[1]["content"])
    assert len(payload["allowed_terms"]) == 3  # Includes terms with no string matches.
    assert payload["known_strat_name"] == "Example Formation"
    assert schema["properties"]["entities"]["items"]["properties"]["entity_type"]["enum"] == [
        "environ", "lith", "lith_att",
    ]


def test_boundaries_aliases_and_overlap():
    rows = TERMS + [{"macrostrat_terms_id": 9200, "name": "red sandstone", "entity_type": "liths"}]
    hints = find_dictionary_mentions(
        "Sandstone red sandstone sand rock sandstones _sandstone", prepare_terms(rows, ONTOLOGY),
    )
    assert [hint["text"] for hint in hints] == ["Sandstone", "red sandstone", "sandstone", "sand rock"]


def test_fuzzy_is_optional_and_never_auto_accepted():
    terms = prepare_terms(TERMS, ONTOLOGY)
    assert not find_dictionary_mentions("sandstome", terms)
    hints = find_dictionary_mentions("sandstome", terms, fuzzy_min_score=0.88)
    assert hints[0]["candidates"][0]["method"] == "fuzzy"
    assert not validate("sandstome")[0]  # No LLM selection => no entity.


def test_occurrences_are_independent():
    entities, _, _, _ = validate("sandstone sandstone", draft(occurrence=1))
    assert len(entities) == 1
    assert entities[0].start == 10
    entities, _, _, _ = validate("sandstone sandstone", draft(), draft(occurrence=1))
    assert [entity.id for entity in entities] == ["e0_9", "e10_19"]


@pytest.mark.parametrize("prediction", [
    draft(term_id=999), draft(entity_type="mineral"), draft(occurrence=1),
    draft(text="Sandstone"), draft(occurrence=True), draft(term_id=True),
    draft(term_id="8199"), draft(candidate_term_ids=[8199, 8199]),
])
def test_invalid_predictions_are_rejected(prediction):
    entities, rejected, _, _ = validate("sandstone", prediction)
    assert not entities
    assert len(rejected) == 1


def test_identical_names_cannot_be_arbitrarily_linked():
    rows = TERMS + [{"macrostrat_terms_id": 9900, "name": "marine", "entity_type": "environs"}]
    entities, _, stats, grounding = validate("marine", draft("marine", 8080, "environ"), rows=rows)
    assert entities[0].macrostrat_terms_id is None
    assert grounding[0]["candidate_term_ids"] == [8080, 9900]
    assert stats["ambiguous"] == 1


def test_conflicting_ids_merge_to_ambiguity():
    rows = TERMS + [{"macrostrat_terms_id": 9901, "name": "quartz sandstone", "entity_type": "liths"}]
    entities, _, _, grounding = validate("sandstone", draft(), draft(term_id=9901), rows=rows)
    assert len(entities) == 1
    assert entities[0].macrostrat_terms_id is None
    assert grounding[0]["candidate_term_ids"] == [8199, 9901]


def test_explicit_ambiguity_is_preserved():
    rows = TERMS + [{"macrostrat_terms_id": 9901, "name": "quartz sandstone", "entity_type": "liths"}]
    entities, _, _, grounding = validate("sandstone", draft(
        term_id=None, status="ambiguous", candidate_term_ids=[8199, 9901],
    ), rows=rows)
    assert entities[0].macrostrat_terms_id is None
    assert grounding[0]["status"] == "ambiguous"


def test_duplicate_mentions_and_conflicting_types():
    entities, _, _, _ = validate("sandstone", draft(), draft())
    assert len(entities) == 1
    entities, rejected, _, _ = validate("sandstone", draft(), draft(term_id=9000, entity_type="lith_att"), draft())
    assert not entities
    assert rejected


def test_vocabulary_scope_and_no_fixed_count():
    rows = [{"macrostrat_terms_id": i + 1, "name": f"term {i}", "entity_type": "liths"} for i in range(482)]
    assert len(prepare_terms(rows, ONTOLOGY)) == 482
    with pytest.raises(ValueError, match="out-of-scope"):
        prepare_terms([{"macrostrat_terms_id": 1, "name": "quartz", "entity_type": "minerals"}], ONTOLOGY)


class FakeAPI:
    def __init__(self, *, invalid_ner=False, ignore_filter=False):
        self.calls = []
        self.term_pages = []
        self.invalid_ner = invalid_ner
        self.ignore_filter = ignore_filter

    def __call__(self, request):
        path, params = request.url.path, request.url.params
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen3"}, {"id": "glm-5"}]})
        if path.endswith("/chat/completions"):
            body = json.loads(request.content)
            self.calls.append(body)
            payload = json.loads(body["messages"][1]["content"])
            assert payload["known_strat_name"] == "Example Formation"
            if "paragraph" in payload:
                assert {row["entity_type"] for row in payload["allowed_terms"]} == {"lith", "lith_att", "environ"}
                assert len(payload["allowed_terms"]) == 3
                output = {"entities": [draft("Thin", 9000, "lith_att"), draft(), draft("marine", 8080, "environ")]}
                if self.invalid_ner:
                    output = {"entities": [draft(term_id=999)]}
            else:
                output = {"relations": [{"candidate_id": item["candidate_id"]} for item in payload["candidates"]]}
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}]})
        if path.endswith("/kg_source_text"):
            assert params["id"] == "eq.32749"
            return httpx.Response(200, json=[{
                "id": 32749, "paragraph_text": "Thin sandstone formed in a marine setting.",
                "source_text_type": "map_description",
            }])
        if path.endswith("/kg_macrostrat_terms"):
            assert params["entity_type"] == "in.(liths,lith_atts,environs)"
            self.term_pages.append(dict(params))
            cursor = int(params.get("macrostrat_terms_id", "gt.0").split(".")[1])
            page = [row for row in TERMS if row["macrostrat_terms_id"] > cursor][:1]
            if self.ignore_filter and page:
                page = [{**page[0], "entity_type": "minerals"}]
            return httpx.Response(200, json=page)
        if path.endswith("/kg_entity_type") or path.endswith("/kg_relationship_type"):
            rows = ENTITY_TYPES if path.endswith("/kg_entity_type") else RELATION_TYPES
            cursor = int(params.get("id", "gt.-1").split(".")[1])
            return httpx.Response(200, json=sorted([row for row in rows if row["id"] > cursor], key=lambda row: row["id"]))
        raise AssertionError(f"Unexpected HTTP request: {request.method} {request.url}")


@pytest.mark.parametrize("model", ["qwen3", "glm-5"])
def test_full_graph_dry_run(tmp_path, monkeypatch, model):
    monkeypatch.setattr("pipeline.nodes.re.save_kg_image", lambda *args, **kwargs: None)
    api = FakeAPI()
    config = Config(
        model=model, source_ids=[32749], strat_name="Example Formation",
        output=tmp_path / "run", structured_output="none", fuzzy_threshold=0.9,
    )
    summary = run_pipeline(config, transport=httpx.MockTransport(api))
    assert summary["status"] == "completed"
    assert summary["uploaded"] == 0
    assert len(api.term_pages) == 4  # Continue past short pages until empty.
    result = json.loads((config.output / "sources/32749/result.json").read_text())
    assert result["strat_name"] == "Example Formation"
    assert result["source"]["paragraph_text"] == "Thin sandstone formed in a marine setting."
    assert len(result["relations"]) == 2
    assert {entity["entity_type"] for entity in result["entities"]} == {"lith", "lith_att", "environ"}
    assert {entity["macrostrat_terms_id"] for entity in result["entities"]} == {8080, 8199, 9000}
    assert len(api.calls) == 2  # NER and RE both receive the supplied context.
    expected = {"enable_thinking": False} if model == "qwen3" else {"reasoning_effort": "low"}
    assert all(call["chat_template_kwargs"] == expected for call in api.calls)
    assert all(call["model"] == model for call in api.calls)
    # Context is saved locally; no fake source span or undocumented API field.
    payload = result["payload"]["results"][0]
    assert all(entity["entity_type"] != "strat_name" for entity in payload["just_entities"])
    assert "strat_name" not in payload


def test_all_rejected_entities_do_not_fail_source(tmp_path):
    api = FakeAPI(invalid_ner=True)
    config = Config(model="qwen3", source_ids=[32749], strat_name="Example Formation", output=tmp_path / "run")
    summary = run_pipeline(config, transport=httpx.MockTransport(api))
    assert summary["status"] == "completed"
    result = json.loads((config.output / "sources/32749/result.json").read_text())
    assert not result["entities"]
    assert result["rejected"]
    assert len(api.calls) == 1


def test_server_ignoring_filter_fails_before_llm(tmp_path):
    api = FakeAPI(ignore_filter=True)
    summary = run_pipeline(Config(output=tmp_path / "run"), transport=httpx.MockTransport(api))
    assert summary["status"] == "failed"
    assert not api.calls
