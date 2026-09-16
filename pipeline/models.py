"""Data contracts and editable extraction constraints.

Change fields here to change the contract. LLM JSON schemas are generated from
NEROutput/REOutput, so prompts and validation do not maintain separate shapes.
Change constraints.json for limits and graph rules; ontology names and endpoint
types come from the database on every run.
"""
from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

Identifier = int | str


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RankingWeights(Contract):
    same_sentence: float = Field(default=0.5, ge=0)
    proximity: float = Field(default=0.3, ge=0)
    evidence_compactness: float = Field(default=0.2, ge=0)

    @model_validator(mode="after")
    def positive_total(self) -> Self:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("At least one ranking weight must be positive")
        return self


class Constraints(Contract):
    version: str = "1"
    max_entities: int = Field(default=32, ge=1)
    max_relations: int = Field(default=64, ge=1)
    max_candidates: int = Field(default=1024, ge=1)
    max_evidence_chars: int = Field(default=1200, ge=1)
    max_incoming_relations: int | None = Field(default=1, ge=1)
    allow_cycles: bool = False
    allow_self_relations: bool = False
    evidence_must_include_endpoints: bool = True
    min_relation_score: float = Field(default=0.0, ge=0, le=1)
    ranking: RankingWeights = Field(default_factory=RankingWeights)

    @model_validator(mode="after")
    def consistent_graph_rules(self) -> Self:
        if self.allow_self_relations and not self.allow_cycles:
            raise ValueError("Self relations are cycles; allow_cycles must also be true")
        return self


class SourceText(Contract):
    # Preserve new source columns; only extraction-relevant columns are required.
    model_config = ConfigDict(extra="allow", strict=True)
    id: int = Field(ge=0)
    paragraph_text: str = Field(min_length=1)
    source_text_type: Identifier
    preprocessor_id: Identifier | None = None
    paper_id: Identifier | None = None
    hashed_text: str | None = None
    weaviate_id: str | None = None
    map_legend_id: Identifier | None = None

    @model_validator(mode="after")
    def nonblank_text(self) -> Self:
        if not self.paragraph_text.strip():
            raise ValueError("paragraph_text is blank")
        return self


class EntityType(Contract):
    model_config = ConfigDict(extra="ignore", strict=True)
    id: int = Field(ge=0)
    name: str = Field(min_length=1)
    description: str | None = None


class RelationshipType(EntityType):
    src_entity_type_id: int = Field(ge=0)
    dst_entity_type_id: int = Field(ge=0)


class Ontology(Contract):
    entity_types: dict[str, EntityType]
    relationship_types: dict[str, RelationshipType]

    @classmethod
    def from_rows(cls, entities: list[dict], relations: list[dict]) -> Ontology:
        entity_models = [EntityType.model_validate(row) for row in entities]
        relation_models = [RelationshipType.model_validate(row) for row in relations]
        for label, rows in (("entity", entity_models), ("relationship", relation_models)):
            if not rows or any(not row.name.strip() for row in rows):
                raise ValueError(f"Missing or blank {label} vocabulary")
            if len({r.name for r in rows}) != len(rows) or len({r.id for r in rows}) != len(rows):
                raise ValueError(f"Duplicate {label} IDs/names")
        ids = {e.id for e in entity_models}
        for relation in relation_models:
            if relation.src_entity_type_id not in ids or relation.dst_entity_type_id not in ids:
                raise ValueError(f"Unknown endpoint entity type in {relation.name!r}")
        return cls(entity_types={e.name: e for e in entity_models},
                   relationship_types={r.name: r for r in relation_models})


class EntityDraft(Contract):
    text: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    occurrence: int = Field(ge=0, description="Zero-based exact occurrence of text in the paragraph")


class NEROutput(Contract):
    entities: list[EntityDraft]

    @model_validator(mode="after")
    def check_limit(self, info: ValidationInfo) -> Self:
        rules = (info.context or {}).get("constraints", Constraints())
        if len(self.entities) > rules.max_entities:
            raise ValueError(f"NER exceeds max_entities={rules.max_entities}")
        return self


class Entity(Contract):
    id: str
    text: str = Field(min_length=1)
    entity_type: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    macrostrat_terms_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def check_source(self, info: ValidationInfo) -> Self:
        if self.end <= self.start or self.id != f"e{self.start}_{self.end}":
            raise ValueError("Invalid entity span/ID")
        context = info.context or {}
        if "text" in context and context["text"][self.start:self.end] != self.text:
            raise ValueError("Entity is not an exact source-text span")
        if "ontology" in context and self.entity_type not in context["ontology"].entity_types:
            raise ValueError(f"Unknown entity type: {self.entity_type!r}")
        return self


class Candidate(Contract):
    id: str
    src_id: str
    relationship_type: str
    dst_id: str


class RelationDraft(Contract):
    candidate_id: str
    evidence: str = Field(min_length=1)
    evidence_occurrence: int = Field(ge=0)


class REOutput(Contract):
    relations: list[RelationDraft]

    @model_validator(mode="after")
    def check_limit(self, info: ValidationInfo) -> Self:
        rules = (info.context or {}).get("constraints", Constraints())
        if len(self.relations) > rules.max_relations:
            raise ValueError(f"RE exceeds max_relations={rules.max_relations}")
        return self


class Relation(Contract):
    src_id: str
    relationship_type: str
    dst_id: str
    evidence: str = Field(min_length=1)
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(gt=0)

    @model_validator(mode="after")
    def check_relation(self, info: ValidationInfo) -> Self:
        context = info.context or {}
        rules = context.get("constraints", Constraints())
        if self.src_id == self.dst_id and not rules.allow_self_relations:
            raise ValueError("Self relation is forbidden")
        if self.evidence_end <= self.evidence_start or len(self.evidence) > rules.max_evidence_chars:
            raise ValueError("Invalid evidence span or evidence exceeds max_evidence_chars")
        if "text" in context and context["text"][self.evidence_start:self.evidence_end] != self.evidence:
            raise ValueError("Evidence is not an exact quote from the source")
        if "entities" in context:
            entities = context["entities"]
            if self.src_id not in entities or self.dst_id not in entities:
                raise ValueError("Relation references an unvalidated entity")
            src, dst = entities[self.src_id], entities[self.dst_id]
            if "ontology" in context:
                ontology = context["ontology"]
                kind = ontology.relationship_types.get(self.relationship_type)
                if kind is None:
                    raise ValueError("Unknown relationship type")
                src_type = ontology.entity_types[src.entity_type].id
                dst_type = ontology.entity_types[dst.entity_type].id
                if (src_type, dst_type) != (kind.src_entity_type_id, kind.dst_entity_type_id):
                    raise ValueError("Endpoint types/direction violate the ontology")
            if rules.evidence_must_include_endpoints and not all(
                self.evidence_start <= e.start < e.end <= self.evidence_end for e in (src, dst)
            ):
                raise ValueError("Evidence must include both specific endpoint mentions")
        return self


class RankedRelation(Contract):
    relation: Relation
    score: float = Field(ge=0, le=1)
    features: dict[str, float]


class Issue(Contract):
    stage: str
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# These models describe the EXISTING record_run API; edit here if its contract changes.
class RecordText(Contract):
    preprocessor_id: Identifier | None
    paper_id: Identifier | None
    hashed_text: str | None
    weaviate_id: str | None
    paragraph_text: str
    text_type: Identifier
    legend_id: Identifier | None


class RecordRelationship(Contract):
    src: str
    relationship_type: str
    dst: str
    reasoning: str = Field(max_length=160)
    src_start_idx: int = Field(ge=0)
    src_end_idx: int = Field(gt=0)
    dst_start_idx: int = Field(ge=0)
    dst_end_idx: int = Field(gt=0)


class RecordEntity(Contract):
    entity: str
    entity_type: str
    start_idx: int = Field(ge=0)
    end_idx: int = Field(gt=0)
    macrostrat_terms_id: int | None = Field(default=None, ge=1)


class RecordResult(Contract):
    text: RecordText
    relationships: list[RecordRelationship]
    just_entities: list[RecordEntity]


class RecordRunPayload(Contract):
    run_id: str
    extraction_pipeline_id: str
    model_name: str
    model_version: int
    results: list[RecordResult] = Field(min_length=1)


class SourceResult(Contract):
    source_id: int
    source: dict[str, Any]
    status: Literal["pending", "ner_complete", "re_complete", "success", "failed"] = "pending"
    entities: list[Entity] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    ranked_relations: list[RankedRelation] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    rejected: list[Issue] = Field(default_factory=list)
    errors: list[Issue] = Field(default_factory=list)
    models: dict[str, str] = Field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    payload: RecordRunPayload | None = None
    upload_status: Literal["not_started", "dry_run", "skipped_failed", "started", "success", "failed", "unknown"] = "not_started"
    receipt: dict[str, Any] | None = None
