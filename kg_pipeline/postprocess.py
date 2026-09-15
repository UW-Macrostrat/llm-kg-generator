"""Validate model JSON, ground mentions in the source, and aggregate chunks."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import Config
from .models import LLMResult, SourceWork, now
from .prompts import Vocabulary

RECORD_RUN_SOURCE_FIELDS = (
    "source_text_type",
    "preprocessor_id",
    "paper_id",
    "hashed_text",
    "weaviate_id",
    "paragraph_text",
)

def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def find_span(text: str, phrase: str) -> tuple[int, int] | None:
    if not phrase:
        return None
    start = text.find(phrase)
    if start >= 0:
        return start, start + len(phrase)
    match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else None


def validate_shape(raw: Any) -> None:
    if not isinstance(raw, dict) or set(raw) != {"relationships", "just_entities"}:
        raise ValueError("Expected an object with relationships and just_entities arrays")
    for name, fields, maximum in (
        ("relationships", {"src", "relationship_type", "dst", "reasoning"}, 6),
        ("just_entities", {"entity", "entity_type"}, 12),
    ):
        items = raw[name]
        if not isinstance(items, list) or len(items) > maximum:
            raise ValueError(f"Invalid {name} array or item limit exceeded")
        for item in items:
            if not isinstance(item, dict) or set(item) != fields:
                raise ValueError(f"Invalid object in {name}")
            if any(not isinstance(item[key], str) for key in fields):
                raise ValueError(f"All fields in {name} must be strings")
            if name == "relationships" and len(item["reasoning"]) > 160:
                raise ValueError("Relationship reasoning exceeds 160 characters")


def clean_extraction(raw: dict, text: str, vocabulary: Vocabulary, offset: int = 0) -> dict:
    allowed_entities = set(vocabulary.entity_names)
    allowed_relationships = set(vocabulary.relationship_names)
    type_labels = {normalize(name) for name in allowed_entities}
    relationships, entities, rejected = [], [], []
    seen_relationships: set[tuple] = set()
    for item in raw["relationships"]:
        src, dst = item["src"].strip(), item["dst"].strip()
        kind = item["relationship_type"].strip()
        src_span, dst_span = find_span(text, src), find_span(text, dst)
        if (not src_span or not dst_span or normalize(src) in type_labels
                or normalize(dst) in type_labels or kind not in allowed_relationships):
            rejected.append({"kind": "relationship", "item": item,
                             "reason": "Unsupported type, type-label mention, or phrase absent from text"})
            continue
        key = (src_span, kind, dst_span)
        if key in seen_relationships:
            continue
        seen_relationships.add(key)
        relationships.append({
            "src": text[slice(*src_span)], "relationship_type": kind,
            "dst": text[slice(*dst_span)], "reasoning": item["reasoning"].strip(),
            "src_start_idx": offset + src_span[0], "src_end_idx": offset + src_span[1],
            "dst_start_idx": offset + dst_span[0], "dst_end_idx": offset + dst_span[1],
        })
    endpoint_spans = {
        (r["src_start_idx"], r["src_end_idx"])
        for r in relationships
    } | {
        (r["dst_start_idx"], r["dst_end_idx"])
        for r in relationships
    }
    seen_entities: set[tuple] = set()
    for item in raw["just_entities"]:
        entity, kind = item["entity"].strip(), item["entity_type"].strip()
        span = find_span(text, entity)
        if not span or normalize(entity) in type_labels or kind not in allowed_entities:
            rejected.append({"kind": "entity", "item": item,
                             "reason": "Unsupported type, type-label mention, or phrase absent from text"})
            continue
        key = (span, kind)
        global_span = (
            offset + span[0],
            offset + span[1],
        )

        if global_span in endpoint_spans or key in seen_entities:
            continue
        seen_entities.add(key)
        entities.append({"entity": text[slice(*span)], "entity_type": kind,
                         "start_idx": offset + span[0], "end_idx": offset + span[1]})
    return {"relationships": relationships, "just_entities": entities, "rejected": rejected}


def process_result(result: LLMResult, vocabulary: Vocabulary) -> dict:
    chunk = result.chunk
    entry: dict[str, Any] = {
        "chunk_index": chunk.index, "start_idx": chunk.start, "end_idx": chunk.end,
        "prompt_tokens": chunk.prompt_tokens, "started_at": result.started_at,
        "finished_at": result.finished_at, "llm_seconds": result.seconds,
        "attempts": result.attempts, "status": "failed", "error": result.error,
    }
    if result.error:
        return entry
    try:
        envelope = json.loads(result.response or b"")
        # Keep the exact response, including truncation/invalid-output cases, for inspection.
        entry["response"] = envelope
        if not isinstance(envelope, dict) or not isinstance(envelope.get("choices"), list) or not envelope["choices"]:
            raise ValueError("Invalid chat-completion response envelope")
        choice = envelope["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("Invalid chat-completion choice/message")
        if choice.get("finish_reason") != "stop":
            raise ValueError(f"Incomplete LLM output: finish_reason={choice.get('finish_reason')!r}; "
                             "if 'length', raise --max-tokens or reduce extraction caps")
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content:
            raise ValueError("LLM returned an empty response")
        raw = json.loads(content)
        entry["raw_extraction"] = raw
        validate_shape(raw)
        text = chunk.work.source["paragraph_text"][chunk.start:chunk.end]
        entry["extraction"] = clean_extraction(raw, text, vocabulary, offset=chunk.start)
        entry["usage"] = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
        entry.update(status="success", error=None)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
        if "response" not in entry:
            entry["response_text"] = (result.response or b"").decode("utf-8", errors="replace")
    return entry


def merge_extractions(chunks: list[dict]) -> dict:
    relationships, entities = {}, {}
    for chunk in chunks:
        if chunk["status"] != "success":
            continue
        for item in chunk["extraction"]["relationships"]:
            key = (item["src_start_idx"], item["src_end_idx"], item["relationship_type"],
                   item["dst_start_idx"], item["dst_end_idx"])
            relationships.setdefault(key, item)
        for item in chunk["extraction"]["just_entities"]:
            key = (item["start_idx"], item["end_idx"], item["entity_type"])
            entities.setdefault(key, item)
    endpoint_names = {normalize(item[k]) for item in relationships.values() for k in ("src", "dst")}
    return {
        "relationships": sorted(relationships.values(), key=lambda x: (x["src_start_idx"], x["dst_start_idx"], x["relationship_type"])),
        "just_entities": sorted((item for item in entities.values() if normalize(item["entity"]) not in endpoint_names),
                                key=lambda x: (x["start_idx"], x["entity_type"])),
    }

def validate_tree_structure(relationships: list[dict]) -> list[str]:
    """
    Validate that relationship endpoints form a directed forest.

    Nodes are identified by their grounded source-text spans.
    Returns a list of validation errors.
    """
    errors: list[str] = []

    # dst node -> src node
    parent_of: dict[tuple[int, int], tuple[int, int]] = {}

    # adjacency list for cycle detection
    children: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for rel in relationships:
        src_node = (
            rel["src_start_idx"],
            rel["src_end_idx"],
        )
        dst_node = (
            rel["dst_start_idx"],
            rel["dst_end_idx"],
        )

        # Same mention cannot point to itself.
        if src_node == dst_node:
            errors.append(
                f"Self relationship: "
                f"{rel['src']!r} -> {rel['relationship_type']} -> {rel['dst']!r}"
            )
            continue

        # A destination may have only one parent.
        if dst_node in parent_of:
            previous_parent = parent_of[dst_node]

            if previous_parent != src_node:
                errors.append(
                    f"Multiple parents for destination {rel['dst']!r} "
                    f"at span {dst_node}"
                )
        else:
            parent_of[dst_node] = src_node

        children.setdefault(src_node, []).append(dst_node)

    # Detect cycles using DFS.
    visiting: set[tuple[int, int]] = set()
    visited: set[tuple[int, int]] = set()

    def visit(node: tuple[int, int]) -> None:
        if node in visiting:
            errors.append(
                f"Cycle detected involving entity span {node}"
            )
            return

        if node in visited:
            return

        visiting.add(node)

        for child in children.get(node, []):
            visit(child)

        visiting.remove(node)
        visited.add(node)

    all_nodes = set(children) | set(parent_of)

    for node in all_nodes:
        if node not in visited:
            visit(node)

    return errors

def build_record_run_payload(
        source: dict,
        extraction: dict,
        model: str,
        run_id: str,
        config: Config,
    ) -> dict:
        text = {
            "preprocessor_id": source.get("preprocessor_id"),
            "paper_id": source.get("paper_id"),
            "hashed_text": source.get("hashed_text"),
            "weaviate_id": source.get("weaviate_id"),
            "paragraph_text": source["paragraph_text"],
            "text_type": source["source_text_type"],
            "legend_id": source.get("map_legend_id"),
        }

        return {
            "run_id": run_id,
            "extraction_pipeline_id": config.extraction_pipeline_id,
            "model_name": model,
            "model_version": config.model_version,
            "results": [
                {
                    "text": text,
                    "relationships": extraction["relationships"],
                    "just_entities": extraction["just_entities"],
                }
            ],
        }

def finish_source(work: SourceWork, model: str, config: Config,
                  preprocessing_error: str | None = None) -> dict:
    chunks = [work.chunks[index] for index in sorted(work.chunks)]
    errors = ([{"stage": "preprocess", "error": preprocessing_error}] if preprocessing_error else [])
    errors.extend({"stage": "llm/postprocess", "chunk_index": c["chunk_index"], "error": c["error"]}
                  for c in chunks if c["status"] != "success")
    extraction = merge_extractions(chunks)

    tree_errors = validate_tree_structure(extraction["relationships"])

    errors.extend(
        {
            "stage": "tree_validation",
            "error": error,
        }
        for error in tree_errors
    )

    payload = (
        None
        if errors
        else build_record_run_payload(
            work.source,
            extraction,
            model,
            work.run_id,
            config,
        )
    )
    return {
        "source_id": work.source_id, "source": work.source, "run_id": work.run_id, "model": model,
        "status": "failed" if errors else "success", "errors": errors,
        "fetched_at": work.fetched_at, "prepared_at": work.prepared_at,
        "preprocess_seconds": work.preprocess_seconds, "postprocessed_at": now(),
        "chunks": chunks, "extraction": extraction, "record_run_payload": payload,
        "upload_status": "skipped_failed" if errors else ("skipped_dry_run" if config.dry_run else "pending"),
    }
