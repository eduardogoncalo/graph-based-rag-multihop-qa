"""Reader-context knob for lightrag_neo4j: 'chunks' (default, byte-identical) vs
'structured' (entities + relationships + chunks). Locks the well-defined architecture."""

from __future__ import annotations

import pytest

from benchmark.methods.lightrag_neo4j.config_builder import (
    LIGHTRAG_NEO4J_READER_CONTEXT_ENV,
    READER_CONTEXT_CHUNKS,
    READER_CONTEXT_STRUCTURED,
    LightRAGNeo4jConfig,
    resolve_lightrag_reader_context,
)
from benchmark.methods.lightrag_neo4j.output_parser import parse_query_output

# query_data-shaped payload (entities + relationships + chunks, no generation).
_STRUCTURED_RAW = {
    "status": "success",
    "data": {
        "entities": [
            {"entity_name": "Ancilla College", "entity_type": "ORG",
             "description": "Owns The Collegian.", "reference_id": "r1"},
        ],
        "relationships": [
            {"src_id": "Ancilla College", "tgt_id": "The Collegian",
             "description": "publishes", "keywords": "owns, publishes", "reference_id": "r2"},
        ],
        "chunks": [
            {"content": "Ancilla College was founded in 1937.",
             "chunk_id": "doc_abc123-chunk-004", "reference_id": "r3"},
        ],
    },
    "metadata": {"query_mode": "mix"},
}


# --- resolver -----------------------------------------------------------------

def test_resolver_defaults_to_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LIGHTRAG_NEO4J_READER_CONTEXT_ENV, raising=False)
    assert resolve_lightrag_reader_context() == READER_CONTEXT_CHUNKS


@pytest.mark.parametrize("value", ["structured", "graph", "FULL", " Structured "])
def test_resolver_env_structured(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(LIGHTRAG_NEO4J_READER_CONTEXT_ENV, value)
    assert resolve_lightrag_reader_context() == READER_CONTEXT_STRUCTURED


def test_resolver_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LIGHTRAG_NEO4J_READER_CONTEXT_ENV, "structured")
    assert resolve_lightrag_reader_context("chunks") == READER_CONTEXT_CHUNKS


def test_config_field_defaults_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LIGHTRAG_NEO4J_READER_CONTEXT_ENV, raising=False)
    assert LightRAGNeo4jConfig().reader_context == READER_CONTEXT_CHUNKS


# --- parser: chunks mode unchanged (default) ----------------------------------

def test_chunks_mode_default_uses_data_chunks() -> None:
    result = parse_query_output(
        query="q", raw_response=_STRUCTURED_RAW, query_mode="mix",
        latency_ms=1.0, top_k=5, method_id="lightrag_neo4j",
    )
    # default == chunks: only the chunk survives, no entity/relationship items.
    assert [i.metadata.get("kind") for i in result.items] == [None] or all(
        i.text == "Ancilla College was founded in 1937." for i in result.items
    )
    assert len(result.items) == 1
    assert result.items[0].source_document_id == "doc_abc123"


# --- parser: structured mode (the new architecture) ---------------------------

def test_structured_mode_emits_entities_relationships_chunks_in_order() -> None:
    result = parse_query_output(
        query="q", raw_response=_STRUCTURED_RAW, query_mode="mix",
        latency_ms=1.0, top_k=5, method_id="lightrag_neo4j",
        context_mode=READER_CONTEXT_STRUCTURED,
    )
    kinds = [i.metadata["kind"] for i in result.items]
    assert kinds == ["entity", "relationship", "chunk"]

    entity, relationship, chunk = result.items
    assert entity.text == "[entity] Ancilla College (ORG): Owns The Collegian."
    assert relationship.text == "[relationship] Ancilla College → The Collegian [owns, publishes]: publishes"
    # only the chunk carries a doc id (evidence-recall parity with chunks mode)
    assert chunk.source_document_id == "doc_abc123"
    assert entity.source_document_id is None and relationship.source_document_id is None


def test_structured_mode_empty_data_is_safe() -> None:
    result = parse_query_output(
        query="q", raw_response={"status": "failure", "data": {}}, query_mode="mix",
        latency_ms=1.0, top_k=5, method_id="lightrag_neo4j",
        context_mode=READER_CONTEXT_STRUCTURED,
    )
    assert result.items == []
