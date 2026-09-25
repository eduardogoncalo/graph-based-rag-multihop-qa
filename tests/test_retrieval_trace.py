from __future__ import annotations

from benchmark.retrieval_trace import (
    build_retrieval_trace,
    normalize_retrieval_trace_payload,
)


def test_trace_complete_counts_nodes_relationships_chunks_and_context() -> None:
    trace = build_retrieval_trace(
        run_id="run_1",
        framework_id="lightrag_neo4j",
        query_mode="mix",
        retrieval_strategy="graph_rag_mix",
        nodes=[{"id": "entity_1", "label": "Termination"}],
        relationships=[{"id": "rel_1", "source": "entity_1", "target": "entity_2"}],
        chunks=[{"chunk_id": "chunk_1", "text": "source text"}],
        final_context={"text": "assembled context"},
        raw_trace={"framework": "fake"},
    )

    assert trace.trace_status == "complete"
    assert trace.trace["counts"] == {"nodes": 1, "relationships": 1, "chunks": 1}
    assert trace.trace["final_context"]["text"] == "assembled context"
    assert trace.metadata["instrumentation_version"] == "lightrag_trace_v1"


def test_trace_partial_when_only_final_context_exists() -> None:
    trace = normalize_retrieval_trace_payload(
        run_id="run_2",
        framework_id="lightrag_neo4j",
        query_mode="mix",
        retrieval_strategy="graph_rag_mix",
        payload={"final_context": {"text": "context only"}},
    )

    assert trace.trace_status == "partial"
    assert trace.trace["counts"] == {"nodes": 0, "relationships": 0, "chunks": 0}
    assert trace.trace["final_context"]["text"] == "context only"


def test_trace_missing_when_payload_absent() -> None:
    trace = normalize_retrieval_trace_payload(
        run_id="run_3",
        framework_id="lightrag_neo4j",
        payload=None,
    )

    assert trace.trace_status == "missing"
    assert trace.trace["nodes"] == []
    assert trace.trace["relationships"] == []
    assert trace.trace["chunks"] == []


def test_trace_normalizer_accepts_framework_aliases() -> None:
    trace = normalize_retrieval_trace_payload(
        run_id="run_4",
        framework_id="lightrag_neo4j",
        payload={
            "entities": [{"id": "entity_1"}],
            "relations": [{"id": "rel_1"}],
            "contexts": [{"chunk_id": "chunk_1"}],
            "context_text": "final context",
        },
    )

    assert trace.trace_status == "complete"
    assert trace.trace["counts"] == {"nodes": 1, "relationships": 1, "chunks": 1}
    assert trace.trace["final_context"]["text"] == "final context"
