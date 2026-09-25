from __future__ import annotations

from typing import Any

from benchmark.core.ids import deterministic_id
from benchmark.core.schemas import RetrievalTrace

TRACE_INSTRUMENTATION_VERSION = "lightrag_trace_v1"
TRACE_STATUSES = {"complete", "partial", "missing", "not_supported", "error"}


def build_retrieval_trace(
    *,
    run_id: str,
    framework_id: str,
    query_mode: str | None = None,
    retrieval_strategy: str | None = None,
    nodes: list[dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    final_context: dict[str, Any] | None = None,
    raw_trace: dict[str, Any] | None = None,
    trace_status: str | None = None,
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> RetrievalTrace:
    normalized_nodes = [_clean_mapping(item) for item in nodes or [] if isinstance(item, dict)]
    normalized_relationships = [
        _clean_mapping(item) for item in relationships or [] if isinstance(item, dict)
    ]
    normalized_chunks = [_clean_mapping(item) for item in chunks or [] if isinstance(item, dict)]
    normalized_context = _clean_mapping(final_context or {})
    status = trace_status or infer_trace_status(
        nodes=normalized_nodes,
        relationships=normalized_relationships,
        chunks=normalized_chunks,
        final_context=normalized_context,
    )
    if status not in TRACE_STATUSES:
        raise ValueError(f"Unsupported retrieval trace status: {status}")
    trace = {
        "framework_id": framework_id,
        "query_mode": query_mode,
        "retrieval_strategy": retrieval_strategy,
        "nodes": normalized_nodes,
        "relationships": normalized_relationships,
        "chunks": normalized_chunks,
        "final_context": normalized_context,
        "counts": {
            "nodes": len(normalized_nodes),
            "relationships": len(normalized_relationships),
            "chunks": len(normalized_chunks),
        },
    }
    trace = {key: value for key, value in trace.items() if value is not None}
    resolved_trace_id = trace_id or deterministic_id("retrieval_trace", [run_id])
    return RetrievalTrace(
        trace_id=resolved_trace_id,
        run_id=run_id,
        trace_status=status,
        trace=trace,
        raw_trace=_clean_mapping(raw_trace or {}),
        metadata={
            "instrumentation_version": TRACE_INSTRUMENTATION_VERSION,
            **(metadata or {}),
        },
    )


def normalize_retrieval_trace_payload(
    *,
    run_id: str,
    framework_id: str,
    payload: dict[str, Any] | None,
    query_mode: str | None = None,
    retrieval_strategy: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RetrievalTrace:
    if not payload:
        return build_retrieval_trace(
            run_id=run_id,
            framework_id=framework_id,
            query_mode=query_mode,
            retrieval_strategy=retrieval_strategy,
            trace_status="missing",
            raw_trace={},
            metadata=metadata,
        )
    nodes = _first_list(payload, "nodes", "entities", "retrieved_nodes")
    relationships = _first_list(
        payload,
        "relationships",
        "relations",
        "edges",
        "retrieved_relationships",
    )
    chunks = _first_list(
        payload,
        "chunks",
        "context_chunks",
        "retrieved_chunks",
        "sources",
        "contexts",
    )
    final_context = _first_mapping(payload, "final_context", "context", "assembled_context")
    if not final_context:
        context_text = payload.get("final_context_text") or payload.get("context_text")
        if isinstance(context_text, str):
            final_context = {"text": context_text}
    return build_retrieval_trace(
        run_id=run_id,
        framework_id=str(payload.get("framework_id") or framework_id),
        query_mode=str(payload.get("query_mode") or query_mode or ""),
        retrieval_strategy=str(payload.get("retrieval_strategy") or retrieval_strategy or ""),
        nodes=nodes,
        relationships=relationships,
        chunks=chunks,
        final_context=final_context,
        raw_trace=payload,
        trace_status=payload.get("trace_status"),
        metadata=metadata,
    )


def infer_trace_status(
    *,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    final_context: dict[str, Any],
) -> str:
    has_graph = bool(nodes or relationships)
    has_context = bool(chunks or final_context)
    if has_graph and has_context:
        return "complete"
    if has_graph or has_context:
        return "partial"
    return "missing"


def _first_list(payload: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [_coerce_mapping(item) for item in value]
    return []


def _first_mapping(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return _clean_mapping(value)
        if isinstance(value, str):
            return {"text": value}
    return {}


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return _clean_mapping(value)
    return {"text": str(value)}


def _clean_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
