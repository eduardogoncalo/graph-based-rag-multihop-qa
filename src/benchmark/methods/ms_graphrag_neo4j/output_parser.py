from __future__ import annotations

from typing import Any

from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.methods.ms_graphrag_neo4j.config_builder import MS_GRAPHRAG_NEO4J_METHOD_ID


def parse_query_output(
    *,
    query: str,
    raw_response: dict[str, Any] | list[Any] | str | None,
    latency_ms: float,
    top_k: int,
) -> RetrievalResult:
    contexts = _extract_contexts(raw_response)
    items = [
        RetrievedItem(
            item_id=f"ms_graphrag_neo4j:{index}",
            text=context.get("text", ""),
            score=context.get("score"),
            source_document_id=context.get("source_document_id"),
            source_chunk_id=context.get("source_chunk_id"),
            metadata={
                "method_id": MS_GRAPHRAG_NEO4J_METHOD_ID,
                **{
                    key: value
                    for key, value in context.items()
                    if key not in {"text", "score", "source_document_id", "source_chunk_id"}
                },
            },
        )
        for index, context in enumerate(contexts[:top_k], start=1)
    ]
    if not items:
        answer = _extract_answer(raw_response)
        if answer:
            items = [
                RetrievedItem(
                    item_id="ms_graphrag_neo4j:answer",
                    text=answer,
                    metadata={"method_id": MS_GRAPHRAG_NEO4J_METHOD_ID, "kind": "answer"},
                )
            ]
    return RetrievalResult(
        method_id=MS_GRAPHRAG_NEO4J_METHOD_ID,
        query=query,
        items=items,
        raw_response=raw_response,
        latency_ms=latency_ms,
    )


def _extract_contexts(
    raw_response: dict[str, Any] | list[Any] | str | None,
) -> list[dict[str, Any]]:
    if isinstance(raw_response, list):
        return [_normalize_context(item) for item in raw_response]
    if isinstance(raw_response, dict):
        for key in ("contexts", "sources", "retrieved_context", "items", "results"):
            value = raw_response.get(key)
            if isinstance(value, list):
                return [_normalize_context(item) for item in value]
        for key in ("entities", "relationships"):
            value = raw_response.get(key)
            if isinstance(value, list):
                return [_normalize_context(item) for item in value]
    if isinstance(raw_response, str) and raw_response.strip():
        return [{"text": raw_response.strip(), "kind": "answer"}]
    return []


def _extract_answer(raw_response: dict[str, Any] | list[Any] | str | None) -> str:
    if isinstance(raw_response, str):
        return raw_response.strip()
    if isinstance(raw_response, dict):
        for key in ("answer", "response", "result", "output"):
            value = raw_response.get(key)
            if isinstance(value, str):
                return value.strip()
    return ""


def _normalize_context(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"text": item}
    if not isinstance(item, dict):
        return {"text": str(item)}
    text = (
        item.get("text")
        or item.get("content")
        or item.get("source_text")
        or item.get("description")
        or item.get("answer")
        or ""
    )
    return {
        "text": str(text),
        "score": item.get("score"),
        "source_document_id": item.get("document_id") or item.get("source_document_id"),
        "source_chunk_id": item.get("chunk_id") or item.get("source_chunk_id"),
        **item,
    }
