from __future__ import annotations

from typing import Any

from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.methods.lightrag.config_builder import LIGHTRAG_METHOD_ID


def parse_query_output(
    *,
    query: str,
    raw_response: dict[str, Any] | str,
    query_mode: str,
    latency_ms: float,
    top_k: int,
) -> RetrievalResult:
    answer = _extract_answer(raw_response)
    contexts = _extract_contexts(raw_response, answer)
    items = [
        RetrievedItem(
            item_id=f"lightrag:{query_mode}:{index}",
            text=context.get("text", ""),
            score=context.get("score"),
            source_document_id=context.get("source_document_id"),
            source_chunk_id=context.get("source_chunk_id"),
            metadata={
                "query_mode": query_mode,
                **{key: value for key, value in context.items() if key not in {"text", "score"}},
            },
        )
        for index, context in enumerate(contexts[:top_k], start=1)
    ]
    if not items and answer:
        items = [
            RetrievedItem(
                item_id=f"lightrag:{query_mode}:answer",
                text=answer,
                metadata={"query_mode": query_mode, "kind": "answer"},
            )
        ]
    return RetrievalResult(
        method_id=LIGHTRAG_METHOD_ID,
        query=query,
        items=items,
        raw_response=raw_response,
        latency_ms=latency_ms,
    )


def _extract_answer(raw_response: dict[str, Any] | str) -> str:
    if isinstance(raw_response, str):
        return raw_response
    for key in ("answer", "response", "result", "output"):
        value = raw_response.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_contexts(raw_response: dict[str, Any] | str, answer: str) -> list[dict[str, Any]]:
    if isinstance(raw_response, dict):
        for key in ("contexts", "sources", "retrieved_context", "context_data"):
            value = raw_response.get(key)
            if isinstance(value, list):
                return [_normalize_context(item) for item in value]
        citations = raw_response.get("citations")
        if isinstance(citations, list):
            return [_normalize_context(item) for item in citations]
    return [{"text": answer, "kind": "answer"}] if answer else []


def _normalize_context(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"text": item}
    if not isinstance(item, dict):
        return {"text": str(item)}
    text = item.get("text") or item.get("content") or item.get("source_text") or item.get("answer")
    return {
        "text": str(text or ""),
        "score": item.get("score"),
        "source_document_id": item.get("document_id") or item.get("source_document_id"),
        "source_chunk_id": item.get("chunk_id") or item.get("source_chunk_id"),
        **item,
    }
