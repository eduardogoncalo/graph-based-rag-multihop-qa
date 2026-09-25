from __future__ import annotations

import json
import re
from typing import Any

from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.methods.ms_graphrag.config_builder import MS_GRAPHRAG_METHOD_ID

# Benchmark document namespace (same convention the cognee parser recovers).
DOCUMENT_ID_RE = re.compile(r"\b(doc_[0-9a-f]{6,})\b")


def parse_query_output(
    *,
    query: str,
    stdout: str,
    query_method: str,
    latency_ms: float,
    top_k: int,
) -> RetrievalResult:
    parsed = _parse_json(stdout)

    # Primary path: structured payload from scripts/graphrag_query_runner.py
    # (retrieval-only context build) — {"context_text", "items": [...], "stats"}.
    if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        return _parse_runner_payload(
            query=query, parsed=parsed, query_method=query_method,
            latency_ms=latency_ms, top_k=top_k,
        )

    answer = _extract_answer(parsed, stdout)
    contexts = _extract_contexts(parsed, answer)
    items = [
        RetrievedItem(
            item_id=f"ms_graphrag:{query_method}:{index}",
            text=context.get("text", ""),
            score=context.get("score"),
            source_document_id=context.get("source_document_id"),
            source_chunk_id=context.get("source_chunk_id"),
            metadata={
                "query_method": query_method,
                "document_ids": _document_ids(context),
                **{key: value for key, value in context.items() if key not in {"text", "score"}},
            },
        )
        for index, context in enumerate(contexts[:top_k], start=1)
    ]
    if not items and answer:
        items = [
            RetrievedItem(
                item_id=f"ms_graphrag:{query_method}:answer",
                text=answer,
                metadata={
                    "query_method": query_method,
                    "kind": "answer",
                    "document_ids": sorted(set(DOCUMENT_ID_RE.findall(answer))),
                },
            )
        ]
    return RetrievalResult(
        method_id=MS_GRAPHRAG_METHOD_ID,
        query=query,
        items=items,
        raw_response=parsed if parsed is not None else stdout,
        latency_ms=latency_ms,
    )


def _parse_runner_payload(
    *,
    query: str,
    parsed: dict[str, Any],
    query_method: str,
    latency_ms: float,
    top_k: int,
) -> RetrievalResult:
    items: list[RetrievedItem] = []
    for index, raw in enumerate(parsed["items"][:top_k], start=1):
        if not isinstance(raw, dict):
            continue
        document_ids = _document_ids(raw)
        items.append(
            RetrievedItem(
                item_id=str(raw.get("unit_id") or f"ms_graphrag:{query_method}:{index}"),
                text=str(raw.get("text") or ""),
                score=raw.get("score"),
                source_document_id=document_ids[0] if document_ids else None,
                source_chunk_id=str(raw.get("unit_short_id")) if raw.get("unit_short_id") is not None else None,
                metadata={
                    "query_method": query_method,
                    "document_ids": document_ids,
                    "in_context": raw.get("in_context", True),
                },
            )
        )
    stats = parsed.get("stats") if isinstance(parsed.get("stats"), dict) else {}
    return RetrievalResult(
        method_id=MS_GRAPHRAG_METHOD_ID,
        query=query,
        items=items,
        raw_response=parsed,
        latency_ms=latency_ms,
        metadata={
            "query_method": query_method,
            "context_text_chars": len(parsed.get("context_text") or ""),
            **stats,
        },
    )


def _document_ids(payload: dict[str, Any]) -> list[str]:
    """Every benchmark doc_* id attributable to this item, in stable order."""
    found: list[str] = []
    for key in ("document_id", "source_document_id", "title", "document_ids"):
        value = payload.get(key)
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, str):
                for match in DOCUMENT_ID_RE.findall(candidate):
                    if match not in found:
                        found.append(match)
    text = payload.get("text")
    if isinstance(text, str):
        for match in DOCUMENT_ID_RE.findall(text):
            if match not in found:
                found.append(match)
    return found


def _parse_json(stdout: str) -> Any | None:
    stripped = stdout.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _extract_answer(parsed: Any | None, stdout: str) -> str:
    if isinstance(parsed, dict):
        for key in ("answer", "response", "result", "output"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    return stdout.strip()


def _extract_contexts(parsed: Any | None, answer: str) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        for key in ("contexts", "sources", "context_data", "retrieved_context"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [_normalize_context(item) for item in value]
        citations = parsed.get("citations")
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
