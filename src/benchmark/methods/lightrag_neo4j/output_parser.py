from __future__ import annotations

import re
from typing import Any

from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.methods.lightrag_neo4j.config_builder import (
    LIGHTRAG_NEO4J_METHOD_ID,
    READER_CONTEXT_CHUNKS,
    READER_CONTEXT_NATIVE,
    READER_CONTEXT_STRUCTURED,
    LightRAGReaderContext,
)

# LightRAG emits positional chunk ids ("doc_<hash>-chunk-NNN") and no document
# id; evidence metrics match document-level gold ids, so the document id must
# be recovered from the chunk id prefix.
_CHUNK_ID_DOC_RE = re.compile(r"^(doc_[0-9a-f]+)-chunk-\d+$")


def parse_query_output(
    *,
    query: str,
    raw_response: dict[str, Any] | list[Any] | str | None,
    query_mode: str,
    latency_ms: float,
    top_k: int,
    method_id: str = LIGHTRAG_NEO4J_METHOD_ID,
    context_mode: LightRAGReaderContext = READER_CONTEXT_CHUNKS,
) -> RetrievalResult:
    if context_mode == READER_CONTEXT_NATIVE:
        return RetrievalResult(
            method_id=method_id,
            query=query,
            items=_native_context_items(raw_response, method_id=method_id, query_mode=query_mode),
            raw_response=raw_response,
            latency_ms=latency_ms,
        )
    if context_mode == READER_CONTEXT_STRUCTURED:
        return RetrievalResult(
            method_id=method_id,
            query=query,
            items=_structured_items(raw_response, method_id=method_id, query_mode=query_mode),
            raw_response=raw_response,
            latency_ms=latency_ms,
        )
    answer = _extract_answer(raw_response)
    contexts = _extract_contexts(raw_response, answer)
    items = [
        RetrievedItem(
            item_id=f"{method_id}:{query_mode}:{index}",
            text=context.get("text", ""),
            score=context.get("score"),
            source_document_id=context.get("source_document_id"),
            source_chunk_id=context.get("source_chunk_id"),
            metadata={
                "method_id": method_id,
                "query_mode": query_mode,
                **{
                    key: value
                    for key, value in context.items()
                    if key not in {"text", "score", "source_document_id", "source_chunk_id"}
                },
            },
        )
        for index, context in enumerate(contexts[:top_k], start=1)
    ]
    if not items and answer:
        items = [
            RetrievedItem(
                item_id=f"{method_id}:{query_mode}:answer",
                text=answer,
                metadata={
                    "method_id": method_id,
                    "query_mode": query_mode,
                    "kind": "answer",
                },
            )
        ]
    return RetrievalResult(
        method_id=method_id,
        query=query,
        items=items,
        raw_response=raw_response,
        latency_ms=latency_ms,
    )


def _native_context_items(
    raw_response: dict[str, Any] | list[Any] | str | None,
    *,
    method_id: str,
    query_mode: str,
) -> list[RetrievedItem]:
    """native_context: the reader receives LightRAG's OWN assembled context STRING as a SINGLE
    block (``raw_response['native_context']``, set by the adapter via only_need_context=True) —
    the analogue of cognee's serialized blob. ``llm_response`` is never read here (the adapter
    already dropped it), so no generated answer can reach the reader. doc_ids are recovered from
    ``raw_response['data']['chunks']`` and exposed as ``metadata['document_ids']`` so
    evidence_recall (which reads item.source_document_id + item.metadata['document_ids']) stays
    calculable. Item id: ``lightrag_neo4j:native_context:1`` (single item, LightRAG's format)."""
    if not isinstance(raw_response, dict):
        return []
    context_string = raw_response.get("native_context")
    if not isinstance(context_string, str) or not context_string.strip():
        return []
    data = raw_response.get("data") if isinstance(raw_response.get("data"), dict) else {}
    chunks = _as_dict_list(data.get("chunks"))
    doc_ids: list[str] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") if isinstance(chunk.get("chunk_id"), str) else None
        doc_id = _document_id_from_chunk_id(chunk_id)
        if not doc_id and isinstance(chunk.get("document_id"), str):
            doc_id = chunk["document_id"]
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return [
        RetrievedItem(
            item_id=f"{method_id}:native_context:1",
            text=context_string,
            metadata=_clean(
                {
                    "method_id": method_id,
                    "query_mode": query_mode,
                    "kind": "native_context",
                    "document_ids": doc_ids or None,
                    "n_entities": len(_as_dict_list(data.get("entities"))),
                    "n_relationships": len(_as_dict_list(data.get("relationships"))),
                    "n_chunks": len(chunks),
                    "only_need_context": True,
                }
            ),
        )
    ]


def _structured_items(
    raw_response: dict[str, Any] | list[Any] | str | None,
    *,
    method_id: str,
    query_mode: str,
) -> list[RetrievedItem]:
    """Serialize LightRAG's full retrieved context (entities + relationships +
    chunks, in that order — the same order LightRAG assembles its own context)
    into reader items. Input is a ``query_data`` payload
    (``{status, data:{entities, relationships, chunks, references}, metadata}``);
    generation is not run in this mode. Lists are already truncated by LightRAG's
    token budget (via top_k at query time), so we do NOT re-slice here — every
    retrieved element reaches the reader. Only chunk items carry a document id, so
    evidence-recall stays identical to ``chunks`` mode plus the graph structure."""
    data = raw_response.get("data") if isinstance(raw_response, dict) else None
    if not isinstance(data, dict):
        return []
    items: list[RetrievedItem] = []

    for index, entity in enumerate(_as_dict_list(data.get("entities")), start=1):
        name = _s(entity.get("entity_name"))
        etype = _s(entity.get("entity_type"))
        desc = _s(entity.get("description"))
        text = f"[entity] {name}" + (f" ({etype})" if etype else "") + (f": {desc}" if desc else "")
        items.append(
            RetrievedItem(
                item_id=f"{method_id}:{query_mode}:entity:{index}",
                text=text.strip(),
                metadata=_clean(
                    {
                        "method_id": method_id,
                        "query_mode": query_mode,
                        "kind": "entity",
                        "entity_name": name or None,
                        "entity_type": etype or None,
                        "reference_id": entity.get("reference_id"),
                    }
                ),
            )
        )

    for index, rel in enumerate(_as_dict_list(data.get("relationships")), start=1):
        src = _s(rel.get("src_id"))
        tgt = _s(rel.get("tgt_id"))
        desc = _s(rel.get("description"))
        keywords = _s(rel.get("keywords"))
        text = (
            f"[relationship] {src} → {tgt}"
            + (f" [{keywords}]" if keywords else "")
            + (f": {desc}" if desc else "")
        )
        items.append(
            RetrievedItem(
                item_id=f"{method_id}:{query_mode}:relationship:{index}",
                text=text.strip(),
                metadata=_clean(
                    {
                        "method_id": method_id,
                        "query_mode": query_mode,
                        "kind": "relationship",
                        "src_id": src or None,
                        "tgt_id": tgt or None,
                        "reference_id": rel.get("reference_id"),
                    }
                ),
            )
        )

    for index, chunk in enumerate(_as_dict_list(data.get("chunks")), start=1):
        content = _s(chunk.get("content"))
        chunk_id = chunk.get("chunk_id") if isinstance(chunk.get("chunk_id"), str) else None
        items.append(
            RetrievedItem(
                item_id=f"{method_id}:{query_mode}:chunk:{index}",
                text=content,
                source_chunk_id=chunk_id,
                source_document_id=_document_id_from_chunk_id(chunk_id),
                metadata=_clean(
                    {
                        "method_id": method_id,
                        "query_mode": query_mode,
                        "kind": "chunk",
                        "reference_id": chunk.get("reference_id"),
                    }
                ),
            )
        )

    return items


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _s(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value is not None}


def _extract_answer(raw_response: dict[str, Any] | list[Any] | str | None) -> str:
    if isinstance(raw_response, str):
        return raw_response.strip()
    if isinstance(raw_response, dict):
        llm_response = raw_response.get("llm_response")
        if isinstance(llm_response, dict):
            content = llm_response.get("content")
            if isinstance(content, str):
                return content.strip()
        for key in ("answer", "response", "result", "output"):
            value = raw_response.get(key)
            if isinstance(value, str):
                return value.strip()
    return ""


def _extract_contexts(
    raw_response: dict[str, Any] | list[Any] | str | None,
    answer: str,
) -> list[dict[str, Any]]:
    if isinstance(raw_response, list):
        return [_normalize_context(item) for item in raw_response]
    if isinstance(raw_response, dict):
        data = raw_response.get("data")
        if isinstance(data, dict):
            chunks = data.get("chunks")
            if isinstance(chunks, list):
                return [_normalize_context(item) for item in chunks]
        for key in ("contexts", "sources", "retrieved_context", "context_data", "items"):
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
    chunk_id = item.get("chunk_id") or item.get("source_chunk_id")
    normalized = {**item}
    normalized["text"] = str(text or "")
    normalized["score"] = item.get("score")
    normalized["source_chunk_id"] = chunk_id
    normalized["source_document_id"] = (
        item.get("document_id")
        or item.get("source_document_id")
        or _document_id_from_chunk_id(chunk_id)
    )
    return normalized


def _document_id_from_chunk_id(chunk_id: Any) -> str | None:
    if not isinstance(chunk_id, str):
        return None
    match = _CHUNK_ID_DOC_RE.match(chunk_id)
    return match.group(1) if match else None
