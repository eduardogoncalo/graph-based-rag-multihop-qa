from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.methods.cognee.config_builder import COGNEE_METHOD_ID

DOCUMENT_ID_RE = re.compile(r"(?:document_id\s*[:=]\s*)?(doc_[0-9a-f]+)", re.IGNORECASE)


def parse_cognee_result(
    *,
    query: str,
    raw_result: Any,
    top_k: int,
    latency_ms: float | None = None,
    chunk_mappings: list[dict[str, Any]] | None = None,
) -> RetrievalResult:
    started = time.perf_counter()
    payload = _as_payload(raw_result)
    chunk_lookup = _ChunkMappingLookup(chunk_mappings or [])
    items = _extract_items(payload, chunk_lookup=chunk_lookup)[:top_k]
    generated_answer = _extract_generated_answer(payload)
    raw_trace = _extract_raw_trace(payload)
    nodes = _extract_nodes(payload)
    relationships = _extract_relationships(payload)
    retriever_purity = _extract_retriever_purity(payload, generated_answer=generated_answer)
    metadata = {
        "retrieval_trace_available": bool(raw_trace or items or nodes or relationships),
        "retriever_purity": retriever_purity,
        "raw_result_available": raw_result is not None,
        "trace_available": bool(raw_trace or items or nodes or relationships),
        "textual_context_available": bool(items),
        "node_trace_available": bool(nodes),
        "relationship_trace_available": bool(relationships),
    }
    if generated_answer:
        metadata["generated_answer"] = generated_answer
    if raw_trace is not None:
        metadata["raw_trace"] = raw_trace
        metadata["retrieval_trace"] = raw_trace
    if nodes:
        metadata["cognee_nodes"] = nodes
    if relationships:
        metadata["cognee_relationships"] = relationships
    if not items:
        metadata.setdefault("retrieval_trace_available", False)
    return RetrievalResult(
        method_id=COGNEE_METHOD_ID,
        query=query,
        items=items,
        raw_response=_jsonable(raw_result),
        latency_ms=latency_ms if latency_ms is not None else (time.perf_counter() - started) * 1000,
        metadata=metadata,
    )


def _as_payload(raw_result: Any) -> dict[str, Any]:
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, list):
        return {"result": [_jsonable(item) for item in raw_result]}
    if hasattr(raw_result, "model_dump"):
        dumped = raw_result.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {"result": dumped}
    if hasattr(raw_result, "__dict__"):
        return dict(raw_result.__dict__)
    return {"result": raw_result}


def _extract_items(
    payload: dict[str, Any],
    *,
    chunk_lookup: "_ChunkMappingLookup | None" = None,
) -> list[RetrievedItem]:
    candidate_lists = [
        payload.get("retrieval_items"),
        payload.get("items"),
        payload.get("contexts"),
        payload.get("chunks"),
        _nested(payload, ["result", "retrieval_items"]),
        _nested(payload, ["result", "items"]),
        _nested(payload, ["result", "contexts"]),
        _nested(payload, ["result", "chunks"]),
        _nested(payload, ["retrieval", "items"]),
        _nested(payload, ["context", "items"]),
        payload.get("result"),
        payload.get("raw_trace"),
    ]
    for candidate in candidate_lists:
        if isinstance(candidate, list):
            flattened = _flatten_result_entries(_jsonable(candidate))
            items = [
                _item_from_raw(item, index, chunk_lookup=chunk_lookup)
                for index, item in enumerate(flattened, start=1)
            ]
            return [item for item in items if item is not None]
    result = payload.get("result")
    if isinstance(result, dict):
        text = result.get("text") or result.get("context") or result.get("content")
    else:
        text = payload.get("text") or payload.get("context") or payload.get("content")
    if isinstance(text, str) and text.strip():
        return [
            RetrievedItem(
                item_id="cognee:context:1",
                text=text,
                metadata={"kind": "context"},
            )
        ]
    return []


def _item_from_raw(
    raw_item: Any,
    index: int,
    *,
    chunk_lookup: "_ChunkMappingLookup | None" = None,
) -> RetrievedItem | None:
    if isinstance(raw_item, str):
        text = raw_item.strip()
        if not text:
            return None
        mapping = chunk_lookup.match(text) if chunk_lookup is not None else {}
        document_id = extract_document_id(text) or _first_str(mapping, ["source_document_id", "document_id"])
        source_dataset_name = _first_str(mapping, ["source_dataset_name", "dataset_name"])
        cognee_chunk_id = _first_str(mapping, ["source_cognee_chunk_id", "cognee_chunk_id", "id"])
        source_content_hash = _first_str(mapping, ["source_content_hash", "content_hash"])
        chunk_id, chunk_strategy = _resolve_chunk_id(
            canonical_chunk_id=None,
            source_dataset_name=source_dataset_name,
            cognee_chunk_id=cognee_chunk_id,
            source_content_hash=source_content_hash,
            text=text,
        )
        all_document_ids = extract_all_document_ids(text)
        metadata = _clean_metadata(
            {
                "kind": "text",
                "source_chunk_id_strategy": chunk_strategy,
                "source_dataset_name": source_dataset_name,
                "source_cognee_chunk_id": cognee_chunk_id,
                "source_content_hash": source_content_hash,
                "source_text_preview": text[:240],
                "raw_trace_available": True,
                "document_ids": all_document_ids or None,
            }
        )
        return RetrievedItem(
            item_id=cognee_chunk_id or f"cognee:item:{index}",
            text=text,
            source_document_id=document_id,
            source_chunk_id=chunk_id,
            metadata=metadata,
        )
    if not isinstance(raw_item, dict):
        return None
    if raw_item.get("source") == "graph_context" and isinstance(raw_item.get("content"), str):
        raw_item = {"text": raw_item["content"], **raw_item}
    if "search_result" in raw_item and isinstance(raw_item["search_result"], str):
        raw_item = {"text": raw_item["search_result"], **raw_item}
    if "search_result" in raw_item and isinstance(raw_item["search_result"], list):
        joined = "\n\n".join(str(part) for part in raw_item["search_result"] if part)
        raw_item = {"text": joined, **raw_item}
    raw_payload = raw_item.get("raw")
    if isinstance(raw_payload, dict):
        raw_item = {**raw_payload, **raw_item}
    text = _first_str(raw_item, ["text", "content", "chunk", "context", "page_content", "search_result", "value"])
    if not text:
        return None
    mapping = chunk_lookup.match(text) if chunk_lookup is not None else {}
    item_id = _first_str(raw_item, ["item_id", "id", "node_id", "memory_id", "chunk_id"])
    raw_metadata = raw_item.get("metadata") if isinstance(raw_item.get("metadata"), dict) else {}
    document_id = (
        _first_str(raw_item, ["source_document_id", "document_id", "doc_id"])
        or _first_str(raw_metadata, ["source_document_id", "document_id", "doc_id"])
        or extract_document_id(text)
        or _first_str(mapping, ["source_document_id", "document_id"])
    )
    canonical_chunk_id = _first_str(raw_item, ["source_chunk_id", "chunk_id", "canonical_chunk_id"]) or _first_str(
        raw_metadata,
        ["source_chunk_id", "chunk_id", "canonical_chunk_id"],
    )
    source_dataset_name = (
        _first_str(raw_item, ["source_dataset_name", "dataset_name"])
        or _first_str(raw_metadata, ["source_dataset_name", "dataset_name"])
        or _first_str(mapping, ["source_dataset_name", "dataset_name"])
    )
    cognee_chunk_id = (
        _first_str(raw_item, ["source_cognee_chunk_id", "cognee_chunk_id"])
        or _first_str(raw_metadata, ["source_cognee_chunk_id", "cognee_chunk_id"])
        or _first_str(mapping, ["source_cognee_chunk_id", "cognee_chunk_id", "id"])
    )
    source_content_hash = (
        _first_str(raw_item, ["source_content_hash", "content_hash"])
        or _first_str(raw_metadata, ["source_content_hash", "content_hash"])
        or _first_str(mapping, ["source_content_hash", "content_hash"])
    )
    chunk_id, chunk_strategy = _resolve_chunk_id(
        canonical_chunk_id=canonical_chunk_id,
        source_dataset_name=source_dataset_name,
        cognee_chunk_id=cognee_chunk_id,
        source_content_hash=source_content_hash,
        text=text,
    )
    score = _first_float(raw_item, ["score", "similarity", "relevance", "distance"])
    metadata = dict(raw_metadata)
    metadata.update(
        {
            key: value
            for key, value in raw_item.items()
            if key not in {"text", "content", "chunk", "context", "page_content", "metadata"}
            and value is not None
        }
    )
    metadata.update(
        {
            "source_chunk_id_strategy": chunk_strategy,
            "source_dataset_name": source_dataset_name,
            "source_cognee_chunk_id": cognee_chunk_id,
            "source_content_hash": source_content_hash,
            "source_text_preview": text[:240],
            "raw_trace_available": True,
            "document_ids": extract_all_document_ids(text) or None,
        }
    )
    return RetrievedItem(
        item_id=item_id or f"cognee:item:{index}",
        text=text,
        score=score,
        source_document_id=document_id,
        source_chunk_id=chunk_id,
        metadata=_clean_metadata(metadata),
    )


def _extract_generated_answer(payload: dict[str, Any]) -> str | None:
    answer = _first_str(payload, ["generated_answer", "answer", "response", "output", "completion"])
    if answer:
        return answer
    result = payload.get("result")
    if isinstance(result, dict):
        answer = _first_str(result, ["generated_answer", "answer", "response", "output", "completion"])
        if answer:
            return answer
    if isinstance(result, list) and result and all(isinstance(item, str) for item in result):
        return "\n\n".join(item for item in result if item.strip())
    return None


def _extract_raw_trace(payload: dict[str, Any]) -> Any:
    for key in ("raw_trace", "retrieval_trace", "trace"):
        value = payload.get(key)
        if value:
            return value
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ("raw_trace", "retrieval_trace", "trace"):
            value = result.get(key)
            if value:
                return value
    result = payload.get("result")
    if result:
        return result
    return None


def _extract_retriever_purity(
    payload: dict[str, Any],
    *,
    generated_answer: str | None,
) -> str:
    value = _first_str(payload, ["retriever_purity"])
    if value in {"context_only", "mixed", "system_level", "unknown"}:
        return value
    if generated_answer:
        return "mixed"
    if _extract_items(payload):
        return "context_only"
    return "unknown"


def _extract_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        payload.get("nodes"),
        _nested(payload, ["subgraph", "nodes"]),
        _nested(payload, ["graph", "nodes"]),
        _nested(payload, ["raw_trace", "nodes"]),
    ]
    nodes: list[dict[str, Any]] = []
    for candidate in candidates:
        nodes.extend(_dict_list(_jsonable(candidate)))
    for entry in _flatten_result_entries(_jsonable(payload.get("result"))):
        if isinstance(entry, dict):
            raw = entry.get("raw") if isinstance(entry.get("raw"), dict) else entry
            nodes.extend(_dict_list(raw.get("nodes")))
            nodes.extend(_dict_list(raw.get("vertices")))
    return nodes


def _extract_relationships(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        payload.get("relationships"),
        payload.get("edges"),
        _nested(payload, ["subgraph", "relationships"]),
        _nested(payload, ["subgraph", "edges"]),
        _nested(payload, ["graph", "relationships"]),
        _nested(payload, ["graph", "edges"]),
        _nested(payload, ["raw_trace", "relationships"]),
    ]
    relationships: list[dict[str, Any]] = []
    for candidate in candidates:
        relationships.extend(_dict_list(_jsonable(candidate)))
    for entry in _flatten_result_entries(_jsonable(payload.get("result"))):
        if isinstance(entry, dict):
            raw = entry.get("raw") if isinstance(entry.get("raw"), dict) else entry
            relationships.extend(_dict_list(raw.get("relationships")))
            relationships.extend(_dict_list(raw.get("edges")))
    return relationships


def _flatten_result_entries(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        if "result" in value:
            return _flatten_result_entries(value["result"])
        if "search_result" in value:
            return _flatten_result_entries(value["search_result"])
        return [value]
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            if isinstance(item, dict) and "search_result" in item:
                nested = item["search_result"]
                if isinstance(nested, list):
                    for nested_item in nested:
                        if isinstance(nested_item, dict):
                            flattened.append({**nested_item, "dataset_id": item.get("dataset_id"), "dataset_name": item.get("dataset_name")})
                        else:
                            flattened.append(nested_item)
                else:
                    flattened.append(item)
            else:
                flattened.append(item)
        return flattened
    return [value]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        item = _jsonable(item)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _first_str(payload: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _nested(payload: dict[str, Any], keys: list[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def extract_all_document_ids(text: str) -> list[str]:
    """All distinct document ids embedded in a text blob, in first-seen order.

    GRAPH_COMPLETION serializes the whole retrieved subgraph into ONE item, so
    the single ``source_document_id`` (first match) caps document-level recall
    at one hit; this recovers every contributing document for the metrics.
    """
    seen: list[str] = []
    for match in DOCUMENT_ID_RE.finditer(text):
        document_id = match.group(1)
        if document_id not in seen:
            seen.append(document_id)
    return seen


def extract_document_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        direct = _first_str(value, ["source_document_id", "document_id", "doc_id"])
        if direct:
            return direct
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            nested = extract_document_id(metadata)
            if nested:
                return nested
        text = _first_str(value, ["text", "content", "chunk", "context", "page_content", "search_result", "value"])
        return extract_document_id(text) if text else None
    match = DOCUMENT_ID_RE.search(str(value))
    return match.group(1) if match else None


def _resolve_chunk_id(
    *,
    canonical_chunk_id: str | None,
    source_dataset_name: str | None,
    cognee_chunk_id: str | None,
    source_content_hash: str | None,
    text: str,
) -> tuple[str | None, str]:
    if canonical_chunk_id:
        return canonical_chunk_id, "canonical"
    stable_part = cognee_chunk_id or source_content_hash
    if not stable_part:
        stable_part = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    dataset = source_dataset_name or "unknown_dataset"
    return f"cognee::{dataset}::{stable_part}", "deterministic_fallback"


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value is not None}


def _normalize_match_text(text: str) -> str:
    return " ".join(text.split())


class _ChunkMappingLookup:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.by_exact_text: dict[str, dict[str, Any]] = {}
        for row in rows:
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                self.by_exact_text[text.strip()] = row

    def match(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        exact = self.by_exact_text.get(stripped)
        if exact is not None:
            return exact
        normalized = _normalize_match_text(stripped)
        if not normalized:
            return {}
        requested_document_id = extract_document_id(stripped)
        for row in self.rows:
            if requested_document_id and row.get("source_document_id") != requested_document_id:
                continue
            candidate = row.get("text")
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            normalized_candidate = _normalize_match_text(candidate)
            if normalized == normalized_candidate:
                return row
            if requested_document_id and normalized_candidate in normalized:
                return row
            if len(normalized) >= 80 and normalized in normalized_candidate:
                return row
            if len(normalized_candidate) >= 80 and normalized_candidate in normalized:
                return row
        return {}
