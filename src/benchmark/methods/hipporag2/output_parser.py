from __future__ import annotations

import re
from typing import Any

from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.methods.hipporag2.config_builder import HIPPORAG2_METHOD_ID

# Benchmark document namespace (same convention the ms_graphrag parser recovers).
DOCUMENT_ID_RE = re.compile(r"\b(doc_[0-9a-f]{6,})\b")


def parse_query_payload(
    *,
    query: str,
    payload: dict[str, Any],
    latency_ms: float,
    top_k: int,
) -> RetrievalResult:
    """Convert the runner payload into the benchmark ``RetrievalResult``.

    Runner payload contract (scripts/hipporag2_query_runner.py):
      {"items": [{"rank": 1, "text": ..., "score": ..., "document_id": "doc_*"|null}],
       "stats": {...}}

    ``source_document_id`` is REQUIRED for document-level evidence recall —
    populated from the passage-map lookup done inside the runner, with a
    doc_*-regex fallback over the text (defensive; MuSiQue passages do not
    contain doc ids).
    """
    items: list[RetrievedItem] = []
    for raw in (payload.get("items") or [])[:top_k]:
        if not isinstance(raw, dict):
            continue
        rank = raw.get("rank")
        text = str(raw.get("text") or "")
        document_id = raw.get("document_id")
        if not document_id:
            found = DOCUMENT_ID_RE.findall(text)
            document_id = found[0] if found else None
        items.append(
            RetrievedItem(
                item_id=f"hipporag2:ppr:{rank}",
                text=text,
                score=float(raw["score"]) if raw.get("score") is not None else None,
                source_document_id=document_id,
                source_chunk_id=document_id,
                metadata={
                    "document_ids": [document_id] if document_id else [],
                    "retrieval": "ppr",
                },
            )
        )
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    return RetrievalResult(
        method_id=HIPPORAG2_METHOD_ID,
        query=query,
        items=items,
        raw_response=payload,
        latency_ms=latency_ms,
        metadata={"unmapped_items": sum(1 for i in items if i.source_document_id is None), **stats},
    )
