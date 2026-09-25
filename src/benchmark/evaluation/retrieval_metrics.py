from __future__ import annotations

from benchmark.core.schemas import RetrievedItem


def evidence_recall_at_k(
    retrieved_items: list[RetrievedItem],
    gold_ids: list[str],
    k: int,
) -> float:
    gold = set(gold_ids)
    if not gold:
        return 1.0
    retrieved = _retrieved_ids(retrieved_items[:k])
    return len(gold & retrieved) / len(gold)


def precision_at_k(retrieved_items: list[RetrievedItem], gold_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top_items = retrieved_items[:k]
    if not top_items:
        return 0.0
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = sum(1 for item in top_items if _item_ids(item) & gold)
    return hits / len(top_items)


def citation_accuracy(citations: list[str], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    if not citations and not gold:
        return 1.0
    if not citations:
        return 0.0
    return sum(1 for citation in citations if citation in gold) / len(citations)


def _retrieved_ids(items: list[RetrievedItem]) -> set[str]:
    ids: set[str] = set()
    for item in items:
        ids.update(_item_ids(item))
    return ids


def _item_ids(item: RetrievedItem) -> set[str]:
    ids = {item.item_id}
    if item.source_chunk_id:
        ids.add(item.source_chunk_id)
    if item.source_document_id:
        ids.add(item.source_document_id)
    evidence_id = item.metadata.get("evidence_id")
    if isinstance(evidence_id, str):
        ids.add(evidence_id)
    evidence_ids = item.metadata.get("evidence_ids")
    if isinstance(evidence_ids, list):
        ids.update(str(value) for value in evidence_ids)
    # Multi-document items (e.g. cognee's serialized subgraph, which is one
    # item embedding several source documents) expose every contributing
    # document here so document-level recall is not capped at one hit.
    document_ids = item.metadata.get("document_ids")
    if isinstance(document_ids, list):
        ids.update(str(value) for value in document_ids)
    return ids
