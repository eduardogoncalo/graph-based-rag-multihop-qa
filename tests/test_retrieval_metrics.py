from benchmark.core.schemas import RetrievedItem
from benchmark.evaluation.retrieval_metrics import (
    citation_accuracy,
    evidence_recall_at_k,
    precision_at_k,
)


def test_evidence_recall_at_k_counts_gold_hits() -> None:
    items = [
        RetrievedItem(item_id="item_1", text="x", source_chunk_id="chunk_1"),
        RetrievedItem(item_id="item_2", text="y", source_chunk_id="chunk_2"),
    ]

    assert evidence_recall_at_k(items, ["chunk_1", "chunk_3"], 2) == 0.5
    assert evidence_recall_at_k(items, ["chunk_1"], 1) == 1.0


def test_precision_at_k_counts_top_k_hits() -> None:
    items = [
        RetrievedItem(item_id="item_1", text="x", source_chunk_id="chunk_1"),
        RetrievedItem(item_id="item_2", text="y", source_chunk_id="chunk_2"),
    ]

    assert precision_at_k(items, ["chunk_1"], 2) == 0.5


def test_citation_accuracy_scores_citations_against_gold() -> None:
    assert citation_accuracy(["chunk_1", "chunk_2"], ["chunk_1"]) == 0.5
    assert citation_accuracy([], ["chunk_1"]) == 0.0
    assert citation_accuracy([], []) == 1.0
