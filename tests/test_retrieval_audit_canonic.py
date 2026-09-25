from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "retrieval_audit_canonic",
    Path(__file__).resolve().parents[1] / "scripts" / "retrieval_audit_canonic.py",
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def test_recall_at_5_follows_the_thesis_footnote() -> None:
    # Two of the three gold documents among the first five -> 2/3.
    docs = ["doc_a1", "doc_x1", "doc_b2", "doc_x2", "doc_x3", "doc_c3"]
    metrics = audit.per_question_metrics(docs, {"doc_a1", "doc_b2", "doc_c3"})
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["all_gold@5"] == 0.0
    assert metrics["recall@inf"] == 1.0
    assert metrics["all_gold@inf"] == 1.0
    assert metrics["first_gold_rank"] == 1
    assert metrics["last_gold_rank"] == 6
    assert metrics["retrieved_docs"] == 6


def test_documents_are_deduplicated_in_rank_order() -> None:
    rows = [
        (1, "doc_aaaaaa", "c1", ""),
        (2, "doc_bbbbbb", "c2", ""),
        (3, "doc_aaaaaa", "c3", ""),
    ]
    assert audit.docs_for_method("vector_rag", rows) == ["doc_aaaaaa", "doc_bbbbbb"]


def test_lightrag_takes_the_document_from_the_chunk_prefix() -> None:
    rows = [
        (1, None, "doc_0abc12-chunk-001", ""),
        (2, "doc_ffffff", "not-a-chunk-id", ""),
    ]
    assert audit.docs_for_method("lightrag_neo4j", rows) == ["doc_0abc12", "doc_ffffff"]


def test_cognee_adds_ids_found_in_the_blob_in_order_of_appearance() -> None:
    rows = [(1, None, None, "see doc_111111 then doc_222222, again doc_111111")]
    assert audit.docs_for_method("cognee", rows) == ["doc_111111", "doc_222222"]


def test_aggregate_reports_docs_per_question_and_zero_gold() -> None:
    entries = [
        audit.per_question_metrics(["doc_a", "doc_b"], {"doc_a"}) | {"n_hops": 2},
        audit.per_question_metrics(["doc_x"], {"doc_a"}) | {"n_hops": 3},
    ]
    summary = audit.aggregate(entries, label="t", ranked=True)
    assert summary["mean_retrieved_docs"] == 1.5
    assert summary["zero_gold_questions"] == 1
    assert summary["recall@5"] == 0.5
    assert set(summary["by_hop"]) == {2, 3}


def test_cells_use_the_suffixes_of_the_other_protocol_scripts() -> None:
    names = [experiment for experiment, *_ in audit.cells("musique_eval1k_")]
    assert names == [
        "musique_eval1k_vector_v1free",
        "musique_eval1k_lightrag_v1free",
        "musique_eval1k_ms_graphrag_v1free",
        "musique_eval1k_hipporag2_v1free",
        "musique_eval1k_cognee_v1free_k5",
        "musique_eval1k_oracle_gold_v1free",
    ]
