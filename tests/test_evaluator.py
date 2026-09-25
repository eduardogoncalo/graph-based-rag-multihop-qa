from benchmark.core.schemas import RetrievedItem
from benchmark.evaluation.evaluator import EvaluationInput, evaluate_input, evaluate_run


def test_evaluator_writes_evaluation_results() -> None:
    store = FakeEvaluationStore()
    summary = evaluate_input(
        EvaluationInput(
            run_id="run_1",
            prediction="Delaware law",
            gold_answer="Delaware law",
            retrieved_items=[
                RetrievedItem(
                    item_id="item_1",
                    text="x",
                    source_document_id="doc_1",
                    source_chunk_id="chunk_1",
                )
            ],
            gold_document_ids=["doc_1"],
            gold_evidence_ids=["ev_1"],
            citations=["chunk_1"],
            retrieval_latency_ms=5.0,
            generation_latency_ms=7.0,
            prompt_tokens=10,
            completion_tokens=5,
            estimated_cost=0.001,
        ),
        experiment_store=store,
    )

    assert summary.metrics["exact_match"] == 1.0
    assert summary.metrics["answer_f1"] == 1.0
    assert summary.metrics["evidence_recall_at_5"] == 1.0
    assert len(store.persisted) == len(summary.metrics)


def test_alias_metrics_are_additive_and_do_not_change_baseline() -> None:
    # primary gold mismatches the prediction, but an alias matches: the raw
    # baseline stays 0 while the alias-aware + containment metrics recover it.
    summary = evaluate_input(
        EvaluationInput(
            run_id="run_alias",
            prediction="The narrator ultimately marries Tracy Mosby in the finale.",
            gold_answer="Tracy McConnell",
            gold_answer_aliases=["Tracy Mosby"],
            retrieved_items=[],
            gold_document_ids=["doc_1"],
            gold_evidence_ids=["ev_1"],
            citations=[],
        ),
    )
    # raw baseline (single gold) — computed against "Tracy McConnell" only
    assert summary.metrics["exact_match"] == 0.0
    baseline_f1 = summary.metrics["answer_f1"]
    # alias-aware: F1 strictly improves via the matching alias "Tracy Mosby"
    assert summary.metrics["answer_f1_aliases"] > baseline_f1
    # containment fires (the gold/alias answer is present, just buried)
    assert summary.metrics["answer_recall_containment"] == 1.0
    assert summary.metrics["answer_substring_containment"] == 1.0
    # exact_match_aliases still 0 here (verbose, not an exact span)
    assert summary.metrics["exact_match_aliases"] == 0.0


def test_recall_uses_document_namespace_not_evidence_namespace() -> None:
    """Regression for the ev_*/doc_* namespace mismatch (recall was always 0).

    The retriever stamps ``source_document_id=doc_*``; the question's gold
    evidence ids are ``ev_*``. Recall must compare against the DOCUMENT-level
    gold (so it is 1.0 here) and ignore the evidence-level ids entirely.
    """
    summary = evaluate_input(
        EvaluationInput(
            run_id="run_ns",
            prediction="irrelevant",
            gold_answer="irrelevant",
            retrieved_items=[
                RetrievedItem(
                    item_id="vector_rag:chunk_x",
                    text="x",
                    source_document_id="doc_a",
                    source_chunk_id="chunk_x",
                )
            ],
            gold_document_ids=["doc_a"],
            gold_evidence_ids=["ev_hash_abc"],
            citations=["chunk_x"],
        ),
    )

    # Matches on the document namespace despite ev_* gold sharing nothing.
    assert summary.metrics["evidence_recall_at_5"] == 1.0
    assert summary.metrics["precision_at_5"] == 1.0
    # Citation stays on the evidence namespace: chunk_x is not an ev_* id.
    assert summary.metrics["citation_accuracy"] == 0.0


def test_musique_like_partial_recall() -> None:
    """Two supporting documents, only one retrieved in the top-5 -> recall 0.5."""
    summary = evaluate_input(
        EvaluationInput(
            run_id="run_mq",
            prediction="answer",
            gold_answer="answer",
            retrieved_items=[
                RetrievedItem(
                    item_id=f"vector_rag:chunk_{i}",
                    text="x",
                    source_document_id=doc_id,
                    source_chunk_id=f"chunk_{i}",
                )
                for i, doc_id in enumerate(
                    ["doc_a", "doc_x", "doc_y", "doc_z", "doc_w"]
                )
            ],
            gold_document_ids=["doc_a", "doc_b"],
            gold_evidence_ids=["ev_a", "ev_b"],
            citations=[],
        ),
    )

    assert summary.metrics["evidence_recall_at_5"] == 0.5


def test_evaluate_run_loads_input_from_store() -> None:
    store = FakeEvaluationStore()

    summary = evaluate_run(run_id="run_1", store=store)

    assert store.loaded_run_id == "run_1"
    assert summary.metrics["citation_accuracy"] == 1.0


class FakeEvaluationStore:
    def __init__(self) -> None:
        self.persisted = []
        self.loaded_run_id = ""

    def load_evaluation_input(self, *, run_id: str, k: int = 5) -> EvaluationInput:
        self.loaded_run_id = run_id
        return EvaluationInput(
            run_id=run_id,
            prediction="Delaware law",
            gold_answer="Delaware",
            retrieved_items=[
                RetrievedItem(
                    item_id="item_1",
                    text="x",
                    source_document_id="doc_1",
                    source_chunk_id="chunk_1",
                )
            ],
            gold_document_ids=["doc_1"],
            gold_evidence_ids=["chunk_1"],
            citations=["chunk_1"],
            k=k,
        )

    def persist_evaluation_result(self, **kwargs: object) -> str:
        self.persisted.append(kwargs)
        return "eval_1"
