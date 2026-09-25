from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.core.schemas import RetrievedItem
from benchmark.evaluation.answer_metrics import (
    answer_f1,
    answer_f1_aliases,
    answer_recall_containment,
    answer_substring_containment,
    exact_match,
    exact_match_aliases,
)
from benchmark.evaluation.latency_cost import aggregate_latency_cost
from benchmark.evaluation.retrieval_metrics import (
    citation_accuracy,
    evidence_recall_at_k,
    precision_at_k,
)


@dataclass(frozen=True)
class EvaluationInput:
    run_id: str
    prediction: str
    gold_answer: str | list[str] | bool | None
    retrieved_items: list[RetrievedItem]
    # Document-level support set (canonical ``doc_*`` ids) used for retrieval
    # recall/precision. Distinct from ``gold_evidence_ids`` (evidence-level
    # ``ev_*`` ids) which is reserved for citation/attribution. See
    # docs/decisions/0001-retrieval-evaluation-granularity.md.
    gold_document_ids: list[str]
    gold_evidence_ids: list[str]
    citations: list[str]
    # MuSiQue answer aliases (q.metadata->'answer_aliases'). Optional/back-compat:
    # used only by the additive alias/containment metrics, never by the raw
    # exact_match/answer_f1 baseline. See ADR 0001 / answer_quality_evaluation_plan.md.
    gold_answer_aliases: list[str] = field(default_factory=list)
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None
    k: int = 5


@dataclass(frozen=True)
class EvaluationSummary:
    run_id: str
    metrics: dict[str, float]


def evaluate_input(
    evaluation_input: EvaluationInput,
    *,
    experiment_store: Any | None = None,
) -> EvaluationSummary:
    gold_answer = _first_gold_answer(evaluation_input.gold_answer)
    latency = aggregate_latency_cost(
        retrieval_latency=evaluation_input.retrieval_latency_ms,
        generation_latency=evaluation_input.generation_latency_ms,
        prompt_tokens=evaluation_input.prompt_tokens,
        completion_tokens=evaluation_input.completion_tokens,
        total_tokens=evaluation_input.total_tokens,
        estimated_cost=evaluation_input.estimated_cost,
    )
    # gold + aliases for the additive (non-baseline) answer metrics.
    alias_golds = [gold_answer, *evaluation_input.gold_answer_aliases]
    metrics = {
        # --- raw baseline (single gold) — NEVER changed/overwritten ---
        "exact_match": exact_match(evaluation_input.prediction, gold_answer),
        "answer_f1": answer_f1(evaluation_input.prediction, gold_answer),
        # --- additive alias-aware metrics (MuSiQue protocol: gold + aliases) ---
        "exact_match_aliases": exact_match_aliases(evaluation_input.prediction, alias_golds),
        "answer_f1_aliases": answer_f1_aliases(evaluation_input.prediction, alias_golds),
        # --- additive diagnostic containment metrics (NOT final metrics) ---
        "answer_recall_containment": answer_recall_containment(
            evaluation_input.prediction, alias_golds
        ),
        "answer_substring_containment": answer_substring_containment(
            evaluation_input.prediction, alias_golds
        ),
        # Retrieval recall/precision compare retrieved items against the
        # DOCUMENT-level gold (``gold_document_ids``). The metric key is kept as
        # ``evidence_recall_at_k`` for back-compat and because, in MuSiQue, a
        # supporting evidence passage IS a canonical document.
        f"evidence_recall_at_{evaluation_input.k}": evidence_recall_at_k(
            evaluation_input.retrieved_items,
            evaluation_input.gold_document_ids,
            evaluation_input.k,
        ),
        f"precision_at_{evaluation_input.k}": precision_at_k(
            evaluation_input.retrieved_items,
            evaluation_input.gold_document_ids,
            evaluation_input.k,
        ),
        # Citation/attribution stays at the evidence-level namespace.
        "citation_accuracy": citation_accuracy(
            evaluation_input.citations,
            evaluation_input.gold_evidence_ids,
        ),
        "retrieval_latency": latency.retrieval_latency,
        "generation_latency": latency.generation_latency,
        "total_latency": latency.total_latency,
        "prompt_tokens": float(latency.prompt_tokens),
        "completion_tokens": float(latency.completion_tokens),
        "total_tokens": float(latency.total_tokens),
        "estimated_cost": latency.estimated_cost,
    }
    if experiment_store is not None:
        for metric_name, metric_value in metrics.items():
            experiment_store.persist_evaluation_result(
                run_id=evaluation_input.run_id,
                metric_name=metric_name,
                metric_value=metric_value,
            )
    return EvaluationSummary(run_id=evaluation_input.run_id, metrics=metrics)


def evaluate_run(
    *,
    run_id: str,
    store: Any,
    k: int = 5,
) -> EvaluationSummary:
    payload = store.load_evaluation_input(run_id=run_id, k=k)
    return evaluate_input(payload, experiment_store=store)


def _first_gold_answer(value: str | list[str] | bool | None) -> object:
    if isinstance(value, list):
        return value[0] if value else None
    return value
