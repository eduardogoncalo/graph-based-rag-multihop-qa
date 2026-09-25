from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComparisonRow:
    method_id: str
    agent_mode: str
    question_type: str
    exact_match: float
    answer_f1: float
    evidence_recall_at_5: float
    citation_accuracy: float
    latency_ms: float
    total_tokens: float
    estimated_cost: float


METRIC_COLUMNS = [
    "exact_match",
    "answer_f1",
    "evidence_recall_at_5",
    "citation_accuracy",
    "latency_ms",
    "total_tokens",
    "estimated_cost",
]


def build_comparison_table(metric_rows: list[dict[str, Any]]) -> list[ComparisonRow]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in metric_rows:
        key = (
            str(row["method_id"]),
            str(row["agent_mode"]),
            str(row.get("question_type") or "unknown"),
        )
        metric_name = _normalize_metric_name(str(row["metric_name"]))
        if metric_name in METRIC_COLUMNS:
            grouped[key][metric_name].append(float(row["metric_value"]))

    table: list[ComparisonRow] = []
    for (method_id, agent_mode, question_type), metrics in sorted(grouped.items()):
        table.append(
            ComparisonRow(
                method_id=method_id,
                agent_mode=agent_mode,
                question_type=question_type,
                exact_match=_mean(metrics["exact_match"]),
                answer_f1=_mean(metrics["answer_f1"]),
                evidence_recall_at_5=_mean(metrics["evidence_recall_at_5"]),
                citation_accuracy=_mean(metrics["citation_accuracy"]),
                latency_ms=_mean(metrics["latency_ms"]),
                total_tokens=sum(metrics["total_tokens"]),
                estimated_cost=sum(metrics["estimated_cost"]),
            )
        )
    return table


def compare_experiment(*, experiment_id: str, store: Any) -> list[ComparisonRow]:
    return build_comparison_table(store.load_compare_rows(experiment_id=experiment_id))


def _normalize_metric_name(metric_name: str) -> str:
    if metric_name == "evidence_recall_at_5":
        return "evidence_recall_at_5"
    if metric_name == "total_latency":
        return "latency_ms"
    return metric_name


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
