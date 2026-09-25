from __future__ import annotations

from benchmark.experiments.compare import ComparisonRow, compare_experiment


def render_markdown_report(*, experiment_id: str, rows: list[ComparisonRow]) -> str:
    lines = [
        f"# Experiment Report: {experiment_id}",
        "",
        "| method_id | agent_mode | question_type | exact_match | answer_f1 | "
        "evidence_recall@5 | citation_accuracy | latency_ms | total_tokens | estimated_cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.method_id} | "
            f"{row.agent_mode} | "
            f"{row.question_type} | "
            f"{row.exact_match:.4f} | "
            f"{row.answer_f1:.4f} | "
            f"{row.evidence_recall_at_5:.4f} | "
            f"{row.citation_accuracy:.4f} | "
            f"{row.latency_ms:.2f} | "
            f"{row.total_tokens:.0f} | "
            f"{row.estimated_cost:.6f} |"
        )
    return "\n".join(lines) + "\n"


def generate_report(*, experiment_id: str, store: object) -> str:
    return render_markdown_report(
        experiment_id=experiment_id,
        rows=compare_experiment(experiment_id=experiment_id, store=store),
    )
