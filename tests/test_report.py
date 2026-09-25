from benchmark.experiments.compare import ComparisonRow
from benchmark.experiments.report import generate_report, render_markdown_report


def test_report_generation_renders_markdown_table() -> None:
    report = render_markdown_report(
        experiment_id="exp_1",
        rows=[
            ComparisonRow(
                method_id="vector_rag",
                agent_mode="single_agent",
                question_type="span_extraction",
                exact_match=1.0,
                answer_f1=1.0,
                evidence_recall_at_5=1.0,
                citation_accuracy=1.0,
                latency_ms=25.0,
                total_tokens=10,
                estimated_cost=0.001,
            )
        ],
    )

    assert "# Experiment Report: exp_1" in report
    assert "| vector_rag | single_agent | span_extraction |" in report


def test_generate_report_uses_store() -> None:
    report = generate_report(experiment_id="exp_1", store=FakeReportStore())

    assert "vector_rag" in report


class FakeReportStore:
    def load_compare_rows(self, *, experiment_id: str):
        assert experiment_id == "exp_1"
        return [
            {
                "method_id": "vector_rag",
                "agent_mode": "multi_agent",
                "question_type": "span_extraction",
                "metric_name": "exact_match",
                "metric_value": 1.0,
            }
        ]
