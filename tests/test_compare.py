from benchmark.experiments.compare import build_comparison_table, compare_experiment


def test_compare_table_generation_aggregates_metrics() -> None:
    rows = build_comparison_table(
        [
            _row("vector_rag", "single_agent", "span_extraction", "exact_match", 1.0),
            _row("vector_rag", "single_agent", "span_extraction", "answer_f1", 0.5),
            _row("vector_rag", "single_agent", "span_extraction", "evidence_recall_at_5", 1.0),
            _row("vector_rag", "single_agent", "span_extraction", "citation_accuracy", 1.0),
            _row("vector_rag", "single_agent", "span_extraction", "total_latency", 25.0),
            _row("vector_rag", "single_agent", "span_extraction", "total_tokens", 10.0),
            _row("vector_rag", "single_agent", "span_extraction", "estimated_cost", 0.001),
        ]
    )

    assert len(rows) == 1
    assert rows[0].method_id == "vector_rag"
    assert rows[0].latency_ms == 25.0
    assert rows[0].total_tokens == 10.0


def test_compare_experiment_loads_rows_from_store() -> None:
    table = compare_experiment(experiment_id="exp_1", store=FakeCompareStore())

    assert table[0].method_id == "vector_rag"


def _row(method_id: str, agent_mode: str, question_type: str, metric_name: str, value: float):
    return {
        "method_id": method_id,
        "agent_mode": agent_mode,
        "question_type": question_type,
        "metric_name": metric_name,
        "metric_value": value,
    }


class FakeCompareStore:
    def load_compare_rows(self, *, experiment_id: str):
        assert experiment_id == "exp_1"
        return [_row("vector_rag", "multi_agent", "span_extraction", "exact_match", 1.0)]
