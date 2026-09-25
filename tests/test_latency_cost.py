from benchmark.evaluation.latency_cost import aggregate_latency_cost


def test_latency_cost_aggregates_latency_tokens_and_cost() -> None:
    summary = aggregate_latency_cost(
        retrieval_latency=10.0,
        generation_latency=15.0,
        prompt_tokens=7,
        completion_tokens=3,
        estimated_cost=0.002,
    )

    assert summary.total_latency == 25.0
    assert summary.total_tokens == 10
    assert summary.estimated_cost == 0.002


def test_latency_cost_uses_explicit_total_tokens() -> None:
    summary = aggregate_latency_cost(prompt_tokens=7, completion_tokens=3, total_tokens=20)

    assert summary.total_tokens == 20
