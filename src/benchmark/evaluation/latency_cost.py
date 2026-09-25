from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyCostSummary:
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


def aggregate_latency_cost(
    *,
    retrieval_latency: float | None = None,
    generation_latency: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    estimated_cost: float | None = None,
) -> LatencyCostSummary:
    resolved_prompt_tokens = prompt_tokens or 0
    resolved_completion_tokens = completion_tokens or 0
    resolved_total_tokens = total_tokens
    if resolved_total_tokens is None:
        resolved_total_tokens = resolved_prompt_tokens + resolved_completion_tokens

    resolved_retrieval_latency = retrieval_latency or 0.0
    resolved_generation_latency = generation_latency or 0.0
    return LatencyCostSummary(
        retrieval_latency=resolved_retrieval_latency,
        generation_latency=resolved_generation_latency,
        total_latency=resolved_retrieval_latency + resolved_generation_latency,
        prompt_tokens=resolved_prompt_tokens,
        completion_tokens=resolved_completion_tokens,
        total_tokens=resolved_total_tokens,
        estimated_cost=estimated_cost or 0.0,
    )
