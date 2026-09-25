from __future__ import annotations

import time
from typing import Any

from benchmark.agents.schemas import AgentGraphState

# USD per 1M tokens (input, output) — OpenAI list prices, checked 2026-07-02.
_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class OpenAIAgentLLM:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        client: Any | None = None,
        temperature: float | None = 0.0,
    ) -> None:
        if not model:
            raise ValueError("OPENAI_CHAT_MODEL is required for OpenAI chat generation")
        self.model_id = model
        self.temperature = temperature
        self.client = client or _openai_client(api_key)
        self.usage_log: list[dict[str, Any]] = []

    def generate(self, *, agent_name: str, prompt: str, state: AgentGraphState) -> str:
        request: dict[str, Any] = {
            "model": self.model_id,
            "input": _agent_prompt(agent_name=agent_name, prompt=prompt, state=state),
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        started = time.perf_counter()
        response = self.client.responses.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.usage_log.append(_usage_record(response, agent_name=agent_name, model=self.model_id, latency_ms=latency_ms))
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return str(output_text)
        return _extract_output_text(response)

    def drain_usage(self) -> list[dict[str, Any]]:
        drained = self.usage_log
        self.usage_log = []
        return drained


def drain_llm_usage(llm: Any) -> list[dict[str, Any]]:
    """Drain per-call usage records from any AgentLLM that tracks them."""
    drain = getattr(llm, "drain_usage", None)
    if callable(drain):
        return drain()
    return []


def estimate_cost_usd(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    pricing = _PRICING_USD_PER_1M.get(model)
    if pricing is None or prompt_tokens is None or completion_tokens is None:
        return None
    input_price, output_price = pricing
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


def _usage_record(response: Any, *, agent_name: str, model: str, latency_ms: float) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    prompt_tokens = _usage_field(usage, "input_tokens", "prompt_tokens")
    completion_tokens = _usage_field(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_field(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    # `model` é o alias que pedimos, por exemplo "gpt-4o-mini". A API devolve em
    # `response.model` o snapshot que serviu de facto, por exemplo
    # "gpt-4o-mini-2024-07-18". Sem os dois campos não é possível saber, mais
    # tarde, se duas execuções correram no mesmo modelo. As células geradas até
    # 2026-08-08 têm apenas o alias, e essa limitação não é reparável a posteriori.
    served_model = getattr(response, "model", None)
    if served_model is None and isinstance(response, dict):
        served_model = response.get("model")
    return {
        "agent_name": agent_name,
        "model": model,
        "requested_model": model,
        "served_model": served_model,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimate_cost_usd(model, prompt_tokens, completion_tokens),
    }


def _usage_field(usage: Any, *names: str) -> int | None:
    if usage is None:
        return None
    for name in names:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        if value is not None:
            return int(value)
    return None


def _agent_prompt(*, agent_name: str, prompt: str, state: AgentGraphState) -> str:
    # method_id is deliberately NOT included: the generator must not condition
    # its behaviour on which retrieval method produced the context.
    return (
        f"Agent: {agent_name}\n"
        f"Dataset: {state.get('dataset_id')} {state.get('dataset_version')}\n"
        f"Question: {state.get('question')}\n\n"
        f"{prompt}"
    )


def _extract_output_text(response: Any) -> str:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not output:
        return ""

    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not content:
            continue
        for content_item in content:
            text = getattr(content_item, "text", None)
            if text is None and isinstance(content_item, dict):
                text = content_item.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _openai_client(api_key: str | None) -> Any:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to use MODEL_PROVIDER=openai") from exc
    return OpenAI(api_key=api_key)
