from __future__ import annotations

import inspect
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

INSTRUMENTATION_VERSION = "lightrag_usage_v2"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
CHAT_INPUT_PRICES_PER_1M = {
    "gpt-4o-mini": 0.15,
    "gpt-4.1-mini": 0.40,
    "gpt-5.4-mini": 0.75,
}
CHAT_OUTPUT_PRICES_PER_1M = {
    "gpt-4o-mini": 0.60,
    "gpt-4.1-mini": 1.60,
    "gpt-5.4-mini": 4.50,
}


@dataclass(frozen=True)
class LightRAGUsageSummary:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_cost: float | None
    generation_latency_ms: float | None
    usage_source: str
    metadata: dict[str, Any]


class LightRAGUsageTracker:
    def __init__(self, *, model: str | None = None) -> None:
        self.model = model or current_chat_model()
        self._calls: list[LightRAGUsageSummary] = []

    def wrap(self, llm_model_func: Callable[..., Any] | None) -> Callable[..., Any] | None:
        if llm_model_func is None:
            return None

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            prompt_text = extract_prompt_text(args=args, kwargs=kwargs)
            started = time.perf_counter()
            result = llm_model_func(*args, **kwargs)
            if inspect.isawaitable(result):
                return self._record_awaitable(result, prompt_text, started)
            self._record_call(prompt_text, result, started)
            return result

        return wrapped

    async def _record_awaitable(
        self,
        awaitable: Awaitable[Any],
        prompt_text: str,
        started: float,
    ) -> Any:
        result = await awaitable
        self._record_call(prompt_text, result, started)
        return result

    def _record_call(self, prompt_text: str, result: Any, started: float) -> None:
        latency_ms = (time.perf_counter() - started) * 1000
        self._calls.append(
            usage_from_payload(
                prompt_text=prompt_text,
                response_payload=result,
                generation_latency_ms=latency_ms,
                model=self.model,
            )
        )

    def summary(self) -> LightRAGUsageSummary:
        if not self._calls:
            return missing_usage_summary(
                model=self.model,
                reason="llm_model_func_was_not_observed",
            )
        prompt_tokens = sum(value.prompt_tokens or 0 for value in self._calls)
        completion_tokens = sum(value.completion_tokens or 0 for value in self._calls)
        total_tokens = sum(
            value.total_tokens
            if value.total_tokens is not None
            else (value.prompt_tokens or 0) + (value.completion_tokens or 0)
            for value in self._calls
        )
        cost = sum(value.estimated_cost or 0.0 for value in self._calls)
        latency = sum(value.generation_latency_ms or 0.0 for value in self._calls)
        usage_sources = {value.usage_source for value in self._calls}
        usage_source = "direct" if usage_sources == {"direct"} else "estimated"
        return LightRAGUsageSummary(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            generation_latency_ms=latency,
            usage_source=usage_source,
            metadata={
                "usage_source": usage_source,
                "instrumentation_version": INSTRUMENTATION_VERSION,
                "model": self.model,
                "llm_call_count": len(self._calls),
                "token_estimator": "character_heuristic"
                if usage_source == "estimated"
                else None,
            },
        )


def current_chat_model() -> str:
    return os.environ.get("OPENAI_CHAT_MODEL") or DEFAULT_CHAT_MODEL


def usage_from_payload(
    *,
    prompt_text: str,
    response_payload: Any,
    generation_latency_ms: float | None,
    model: str | None = None,
) -> LightRAGUsageSummary:
    resolved_model = model or current_chat_model()
    direct = extract_direct_usage(response_payload)
    if direct is not None:
        prompt_tokens, completion_tokens, total_tokens = direct
        return LightRAGUsageSummary(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimate_chat_cost(prompt_tokens, completion_tokens, resolved_model),
            generation_latency_ms=generation_latency_ms,
            usage_source="direct",
            metadata={
                "usage_source": "direct",
                "instrumentation_version": INSTRUMENTATION_VERSION,
                "model": resolved_model,
            },
        )

    response_text = extract_response_text(response_payload)
    if prompt_text or response_text:
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens
        return LightRAGUsageSummary(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimate_chat_cost(prompt_tokens, completion_tokens, resolved_model),
            generation_latency_ms=generation_latency_ms,
            usage_source="estimated",
            metadata={
                "usage_source": "estimated",
                "instrumentation_version": INSTRUMENTATION_VERSION,
                "token_estimator": "character_heuristic",
                "model": resolved_model,
            },
        )

    return missing_usage_summary(model=resolved_model, reason="no_prompt_or_response_text")


def usage_from_query_result(
    *,
    query: str,
    raw_response: Any,
    tracked_usage: LightRAGUsageSummary,
    generation_latency_ms: float | None,
    model: str | None = None,
) -> LightRAGUsageSummary:
    if tracked_usage.usage_source in {"direct", "estimated"}:
        return tracked_usage
    if not extract_response_text(raw_response):
        return missing_usage_summary(
            model=model,
            reason="no_response_text_for_query_usage_estimate",
        )
    return usage_from_payload(
        prompt_text=query,
        response_payload=raw_response,
        generation_latency_ms=generation_latency_ms,
        model=model,
    )


def missing_usage_summary(*, model: str | None = None, reason: str) -> LightRAGUsageSummary:
    return LightRAGUsageSummary(
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        estimated_cost=None,
        generation_latency_ms=None,
        usage_source="missing",
        metadata={
            "usage_source": "missing",
            "instrumentation_version": INSTRUMENTATION_VERSION,
            "model": model or current_chat_model(),
            "reason": reason,
        },
    )


def summary_to_metadata(summary: LightRAGUsageSummary) -> dict[str, Any]:
    metadata = {
        key: value for key, value in summary.metadata.items() if value is not None
    }
    metadata.update(
        {
            "prompt_tokens": summary.prompt_tokens,
            "completion_tokens": summary.completion_tokens,
            "total_tokens": summary.total_tokens,
            "estimated_cost": summary.estimated_cost,
            "generation_latency_ms": summary.generation_latency_ms,
        }
    )
    return {key: value for key, value in metadata.items() if value is not None}


def extract_prompt_text(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in args:
        if isinstance(value, str):
            parts.append(value)
    for key in (
        "prompt",
        "query",
        "input",
        "user_prompt",
        "system_prompt",
        "history_messages",
        "keyword_extraction",
    ):
        value = kwargs.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return "\n".join(parts)


def extract_response_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("answer", "response", "result", "output", "output_text", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    value = getattr(payload, "output_text", None)
    if isinstance(value, str):
        return value
    value = getattr(payload, "content", None)
    if isinstance(value, str):
        return value
    return ""


def extract_direct_usage(payload: Any) -> tuple[int, int, int | None] | None:
    usage = getattr(payload, "usage", None)
    if usage is None and isinstance(payload, dict):
        usage = payload.get("usage") or payload.get("token_usage")
    if usage is None:
        return None
    prompt_tokens = usage_value(usage, "input_tokens", "prompt_tokens")
    completion_tokens = usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = usage_value(usage, "total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
        if isinstance(usage, dict) and usage.get(name) is not None:
            return int(usage[name])
    return None


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_chat_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    model: str | None = None,
) -> float:
    resolved_model = model or current_chat_model()
    input_price = CHAT_INPUT_PRICES_PER_1M.get(
        resolved_model,
        CHAT_INPUT_PRICES_PER_1M[DEFAULT_CHAT_MODEL],
    )
    output_price = CHAT_OUTPUT_PRICES_PER_1M.get(
        resolved_model,
        CHAT_OUTPUT_PRICES_PER_1M[DEFAULT_CHAT_MODEL],
    )
    return (prompt_tokens or 0) / 1_000_000 * input_price + (
        completion_tokens or 0
    ) / 1_000_000 * output_price
