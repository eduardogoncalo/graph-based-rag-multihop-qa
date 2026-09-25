from __future__ import annotations

from typing import Any

from benchmark.agents.llm import AgentLLM, FakeAgentLLM
from benchmark.agents.openai import OpenAIAgentLLM
from benchmark.core.settings import Settings, load_settings
from benchmark.embeddings.factory import FAKE_PROVIDER_NAMES, OPENAI_PROVIDER_NAME


def create_agent_llm(
    *,
    settings: Settings | None = None,
    provider_name: str | None = None,
    client: Any | None = None,
) -> AgentLLM:
    resolved_settings = settings or load_settings()
    resolved_name = (provider_name or resolved_settings.model_provider).strip().lower()

    if resolved_name in FAKE_PROVIDER_NAMES:
        return FakeAgentLLM()

    if resolved_name == OPENAI_PROVIDER_NAME:
        if not resolved_settings.openai_chat_model:
            raise ValueError("OPENAI_CHAT_MODEL is required when MODEL_PROVIDER=openai")
        return OpenAIAgentLLM(
            api_key=resolved_settings.openai_api_key,
            model=resolved_settings.openai_chat_model,
            client=client,
            temperature=resolved_settings.openai_chat_temperature,
        )

    raise ValueError("Supported model providers: fake, deterministic, openai")
