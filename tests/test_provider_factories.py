from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.agents import FakeAgentLLM, OpenAIAgentLLM, create_agent_llm
from benchmark.core.settings import Settings
from benchmark.embeddings import (
    DeterministicEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)


def test_settings_parse_openai_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "chat-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "32")

    settings = Settings(_env_file=None)

    assert settings.model_provider == "openai"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_chat_model == "chat-model"
    assert settings.openai_embedding_model == "embedding-model"
    assert settings.openai_embedding_dimensions == 32


def test_embedding_factory_defaults_to_fake_provider() -> None:
    provider = create_embedding_provider(
        settings=_settings(model_provider="fake"),
        method_config=SimpleNamespace(embedding_dimension=7),
    )

    assert isinstance(provider, DeterministicEmbeddingProvider)
    assert provider.dimension == 7


def test_agent_factory_defaults_to_fake_provider() -> None:
    provider = create_agent_llm(settings=_settings(model_provider="fake"))

    assert isinstance(provider, FakeAgentLLM)


def test_embedding_factory_selects_openai_provider_with_fake_client() -> None:
    provider = create_embedding_provider(
        settings=_settings(
            model_provider="openai",
            openai_api_key="test-key",
            openai_embedding_model="embedding-model",
            openai_embedding_dimensions=12,
        ),
        client=FakeOpenAIClient(),
    )

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model_id == "embedding-model"
    assert provider.dimension == 12


def test_agent_factory_selects_openai_provider_with_fake_client() -> None:
    provider = create_agent_llm(
        settings=_settings(
            model_provider="openai",
            openai_api_key="test-key",
            openai_chat_model="chat-model",
        ),
        client=FakeOpenAIClient(),
    )

    assert isinstance(provider, OpenAIAgentLLM)
    assert provider.model_id == "chat-model"


def test_openai_embedding_factory_requires_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="OPENAI_EMBEDDING_DIMENSIONS"):
        create_embedding_provider(
            settings=_settings(
                model_provider="openai",
                openai_api_key="test-key",
                openai_embedding_model="embedding-model",
            ),
            client=FakeOpenAIClient(),
        )


def test_openai_agent_factory_requires_api_key_without_injected_client() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_agent_llm(
            settings=_settings(
                model_provider="openai",
                openai_chat_model="chat-model",
            )
        )


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Supported model providers"):
        create_agent_llm(settings=_settings(model_provider="unknown"))


class FakeOpenAIClient:
    pass


def _settings(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)
