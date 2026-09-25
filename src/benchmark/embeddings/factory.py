from __future__ import annotations

from typing import Any

from benchmark.core.settings import Settings, load_settings
from benchmark.embeddings.base import EmbeddingProvider
from benchmark.embeddings.fake import DeterministicEmbeddingProvider
from benchmark.embeddings.openai import OpenAIEmbeddingProvider

FAKE_PROVIDER_NAMES = {"fake", "deterministic"}
OPENAI_PROVIDER_NAME = "openai"


def create_embedding_provider(
    *,
    settings: Settings | None = None,
    provider_name: str | None = None,
    method_config: Any | None = None,
    client: Any | None = None,
) -> EmbeddingProvider:
    resolved_settings = settings or load_settings()
    resolved_name = (provider_name or resolved_settings.model_provider).strip().lower()

    if resolved_name in FAKE_PROVIDER_NAMES:
        dimension = int(getattr(method_config, "embedding_dimension", 16))
        return DeterministicEmbeddingProvider(dimension=dimension)

    if resolved_name == OPENAI_PROVIDER_NAME:
        if resolved_settings.openai_embedding_dimensions is None:
            raise ValueError(
                "OPENAI_EMBEDDING_DIMENSIONS is required when MODEL_PROVIDER=openai"
            )
        return OpenAIEmbeddingProvider(
            api_key=resolved_settings.openai_api_key,
            model=_required_setting(
                resolved_settings.openai_embedding_model,
                "OPENAI_EMBEDDING_MODEL",
            ),
            dimension=resolved_settings.openai_embedding_dimensions,
            client=client,
        )

    raise ValueError("Supported model providers: fake, deterministic, openai")


def _required_setting(value: str | None, env_name: str) -> str:
    if not value:
        raise ValueError(f"{env_name} is required when MODEL_PROVIDER=openai")
    return value
