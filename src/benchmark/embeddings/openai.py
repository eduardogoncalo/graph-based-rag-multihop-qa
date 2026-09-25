from __future__ import annotations

from typing import Any


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        dimension: int,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("OPENAI_EMBEDDING_MODEL is required for OpenAI embeddings")
        if dimension <= 0:
            raise ValueError("OPENAI_EMBEDDING_DIMENSIONS must be a positive integer")
        self.model_id = model
        self.dimension = dimension
        self.client = client or _openai_client(api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model_id,
            input=texts,
            dimensions=self.dimension,
            encoding_format="float",
        )
        return [list(item.embedding) for item in response.data]


def _openai_client(api_key: str | None) -> Any:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to use MODEL_PROVIDER=openai") from exc
    return OpenAI(api_key=api_key)
