from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text."""
