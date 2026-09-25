from __future__ import annotations

import hashlib
import math


class DeterministicEmbeddingProvider:
    model_id = "deterministic-test-embedding"

    def __init__(self, dimension: int = 16) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        buckets = [0.0 for _ in range(self.dimension)]
        tokens = text.lower().split()
        if not tokens:
            tokens = [""]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.dimension):
                value = digest[index % len(digest)] / 255.0
                buckets[index] += (value * 2.0) - 1.0

        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0:
            return buckets
        return [value / norm for value in buckets]
