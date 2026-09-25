from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.core.schemas import Chunk
from benchmark.embeddings import EmbeddingProvider
from benchmark.storage.pgvector_store import VECTOR_RAG_METHOD_ID, PgVectorStore, VectorRecord


@dataclass(frozen=True)
class VectorIndexSummary:
    dataset_id: str
    dataset_version: str
    method_id: str
    chunk_count: int
    embedding_model: str
    embedding_dim: int


def load_canonical_chunks(canonical_path: str | Path) -> list[Chunk]:
    path = Path(canonical_path) / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical chunks not found at {path}. Run benchmark ingest before indexing."
        )

    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks


def index_chunks(
    *,
    chunks: list[Chunk],
    embedding_provider: EmbeddingProvider,
    store: PgVectorStore,
    dataset_id: str,
    dataset_version: str,
    method_id: str = VECTOR_RAG_METHOD_ID,
    batch_size: int = 128,
) -> VectorIndexSummary:
    if method_id != VECTOR_RAG_METHOD_ID:
        raise ValueError("Vector indexer only supports method_id='vector_rag'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    texts = [chunk.text for chunk in chunks]
    embeddings = []
    for start in range(0, len(texts), batch_size):
        embeddings.extend(embedding_provider.embed_texts(texts[start : start + batch_size]))
    if len(embeddings) != len(chunks):
        raise ValueError("embedding provider returned a different number of vectors than chunks")

    records = [
        VectorRecord(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            method_id=method_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            embedding_model=embedding_provider.model_id,
            embedding_dim=embedding_provider.dimension,
            text=chunk.text,
            metadata={**chunk.metadata, "method_id": method_id},
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    store.ensure_schema()
    store.upsert_embeddings(records)
    store.ensure_index()
    return VectorIndexSummary(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        method_id=method_id,
        chunk_count=len(records),
        embedding_model=embedding_provider.model_id,
        embedding_dim=embedding_provider.dimension,
    )
