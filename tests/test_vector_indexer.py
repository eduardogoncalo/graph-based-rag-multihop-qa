import json
from pathlib import Path

from benchmark.core.schemas import Chunk
from benchmark.embeddings import DeterministicEmbeddingProvider
from benchmark.methods.vector_rag.indexer import index_chunks, load_canonical_chunks
from benchmark.storage.pgvector_store import VECTOR_RAG_METHOD_ID


def test_load_canonical_chunks_reads_jsonl(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical" / "musique_smoke_20_v1"
    canonical_path.mkdir(parents=True)
    chunk = Chunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Delaware law",
    )
    (canonical_path / "chunks.jsonl").write_text(
        json.dumps(chunk.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    chunks = load_canonical_chunks(canonical_path)

    assert chunks == [chunk]


def test_index_chunks_smoke_uses_vector_rag_store() -> None:
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            text="Delaware law",
        )
    ]
    store = FakeVectorStore()
    provider = DeterministicEmbeddingProvider(dimension=4)

    summary = index_chunks(
        chunks=chunks,
        embedding_provider=provider,
        store=store,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    assert summary.method_id == VECTOR_RAG_METHOD_ID
    assert summary.chunk_count == 1
    assert store.schema_created is True
    assert store.index_created_after_upsert is True
    assert store.records[0].method_id == VECTOR_RAG_METHOD_ID
    assert store.records[0].metadata["method_id"] == VECTOR_RAG_METHOD_ID


class FakeVectorStore:
    def __init__(self) -> None:
        self.schema_created = False
        self.index_created_after_upsert = False
        self.records = []

    def ensure_schema(self) -> None:
        self.schema_created = True

    def upsert_embeddings(self, records: list[object]) -> int:
        self.records.extend(records)
        return len(records)

    def ensure_index(self) -> None:
        self.index_created_after_upsert = bool(self.records)
