import pytest

from benchmark.core.schemas import Chunk
from benchmark.embeddings import DeterministicEmbeddingProvider
from benchmark.methods.vector_rag.indexer import index_chunks


def test_vector_indexer_rejects_other_methods() -> None:
    chunk = Chunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Delaware law",
    )

    with pytest.raises(ValueError, match="vector_rag"):
        index_chunks(
            chunks=[chunk],
            embedding_provider=DeterministicEmbeddingProvider(dimension=4),
            store=FakeVectorStore(),
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            method_id="lightrag_neo4j",
        )


class FakeVectorStore:
    def ensure_schema(self) -> None:
        raise AssertionError("should fail before touching storage")

    def upsert_embeddings(self, records: list[object]) -> int:
        raise AssertionError("should fail before touching storage")
