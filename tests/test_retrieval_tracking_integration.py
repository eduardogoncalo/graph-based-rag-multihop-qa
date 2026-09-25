from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.embeddings import DeterministicEmbeddingProvider
from benchmark.methods.vector_rag.retriever import retrieve
from benchmark.storage.pgvector_store import VECTOR_TABLE


def test_vector_retrieval_can_persist_shared_tracking() -> None:
    experiment_store = FakeExperimentStore()

    result = retrieve(
        query="law",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        embedding_provider=DeterministicEmbeddingProvider(dimension=4),
        store=FakeVectorStore(),
        top_k=1,
        experiment_store=experiment_store,
        run_id="run_1",
    )

    assert result.method_id == "vector_rag"
    assert experiment_store.calls[0]["result"].method_id == "vector_rag"
    assert experiment_store.calls[0]["run_id"] == "run_1"


def test_vector_native_storage_uses_method_scoped_table() -> None:
    assert VECTOR_TABLE == "vector_rag_chunk_embeddings"


class FakeVectorStore:
    def search(
        self,
        *,
        dataset_id: str,
        dataset_version: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> RetrievalResult:
        return RetrievalResult(
            method_id="vector_rag",
            query=query,
            latency_ms=1.0,
            items=[
                RetrievedItem(
                    item_id="vector_rag:chunk_1",
                    text=f"{dataset_id}_{dataset_version}: Delaware law",
                    source_chunk_id="chunk_1",
                    score=0.9,
                    metadata={"query_embedding_dim": len(query_embedding), "top_k": top_k},
                )
            ],
        )


class FakeExperimentStore:
    def __init__(self) -> None:
        self.calls = []

    def persist_retrieval_result(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return "retrieval_1"
