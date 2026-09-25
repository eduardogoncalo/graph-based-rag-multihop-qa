from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.embeddings import DeterministicEmbeddingProvider
from benchmark.methods.vector_rag.prompt_builder import build_retrieval_prompt
from benchmark.methods.vector_rag.retriever import retrieve
from benchmark.storage.pgvector_store import VECTOR_RAG_METHOD_ID


def test_vector_retrieve_smoke_returns_retrieval_result() -> None:
    store = FakeSearchStore()
    provider = DeterministicEmbeddingProvider(dimension=4)

    result = retrieve(
        query="governing law",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        embedding_provider=provider,
        store=store,
        top_k=2,
    )

    assert result.method_id == VECTOR_RAG_METHOD_ID
    assert result.items[0].source_chunk_id == "chunk_1"
    assert store.last_top_k == 2
    assert len(store.last_query_embedding) == 4


def test_build_retrieval_prompt_includes_chunk_citations() -> None:
    item = RetrievedItem(
        item_id="vector_rag:chunk_1",
        text="Governing law is Delaware.",
        source_chunk_id="chunk_1",
    )

    prompt = build_retrieval_prompt("What is the governing law?", [item])

    assert "What is the governing law?" in prompt
    assert "chunk_id=chunk_1" in prompt
    assert "Governing law is Delaware." in prompt


class FakeSearchStore:
    def __init__(self) -> None:
        self.last_top_k = 0
        self.last_query_embedding: list[float] = []

    def search(
        self,
        *,
        dataset_id: str,
        dataset_version: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> RetrievalResult:
        self.last_top_k = top_k
        self.last_query_embedding = query_embedding
        return RetrievalResult(
            method_id=VECTOR_RAG_METHOD_ID,
            query=query,
            latency_ms=1.0,
            items=[
                RetrievedItem(
                    item_id="vector_rag:chunk_1",
                    text=f"{dataset_id}_{dataset_version}: Delaware law",
                    source_document_id="doc_1",
                    source_chunk_id="chunk_1",
                    score=0.9,
                )
            ],
        )
