from __future__ import annotations

from benchmark.core.schemas import RetrievalResult
from benchmark.embeddings import EmbeddingProvider
from benchmark.storage.experiment_store import ExperimentStore
from benchmark.storage.pgvector_store import PgVectorStore


def retrieve(
    *,
    query: str,
    dataset_id: str,
    dataset_version: str,
    embedding_provider: EmbeddingProvider,
    store: PgVectorStore,
    top_k: int = 5,
    experiment_store: ExperimentStore | None = None,
    run_id: str | None = None,
) -> RetrievalResult:
    query_embedding = embedding_provider.embed_texts([query])[0]
    result = store.search(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        query=query,
        query_embedding=query_embedding,
        top_k=top_k,
    )
    if experiment_store is not None:
        experiment_store.persist_retrieval_result(
            result=result,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            top_k=top_k,
            run_id=run_id,
        )
    return result
