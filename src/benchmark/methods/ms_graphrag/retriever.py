from __future__ import annotations

import time

from benchmark.core.schemas import RetrievalResult
from benchmark.methods.ms_graphrag.adapter import MicrosoftGraphRAGAdapter
from benchmark.methods.ms_graphrag.output_parser import parse_query_output
from benchmark.storage.experiment_store import ExperimentStore


def retrieve(
    *,
    adapter: MicrosoftGraphRAGAdapter,
    query: str,
    top_k: int = 5,
    query_method: str = "local",
    experiment_store: ExperimentStore | None = None,
    run_id: str | None = None,
) -> RetrievalResult:
    started = time.perf_counter()
    # Retrieval-only: context building costs embeddings, never a chat call.
    # The fixed benchmark reader (READER_GROUNDING; v1 by default) does the generation.
    command_result = adapter.query_context(query=query, query_method=query_method)
    result = parse_query_output(
        query=query,
        stdout=command_result.stdout,
        query_method=query_method,
        latency_ms=(time.perf_counter() - started) * 1000,
        top_k=top_k,
    )
    if experiment_store is not None:
        experiment_store.persist_retrieval_result(
            result=result,
            dataset_id=adapter.workspace.dataset_id,
            dataset_version=adapter.workspace.dataset_version,
            top_k=top_k,
            run_id=run_id,
        )
    return result
