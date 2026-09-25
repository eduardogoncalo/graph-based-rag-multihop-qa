from __future__ import annotations

import time

from benchmark.core.schemas import RetrievalResult
from benchmark.methods.ms_graphrag_neo4j.adapter import GraphRAGNeo4jAdapter
from benchmark.methods.ms_graphrag_neo4j.output_parser import parse_query_output
from benchmark.storage.experiment_store import ExperimentStore


def retrieve(
    *,
    adapter: GraphRAGNeo4jAdapter,
    query: str,
    top_k: int = 5,
    experiment_store: ExperimentStore | None = None,
    run_id: str | None = None,
) -> RetrievalResult:
    started = time.perf_counter()
    raw_result = adapter.query(query=query, top_k=top_k)
    result = parse_query_output(
        query=query,
        raw_response=raw_result.raw_response,
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
