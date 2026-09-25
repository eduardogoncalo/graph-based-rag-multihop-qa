from __future__ import annotations

from benchmark.core.schemas import RetrievalResult
from benchmark.methods.cognee.adapter import CogneeAdapter
from benchmark.methods.cognee.trace_writer import CogneeTraceWriter
from benchmark.storage.experiment_store import ExperimentStore


def retrieve(
    *,
    adapter: CogneeAdapter,
    query: str,
    top_k: int = 5,
    experiment_store: ExperimentStore | None = None,
    run_id: str | None = None,
    question_id: str | None = None,
    trace_writer: CogneeTraceWriter | None = None,
) -> RetrievalResult:
    result = adapter.retrieve(query=query, top_k=top_k)
    if trace_writer is not None and run_id is not None and question_id is not None:
        trace_writer.write_result(run_id=run_id, question_id=question_id, result=result)
    if experiment_store is not None:
        experiment_store.persist_retrieval_result(
            result=result,
            dataset_id=adapter.workspace.dataset_id,
            dataset_version=adapter.workspace.dataset_version,
            top_k=top_k,
            run_id=run_id,
        )
    return result
