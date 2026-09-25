import pytest

from benchmark.agents import build_agent_graph
from benchmark.core.schemas import RetrievalResult


def test_retriever_node_rejects_unsupported_method() -> None:
    workflow = build_agent_graph(retriever=matching_retriever)

    with pytest.raises(ValueError, match="Retriever supports only vector_rag"):
        workflow.invoke(
            {
                "dataset_id": "musique_smoke_20",
                "dataset_version": "v1",
                "method_id": "definitely_unsupported_method",
                "experiment_id": "exp_1",
                "run_id": "run_1",
                "agent_mode": "multi_agent",
                "question_id": "q_1",
                "question": "What is the governing law?",
                "agent_messages": [],
            }
        )


def test_retriever_node_rejects_cross_method_results() -> None:
    workflow = build_agent_graph(retriever=wrong_method_retriever)

    with pytest.raises(ValueError, match="different method"):
        workflow.invoke(
            {
                "dataset_id": "musique_smoke_20",
                "dataset_version": "v1",
                "method_id": "lightrag_neo4j",
                "experiment_id": "exp_1",
                "run_id": "run_1",
                "agent_mode": "multi_agent",
                "question_id": "q_1",
                "question": "What is the governing law?",
                "agent_messages": [],
            }
        )


def matching_retriever(state, query: str) -> RetrievalResult:
    return RetrievalResult(method_id=state["method_id"], query=query, latency_ms=1.0)


def wrong_method_retriever(state, query: str) -> RetrievalResult:
    return RetrievalResult(method_id="vector_rag", query=query, latency_ms=1.0)
