from __future__ import annotations

from benchmark.agents import build_agent_graph
from benchmark.core.schemas import RetrievalResult


def test_agent_workflow_accepts_cognee_method_id() -> None:
    workflow = build_agent_graph(retriever=matching_retriever)

    state = workflow.invoke(
        {
            "dataset_id": "musique_smoke_20",
            "dataset_version": "v1",
            "method_id": "cognee",
            "experiment_id": "exp_1",
            "run_id": "run_1",
            "agent_mode": "multi_agent",
            "question_id": "q_1",
            "question": "What is the governing law?",
            "agent_messages": [],
        }
    )

    assert state["retrieval_result"].method_id == "cognee"
    assert state["final_answer"].answer_text


def matching_retriever(state, query: str) -> RetrievalResult:
    return RetrievalResult(method_id=state["method_id"], query=query, latency_ms=1.0)
