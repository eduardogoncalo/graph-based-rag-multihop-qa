import pytest

from benchmark.agents import build_agent_graph
from benchmark.core.schemas import RetrievalResult, RetrievedItem


def test_langgraph_compiles_with_default_nodes() -> None:
    workflow = build_agent_graph(retriever=fake_retriever)

    assert workflow.node_names == [
        "query_planner",
        "retriever",
        "domain_reasoner",
        "evidence_auditor",
        "final_answer",
    ]
    assert workflow.critic_enabled is False


def test_langgraph_executes_nodes_in_order() -> None:
    workflow = build_agent_graph(retriever=fake_retriever)

    result = workflow.invoke(base_state(method_id="vector_rag"))

    assert [message.agent_name for message in result["agent_messages"]] == [
        "query_planner",
        "retriever",
        "domain_reasoner",
        "evidence_auditor",
        "final_answer",
    ]
    assert result["final_answer"].citations == ["chunk_1"]


def test_critic_disabled_by_default_and_rejected_when_enabled() -> None:
    workflow = build_agent_graph(retriever=fake_retriever)

    assert "critic" not in workflow.node_names
    with pytest.raises(ValueError, match="Critic"):
        build_agent_graph(retriever=fake_retriever, enable_critic=True)


def fake_retriever(state, query: str) -> RetrievalResult:
    return RetrievalResult(
        method_id=state["method_id"],
        query=query,
        latency_ms=1.0,
        items=[
            RetrievedItem(
                item_id=f"{state['method_id']}:chunk_1",
                text="Governing law is Delaware.",
                source_document_id="doc_1",
                source_chunk_id="chunk_1",
                score=0.9,
            )
        ],
    )


def base_state(method_id: str):
    return {
        "dataset_id": "musique_smoke_20",
        "dataset_version": "v1",
        "method_id": method_id,
        "experiment_id": "exp_1",
        "run_id": "run_1",
        "agent_mode": "multi_agent",
        "question_id": "q_1",
        "question": "What is the governing law?",
        "agent_messages": [],
    }
