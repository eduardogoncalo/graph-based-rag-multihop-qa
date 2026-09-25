from benchmark.agents.llm import FakeAgentLLM
from benchmark.agents.nodes import AgentNodeRunner
from benchmark.core.schemas import RetrievalResult, RetrievedItem


def test_planner_node_outputs_configured_method() -> None:
    runner = AgentNodeRunner(llm=FakeAgentLLM(), retriever=fake_retriever)

    update = runner.query_planner(base_state("vector_rag"))

    assert update["planner_output"].query == "What is the governing law?"
    assert update["planner_output"].method_id == "vector_rag"
    assert update["agent_messages"][0].agent_name == "query_planner"


def test_fake_multi_agent_run_with_vector_rag() -> None:
    state = run_graph_state("vector_rag")

    assert state["retrieval_result"].method_id == "vector_rag"
    assert state["final_answer"].answer_text.startswith("Final answer")
    assert state["final_answer"].citations == ["chunk_1"]


def test_fake_multi_agent_run_with_lightrag_neo4j() -> None:
    state = run_graph_state("lightrag_neo4j")

    assert state["retrieval_result"].method_id == "lightrag_neo4j"
    assert state["final_answer"].citations == ["chunk_1"]


def run_graph_state(method_id: str):
    from benchmark.agents import build_agent_graph

    return build_agent_graph(retriever=fake_retriever).invoke(base_state(method_id))


def fake_retriever(state, query: str) -> RetrievalResult:
    return RetrievalResult(
        method_id=state["method_id"],
        query=query,
        latency_ms=1.0,
        items=[
            RetrievedItem(
                item_id=f"{state['method_id']}:chunk_1",
                text="Governing law is Delaware.",
                source_chunk_id="chunk_1",
                score=0.8,
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
