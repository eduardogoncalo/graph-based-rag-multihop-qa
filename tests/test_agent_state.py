from benchmark.agents.schemas import (
    AgentGraphState,
    AgentMessage,
    EvidenceAudit,
    FinalAnswer,
    PlannerOutput,
)


def test_agent_graph_state_accepts_phase_four_fields() -> None:
    planner_output = PlannerOutput(query="governing law", method_id="vector_rag")
    audit = EvidenceAudit(supported=True, cited_chunk_ids=["chunk_1"])
    final_answer = FinalAnswer(answer_text="Delaware.", citations=["chunk_1"], evidence_audit=audit)
    message = AgentMessage(agent_name="query_planner", content="planned")

    state = AgentGraphState(
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        method_id="vector_rag",
        experiment_id="exp_1",
        run_id="run_1",
        agent_mode="multi_agent",
        question_id="q_1",
        question="What is the governing law?",
        planner_output=planner_output,
        evidence_audit=audit,
        final_answer=final_answer,
        agent_messages=[message],
    )

    assert state["planner_output"].method_id == "vector_rag"
    assert state["final_answer"].citations == ["chunk_1"]
    assert state["agent_messages"][0].agent_name == "query_planner"
