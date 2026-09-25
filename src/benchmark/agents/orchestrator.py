from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from benchmark.agents.llm import AgentLLM, FakeAgentLLM
from benchmark.agents.nodes import AgentNodeRunner, RetrieverFn
from benchmark.agents.schemas import AgentGraphState, FinalAnswer
from benchmark.storage.experiment_store import ExperimentStore


@dataclass(frozen=True)
class AgentWorkflow:
    graph: object
    node_names: list[str]
    critic_enabled: bool = False

    def invoke(self, state: AgentGraphState) -> AgentGraphState:
        return self.graph.invoke(state)

    def run(self, state: AgentGraphState) -> FinalAnswer:
        result = self.invoke(state)
        return result["final_answer"]


def build_agent_graph(
    *,
    retriever: RetrieverFn,
    llm: AgentLLM | None = None,
    experiment_store: ExperimentStore | None = None,
    top_k: int = 5,
    enable_critic: bool = False,
) -> AgentWorkflow:
    if enable_critic:
        raise ValueError("Critic agent is optional and disabled in the Phase 4 default workflow")

    runner = AgentNodeRunner(
        llm=llm or FakeAgentLLM(),
        retriever=retriever,
        experiment_store=experiment_store,
        top_k=top_k,
    )
    node_names = [
        "query_planner",
        "retriever",
        "domain_reasoner",
        "evidence_auditor",
        "final_answer",
    ]
    builder = StateGraph(AgentGraphState)
    builder.add_node("query_planner", runner.query_planner)
    builder.add_node("retriever", runner.retriever_node)
    builder.add_node("domain_reasoner", runner.domain_reasoner)
    builder.add_node("evidence_auditor", runner.evidence_auditor)
    builder.add_node("final_answer", runner.final_answer)
    builder.add_edge(START, "query_planner")
    builder.add_edge("query_planner", "retriever")
    builder.add_edge("retriever", "domain_reasoner")
    builder.add_edge("domain_reasoner", "evidence_auditor")
    builder.add_edge("evidence_auditor", "final_answer")
    builder.add_edge("final_answer", END)
    return AgentWorkflow(
        graph=builder.compile(),
        node_names=node_names,
        critic_enabled=False,
    )
