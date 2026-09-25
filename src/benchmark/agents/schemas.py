from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from benchmark.core.schemas import RetrievalResult

AgentMode = Literal["single_agent", "multi_agent"]
RetrievalMethod = Literal[
    "vector_rag",
    "lightrag_neo4j",
    "cognee",
    "ms_graphrag",
    "hipporag2",
    "zero_shot_no_context",
    "single_document_context",
]


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentMessage(AgentModel):
    agent_name: str
    role: str = "assistant"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlannerOutput(AgentModel):
    query: str
    method_id: RetrievalMethod
    constraints: list[str] = Field(default_factory=list)


class EvidenceAudit(AgentModel):
    supported: bool
    cited_chunk_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class FinalAnswer(AgentModel):
    answer_text: str
    citations: list[str] = Field(default_factory=list)
    evidence_audit: EvidenceAudit | None = None


class AgentGraphState(TypedDict, total=False):
    dataset_id: str
    dataset_version: str
    method_id: RetrievalMethod
    experiment_id: str
    run_id: str
    agent_mode: AgentMode
    question_id: str | None
    question: str
    planner_output: PlannerOutput
    retrieval_result: RetrievalResult
    draft_answer: str
    evidence_audit: EvidenceAudit
    final_answer: FinalAnswer
    agent_messages: list[AgentMessage]
