from __future__ import annotations

import json
import os
from collections.abc import Callable

from benchmark.agents.llm import AgentLLM
from benchmark.agents.openai import drain_llm_usage
from benchmark.agents.schemas import (
    AgentGraphState,
    AgentMessage,
    EvidenceAudit,
    FinalAnswer,
    PlannerOutput,
)
from benchmark.core.schemas import RetrievalResult
from benchmark.methods.baselines import ORACLE_METHOD_ID, ZERO_SHOT_METHOD_ID
from benchmark.methods.cognee.config_builder import COGNEE_METHOD_ID
from benchmark.methods.hipporag2.config_builder import HIPPORAG2_METHOD_ID
from benchmark.methods.lightrag_neo4j.config_builder import LIGHTRAG_NEO4J_METHOD_ID
from benchmark.methods.ms_graphrag.config_builder import MS_GRAPHRAG_METHOD_ID
from benchmark.storage.experiment_store import ExperimentStore
from benchmark.storage.pgvector_store import VECTOR_RAG_METHOD_ID

RetrieverFn = Callable[[AgentGraphState, str], RetrievalResult]


class AgentNodeRunner:
    def __init__(
        self,
        *,
        llm: AgentLLM,
        retriever: RetrieverFn,
        experiment_store: ExperimentStore | None = None,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.experiment_store = experiment_store
        self.top_k = top_k

    def query_planner(self, state: AgentGraphState) -> AgentGraphState:
        method_id = state["method_id"]
        _ensure_supported_method(method_id)
        drain_llm_usage(self.llm)  # discard leftovers from a previous question
        output = PlannerOutput(
            query=state["question"],
            method_id=method_id,
            constraints=["use_configured_retrieval_method_only"],
        )
        return self._with_message(
            state,
            agent_name="query_planner",
            output={"planner_output": output.model_dump(mode="json")},
            updates={"planner_output": output},
        )

    def retriever_node(self, state: AgentGraphState) -> AgentGraphState:
        method_id = state["method_id"]
        _ensure_supported_method(method_id)
        planner_output = state["planner_output"]
        if planner_output.method_id != method_id:
            raise ValueError("Planner output method does not match run method_id")

        result = self.retriever(state, planner_output.query)
        if result.method_id != method_id:
            raise ValueError("Retriever returned results for a different method")
        if self.experiment_store is not None:
            self.experiment_store.persist_retrieval_result(
                result=result,
                dataset_id=state["dataset_id"],
                dataset_version=state["dataset_version"],
                top_k=self.top_k,
                run_id=state.get("run_id"),
            )
        return self._with_message(
            state,
            agent_name="retriever",
            output={
                "method_id": result.method_id,
                "items": [item.model_dump(mode="json") for item in result.items],
            },
            updates={"retrieval_result": result},
        )

    def domain_reasoner(self, state: AgentGraphState) -> AgentGraphState:
        prompt = _retrieval_context_prompt(state)
        draft = self.llm.generate(agent_name="domain_reasoner", prompt=prompt, state=state)
        return self._with_message(
            state,
            agent_name="domain_reasoner",
            output={"draft_answer": draft},
            updates={"draft_answer": draft},
        )

    def evidence_auditor(self, state: AgentGraphState) -> AgentGraphState:
        retrieval_result = state.get("retrieval_result")
        cited_chunk_ids = []
        if retrieval_result is not None:
            cited_chunk_ids = [
                item.source_chunk_id or item.item_id
                for item in retrieval_result.items
                if item.source_chunk_id or item.item_id
            ]
        audit = EvidenceAudit(
            supported=bool(cited_chunk_ids),
            cited_chunk_ids=cited_chunk_ids,
            notes="Fake audit: answer is supported when retrieved chunks exist.",
        )
        return self._with_message(
            state,
            agent_name="evidence_auditor",
            output={"evidence_audit": audit.model_dump(mode="json")},
            updates={"evidence_audit": audit},
        )

    def final_answer(self, state: AgentGraphState) -> AgentGraphState:
        answer_text = self.llm.generate(
            agent_name="final_answer",
            prompt=state.get("draft_answer", ""),
            state=state,
        )
        audit = state["evidence_audit"]
        final = FinalAnswer(
            answer_text=answer_text,
            citations=audit.cited_chunk_ids,
            evidence_audit=audit,
        )
        usage_calls = drain_llm_usage(self.llm)
        totals = _sum_usage(usage_calls)
        if self.experiment_store is not None:
            metadata: dict[str, object] = {"evidence_audit": audit.model_dump(mode="json")}
            if usage_calls:
                metadata["usage_calls"] = usage_calls
            self.experiment_store.persist_answer(
                run_id=state["run_id"],
                question_id=state.get("question_id"),
                method_id=state["method_id"],
                agent_mode=state["agent_mode"],
                answer_text=final.answer_text,
                citations=final.citations,
                latency_ms=totals["latency_ms"],
                prompt_tokens=totals["prompt_tokens"],
                completion_tokens=totals["completion_tokens"],
                total_tokens=totals["total_tokens"],
                estimated_cost=totals["estimated_cost"],
                metadata=metadata,
            )
        return self._with_message(
            state,
            agent_name="final_answer",
            output={"final_answer": final.model_dump(mode="json")},
            updates={"final_answer": final},
        )

    def _with_message(
        self,
        state: AgentGraphState,
        *,
        agent_name: str,
        output: dict[str, object],
        updates: AgentGraphState,
    ) -> AgentGraphState:
        content = json.dumps(
            {
                "input": _message_input_snapshot(state),
                "output": output,
            },
            sort_keys=True,
            default=str,
        )
        message = AgentMessage(agent_name=agent_name, content=content)
        messages = [*state.get("agent_messages", []), message]
        if self.experiment_store is not None:
            self.experiment_store.persist_agent_message(
                run_id=state["run_id"],
                agent_name=message.agent_name,
                role=message.role,
                content=message.content,
                metadata=message.metadata,
            )
        return {**updates, "agent_messages": messages}


def _sum_usage(usage_calls: list[dict[str, object]]) -> dict[str, float | int | None]:
    def total(field: str) -> float | int | None:
        values = [call.get(field) for call in usage_calls]
        present = [value for value in values if isinstance(value, (int, float))]
        if not present:
            return None
        return sum(present)

    return {
        "latency_ms": total("latency_ms"),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "estimated_cost": total("estimated_cost"),
    }


_SUPPORTED_METHODS = {
    VECTOR_RAG_METHOD_ID,
    LIGHTRAG_NEO4J_METHOD_ID,
    COGNEE_METHOD_ID,
    MS_GRAPHRAG_METHOD_ID,
    HIPPORAG2_METHOD_ID,
    ZERO_SHOT_METHOD_ID,
    ORACLE_METHOD_ID,
}


def _ensure_supported_method(method_id: str) -> None:
    if method_id not in _SUPPORTED_METHODS:
        raise ValueError(
            "Retriever supports only vector_rag, lightrag_neo4j, cognee, "
            "ms_graphrag, hipporag2, zero_shot_no_context, and single_document_context"
        )


# Reader "v2" — grounded: answer strictly from retrieved context, abstain
# instead of guessing. This is the default and the wording is byte-for-byte the
# instruction shipped since Bloco B; do NOT change it without a new prompt id.
_GROUNDED_READER_INSTRUCTION = (
    "Instructions: answer using ONLY the retrieved chunks above. They reflect "
    "the corpus as it was at collection time; when they disagree with what you "
    "believe is current, trust the chunks and answer from them. If the chunks "
    "do not contain the information needed, state that the retrieved context "
    "is insufficient instead of guessing."
)

# Reader "v1" — free: the SAME scaffold minus the ground-only-in-context
# restriction. The reader may fall back on parametric knowledge and is not told
# to abstain. Selected via READER_GROUNDING so the ONLY variable that changes
# between v1 and v2 is this instruction (single-variable grounding ablation;
# powers the oracle-v1 quadrant and Fase 3 cell 3.1).
_FREE_READER_INSTRUCTION = (
    "Instructions: answer the question. Use the retrieved chunks above when they "
    "help, and otherwise rely on your own knowledge; give your best answer rather "
    "than declining."
)


def reader_grounding_enabled() -> bool:
    """Whether the reader is grounded (v2, default). READER_GROUNDING in
    {v1, off, free, 0, no, none} selects the free reader (v1)."""
    val = os.getenv("READER_GROUNDING", "v2").strip().lower()
    return val not in {"v1", "off", "free", "0", "no", "none"}


def _retrieval_context_prompt(state: AgentGraphState) -> str:
    if state.get("method_id") == ZERO_SHOT_METHOD_ID:
        return _closed_book_prompt(state)
    retrieval_result = state.get("retrieval_result")
    chunks = []
    if retrieval_result is not None:
        chunks = [
            f"{item.source_chunk_id or item.item_id}: {item.text}"
            for item in retrieval_result.items
        ]
    instruction = (
        _GROUNDED_READER_INSTRUCTION
        if reader_grounding_enabled()
        else _FREE_READER_INSTRUCTION
    )
    return ("Question:\n{}\n\nRetrieved chunks:\n{}\n\n{}").format(
        state["question"],
        "\n".join(chunks) if chunks else "No retrieved chunks.",
        instruction,
    )


def _closed_book_prompt(state: AgentGraphState) -> str:
    # Parametric floor: no retrieved context on purpose, and no grounding
    # instruction (there is nothing to ground on).
    return (
        "Question:\n{}\n\n"
        "Instructions: answer the question directly from your own knowledge. "
        "Do not ask for additional context. If you do not know the answer, "
        "say that you do not know instead of guessing."
    ).format(state["question"])


def _message_input_snapshot(state: AgentGraphState) -> dict[str, object]:
    return {
        "dataset_id": state.get("dataset_id"),
        "dataset_version": state.get("dataset_version"),
        "method_id": state.get("method_id"),
        "experiment_id": state.get("experiment_id"),
        "run_id": state.get("run_id"),
        "agent_mode": state.get("agent_mode"),
        "question_id": state.get("question_id"),
        "question": state.get("question"),
    }
