from benchmark.agents import build_agent_graph
from benchmark.core.schemas import RetrievalResult, RetrievedItem


def test_agent_messages_retrieval_and_final_answer_are_persisted() -> None:
    store = FakeExperimentStore()
    workflow = build_agent_graph(retriever=fake_retriever, experiment_store=store)

    result = workflow.invoke(base_state("vector_rag"))

    assert result["final_answer"].answer_text.startswith("Final answer")
    assert len(store.messages) == 5
    assert store.retrieval_results[0]["result"].method_id == "vector_rag"
    assert store.answers[0]["method_id"] == "vector_rag"
    assert store.answers[0]["citations"] == ["chunk_1"]


def test_final_answer_persists_aggregated_llm_usage() -> None:
    store = FakeExperimentStore()
    workflow = build_agent_graph(
        retriever=fake_retriever,
        llm=UsageTrackingFakeLLM(),
        experiment_store=store,
    )

    workflow.invoke(base_state("vector_rag"))

    answer = store.answers[0]
    assert answer["latency_ms"] == 25.0
    assert answer["prompt_tokens"] == 300
    assert answer["completion_tokens"] == 80
    assert answer["total_tokens"] == 380
    assert answer["estimated_cost"] == 0.0002
    assert len(answer["metadata"]["usage_calls"]) == 2


def test_final_answer_persists_null_usage_for_untracked_llm() -> None:
    store = FakeExperimentStore()
    workflow = build_agent_graph(retriever=fake_retriever, experiment_store=store)

    workflow.invoke(base_state("vector_rag"))

    answer = store.answers[0]
    assert answer["latency_ms"] is None
    assert answer["prompt_tokens"] is None
    assert answer["completion_tokens"] is None
    assert answer["total_tokens"] is None
    assert answer["estimated_cost"] is None
    assert "usage_calls" not in answer["metadata"]


class UsageTrackingFakeLLM:
    def __init__(self) -> None:
        self.usage_log = []
        self._calls = 0

    def generate(self, *, agent_name: str, prompt: str, state) -> str:
        self._calls += 1
        self.usage_log.append(
            {
                "agent_name": agent_name,
                "model": "gpt-4o-mini",
                "latency_ms": 12.5,
                "prompt_tokens": 150,
                "completion_tokens": 40,
                "total_tokens": 190,
                "estimated_cost": 0.0001,
            }
        )
        if agent_name == "final_answer":
            return "Final answer using chunk_1."
        return "Draft answer using chunk_1."

    def drain_usage(self):
        drained = self.usage_log
        self.usage_log = []
        return drained


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
                score=0.7,
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


class FakeExperimentStore:
    def __init__(self) -> None:
        self.messages = []
        self.retrieval_results = []
        self.answers = []

    def persist_agent_message(self, **kwargs: object) -> str:
        self.messages.append(kwargs)
        return f"message_{len(self.messages)}"

    def persist_retrieval_result(self, **kwargs: object) -> str:
        self.retrieval_results.append(kwargs)
        return "retrieval_1"

    def persist_answer(self, **kwargs: object) -> str:
        self.answers.append(kwargs)
        return "answer_1"
