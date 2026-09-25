from __future__ import annotations

from types import SimpleNamespace

from benchmark.agents.openai import OpenAIAgentLLM
from benchmark.embeddings.openai import OpenAIEmbeddingProvider


def test_openai_embedding_provider_calls_embeddings_endpoint() -> None:
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="embedding-model",
        dimension=3,
        client=client,
    )

    embeddings = provider.embed_texts(["alpha", "beta"])

    assert embeddings == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert client.embeddings.calls == [
        {
            "model": "embedding-model",
            "input": ["alpha", "beta"],
            "dimensions": 3,
            "encoding_format": "float",
        }
    ]


def test_openai_agent_llm_calls_responses_endpoint() -> None:
    client = FakeOpenAIClient()
    llm = OpenAIAgentLLM(api_key="test-key", model="chat-model", client=client)

    text = llm.generate(
        agent_name="domain_reasoner",
        prompt="Use retrieved evidence.",
        state={
            "dataset_id": "musique_smoke_20",
            "dataset_version": "v1",
            "method_id": "vector_rag",
            "question": "What is the governing law?",
        },
    )

    assert text == "model answer"
    assert client.responses.calls[0]["model"] == "chat-model"
    assert "Agent: domain_reasoner" in client.responses.calls[0]["input"]
    assert "Question: What is the governing law?" in client.responses.calls[0]["input"]
    assert client.responses.calls[0]["temperature"] == 0.0


def test_openai_agent_llm_records_usage_per_call() -> None:
    client = FakeOpenAIClient()
    llm = OpenAIAgentLLM(api_key="test-key", model="gpt-4o-mini", client=client)
    state = {
        "dataset_id": "musique",
        "dataset_version": "eval1k",
        "method_id": "vector_rag",
        "question": "Who founded the company?",
    }

    llm.generate(agent_name="domain_reasoner", prompt="draft", state=state)
    llm.generate(agent_name="final_answer", prompt="final", state=state)

    assert [record["agent_name"] for record in llm.usage_log] == [
        "domain_reasoner",
        "final_answer",
    ]
    first = llm.usage_log[0]
    assert first["prompt_tokens"] == 120
    assert first["completion_tokens"] == 30
    assert first["total_tokens"] == 150
    assert first["latency_ms"] > 0
    expected_cost = (120 * 0.15 + 30 * 0.60) / 1_000_000
    assert first["estimated_cost"] == expected_cost

    drained = llm.drain_usage()
    assert len(drained) == 2
    assert llm.usage_log == []


def test_openai_agent_llm_tolerates_missing_usage() -> None:
    client = FakeOpenAIClient()
    client.responses.usage = None
    llm = OpenAIAgentLLM(api_key="test-key", model="chat-model", client=client)

    llm.generate(
        agent_name="final_answer",
        prompt="final",
        state={"dataset_id": "d", "dataset_version": "v", "question": "q"},
    )

    record = llm.usage_log[0]
    assert record["prompt_tokens"] is None
    assert record["completion_tokens"] is None
    assert record["total_tokens"] is None
    assert record["estimated_cost"] is None
    assert record["latency_ms"] > 0


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsEndpoint()
        self.responses = FakeResponsesEndpoint()


class FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[1.0, 0.0, 0.0]),
                SimpleNamespace(embedding=[0.0, 1.0, 0.0]),
            ]
        )


class FakeResponsesEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.usage: SimpleNamespace | None = SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
        )

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="model answer", usage=self.usage)
