import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmark.core.schemas import Document
from benchmark.methods.lightrag_neo4j.adapter import (
    LightRAGNeo4jAdapter,
    LightRAGNeo4jIndexResult,
    LightRAGNeo4jQueryResult,
    RealLightRAGNeo4jClient,
    reset_lightrag_rag_cache,
    scoped_lightrag_neo4j_env,
)
from benchmark.methods.lightrag_neo4j.config_builder import LightRAGNeo4jConfig, build_workspace
from benchmark.methods.lightrag_neo4j.indexer import index, load_canonical_documents
from benchmark.methods.lightrag_neo4j.output_parser import parse_query_output
from benchmark.methods.lightrag_neo4j.retriever import retrieve


def test_input_preparation_writes_documents_and_manifest(tmp_path: Path) -> None:
    adapter = LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=FakeLightRAGNeo4jClient(),
    )
    document = _document(text="Contract\t\ttext ¶ with   spacing")

    paths = adapter.prepare_inputs([document])

    assert paths == [adapter.workspace.input_dir / "doc_1.txt"]
    assert paths[0].read_text(encoding="utf-8") == "Contract text with spacing"
    assert document.text == "Contract\t\ttext ¶ with   spacing"
    manifest = json.loads((adapter.workspace.input_dir / "manifest.json").read_text())
    assert manifest[0]["document_id"] == "doc_1"
    assert manifest[0]["dataset_id"] == "musique_smoke_20"


def test_fake_indexing_path_preserves_raw_output(tmp_path: Path) -> None:
    client = FakeLightRAGNeo4jClient()
    adapter = LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=client,
    )

    summary = index(adapter=adapter, documents=[_document()])

    assert summary.method_id == "lightrag_neo4j"
    assert summary.document_count == 1
    assert client.index_calls[0]["config"].neo4j_uri_env == "LIGHTRAG_NEO4J_URI"
    assert client.index_calls[0]["documents"][0].text == "Contract text"
    assert (adapter.workspace.raw_dir / "index_result.json").exists()


def test_fake_retrieval_path_converts_and_persists(tmp_path: Path) -> None:
    adapter = LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=FakeLightRAGNeo4jClient(),
    )
    store = FakeExperimentStore()

    result = retrieve(
        adapter=adapter,
        query="governing law",
        top_k=1,
        query_mode="mix",
        experiment_store=store,
        run_id="run_1",
    )

    assert result.method_id == "lightrag_neo4j"
    assert result.items[0].text == "Governing law is Delaware."
    assert result.items[0].source_document_id == "doc_1"
    assert result.items[0].source_chunk_id == "chunk_1"
    assert store.calls[0]["result"].method_id == "lightrag_neo4j"
    assert store.calls[0]["run_id"] == "run_1"
    assert result.metadata["lightrag_usage"]["usage_source"] == "estimated"
    assert result.metadata["lightrag_usage"]["instrumentation_version"] == "lightrag_usage_v2"
    assert result.metadata["lightrag_usage"]["prompt_tokens"] > 0
    assert result.metadata["lightrag_usage"]["completion_tokens"] > 0
    assert (adapter.workspace.raw_dir / "query_mix_result.json").exists()


def test_fake_retrieval_path_attaches_trace_when_framework_exposes_it(tmp_path: Path) -> None:
    adapter = LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=FakeTraceLightRAGNeo4jClient(),
    )

    result = retrieve(adapter=adapter, query="governing law", top_k=1, query_mode="mix")

    assert result.items[0].text == "Governing law is Delaware."
    assert result.items[0].source_chunk_id == "chunk_1"
    trace = result.metadata["retrieval_trace"]
    assert trace["framework_id"] == "lightrag_neo4j"
    assert trace["query_mode"] == "mix"
    assert trace["entities"][0]["id"] == "entity_1"
    assert trace["relations"][0]["id"] == "rel_1"
    assert trace["final_context"]["text"] == "assembled context"


def test_fake_retrieval_path_builds_trace_from_lightrag_query_llm_data(tmp_path: Path) -> None:
    adapter = LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=FakeStructuredTraceLightRAGNeo4jClient(),
    )

    result = retrieve(adapter=adapter, query="governing law", top_k=1, query_mode="mix")

    assert result.items[0].text == "Governing law is Delaware."
    assert result.items[0].source_chunk_id == "chunk_1"
    trace = result.metadata["retrieval_trace"]
    assert trace["framework_id"] == "lightrag_neo4j"
    assert trace["query_mode"] == "mix"
    assert trace["nodes"][0]["entity_name"] == "GOVERNING LAW"
    assert trace["relationships"][0]["src_id"] == "AGREEMENT"
    assert trace["chunks"][0]["chunk_id"] == "chunk_1"
    assert trace["final_context"]["type"] == "observed_system_prompt"
    assert trace["trace_status"] == "complete"


def test_output_parser_handles_plain_text_response() -> None:
    result = parse_query_output(
        query="governing law",
        raw_response="Delaware law.",
        query_mode="mix",
        latency_ms=1.0,
        top_k=5,
    )

    assert result.method_id == "lightrag_neo4j"
    assert result.items[0].text == "Delaware law."
    assert result.items[0].metadata["kind"] == "answer"


def test_output_parser_handles_lightrag_query_llm_response() -> None:
    result = parse_query_output(
        query="governing law",
        raw_response=_structured_lightrag_response(),
        query_mode="mix",
        latency_ms=1.0,
        top_k=5,
    )

    assert result.items[0].text == "Governing law is Delaware."
    assert result.items[0].source_chunk_id == "chunk_1"
    assert result.items[0].metadata["reference_id"] == "1"


def test_load_canonical_documents_supports_max_documents(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    docs = [_document("doc_1"), _document("doc_2")]
    (canonical / "documents.jsonl").write_text(
        "\n".join(json.dumps(doc.model_dump(mode="json")) for doc in docs) + "\n",
        encoding="utf-8",
    )

    loaded = load_canonical_documents(canonical, max_documents=1)

    assert loaded == [docs[0]]


def test_default_adapter_fails_with_actionable_missing_env_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "LIGHTRAG_NEO4J_URI",
        "LIGHTRAG_NEO4J_USER",
        "LIGHTRAG_NEO4J_PASSWORD",
        "LIGHTRAG_NEO4J_DATABASE",
    ):
        monkeypatch.delenv(name, raising=False)
    adapter = LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        )
    )

    with pytest.raises(RuntimeError, match="LightRAG-specific Neo4j environment variables"):
        adapter.index(documents=[_document()])


def test_scoped_env_mapping_restores_values_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    _set_lightrag_env(monkeypatch)
    monkeypatch.setenv("NEO4J_URI", "bolt://previous:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "previous-user")
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    monkeypatch.delenv("NEO4J_WORKSPACE", raising=False)

    with scoped_lightrag_neo4j_env(workspace=workspace, config=LightRAGNeo4jConfig()):
        assert _native_neo4j_env() == {
            "NEO4J_URI": "bolt://localhost:7688",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "benchmark_lightrag",
            "NEO4J_DATABASE": "neo4j",
            "NEO4J_WORKSPACE": "lightrag_neo4j_musique_smoke_20_v1",
        }

    assert os.environ["NEO4J_URI"] == "bolt://previous:7687"
    assert os.environ["NEO4J_USERNAME"] == "previous-user"
    assert "NEO4J_PASSWORD" not in os.environ
    assert "NEO4J_DATABASE" not in os.environ
    assert "NEO4J_WORKSPACE" not in os.environ


def test_scoped_env_mapping_restores_values_after_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    _set_lightrag_env(monkeypatch)
    monkeypatch.setenv("NEO4J_URI", "bolt://previous:7687")

    with pytest.raises(RuntimeError, match="boom"):
        with scoped_lightrag_neo4j_env(workspace=workspace, config=LightRAGNeo4jConfig()):
            raise RuntimeError("boom")

    assert os.environ["NEO4J_URI"] == "bolt://previous:7687"
    assert "NEO4J_WORKSPACE" not in os.environ


def test_real_client_instantiates_lightrag_with_neo4j_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_lightrag_env(monkeypatch)
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAG,
        query_param_cls=FakeQueryParam,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    result = client.index(
        workspace=workspace,
        config=LightRAGNeo4jConfig(),
        documents=[_document()],
    )

    assert result.document_count == 1
    assert len(FakeLightRAG.instances) == 1
    assert FakeLightRAG.instances[0].kwargs["graph_storage"] == "Neo4JStorage"
    assert FakeLightRAG.instances[0].kwargs["working_dir"] == str(workspace.native_dir)
    assert FakeLightRAG.instances[0].kwargs["workspace"] == "lightrag_neo4j_musique_smoke_20_v1"
    assert "enable_llm_cache" not in FakeLightRAG.instances[0].kwargs
    assert FakeLightRAG.instances[0].insert_calls == [
        {"texts": ["Contract text"], "ids": ["doc_1"]}
    ]


def test_real_client_query_uses_query_param_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_lightrag_env(monkeypatch)
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAG,
        query_param_cls=FakeQueryParam,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    result = client.query(
        workspace=workspace,
        config=LightRAGNeo4jConfig(),
        query="governing law",
        mode="mix",
        top_k=5,
    )

    assert result.raw_response == {"answer": "fake answer"}
    assert result.usage is not None
    assert result.usage.usage_source == "estimated"
    assert result.usage.prompt_tokens is not None
    assert result.usage.completion_tokens is not None
    assert FakeLightRAG.instances[0].query_calls == [
        {
            "query": "governing law",
            "param": FakeQueryParam(mode="mix", top_k=5),
        }
    ]


def test_real_client_query_can_disable_lightrag_llm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_lightrag_env(monkeypatch)
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAG,
        query_param_cls=FakeQueryParam,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    config = LightRAGNeo4jConfig(disable_llm_cache_for_query=True)

    result = client.query(
        workspace=workspace,
        config=config,
        query="governing law",
        mode="mix",
        top_k=5,
    )

    assert result.raw_response == {"answer": "fake answer"}
    assert FakeLightRAG.instances[0].kwargs["enable_llm_cache"] is False


def test_retrieve_marks_cache_disabled_metadata_when_config_requests_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_lightrag_env(monkeypatch)
    adapter = LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        config=LightRAGNeo4jConfig(disable_llm_cache_for_query=True),
        client=FakeTraceLightRAGNeo4jClient(),
    )

    result = retrieve(adapter=adapter, query="governing law", top_k=1, query_mode="mix")

    assert result.metadata["llm_cache_disabled"] is True
    assert result.metadata["cache_control_source"] in {"env", "config"}
    assert result.metadata["instrumentation_version"] == "lightrag_trace_v1"


def test_real_client_query_uses_direct_llm_usage_when_exposed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_lightrag_env(monkeypatch)
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAGCallingLlm,
        query_param_cls=FakeQueryParam,
        llm_model_func=fake_llm_with_usage,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    result = client.query(
        workspace=workspace,
        config=LightRAGNeo4jConfig(),
        query="governing law",
        mode="mix",
        top_k=5,
    )

    assert result.usage is not None
    assert result.usage.usage_source == "direct"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 18
    assert result.usage.estimated_cost is not None


def test_real_client_prefers_query_llm_and_attaches_observed_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_lightrag_env(monkeypatch)
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAGWithQueryLlm,
        query_param_cls=FakeQueryParam,
        llm_model_func=fake_llm_with_usage,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    result = client.query(
        workspace=workspace,
        config=LightRAGNeo4jConfig(),
        query="governing law",
        mode="mix",
        top_k=5,
    )

    assert result.raw_response is not None
    assert isinstance(result.raw_response, dict)
    assert result.raw_response["llm_response"]["content"] == "Delaware law."
    assert result.raw_response["retrieval_trace"]["final_context"] == {
        "type": "observed_system_prompt",
        "text": "Context for governing law",
    }
    assert FakeLightRAG.instances[0].query_llm_calls == [
        {
            "query": "governing law",
            "param": FakeQueryParam(mode="mix", top_k=5),
        }
    ]
    assert FakeLightRAG.instances[0].query_calls == []
    assert result.usage is not None
    assert result.usage.usage_source == "direct"


def test_real_client_rebuilds_rag_per_query_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default (no LIGHTRAG_REUSE_RAG): each query builds its own instance.
    monkeypatch.delenv("LIGHTRAG_REUSE_RAG", raising=False)
    _set_lightrag_env(monkeypatch)
    reset_lightrag_rag_cache()
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAG,
        query_param_cls=FakeQueryParam,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    for _ in range(3):
        client.query(
            workspace=workspace, config=LightRAGNeo4jConfig(), query="q", mode="mix", top_k=5
        )

    assert len(FakeLightRAG.instances) == 3


def test_real_client_reuses_single_rag_instance_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With LIGHTRAG_REUSE_RAG=1 the heavy native index loads once: a single
    # LightRAG instance is built and reused across every query in the process.
    monkeypatch.setenv("LIGHTRAG_REUSE_RAG", "1")
    _set_lightrag_env(monkeypatch)
    reset_lightrag_rag_cache()
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAG,
        query_param_cls=FakeQueryParam,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    results = [
        client.query(
            workspace=workspace, config=LightRAGNeo4jConfig(), query="q", mode="mix", top_k=5
        )
        for _ in range(3)
    ]

    assert len(FakeLightRAG.instances) == 1
    assert all(r.raw_response == {"answer": "fake answer"} for r in results)
    # Same workspace + same query reach the same cached instance across queries.
    assert FakeLightRAG.instances[0].query_calls[0]["query"] == "q"
    assert len(FakeLightRAG.instances[0].query_calls) == 3
    reset_lightrag_rag_cache()


def test_reused_rag_keeps_per_query_usage_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A cached rag must still attribute usage per query: each query gets its own
    # tracker via the holder indirection, so totals never accumulate across calls.
    monkeypatch.setenv("LIGHTRAG_REUSE_RAG", "1")
    _set_lightrag_env(monkeypatch)
    reset_lightrag_rag_cache()
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAGCallingLlm,
        query_param_cls=FakeQueryParam,
        llm_model_func=fake_llm_with_usage,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    first = client.query(
        workspace=workspace, config=LightRAGNeo4jConfig(), query="governing law", mode="mix", top_k=5
    )
    second = client.query(
        workspace=workspace, config=LightRAGNeo4jConfig(), query="governing law", mode="mix", top_k=5
    )

    assert len(FakeLightRAG.instances) == 1
    for result in (first, second):
        assert result.usage is not None
        assert result.usage.usage_source == "direct"
        assert result.usage.prompt_tokens == 11
        assert result.usage.completion_tokens == 7
    reset_lightrag_rag_cache()


def test_real_client_uses_only_lightrag_neo4j_user_envs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_lightrag_env(monkeypatch)
    monkeypatch.setenv("GRAPHRAG_NEO4J_URI", "bolt://wrong:7687")
    FakeLightRAG.instances = []
    client = RealLightRAGNeo4jClient(
        lightrag_cls=FakeLightRAG,
        query_param_cls=FakeQueryParam,
    )
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    client.index(workspace=workspace, config=LightRAGNeo4jConfig(), documents=[_document()])

    assert FakeLightRAG.instances[0].env_seen["NEO4J_URI"] == "bolt://localhost:7688"
    assert os.environ["GRAPHRAG_NEO4J_URI"] == "bolt://wrong:7687"


def _document(document_id: str = "doc_1", text: str = "Contract text") -> Document:
    return Document(
        document_id=document_id,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text=text,
    )


def _set_lightrag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTRAG_NEO4J_URI", "bolt://localhost:7688")
    monkeypatch.setenv("LIGHTRAG_NEO4J_USER", "neo4j")
    monkeypatch.setenv("LIGHTRAG_NEO4J_PASSWORD", "benchmark_lightrag")
    monkeypatch.setenv("LIGHTRAG_NEO4J_DATABASE", "neo4j")


def _native_neo4j_env() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in (
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "NEO4J_DATABASE",
            "NEO4J_WORKSPACE",
        )
    }


class FakeLightRAGNeo4jClient:
    def __init__(self) -> None:
        self.index_calls = []
        self.query_calls = []

    def index(self, *, workspace, config, documents):
        self.index_calls.append({"workspace": workspace, "config": config, "documents": documents})
        return LightRAGNeo4jIndexResult(
            document_count=len(documents),
            raw_response={"status": "ok", "documents": len(documents)},
        )

    def query(self, *, workspace, config, query: str, mode: str, top_k: int):
        self.query_calls.append(
            {
                "workspace": workspace,
                "config": config,
                "query": query,
                "mode": mode,
                "top_k": top_k,
            }
        )
        return LightRAGNeo4jQueryResult(
            query=query,
            mode=mode,
            raw_response={
                "answer": "Delaware law.",
                "contexts": [
                    {
                        "text": "Governing law is Delaware.",
                        "document_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "score": 0.9,
                    }
                ],
            },
        )


class FakeTraceLightRAGNeo4jClient(FakeLightRAGNeo4jClient):
    def query(self, *, workspace, config, query: str, mode: str, top_k: int):
        self.query_calls.append(
            {
                "workspace": workspace,
                "config": config,
                "query": query,
                "mode": mode,
                "top_k": top_k,
            }
        )
        return LightRAGNeo4jQueryResult(
            query=query,
            mode=mode,
            raw_response={
                "answer": "Delaware law.",
                "contexts": [
                    {
                        "text": "Governing law is Delaware.",
                        "document_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "score": 0.9,
                    }
                ],
                "retrieval_trace": {
                    "entities": [{"id": "entity_1", "label": "Delaware"}],
                    "relations": [{"id": "rel_1", "source": "entity_1", "target": "entity_2"}],
                    "final_context": {"text": "assembled context"},
                },
            },
        )


class FakeStructuredTraceLightRAGNeo4jClient(FakeLightRAGNeo4jClient):
    def query(self, *, workspace, config, query: str, mode: str, top_k: int):
        self.query_calls.append(
            {
                "workspace": workspace,
                "config": config,
                "query": query,
                "mode": mode,
                "top_k": top_k,
            }
        )
        return LightRAGNeo4jQueryResult(
            query=query,
            mode=mode,
            raw_response=_structured_lightrag_response(),
        )


class FakeExperimentStore:
    def __init__(self) -> None:
        self.calls = []

    def persist_retrieval_result(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return "retrieval_1"


class FakeLightRAG:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.env_seen = _native_neo4j_env()
        self.insert_calls = []
        self.query_calls = []
        self.query_llm_calls = []
        FakeLightRAG.instances.append(self)

    def insert(self, texts, ids):
        self.insert_calls.append({"texts": texts, "ids": ids})
        return {"inserted": len(texts)}

    def query(self, query, param):
        self.query_calls.append({"query": query, "param": param})
        return {"answer": "fake answer"}


class FakeLightRAGWithQueryLlm(FakeLightRAG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_model_func = kwargs["llm_model_func"]

    def query_llm(self, query, param):
        self.query_llm_calls.append({"query": query, "param": param})
        response = self.llm_model_func(
            query,
            system_prompt=f"Context for {query}",
        )
        payload = _structured_lightrag_response()
        payload["llm_response"] = {
            "content": response.output_text,
            "response_iterator": None,
            "is_streaming": False,
        }
        return payload


class FakeLightRAGCallingLlm(FakeLightRAG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_model_func = kwargs["llm_model_func"]

    def query(self, query, param):
        self.query_calls.append({"query": query, "param": param})
        response = self.llm_model_func(f"Answer this: {query}")
        return {"answer": response.output_text}


class FakeUsage:
    input_tokens = 11
    output_tokens = 7
    total_tokens = 18


class FakeLlmResponse:
    output_text = "Delaware law."
    usage = FakeUsage()


def fake_llm_with_usage(prompt: str, **kwargs) -> FakeLlmResponse:
    assert "governing law" in prompt
    return FakeLlmResponse()


@dataclass(frozen=True)
class FakeQueryParam:
    mode: str
    top_k: int


def _structured_lightrag_response() -> dict:
    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "entities": [
                {
                    "entity_name": "GOVERNING LAW",
                    "entity_type": "CLAUSE",
                    "description": "Governing law clause.",
                }
            ],
            "relationships": [
                {
                    "src_id": "AGREEMENT",
                    "tgt_id": "GOVERNING LAW",
                    "description": "contains clause",
                    "weight": 1.0,
                }
            ],
            "chunks": [
                {
                    "reference_id": "1",
                    "content": "Governing law is Delaware.",
                    "file_path": "doc_1.txt",
                    "chunk_id": "chunk_1",
                }
            ],
            "references": [{"reference_id": "1", "file_path": "doc_1.txt"}],
        },
        "metadata": {"query_mode": "mix"},
        "llm_response": {
            "content": "Delaware law.",
            "response_iterator": None,
            "is_streaming": False,
        },
        "retrieval_trace": {
            "final_context": {
                "type": "observed_system_prompt",
                "text": "Context for governing law",
            }
        },
    }
