import os
from pathlib import Path

import pytest

from benchmark.methods.lightrag_neo4j.adapter import (
    LightRAGNeo4jAdapter,
    LightRAGNeo4jQueryResult,
    scoped_lightrag_neo4j_env,
)
from benchmark.methods.lightrag_neo4j.config_builder import (
    LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
    LIGHTRAG_NEO4J_METHOD_ID,
    LightRAGNeo4jConfig,
    build_workspace,
)
from benchmark.methods.lightrag_neo4j.document_scope import (
    GRAPH_FIELD_SEP,
    NOT_ENOUGH_INFORMATION,
    apply_document_scope,
)
from benchmark.methods.lightrag_neo4j.retriever import retrieve


def test_strict_keeps_only_target_document_chunks() -> None:
    scoped = apply_document_scope(_raw_data(), target_document_id="doc_target")

    chunks = scoped.raw_data["data"]["chunks"]

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "doc_target-chunk-1",
        "chunk_from_source_document_id",
        "chunk_from_file_path",
    ]
    assert scoped.scope_metadata["chunks_before_scope"] == 4
    assert scoped.scope_metadata["chunks_after_scope"] == 3


def test_strict_removes_other_document_chunks() -> None:
    scoped = apply_document_scope(_raw_data(), target_document_id="doc_other")

    chunks = scoped.raw_data["data"]["chunks"]

    assert [chunk["chunk_id"] for chunk in chunks] == ["doc_other-chunk-1"]
    assert all("target clause" not in str(chunk.get("content")) for chunk in chunks)


def test_source_id_with_sep_is_split_and_podado_for_node() -> None:
    source_id = GRAPH_FIELD_SEP.join(["doc_target-chunk-1", "doc_other-chunk-1"])
    raw_data = {"data": {"entities": [{"entity_name": "Clause", "source_id": source_id}]}}

    scoped = apply_document_scope(raw_data, target_document_id="doc_target")

    entities = scoped.raw_data["data"]["entities"]
    assert len(entities) == 1
    assert entities[0]["source_id"] == "doc_target-chunk-1"
    assert scoped.scope_metadata["nodes_before_scope"] == 1
    assert scoped.scope_metadata["nodes_after_scope"] == 1


def test_relationship_with_multiple_source_ids_is_kept_and_pruned() -> None:
    source_id = GRAPH_FIELD_SEP.join(["doc_target-chunk-1", "doc_other-chunk-1"])
    raw_data = {
        "data": {
            "relationships": [
                {
                    "src_id": "Agreement",
                    "tgt_id": "Term",
                    "source_id": source_id,
                }
            ]
        }
    }

    scoped = apply_document_scope(raw_data, target_document_id="doc_target")

    relationships = scoped.raw_data["data"]["relationships"]
    assert len(relationships) == 1
    assert relationships[0]["source_id"] == "doc_target-chunk-1"
    assert scoped.scope_metadata["relationships_before_scope"] == 1
    assert scoped.scope_metadata["relationships_after_scope"] == 1


def test_chunk_with_source_document_id_is_kept() -> None:
    raw_data = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "opaque_chunk",
                    "source_document_id": "doc_target",
                    "content": "target text",
                }
            ]
        }
    }

    scoped = apply_document_scope(raw_data, target_document_id="doc_target")

    assert scoped.raw_data["data"]["chunks"][0]["chunk_id"] == "opaque_chunk"


def test_chunk_with_file_path_containing_document_id_is_kept() -> None:
    raw_data = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "opaque_chunk",
                    "file_path": "/tmp/input/doc_target.txt",
                    "content": "target text",
                }
            ]
        }
    }

    scoped = apply_document_scope(raw_data, target_document_id="doc_target")

    assert scoped.raw_data["data"]["chunks"][0]["chunk_id"] == "opaque_chunk"


def test_strict_zero_chunks_returns_deterministic_answer_and_skips_llm(tmp_path: Path) -> None:
    client = FakeDocScopedClient(raw_data=_raw_data(only_other_document=True))
    adapter = _strict_adapter(tmp_path, client)

    result = retrieve(
        adapter=adapter,
        query="What is the agreement name?",
        top_k=5,
        query_mode="mix",
        target_document_id="doc_target",
        method_id=LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
    )

    assert result.method_id == LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID
    assert client.query_calls == []
    assert len(client.query_data_calls) == 1
    assert client.generate_calls == []
    assert result.raw_response["llm_response"]["content"] == NOT_ENOUGH_INFORMATION
    trace = result.metadata["retrieval_trace"]
    assert trace["generation_skipped"] is True
    assert trace["generation_skip_reason"] == "empty_document_scoped_context"
    assert trace["scope_fallback"] == "none"
    assert trace["chunks_after_scope"] == 0


def test_strict_with_filtered_chunks_calls_llm_once_and_records_counts(tmp_path: Path) -> None:
    client = FakeDocScopedClient(raw_data=_raw_data())
    adapter = _strict_adapter(tmp_path, client)

    result = retrieve(
        adapter=adapter,
        query="What is the agreement name?",
        top_k=5,
        query_mode="mix",
        target_document_id="doc_target",
        method_id=LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
    )

    assert result.method_id == LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID
    assert client.query_calls == []
    assert len(client.query_data_calls) == 1
    assert len(client.generate_calls) == 1
    assert "target clause" in client.generate_calls[0]["system_prompt"]
    assert "other clause" not in client.generate_calls[0]["system_prompt"]
    assert result.raw_response["llm_response"]["content"] == "DISTRIBUTOR AGREEMENT"
    trace = result.metadata["retrieval_trace"]
    assert trace["target_document_id"] == "doc_target"
    assert trace["document_scope_applied"] is True
    assert trace["scope_mode"] == "strict"
    assert trace["scope_fallback"] == "none"
    assert trace["nodes_before_scope"] == 2
    assert trace["nodes_after_scope"] == 1
    assert trace["relationships_before_scope"] == 2
    assert trace["relationships_after_scope"] == 1
    assert trace["chunks_before_scope"] == 4
    assert trace["chunks_after_scope"] == 3


def test_vanilla_lightrag_does_not_apply_document_scope(tmp_path: Path) -> None:
    client = FakeDocScopedClient(raw_data=_raw_data())
    adapter = LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=client,
    )

    result = retrieve(
        adapter=adapter,
        query="What is the agreement name?",
        top_k=5,
        query_mode="mix",
        target_document_id="doc_target",
    )

    assert result.method_id == LIGHTRAG_NEO4J_METHOD_ID
    assert len(client.query_calls) == 1
    assert client.query_data_calls == []
    assert "retrieval_trace" not in result.metadata


def test_strict_uses_vanilla_backing_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        method_id=LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
    )
    _set_lightrag_env(monkeypatch)

    assert workspace.method_id == LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID
    assert workspace.artifact_dir == tmp_path / "musique_smoke_20_v1" / LIGHTRAG_NEO4J_METHOD_ID
    with scoped_lightrag_neo4j_env(workspace=workspace, config=LightRAGNeo4jConfig()):
        assert _native_neo4j_env()["NEO4J_WORKSPACE"] == "lightrag_neo4j_musique_smoke_20_v1"


class FakeDocScopedClient:
    def __init__(self, *, raw_data: dict) -> None:
        self.raw_data = raw_data
        self.query_calls = []
        self.query_data_calls = []
        self.generate_calls = []

    def query(self, *, workspace, config, query: str, mode: str, top_k: int):
        self.query_calls.append(
            {"workspace": workspace, "config": config, "query": query, "mode": mode, "top_k": top_k}
        )
        return LightRAGNeo4jQueryResult(query=query, mode=mode, raw_response={"answer": "vanilla"})

    def query_data(self, *, workspace, config, query: str, mode: str, top_k: int):
        self.query_data_calls.append(
            {"workspace": workspace, "config": config, "query": query, "mode": mode, "top_k": top_k}
        )
        return LightRAGNeo4jQueryResult(query=query, mode=mode, raw_response=self.raw_data)

    def generate(self, *, workspace, config, query: str, system_prompt: str):
        self.generate_calls.append(
            {
                "workspace": workspace,
                "config": config,
                "query": query,
                "system_prompt": system_prompt,
            }
        )
        return LightRAGNeo4jQueryResult(
            query=query,
            mode="bypass",
            raw_response={
                "status": "success",
                "llm_response": {
                    "content": "DISTRIBUTOR AGREEMENT",
                    "response_iterator": None,
                    "is_streaming": False,
                },
            },
        )


def _strict_adapter(tmp_path: Path, client: FakeDocScopedClient) -> LightRAGNeo4jAdapter:
    return LightRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            method_id=LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
        ),
        config=LightRAGNeo4jConfig(method_id=LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID),
        client=client,
    )


def _raw_data(*, only_other_document: bool = False) -> dict:
    chunks = [
        {
            "reference_id": "1",
            "content": "target clause: DISTRIBUTOR AGREEMENT",
            "file_path": "/tmp/doc_target.txt",
            "chunk_id": "doc_target-chunk-1",
        },
        {
            "reference_id": "2",
            "content": "other clause",
            "file_path": "/tmp/doc_other.txt",
            "chunk_id": "doc_other-chunk-1",
        },
        {
            "reference_id": "3",
            "content": "target clause from source_document_id",
            "source_document_id": "doc_target",
            "chunk_id": "chunk_from_source_document_id",
        },
        {
            "reference_id": "4",
            "content": "target clause from file path",
            "file_path": "/tmp/input/doc_target.txt",
            "chunk_id": "chunk_from_file_path",
        },
    ]
    if only_other_document:
        chunks = [chunks[1]]
    source_id = GRAPH_FIELD_SEP.join(["doc_target-chunk-1", "doc_other-chunk-1"])
    return {
        "status": "success",
        "data": {
            "entities": [
                {"entity_name": "Agreement", "source_id": source_id},
                {"entity_name": "Other", "source_id": "doc_other-chunk-1"},
            ],
            "relationships": [
                {"src_id": "Agreement", "tgt_id": "Document", "source_id": source_id},
                {"src_id": "Other", "tgt_id": "Document", "source_id": "doc_other-chunk-1"},
            ],
            "chunks": chunks,
            "references": [],
        },
        "metadata": {"query_mode": "mix"},
    }


def _set_lightrag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTRAG_NEO4J_URI", "bolt://localhost:7688")
    monkeypatch.setenv("LIGHTRAG_NEO4J_USER", "neo4j")
    monkeypatch.setenv("LIGHTRAG_NEO4J_PASSWORD", "benchmark_lightrag")
    monkeypatch.setenv("LIGHTRAG_NEO4J_DATABASE", "neo4j")


def _native_neo4j_env() -> dict[str, str]:
    return {
        name: value
        for name in (
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "NEO4J_DATABASE",
            "NEO4J_WORKSPACE",
        )
        if (value := os.environ.get(name)) is not None
    }
