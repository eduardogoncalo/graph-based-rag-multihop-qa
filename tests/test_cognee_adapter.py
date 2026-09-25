from __future__ import annotations

import asyncio
from pathlib import Path

from benchmark.methods.cognee import (
    CogneeAdapter,
    CogneeConfig,
    CogneeTracePaths,
    CogneeTraceWriter,
    build_workspace,
    parse_cognee_result,
)


class FakeCogneeClient:
    def __init__(self) -> None:
        self.add_kwargs = {}
        self.cognify_kwargs = {}
        self.recall_dataset_names = ()

    async def remember(self, data, *, dataset_name: str, **kwargs):
        return {"status": "completed", "dataset_name": dataset_name, "items": data}

    async def recall(
        self,
        query: str,
        *,
        top_k: int,
        dataset_name: str,
        dataset_names: tuple[str, ...] | None = None,
        only_context: bool = True,
    ):
        self.recall_dataset_names = tuple(dataset_names or (dataset_name,))
        return {
            "primary_api": "recall",
            "retrieval_items": [
                {
                    "id": "node_1",
                    "text": f"context for {query}",
                    "document_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.91,
                    "metadata": {"kind": "memory"},
                }
            ],
            "generated_answer": "answer generated inside Cognee",
            "raw_trace": {"nodes": ["node_1"], "relationships": []},
            "retriever_purity": "mixed",
        }

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        dataset_name: str,
        dataset_names: tuple[str, ...] | None = None,
        only_context: bool = True,
    ):
        return []

    async def add(self, data, *, dataset_name: str, **kwargs):
        self.add_kwargs = kwargs
        return {"status": "added"}

    async def cognify(self, *, dataset_name: str, **kwargs):
        self.cognify_kwargs = kwargs
        return {"status": "cognified"}


def test_parser_transforms_fake_result_into_retrieval_result() -> None:
    result = parse_cognee_result(
        query="governing law",
        raw_result={
            "retrieval_items": [
                {
                    "text": "This agreement is governed by New York law.",
                    "document_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.5,
                }
            ],
            "raw_trace": {"nodes": ["n1"]},
        },
        top_k=5,
        latency_ms=1.0,
    )

    assert result.method_id == "cognee"
    assert result.items[0].source_document_id == "doc_1"
    assert result.items[0].source_chunk_id == "chunk_1"
    assert result.items[0].score == 0.5
    assert result.metadata["retrieval_trace_available"] is True
    assert result.metadata["retriever_purity"] == "context_only"


def test_parser_enriches_plain_cognee_chunk_text_from_mapping_index() -> None:
    text = "document_id: doc_abc123\nRelevant clause text."

    result = parse_cognee_result(
        query="clause",
        raw_result=[text],
        top_k=5,
        latency_ms=1.0,
        chunk_mappings=[
            {
                "text": text,
                "source_document_id": "doc_abc123",
                "source_dataset_name": "musique_smoke_20_v1_batch_001",
                "source_cognee_chunk_id": "cognee_chunk_1",
                "source_content_hash": "hash_1",
            }
        ],
    )

    item = result.items[0]
    assert item.source_document_id == "doc_abc123"
    assert item.source_chunk_id == "cognee::musique_smoke_20_v1_batch_001::cognee_chunk_1"
    assert item.metadata["source_chunk_id_strategy"] == "deterministic_fallback"
    assert item.metadata["source_dataset_name"] == "musique_smoke_20_v1_batch_001"
    assert item.metadata["source_cognee_chunk_id"] == "cognee_chunk_1"


def test_parser_recognizes_document_id_patterns_without_mapping_index() -> None:
    result = parse_cognee_result(
        query="clause",
        raw_result=["metadata document_id=doc_feed123 text"],
        top_k=5,
        latency_ms=1.0,
    )

    assert result.items[0].source_document_id == "doc_feed123"
    assert result.items[0].metadata["source_chunk_id_strategy"] == "deterministic_fallback"


def test_parser_mapping_lookup_prefers_matching_document_id_for_graph_context() -> None:
    result = parse_cognee_result(
        query="contracts",
        raw_result=[
            "Nodes:\n__node_content_start__\ndocument_id: doc_222abc\nMain text.\n__node_content_end__\n"
            "__node_content_start__\nOther chunk text.\n__node_content_end__"
        ],
        top_k=5,
        latency_ms=1.0,
        chunk_mappings=[
            {
                "text": "Other chunk text.",
                "source_document_id": "doc_111abc",
                "source_dataset_name": "dataset_1",
                "source_cognee_chunk_id": "wrong_chunk",
            },
            {
                "text": "document_id: doc_222abc\nMain text.",
                "source_document_id": "doc_222abc",
                "source_dataset_name": "dataset_2",
                "source_cognee_chunk_id": "right_chunk",
            },
        ],
    )

    assert result.items[0].source_document_id == "doc_222abc"
    assert result.items[0].source_chunk_id == "cognee::dataset_2::right_chunk"


def test_parser_marks_unstructured_result_metadata() -> None:
    result = parse_cognee_result(
        query="question",
        raw_result={"answer": "Cognee answer only"},
        top_k=5,
        latency_ms=1.0,
    )

    assert result.items == []
    assert result.metadata["retrieval_trace_available"] is False
    assert result.metadata["retriever_purity"] == "mixed"
    assert result.metadata["raw_result_available"] is True
    assert result.metadata["generated_answer"] == "Cognee answer only"


def test_adapter_can_be_instantiated_with_fake_client(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    adapter = CogneeAdapter(workspace=workspace, client=FakeCogneeClient())

    result = adapter.retrieve(query="question", top_k=1)

    assert result.method_id == "cognee"
    assert len(result.items) == 1
    assert result.items[0].source_chunk_id == "chunk_1"
    assert result.metadata["generated_answer"] == "answer generated inside Cognee"


def test_adapter_retrieval_uses_explicit_dataset_list(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    client = FakeCogneeClient()
    adapter = CogneeAdapter(
        workspace=workspace,
        config=CogneeConfig(
            dataset_name="legacy",
            dataset_names=("musique_smoke_20_v1_batch_001", "musique_smoke_20_v1_batch_002"),
        ),
        client=client,
    )

    adapter.retrieve(query="question", top_k=1)

    assert client.recall_dataset_names == (
        "musique_smoke_20_v1_batch_001",
        "musique_smoke_20_v1_batch_002",
    )


def test_adapter_add_and_cognify_uses_lower_level_ingestion(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    adapter = CogneeAdapter(workspace=workspace, client=FakeCogneeClient())

    result = asyncio.run(adapter.add_and_cognify_async(["payload"]))

    assert result["primary_api"] == "add+cognify"
    assert result["add_result"]["status"] == "added"
    assert result["cognify_result"]["status"] == "cognified"


def test_adapter_add_and_cognify_forwards_batch_limits_to_cognify(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    client = FakeCogneeClient()
    adapter = CogneeAdapter(workspace=workspace, client=client)

    asyncio.run(
        adapter.add_and_cognify_async(
            ["payload"],
            data_per_batch=1,
            chunks_per_batch=1,
        )
    )

    assert client.add_kwargs["data_per_batch"] == 1
    assert client.cognify_kwargs["data_per_batch"] == 1
    assert client.cognify_kwargs["chunks_per_batch"] == 1


def test_parser_extracts_nodes_relationships_and_trace() -> None:
    result = parse_cognee_result(
        query="question",
        raw_result={
            "result": [
                {
                    "text": "contract context",
                    "score": 0.7,
                    "raw": {
                        "nodes": [{"id": "n1", "label": "Clause", "properties": {"document_id": "doc_1"}}],
                        "relationships": [{"source": "n1", "target": "n2", "type": "MENTIONS"}],
                    },
                }
            ],
            "raw_trace": {"entities": ["Clause"], "facts": ["fact"]},
        },
        top_k=5,
        latency_ms=1.0,
    )

    assert result.items[0].text == "contract context"
    assert result.metadata["node_trace_available"] is True
    assert result.metadata["relationship_trace_available"] is True
    assert result.metadata["cognee_nodes"][0]["id"] == "n1"
    assert result.metadata["cognee_relationships"][0]["type"] == "MENTIONS"


def test_trace_writer_persists_required_csvs(tmp_path: Path) -> None:
    result = parse_cognee_result(
        query="question",
        raw_result={
            "retrieval_items": [
                {
                    "text": "context",
                    "document_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.4,
                    "similarity": 0.6,
                }
            ],
            "nodes": [{"id": "n1", "label": "Clause", "score": 0.2}],
            "relationships": [{"source": "n1", "target": "n2", "type": "RELATED"}],
            "raw_trace": {"entities": ["e1"], "facts": ["f1"], "reasoning_inputs": ["r1"]},
        },
        top_k=5,
        latency_ms=1.0,
    )
    writer = CogneeTraceWriter(CogneeTracePaths.from_tables_dir(tmp_path))

    writer.write_result(run_id="run_1", question_id="q_1", result=result)

    assert (tmp_path / "cognee_retrieval_items.csv").read_text(encoding="utf-8").splitlines()[0].startswith("run_id,question_id")
    assert "chunk_1" in (tmp_path / "cognee_retrieval_items.csv").read_text(encoding="utf-8")
    assert "n1" in (tmp_path / "cognee_nodes.csv").read_text(encoding="utf-8")
    assert "RELATED" in (tmp_path / "cognee_relationships.csv").read_text(encoding="utf-8")
    assert "reasoning_inputs_json" in (tmp_path / "cognee_graph_trace.csv").read_text(encoding="utf-8")
