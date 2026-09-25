import json
from pathlib import Path

import pytest

from benchmark.core.schemas import Document
from benchmark.methods.ms_graphrag_neo4j.adapter import (
    GraphRAGNeo4jAdapter,
    GraphRAGNeo4jIndexResult,
    GraphRAGNeo4jQueryResult,
)
from benchmark.methods.ms_graphrag_neo4j.config_builder import build_workspace
from benchmark.methods.ms_graphrag_neo4j.indexer import index, load_canonical_documents
from benchmark.methods.ms_graphrag_neo4j.output_parser import parse_query_output
from benchmark.methods.ms_graphrag_neo4j.retriever import retrieve


def test_input_preparation_writes_documents_and_manifest(tmp_path: Path) -> None:
    adapter = GraphRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=FakeGraphRAGNeo4jClient(),
    )
    document = _document()

    paths = adapter.prepare_inputs([document])

    assert paths == [adapter.workspace.input_dir / "doc_1.txt"]
    assert paths[0].read_text(encoding="utf-8") == "Contract text"
    manifest = json.loads((adapter.workspace.input_dir / "manifest.json").read_text())
    assert manifest[0]["document_id"] == "doc_1"
    assert manifest[0]["dataset_id"] == "musique_smoke_20"


def test_fake_indexing_path_preserves_raw_output(tmp_path: Path) -> None:
    client = FakeGraphRAGNeo4jClient()
    adapter = GraphRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=client,
    )

    summary = index(adapter=adapter, documents=[_document()])

    assert summary.method_id == "ms_graphrag_neo4j"
    assert summary.document_count == 1
    assert client.index_calls[0]["config"].neo4j_uri_env == "GRAPHRAG_NEO4J_URI"
    assert (adapter.workspace.raw_dir / "index_result.json").exists()


def test_fake_retrieval_path_converts_and_persists(tmp_path: Path) -> None:
    adapter = GraphRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        client=FakeGraphRAGNeo4jClient(),
    )
    store = FakeExperimentStore()

    result = retrieve(
        adapter=adapter,
        query="governing law",
        top_k=1,
        experiment_store=store,
        run_id="run_1",
    )

    assert result.method_id == "ms_graphrag_neo4j"
    assert result.items[0].text == "Governing law is Delaware."
    assert result.items[0].source_document_id == "doc_1"
    assert result.items[0].source_chunk_id == "chunk_1"
    assert store.calls[0]["result"].method_id == "ms_graphrag_neo4j"
    assert store.calls[0]["run_id"] == "run_1"
    assert (adapter.workspace.raw_dir / "query_result.json").exists()


def test_output_parser_handles_plain_text_response() -> None:
    result = parse_query_output(
        query="governing law",
        raw_response="Delaware law.",
        latency_ms=1.0,
        top_k=5,
    )

    assert result.method_id == "ms_graphrag_neo4j"
    assert result.items[0].text == "Delaware law."
    assert result.items[0].metadata["kind"] == "answer"


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


def test_default_adapter_fails_with_actionable_missing_dependency_error(tmp_path: Path) -> None:
    adapter = GraphRAGNeo4jAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        )
    )

    with pytest.raises(RuntimeError, match="live execution is not configured"):
        adapter.index(documents=[_document()])


def _document(document_id: str = "doc_1") -> Document:
    return Document(
        document_id=document_id,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Contract text",
    )


class FakeGraphRAGNeo4jClient:
    def __init__(self) -> None:
        self.index_calls = []
        self.query_calls = []

    def index(self, *, workspace, config, documents):
        self.index_calls.append({"workspace": workspace, "config": config, "documents": documents})
        return GraphRAGNeo4jIndexResult(
            document_count=len(documents),
            raw_response={"status": "ok", "documents": len(documents)},
        )

    def query(self, *, workspace, config, query: str, top_k: int):
        self.query_calls.append(
            {"workspace": workspace, "config": config, "query": query, "top_k": top_k}
        )
        return GraphRAGNeo4jQueryResult(
            query=query,
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


class FakeExperimentStore:
    def __init__(self) -> None:
        self.calls = []

    def persist_retrieval_result(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return "retrieval_1"
