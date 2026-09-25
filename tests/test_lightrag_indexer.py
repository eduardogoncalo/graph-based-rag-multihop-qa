import json
from pathlib import Path

from benchmark.core.schemas import Document
from benchmark.methods.lightrag.adapter import (
    LightRAGAdapter,
    LightRAGIngestResult,
    LightRAGQueryResult,
)
from benchmark.methods.lightrag.config_builder import build_workspace
from benchmark.methods.lightrag.indexer import index, load_canonical_documents


def test_lightrag_input_file_preparation(tmp_path: Path) -> None:
    adapter = LightRAGAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        runner=FakeLightRAGRunner(),
    )
    document = Document(
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Contract\t\ttext ¶ with   spacing",
    )

    paths = adapter.prepare_inputs([document])

    assert paths == [adapter.workspace.input_dir / "doc_1.txt"]
    assert paths[0].read_text(encoding="utf-8") == "Contract text with spacing"
    assert document.text == "Contract\t\ttext ¶ with   spacing"


def test_lightrag_index_uses_runner_without_real_indexing(tmp_path: Path) -> None:
    runner = FakeLightRAGRunner()
    adapter = LightRAGAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        runner=runner,
    )
    document = Document(
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Contract text",
    )

    summary = index(adapter=adapter, documents=[document], query_mode="mix", top_k=5)

    assert summary.method_id == "lightrag"
    assert summary.document_count == 1
    assert runner.ingested_ids == ["doc_1"]
    assert runner.ingested_texts == ["Contract text"]
    assert (adapter.workspace.raw_dir / "ingest_result.json").exists()
    assert (adapter.workspace.workspace_dir / "config.json").exists()


def test_load_lightrag_canonical_documents_reads_jsonl(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    document = Document(
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Contract text",
    )
    (canonical / "documents.jsonl").write_text(
        json.dumps(document.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    assert load_canonical_documents(canonical) == [document]


class FakeLightRAGRunner:
    def __init__(self) -> None:
        self.ingested_ids: list[str] = []
        self.ingested_texts: list[str] = []

    def ingest(self, *, workspace, documents):
        self.ingested_ids = [document.document_id for document in documents]
        self.ingested_texts = [document.text for document in documents]
        return LightRAGIngestResult(
            document_count=len(documents),
            raw_response={"status": "ok", "document_count": len(documents)},
        )

    def query(self, *, workspace, query: str, mode: str, top_k: int):
        return LightRAGQueryResult(query=query, mode=mode, raw_response="answer")
