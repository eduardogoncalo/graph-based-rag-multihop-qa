import json
from pathlib import Path

from benchmark.core.schemas import Document
from benchmark.methods.ms_graphrag.adapter import GraphRAGCommandResult, MicrosoftGraphRAGAdapter
from benchmark.methods.ms_graphrag.config_builder import build_workspace
from benchmark.methods.ms_graphrag.indexer import index, load_canonical_documents


def test_ms_graphrag_input_file_preparation(tmp_path: Path) -> None:
    adapter = MicrosoftGraphRAGAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        runner=FakeGraphRAGRunner(),
    )
    document = Document(
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Contract text",
    )

    paths = adapter.prepare_inputs([document])

    assert paths == [adapter.workspace.input_dir / "documents.csv"]
    csv_text = paths[0].read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == '"id","title","text"'
    assert '"doc_1","doc_1"' in csv_text
    assert "Contract text" in csv_text


def test_ms_graphrag_index_uses_official_cli_commands(tmp_path: Path) -> None:
    runner = FakeGraphRAGRunner()
    adapter = MicrosoftGraphRAGAdapter(
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

    summary = index(adapter=adapter, documents=[document], index_method="fast")

    assert summary.method_id == "ms_graphrag"
    assert summary.document_count == 1
    assert runner.calls[0][:2] == ["graphrag", "init"]
    assert runner.calls[1][:2] == ["graphrag", "index"]
    assert "--root" in runner.calls[1]
    assert "--method" in runner.calls[1]
    assert "fast" in runner.calls[1]
    assert (adapter.workspace.raw_dir / "index_stdout.txt").exists()


def test_load_canonical_documents_reads_jsonl(tmp_path: Path) -> None:
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


class FakeGraphRAGRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, args: list[str], *, cwd: Path | None = None) -> GraphRAGCommandResult:
        self.calls.append(args)
        return GraphRAGCommandResult(args=args, returncode=0, stdout="ok", stderr="")
