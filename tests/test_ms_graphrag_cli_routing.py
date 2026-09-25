from pathlib import Path

from typer.testing import CliRunner

from benchmark.cli import app as cli_app
from benchmark.methods.ms_graphrag.adapter import GraphRAGIndexSummary


def test_cli_routes_ms_graphrag_index(monkeypatch, tmp_path: Path) -> None:
    calls = []

    monkeypatch.setattr(
        cli_app,
        "load_canonical_documents",
        lambda path: ["doc"],
    )
    monkeypatch.setattr(
        cli_app,
        "_ms_graphrag_adapter",
        lambda dataset_id, dataset_version: "adapter",
    )

    def fake_index_ms_graphrag(*, adapter, documents, index_method):
        calls.append((adapter, documents, index_method))
        return GraphRAGIndexSummary(
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            method_id="ms_graphrag",
            workspace_dir=tmp_path / "workspace",
            document_count=1,
        )

    monkeypatch.setattr(cli_app, "index_ms_graphrag", fake_index_ms_graphrag)

    result = CliRunner().invoke(
        cli_app.app,
        ["index", "--dataset", "musique_smoke_20", "--version", "v1", "--method", "ms_graphrag"],
    )

    assert result.exit_code == 0
    assert calls == [("adapter", ["doc"], "standard")]
    assert "ms_graphrag" in result.output


def test_cli_routes_ms_graphrag_retrieve(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        cli_app,
        "_ms_graphrag_adapter",
        lambda dataset_id, dataset_version: "adapter",
    )

    def fake_retrieve_ms_graphrag(**kwargs):
        calls.append(kwargs)
        from benchmark.core.schemas import RetrievalResult

        return RetrievalResult(method_id="ms_graphrag", query=kwargs["query"], latency_ms=1.0)

    monkeypatch.setattr(cli_app, "retrieve_ms_graphrag", fake_retrieve_ms_graphrag)

    result = CliRunner().invoke(
        cli_app.app,
        [
            "retrieve",
            "--dataset",
            "musique_smoke_20",
            "--version",
            "v1",
            "--method",
            "ms_graphrag",
            "--query",
            "governing law",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["adapter"] == "adapter"
    assert calls[0]["query_method"] == "local"
    assert "retrieved 0 chunks" in result.output
