from pathlib import Path

from typer.testing import CliRunner

from benchmark.cli import app as cli_app
from benchmark.methods.lightrag.indexer import LightRAGIndexSummary


def test_cli_routes_lightrag_index(monkeypatch, tmp_path: Path) -> None:
    calls = []

    monkeypatch.setattr(cli_app, "load_lightrag_documents", lambda path: ["doc"])
    monkeypatch.setattr(
        cli_app,
        "_lightrag_adapter",
        lambda dataset_id, dataset_version: "adapter",
    )

    def fake_index_lightrag(*, adapter, documents, query_mode, top_k):
        calls.append((adapter, documents, query_mode, top_k))
        return LightRAGIndexSummary(
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            method_id="lightrag",
            workspace_dir=tmp_path / "workspace",
            document_count=1,
        )

    monkeypatch.setattr(cli_app, "index_lightrag", fake_index_lightrag)

    result = CliRunner().invoke(
        cli_app.app,
        ["index", "--dataset", "musique_smoke_20", "--version", "v1", "--method", "lightrag"],
    )

    assert result.exit_code == 0
    assert calls == [("adapter", ["doc"], "mix", 5)]
    assert "lightrag" in result.output


def test_cli_routes_lightrag_retrieve(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        cli_app,
        "_lightrag_adapter",
        lambda dataset_id, dataset_version: "adapter",
    )

    def fake_retrieve_lightrag(**kwargs):
        calls.append(kwargs)
        from benchmark.core.schemas import RetrievalResult

        return RetrievalResult(method_id="lightrag", query=kwargs["query"], latency_ms=1.0)

    monkeypatch.setattr(cli_app, "retrieve_lightrag", fake_retrieve_lightrag)

    result = CliRunner().invoke(
        cli_app.app,
        [
            "retrieve",
            "--dataset",
            "musique_smoke_20",
            "--version",
            "v1",
            "--method",
            "lightrag",
            "--query",
            "governing law",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["adapter"] == "adapter"
    assert calls[0]["query_mode"] == "mix"
    assert "retrieved 0 chunks" in result.output
