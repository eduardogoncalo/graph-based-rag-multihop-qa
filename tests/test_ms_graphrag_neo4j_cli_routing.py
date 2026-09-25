from pathlib import Path

from typer.testing import CliRunner

from benchmark.cli import app as cli_app
from benchmark.methods.ms_graphrag_neo4j.indexer import GraphRAGNeo4jIndexSummary


def test_cli_routes_ms_graphrag_neo4j_index(monkeypatch, tmp_path: Path) -> None:
    calls = []

    monkeypatch.setattr(cli_app, "load_graphrag_neo4j_documents", lambda path, **kwargs: ["doc"])
    monkeypatch.setattr(
        cli_app,
        "_ms_graphrag_neo4j_adapter",
        lambda dataset_id, dataset_version, method_config: "adapter",
    )

    def fake_index_ms_graphrag_neo4j(*, adapter, documents):
        calls.append((adapter, documents))
        return GraphRAGNeo4jIndexSummary(
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            method_id="ms_graphrag_neo4j",
            workspace_dir=tmp_path / "workspace",
            document_count=1,
        )

    monkeypatch.setattr(cli_app, "index_ms_graphrag_neo4j", fake_index_ms_graphrag_neo4j)

    result = CliRunner().invoke(
        cli_app.app,
        ["index", "--dataset", "musique_smoke_20", "--version", "v1", "--method", "ms_graphrag_neo4j"],
    )

    assert result.exit_code == 0
    assert calls == [("adapter", ["doc"])]
    assert "ms_graphrag_neo4j" in result.output


def test_cli_routes_ms_graphrag_neo4j_retrieve(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        cli_app,
        "_ms_graphrag_neo4j_adapter",
        lambda dataset_id, dataset_version, method_config: "adapter",
    )

    def fake_retrieve_ms_graphrag_neo4j(**kwargs):
        calls.append(kwargs)
        from benchmark.core.schemas import RetrievalResult

        return RetrievalResult(
            method_id="ms_graphrag_neo4j",
            query=kwargs["query"],
            latency_ms=1.0,
        )

    monkeypatch.setattr(
        cli_app,
        "retrieve_ms_graphrag_neo4j",
        fake_retrieve_ms_graphrag_neo4j,
    )

    result = CliRunner().invoke(
        cli_app.app,
        [
            "retrieve",
            "--dataset",
            "musique_smoke_20",
            "--version",
            "v1",
            "--method",
            "ms_graphrag_neo4j",
            "--query",
            "governing law",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["adapter"] == "adapter"
    assert calls[0]["top_k"] == 5
    assert "retrieved 0 chunks" in result.output
