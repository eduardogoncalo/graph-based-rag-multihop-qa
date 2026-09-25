import json
from pathlib import Path

from benchmark.methods.ms_graphrag.adapter import GraphRAGCommandResult, MicrosoftGraphRAGAdapter
from benchmark.methods.ms_graphrag.config_builder import build_workspace
from benchmark.methods.ms_graphrag.retriever import retrieve


def test_ms_graphrag_query_uses_official_cli_and_persists(tmp_path: Path) -> None:
    runner = FakeGraphRAGRunner()
    adapter = MicrosoftGraphRAGAdapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
        ),
        runner=runner,
    )
    store = FakeExperimentStore()

    result = retrieve(
        adapter=adapter,
        query="governing law",
        query_method="local",
        top_k=1,
        experiment_store=store,
        run_id="run_1",
    )

    assert result.method_id == "ms_graphrag"
    assert result.items[0].source_chunk_id == "42"
    assert result.items[0].source_document_id == "doc_6358038e054ac768"
    assert result.items[0].metadata["document_ids"] == ["doc_6358038e054ac768"]
    # retrieval-only: invokes the venv runner script, not `graphrag query`
    assert runner.calls[0][1].endswith("graphrag_query_runner.py")
    assert "local" in runner.calls[0]
    assert store.calls[0]["result"].method_id == "ms_graphrag"
    assert (adapter.workspace.raw_dir / "context_local_stdout.txt").exists()


class FakeGraphRAGRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, args: list[str], *, cwd: Path | None = None) -> GraphRAGCommandResult:
        self.calls.append(args)
        stdout = json.dumps(
            {
                "context_text": "-----Sources-----\nGoverning law is Delaware.",
                "items": [
                    {
                        "unit_id": "b2c-uuid",
                        "unit_short_id": "42",
                        "document_id": "doc_6358038e054ac768",
                        "title": "doc_6358038e054ac768",
                        "text": "Governing law is Delaware.",
                        "in_context": True,
                    }
                ],
                "stats": {"n_items": 1, "n_in_context": 1},
            }
        )
        return GraphRAGCommandResult(args=args, returncode=0, stdout=stdout, stderr="")


class FakeExperimentStore:
    def __init__(self) -> None:
        self.calls = []

    def persist_retrieval_result(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return "retrieval_1"
