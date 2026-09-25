from pathlib import Path

from benchmark.methods.lightrag.adapter import (
    LightRAGAdapter,
    LightRAGIngestResult,
    LightRAGQueryResult,
)
from benchmark.methods.lightrag.config_builder import build_workspace
from benchmark.methods.lightrag.retriever import retrieve


def test_lightrag_query_uses_runner_and_persists(tmp_path: Path) -> None:
    runner = FakeLightRAGRunner()
    adapter = LightRAGAdapter(
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
        query_mode="mix",
        top_k=1,
        experiment_store=store,
        run_id="run_1",
    )

    assert result.method_id == "lightrag"
    assert result.items[0].source_chunk_id == "chunk_1"
    assert runner.calls == [("governing law", "mix", 1)]
    assert store.calls[0]["result"].method_id == "lightrag"
    assert (adapter.workspace.raw_dir / "query_mix_result.json").exists()


class FakeLightRAGRunner:
    def __init__(self) -> None:
        self.calls = []

    def ingest(self, *, workspace, documents):
        return LightRAGIngestResult(document_count=len(documents), raw_response={"status": "ok"})

    def query(self, *, workspace, query: str, mode: str, top_k: int):
        self.calls.append((query, mode, top_k))
        return LightRAGQueryResult(
            query=query,
            mode=mode,
            raw_response={
                "answer": "Delaware law.",
                "contexts": [
                    {
                        "text": "Governing law is Delaware.",
                        "chunk_id": "chunk_1",
                        "document_id": "doc_1",
                        "score": 0.8,
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
