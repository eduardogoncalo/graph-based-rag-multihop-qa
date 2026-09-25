from pathlib import Path

import pytest

from benchmark.methods.hipporag2 import Hipporag2Adapter, build_workspace
from benchmark.methods.hipporag2.retriever import retrieve


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def query(self, *, query: str, top_k: int) -> dict:
        self.calls.append({"query": query, "top_k": top_k})
        return self.payload


class FakeStore:
    def __init__(self) -> None:
        self.persisted: list[dict] = []

    def persist_retrieval_result(self, **kwargs) -> None:
        self.persisted.append(kwargs)


def _adapter(tmp_path: Path, payload: dict) -> tuple[Hipporag2Adapter, FakeClient]:
    client = FakeClient(payload)
    adapter = Hipporag2Adapter(
        workspace=build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique",
            dataset_version="ans_v1.0_eval1k",
        ),
        client=client,
    )
    return adapter, client


def test_retrieve_returns_parsed_result_and_persists(tmp_path: Path) -> None:
    payload = {
        "items": [{"rank": 1, "text": "t", "score": 0.5, "document_id": "doc_abcdef012345"}]
    }
    adapter, client = _adapter(tmp_path, payload)
    store = FakeStore()

    result = retrieve(
        adapter=adapter, query="who?", top_k=3, experiment_store=store, run_id="run_1"
    )

    assert client.calls == [{"query": "who?", "top_k": 3}]
    assert result.items[0].source_document_id == "doc_abcdef012345"
    assert len(store.persisted) == 1
    persisted = store.persisted[0]
    assert persisted["dataset_id"] == "musique"
    assert persisted["run_id"] == "run_1"
    assert persisted["top_k"] == 3


def test_adapter_surfaces_runner_error(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path, {"error": "IndexError: boom"})
    with pytest.raises(RuntimeError, match="boom"):
        adapter.query_context(query="q", top_k=5)


def test_retrieve_without_store_does_not_require_db(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path, {"items": []})
    result = retrieve(adapter=adapter, query="q", top_k=5)
    assert result.items == []
