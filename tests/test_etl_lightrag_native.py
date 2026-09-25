from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "etl_lightrag_native", Path(__file__).resolve().parents[1] / "scripts" / "etl_lightrag_native.py"
)
etl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(etl)


class FakeStore:
    def __init__(self) -> None:
        self.experiments: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self.answers: dict[str, dict] = {}

    def create_experiment(self, *, experiment_id, **kwargs):
        self.experiments[experiment_id] = kwargs
        return experiment_id

    def create_run(self, *, run_id, **kwargs):
        self.runs[run_id] = kwargs
        return run_id

    def persist_answer(self, *, run_id, **kwargs):
        self.answers[run_id] = kwargs
        return run_id


ROWS = [
    ("q_a", "run_src_a", "rr_a", "Paris"),
    ("q_b", "run_src_b", "rr_b", "The answer is 1839."),
]


def _materialize(store: FakeStore, rows=ROWS) -> int:
    return etl.materialize(
        store,
        rows,
        source_experiment="musique_eval1k_lightrag_v1free",
        target_experiment="musique_eval1k_lightrag_native",
        dataset_id="musique",
        dataset_version="ans_v1.0_eval1k",
    )


def test_copies_one_native_answer_per_question() -> None:
    store = FakeStore()
    assert _materialize(store) == 2
    assert set(store.experiments) == {"musique_eval1k_lightrag_native"}
    texts = sorted(answer["answer_text"] for answer in store.answers.values())
    assert texts == ["Paris", "The answer is 1839."]
    for answer in store.answers.values():
        assert answer["method_id"] == "lightrag_neo4j"
        assert answer["metadata"]["source_experiment_id"] == "musique_eval1k_lightrag_v1free"


def test_run_ids_are_deterministic_so_reruns_upsert() -> None:
    first, second = FakeStore(), FakeStore()
    _materialize(first)
    _materialize(second)
    assert set(first.runs) == set(second.runs)
    assert len(first.runs) == 2


def test_refuses_empty_native_answers_and_writes_nothing() -> None:
    store = FakeStore()
    with pytest.raises(SystemExit, match="empty native answers"):
        _materialize(store, ROWS + [("q_c", "run_src_c", "rr_c", "  ")])
    assert not store.runs and not store.answers and not store.experiments


def test_refuses_an_empty_source() -> None:
    with pytest.raises(SystemExit, match="Run the controlled LightRAG cell first"):
        _materialize(FakeStore(), [])


def test_default_prefix_matches_the_other_protocol_scripts() -> None:
    assert etl.default_prefix("twowiki") == "twowiki_eval1k_"
    assert etl.default_prefix("musique_smoke_20") == "musique_smoke_20_eval1k_"
