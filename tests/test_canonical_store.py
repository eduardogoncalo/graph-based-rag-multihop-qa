import json
from pathlib import Path

from benchmark.core.config_loader import DatasetConfig
from benchmark.core.schemas import Chunk, Document, GoldEvidence, Question
from benchmark.storage.canonical_store import CanonicalStore
from tests.fakes import FakeConnection


def test_canonical_store_persists_jsonl_fixtures(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical" / "musique_smoke_20_v1"
    canonical_path.mkdir(parents=True)
    document = Document(
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        title="Contract",
        text="Governing law is Delaware.",
    )
    chunk = Chunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Governing law is Delaware.",
    )
    question = Question(
        question_id="q_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        document_id="doc_1",
        question="What is the governing law?",
        question_type="span_extraction",
        gold_answer="Delaware",
        gold_evidence=["ev_1"],
    )
    evidence = GoldEvidence(
        evidence_id="ev_1",
        question_id="q_1",
        document_id="doc_1",
        text="Delaware",
    )
    _write_jsonl(canonical_path / "documents.jsonl", [document])
    _write_jsonl(canonical_path / "chunks.jsonl", [chunk])
    _write_jsonl(canonical_path / "questions.jsonl", [question])
    _write_jsonl(canonical_path / "gold_evidence.jsonl", [evidence])

    connection = FakeConnection()
    summary = CanonicalStore(connection).persist_from_path(
        config=DatasetConfig(
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            name="MuSiQue (amostra de smoke)",
            raw_path="raw",
            canonical_path=str(canonical_path),
            splits_path="splits",
        )
    )

    assert summary.documents == 1
    assert summary.chunks == 1
    assert summary.questions == 1
    assert summary.gold_evidence == 1
    assert len(connection.cursor_obj.executed) == 5
    assert connection.commits == 1


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )
