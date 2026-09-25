from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from benchmark.core.config_loader import DatasetConfig
from benchmark.core.schemas import Document, GoldEvidence, Question
from benchmark.ingestion.musique_loader import (
    MusiqueRawDataMissingError,
    find_musique_raw_file,
    load_musique,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "musique_sample.jsonl"


def _read_fixture_rows() -> list[dict]:
    rows = []
    with FIXTURE_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _expected_distinct_passages(rows: list[dict]) -> int:
    keys = {
        (p["title"], p["paragraph_text"])
        for row in rows
        for p in row["paragraphs"]
    }
    return len(keys)


def _expected_supporting_count(rows: list[dict]) -> int:
    return sum(1 for row in rows for p in row["paragraphs"] if p["is_supporting"])


def _make_config(tmp_path: Path) -> DatasetConfig:
    return DatasetConfig(
        dataset_id="musique",
        dataset_version="fixture_v0",
        name="MuSiQue fixture",
        raw_path=str(tmp_path / "raw" / "musique"),
        canonical_path=str(tmp_path / "canonical"),
        splits_path=str(tmp_path / "splits"),
    )


def test_load_musique_parses_jsonl_and_deduplicates_passages(tmp_path: Path) -> None:
    rows = _read_fixture_rows()
    config = _make_config(tmp_path)

    dataset = load_musique(config, raw_file=FIXTURE_PATH)

    assert len(dataset.questions) == len(rows)
    assert isinstance(dataset.documents[0], Document)
    assert isinstance(dataset.questions[0], Question)
    assert isinstance(dataset.gold_evidence[0], GoldEvidence)

    # Distinct (title, paragraph_text) pairs is the dedup target.
    expected_documents = _expected_distinct_passages(rows)
    assert len(dataset.documents) == expected_documents
    # Naive enumeration would have produced 28 documents; dedup collapses
    # the shared MIRROR_PROJECT (film) passage from Q4 and Q5.
    assert sum(len(row["paragraphs"]) for row in rows) - expected_documents == 1


def test_load_musique_questions_are_multi_hop_with_no_document_id(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_musique(config, raw_file=FIXTURE_PATH)

    for question in dataset.questions:
        assert question.document_id is None
        assert question.question_type == "multi_hop_reasoning"
        assert question.gold_evidence
        assert len(question.gold_evidence) >= 2


def test_load_musique_document_text_is_title_plus_passage(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_musique(config, raw_file=FIXTURE_PATH)

    documents_by_title = {doc.title: doc for doc in dataset.documents}
    syntheCorp_doc = documents_by_title["SyntheCorp"]

    assert syntheCorp_doc.text.startswith("SyntheCorp\n\n")
    assert "FIX-P-S02" in syntheCorp_doc.text
    assert syntheCorp_doc.metadata["paragraph_text"].startswith("FIX-P-S02")
    assert syntheCorp_doc.metadata["musique_idx"] == 1


def test_load_musique_evidence_count_matches_supporting_paragraphs(tmp_path: Path) -> None:
    rows = _read_fixture_rows()
    config = _make_config(tmp_path)

    dataset = load_musique(config, raw_file=FIXTURE_PATH)

    expected_evidence = _expected_supporting_count(rows)
    assert len(dataset.gold_evidence) == expected_evidence

    evidence_by_question: Counter[str] = Counter(
        item.question_id for item in dataset.gold_evidence
    )
    for question in dataset.questions:
        source_id = question.metadata["source_question_id"]
        original = next(row for row in rows if row["id"] == source_id)
        expected = sum(1 for p in original["paragraphs"] if p["is_supporting"])
        assert evidence_by_question[question.question_id] == expected


def test_load_musique_shared_paragraph_collapses_to_single_document(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_musique(config, raw_file=FIXTURE_PATH)

    mirror_docs = [doc for doc in dataset.documents if doc.title == "MIRROR_PROJECT (film)"]
    assert len(mirror_docs) == 1
    mirror_doc_id = mirror_docs[0].document_id

    q4 = next(
        q for q in dataset.questions
        if q.metadata["source_question_id"] == "fix__synth_q4_2hop_shared"
    )
    q5 = next(
        q for q in dataset.questions
        if q.metadata["source_question_id"] == "fix__synth_q5_2hop_aliases_shared"
    )

    q4_evidence = [
        item for item in dataset.gold_evidence if item.question_id == q4.question_id
    ]
    q5_evidence = [
        item for item in dataset.gold_evidence if item.question_id == q5.question_id
    ]

    q4_document_ids = {item.document_id for item in q4_evidence}
    q5_document_ids = {item.document_id for item in q5_evidence}

    assert mirror_doc_id in q4_document_ids
    assert mirror_doc_id in q5_document_ids
    # Evidence IDs differ across questions even though the document is shared.
    q4_evidence_ids = {item.evidence_id for item in q4_evidence}
    q5_evidence_ids = {item.evidence_id for item in q5_evidence}
    assert q4_evidence_ids.isdisjoint(q5_evidence_ids)


def test_load_musique_answer_aliases_preserved_in_metadata(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_musique(config, raw_file=FIXTURE_PATH)

    q5 = next(
        q for q in dataset.questions
        if q.metadata["source_question_id"] == "fix__synth_q5_2hop_aliases_shared"
    )
    assert q5.metadata["answer_aliases"] == ["Vermillion Bay", "Vermillion City", "VBay"]

    q1 = next(
        q for q in dataset.questions
        if q.metadata["source_question_id"] == "fix__synth_q1_2hop"
    )
    assert q1.metadata["answer_aliases"] == []


def test_load_musique_decomposition_preserved_in_metadata(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_musique(config, raw_file=FIXTURE_PATH)

    q2 = next(
        q for q in dataset.questions
        if q.metadata["source_question_id"] == "fix__synth_q2_3hop"
    )
    decomposition = q2.metadata["decomposition"]
    assert len(decomposition) == 3
    assert decomposition[0]["question"] == "In which country was SYNTH_INVENTOR_BETA born?"
    assert decomposition[2]["answer"] == "Boreal"

    q3 = next(
        q for q in dataset.questions
        if q.metadata["source_question_id"] == "fix__synth_q3_4hop"
    )
    assert len(q3.metadata["decomposition"]) == 4


def test_load_musique_chunks_are_one_per_document_for_short_passages(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_musique(config, raw_file=FIXTURE_PATH)

    chunks_by_doc: Counter[str] = Counter(chunk.document_id for chunk in dataset.chunks)
    document_ids = {doc.document_id for doc in dataset.documents}

    assert chunks_by_doc.keys() == document_ids
    assert all(count == 1 for count in chunks_by_doc.values())


def test_load_musique_uses_deterministic_ids(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    first = load_musique(config, raw_file=FIXTURE_PATH)
    second = load_musique(config, raw_file=FIXTURE_PATH)

    assert [doc.document_id for doc in first.documents] == [
        doc.document_id for doc in second.documents
    ]
    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
    assert [q.question_id for q in first.questions] == [
        q.question_id for q in second.questions
    ]
    assert [ev.evidence_id for ev in first.gold_evidence] == [
        ev.evidence_id for ev in second.gold_evidence
    ]


def test_load_musique_sampling_is_seed_reproducible(tmp_path: Path) -> None:
    config = DatasetConfig(
        dataset_id="musique",
        dataset_version="fixture_v0",
        name="MuSiQue fixture",
        raw_path=str(tmp_path / "raw" / "musique"),
        canonical_path=str(tmp_path / "canonical"),
        splits_path=str(tmp_path / "splits"),
        sample={"num_questions": 2, "seed": 123},
    )

    first = load_musique(config, raw_file=FIXTURE_PATH)
    second = load_musique(config, raw_file=FIXTURE_PATH)

    assert len(first.questions) == 2
    assert [q.question_id for q in first.questions] == [
        q.question_id for q in second.questions
    ]

    # Different seed → different subset (with overwhelming probability for k=2 of 5).
    other_config = DatasetConfig(
        dataset_id="musique",
        dataset_version="fixture_v0",
        name="MuSiQue fixture",
        raw_path=str(tmp_path / "raw" / "musique"),
        canonical_path=str(tmp_path / "canonical"),
        splits_path=str(tmp_path / "splits"),
        sample={"num_questions": 2, "seed": 999},
    )
    other = load_musique(other_config, raw_file=FIXTURE_PATH)
    assert {q.question_id for q in first.questions} != {q.question_id for q in other.questions}


def test_load_musique_sample_size_argument_overrides_config(tmp_path: Path) -> None:
    config = DatasetConfig(
        dataset_id="musique",
        dataset_version="fixture_v0",
        name="MuSiQue fixture",
        raw_path=str(tmp_path / "raw" / "musique"),
        canonical_path=str(tmp_path / "canonical"),
        splits_path=str(tmp_path / "splits"),
        sample={"num_questions": 100, "seed": 42},
    )

    dataset = load_musique(config, raw_file=FIXTURE_PATH, sample_size=3)

    assert len(dataset.questions) == 3


def test_find_musique_raw_file_returns_first_known_candidate(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "musique"
    raw_dir.mkdir(parents=True)
    target = raw_dir / "musique_ans_v1.0_dev.jsonl"
    target.write_text("", encoding="utf-8")

    resolved = find_musique_raw_file(raw_dir)

    assert resolved == target


def test_load_musique_raises_when_raw_file_missing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "musique"
    raw_dir.mkdir(parents=True)
    config = DatasetConfig(
        dataset_id="musique",
        dataset_version="fixture_v0",
        name="MuSiQue fixture",
        raw_path=str(raw_dir),
        canonical_path=str(tmp_path / "canonical"),
        splits_path=str(tmp_path / "splits"),
    )

    with pytest.raises(MusiqueRawDataMissingError):
        load_musique(config)
