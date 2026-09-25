from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from benchmark.core.config_loader import DatasetConfig
from benchmark.core.ids import deterministic_id
from benchmark.core.schemas import Document, GoldEvidence, Question
from benchmark.ingestion.canonical_export import CanonicalDataset
from benchmark.ingestion.chunking import chunk_document

RAW_FILE_CANDIDATES = (
    "musique_ans_v1.0_dev.jsonl",
    "musique_ans_v1.0_train.jsonl",
    "musique_ans_v1.0_test.jsonl",
)

MUSIQUE_QUESTION_TYPE = "multi_hop_reasoning"


class MusiqueRawDataMissingError(FileNotFoundError):
    pass


def load_musique(
    config: DatasetConfig,
    *,
    sample_size: int | None = None,
    raw_file: str | Path | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> CanonicalDataset:
    """Load MuSiQue raw JSONL into the canonical dataset format.

    Passages are deduplicated by ``(title, paragraph_text)`` via
    ``deterministic_id``: two questions referencing the same passage produce
    one Document. Each Question carries ``document_id=None`` (multi-doc by
    construction) and points to the supporting passages through
    ``gold_evidence``.

    Sampling is driven either by the explicit ``sample_size`` argument or by
    ``config.sample.num_questions`` in the YAML config. When both are absent
    every question in the raw file is loaded.
    """
    resolved_raw_file = (
        Path(raw_file) if raw_file is not None else find_musique_raw_file(config.raw_path)
    )
    raw_questions = _read_jsonl(resolved_raw_file)

    selected_questions = _apply_sampling(
        raw_questions,
        sample_size=sample_size,
        sample_config=_resolve_sample_config(config),
    )

    documents_by_id: dict[str, Document] = {}
    questions: list[Question] = []
    evidence: list[GoldEvidence] = []

    for raw_question in selected_questions:
        if not isinstance(raw_question, dict):
            continue

        question_text = str(raw_question.get("question") or "").strip()
        if not question_text:
            continue

        source_question_id = raw_question.get("id")
        question_id = deterministic_id(
            "q",
            [
                config.dataset_id,
                config.dataset_version,
                source_question_id,
                question_text,
            ],
        )

        question_paragraphs = _collect_paragraphs(
            raw_question.get("paragraphs"),
            config=config,
            source_path=resolved_raw_file,
            documents_by_id=documents_by_id,
        )

        question_evidence = _build_question_evidence(
            question_id=question_id,
            paragraphs=question_paragraphs,
        )

        question = _build_question(
            config=config,
            question_id=question_id,
            question_text=question_text,
            source_question_id=source_question_id,
            raw_question=raw_question,
            evidence_ids=[item.evidence_id for item in question_evidence],
        )

        questions.append(question)
        evidence.extend(question_evidence)

    documents = list(documents_by_id.values())
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]

    return CanonicalDataset(
        documents=documents,
        chunks=chunks,
        questions=questions,
        gold_evidence=evidence,
    )


def find_musique_raw_file(raw_path: str | Path) -> Path:
    root = Path(raw_path)
    for candidate in RAW_FILE_CANDIDATES:
        path = root / candidate
        if path.exists():
            return path

    expected = ", ".join(str(root / candidate) for candidate in RAW_FILE_CANDIDATES)
    raise MusiqueRawDataMissingError(
        "MuSiQue raw data was not found. Download is intentionally not automatic. "
        f"Place a MuSiQue JSONL file at one of: {expected}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _resolve_sample_config(config: DatasetConfig) -> dict[str, Any]:
    raw = getattr(config, "sample", None)
    if isinstance(raw, dict):
        return raw
    return {}


def _apply_sampling(
    raw_questions: list[dict[str, Any]],
    *,
    sample_size: int | None,
    sample_config: dict[str, Any],
) -> list[dict[str, Any]]:
    num_questions = sample_size
    if num_questions is None:
        candidate = sample_config.get("num_questions")
        if isinstance(candidate, int):
            num_questions = candidate

    if num_questions is None or num_questions >= len(raw_questions):
        return list(raw_questions)
    if num_questions <= 0:
        return []

    seed_value = sample_config.get("seed", 42)
    if not isinstance(seed_value, int):
        seed_value = 42

    rng = random.Random(seed_value)
    return rng.sample(raw_questions, num_questions)


def _collect_paragraphs(
    raw_paragraphs: Any,
    *,
    config: DatasetConfig,
    source_path: Path,
    documents_by_id: dict[str, Document],
) -> list[tuple[Document, dict[str, Any]]]:
    if not isinstance(raw_paragraphs, list):
        return []

    collected: list[tuple[Document, dict[str, Any]]] = []
    for raw_paragraph in raw_paragraphs:
        if not isinstance(raw_paragraph, dict):
            continue

        title = str(raw_paragraph.get("title") or "").strip()
        paragraph_text = str(raw_paragraph.get("paragraph_text") or "")
        if not title and not paragraph_text:
            continue

        document_id = deterministic_id(
            "doc",
            [config.dataset_id, config.dataset_version, title, paragraph_text],
        )

        document = documents_by_id.get(document_id)
        if document is None:
            passage = f"{title}\n\n{paragraph_text}" if title else paragraph_text
            document = Document(
                document_id=document_id,
                dataset_id=config.dataset_id,
                dataset_version=config.dataset_version,
                title=title or None,
                text=passage,
                source_path=str(source_path),
                metadata={
                    "paragraph_text": paragraph_text,
                    "musique_idx": raw_paragraph.get("idx"),
                },
            )
            documents_by_id[document_id] = document

        collected.append((document, raw_paragraph))

    return collected


def _build_question_evidence(
    *,
    question_id: str,
    paragraphs: list[tuple[Document, dict[str, Any]]],
) -> list[GoldEvidence]:
    evidence: list[GoldEvidence] = []
    seen: set[str] = set()

    for document, raw_paragraph in paragraphs:
        if not raw_paragraph.get("is_supporting"):
            continue
        if document.document_id in seen:
            continue
        seen.add(document.document_id)

        evidence_id = deterministic_id("ev", [question_id, document.document_id])
        evidence.append(
            GoldEvidence(
                evidence_id=evidence_id,
                question_id=question_id,
                document_id=document.document_id,
                text=str(raw_paragraph.get("paragraph_text") or ""),
                start_char=None,
                end_char=None,
                metadata={"musique_idx": raw_paragraph.get("idx")},
            )
        )

    return evidence


def _build_question(
    *,
    config: DatasetConfig,
    question_id: str,
    question_text: str,
    source_question_id: Any,
    raw_question: dict[str, Any],
    evidence_ids: list[str],
) -> Question:
    answer = raw_question.get("answer")
    raw_aliases = raw_question.get("answer_aliases")
    answer_aliases = list(raw_aliases) if isinstance(raw_aliases, list) else []
    raw_decomposition = raw_question.get("question_decomposition")
    decomposition = (
        list(raw_decomposition) if isinstance(raw_decomposition, list) else []
    )

    return Question(
        question_id=question_id,
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        document_id=None,
        question=question_text,
        question_type=MUSIQUE_QUESTION_TYPE,
        clause_type=None,
        gold_answer=_normalize_gold_answer(answer),
        gold_evidence=evidence_ids or None,
        metadata={
            "source_question_id": source_question_id,
            "answer_aliases": answer_aliases,
            "decomposition": decomposition,
        },
    )


def _normalize_gold_answer(answer: Any) -> str | list[str] | bool | None:
    if isinstance(answer, (str, bool)) or answer is None:
        return answer
    if isinstance(answer, list):
        return [str(value) for value in answer]
    return str(answer)
