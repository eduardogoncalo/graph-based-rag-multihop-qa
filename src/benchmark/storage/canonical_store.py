from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from benchmark.core.config_loader import DatasetConfig
from benchmark.core.schemas import Chunk, Document, GoldEvidence, Question

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class CanonicalPersistSummary:
    dataset_id: str
    dataset_version: str
    documents: int
    chunks: int
    questions: int
    gold_evidence: int


class CanonicalStore:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def persist_from_path(
        self,
        *,
        config: DatasetConfig,
        canonical_path: str | Path | None = None,
    ) -> CanonicalPersistSummary:
        root = Path(canonical_path or config.canonical_path)
        documents = _read_jsonl(root / "documents.jsonl", Document)
        chunks = _read_jsonl(root / "chunks.jsonl", Chunk)
        questions = _read_jsonl(root / "questions.jsonl", Question)
        evidence = _read_jsonl(root / "gold_evidence.jsonl", GoldEvidence)

        with self.connection.cursor() as cursor:
            self._upsert_dataset(cursor, config)
            for document in documents:
                self._upsert_document(cursor, document)
            for chunk in chunks:
                self._upsert_chunk(cursor, chunk)
            for question in questions:
                self._upsert_question(cursor, question)
            for item in evidence:
                self._upsert_gold_evidence(cursor, item)
        self.connection.commit()
        return CanonicalPersistSummary(
            dataset_id=config.dataset_id,
            dataset_version=config.dataset_version,
            documents=len(documents),
            chunks=len(chunks),
            questions=len(questions),
            gold_evidence=len(evidence),
        )

    def _upsert_dataset(self, cursor: Any, config: DatasetConfig) -> None:
        cursor.execute(
            """
            INSERT INTO datasets (dataset_id, dataset_version, name, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (dataset_id, dataset_version)
            DO UPDATE SET name = EXCLUDED.name, metadata = EXCLUDED.metadata, updated_at = now()
            """,
            (config.dataset_id, config.dataset_version, config.name, "{}"),
        )

    def _upsert_document(self, cursor: Any, document: Document) -> None:
        cursor.execute(
            """
            INSERT INTO documents (
                document_id, dataset_id, dataset_version, title, text, source_path, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (document_id)
            DO UPDATE SET
                title = EXCLUDED.title,
                text = EXCLUDED.text,
                source_path = EXCLUDED.source_path,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                document.document_id,
                document.dataset_id,
                document.dataset_version,
                document.title,
                document.text,
                document.source_path,
                _json(document.metadata),
            ),
        )

    def _upsert_chunk(self, cursor: Any, chunk: Chunk) -> None:
        cursor.execute(
            """
            INSERT INTO chunks (
                chunk_id,
                document_id,
                dataset_id,
                dataset_version,
                method_id,
                text,
                start_char,
                end_char,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (chunk_id)
            DO UPDATE SET
                text = EXCLUDED.text,
                start_char = EXCLUDED.start_char,
                end_char = EXCLUDED.end_char,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.dataset_id,
                chunk.dataset_version,
                chunk.method_id,
                chunk.text,
                chunk.start_char,
                chunk.end_char,
                _json(chunk.metadata),
            ),
        )

    def _upsert_question(self, cursor: Any, question: Question) -> None:
        cursor.execute(
            """
            INSERT INTO questions (
                question_id,
                dataset_id,
                dataset_version,
                document_id,
                question,
                question_type,
                clause_type,
                gold_answer,
                gold_evidence,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            ON CONFLICT (question_id)
            DO UPDATE SET
                question = EXCLUDED.question,
                clause_type = EXCLUDED.clause_type,
                gold_answer = EXCLUDED.gold_answer,
                gold_evidence = EXCLUDED.gold_evidence,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                question.question_id,
                question.dataset_id,
                question.dataset_version,
                question.document_id,
                question.question,
                question.question_type,
                question.clause_type,
                _json(question.gold_answer),
                _json(question.gold_evidence),
                _json(question.metadata),
            ),
        )

    def _upsert_gold_evidence(self, cursor: Any, evidence: GoldEvidence) -> None:
        cursor.execute(
            """
            INSERT INTO gold_evidence (
                evidence_id, question_id, document_id, text, start_char, end_char, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (evidence_id)
            DO UPDATE SET
                text = EXCLUDED.text,
                start_char = EXCLUDED.start_char,
                end_char = EXCLUDED.end_char,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                evidence.evidence_id,
                evidence.question_id,
                evidence.document_id,
                evidence.text,
                evidence.start_char,
                evidence.end_char,
                _json(evidence.metadata),
            ),
        )


def _read_jsonl(path: Path, model: type[T]) -> list[T]:
    if not path.exists():
        raise FileNotFoundError(f"Canonical file not found: {path}")
    rows: list[T] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(model.model_validate(json.loads(line)))
    return rows


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True)
