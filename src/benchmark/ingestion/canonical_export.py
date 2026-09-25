from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from benchmark.core.schemas import Chunk, Document, GoldEvidence, Question


@dataclass(frozen=True)
class CanonicalDataset:
    documents: list[Document]
    chunks: list[Chunk]
    questions: list[Question]
    gold_evidence: list[GoldEvidence]


def export_canonical_jsonl(dataset: CanonicalDataset, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "documents": output_path / "documents.jsonl",
        "chunks": output_path / "chunks.jsonl",
        "questions": output_path / "questions.jsonl",
        "gold_evidence": output_path / "gold_evidence.jsonl",
    }
    _write_jsonl(files["documents"], dataset.documents)
    _write_jsonl(files["chunks"], dataset.chunks)
    _write_jsonl(files["questions"], dataset.questions)
    _write_jsonl(files["gold_evidence"], dataset.gold_evidence)
    return files


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
