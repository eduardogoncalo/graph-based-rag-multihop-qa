from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.core.schemas import Document
from benchmark.methods.lightrag.adapter import LightRAGAdapter


@dataclass(frozen=True)
class LightRAGIndexSummary:
    dataset_id: str
    dataset_version: str
    method_id: str
    workspace_dir: Path
    document_count: int


def load_canonical_documents(canonical_path: str | Path) -> list[Document]:
    path = Path(canonical_path) / "documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical documents not found at {path}. Run benchmark ingest before indexing."
        )
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(Document.model_validate(json.loads(line)))
    return documents


def index(
    *,
    adapter: LightRAGAdapter,
    documents: list[Document],
    query_mode: str = "mix",
    top_k: int = 5,
) -> LightRAGIndexSummary:
    result = adapter.ingest(documents=documents, query_mode=query_mode, top_k=top_k)
    workspace = adapter.workspace
    return LightRAGIndexSummary(
        dataset_id=workspace.dataset_id,
        dataset_version=workspace.dataset_version,
        method_id=workspace.method_id,
        workspace_dir=workspace.workspace_dir,
        document_count=result.document_count,
    )
