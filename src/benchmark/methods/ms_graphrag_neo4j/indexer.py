from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.core.schemas import Document
from benchmark.methods.ms_graphrag_neo4j.adapter import GraphRAGNeo4jAdapter


@dataclass(frozen=True)
class GraphRAGNeo4jIndexSummary:
    dataset_id: str
    dataset_version: str
    method_id: str
    workspace_dir: Path
    document_count: int


def load_canonical_documents(
    canonical_path: str | Path,
    *,
    max_documents: int | None = None,
) -> list[Document]:
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
                if max_documents is not None and len(documents) >= max_documents:
                    break
    return documents


def index(
    *,
    adapter: GraphRAGNeo4jAdapter,
    documents: list[Document],
) -> GraphRAGNeo4jIndexSummary:
    result = adapter.index(documents=documents)
    workspace = adapter.workspace
    return GraphRAGNeo4jIndexSummary(
        dataset_id=workspace.dataset_id,
        dataset_version=workspace.dataset_version,
        method_id=workspace.method_id,
        workspace_dir=workspace.workspace_dir,
        document_count=result.document_count,
    )
