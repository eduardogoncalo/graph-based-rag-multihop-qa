from __future__ import annotations

import json
from pathlib import Path

from benchmark.core.schemas import Document
from benchmark.methods.ms_graphrag.adapter import GraphRAGIndexSummary, MicrosoftGraphRAGAdapter


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
    adapter: MicrosoftGraphRAGAdapter,
    documents: list[Document],
    index_method: str = "standard",
    initialize: bool = True,
) -> GraphRAGIndexSummary:
    adapter.prepare_inputs(documents)
    if initialize:
        adapter.initialize_workspace(force=True)
    adapter.index(method=index_method)
    workspace = adapter.workspace
    return GraphRAGIndexSummary(
        dataset_id=workspace.dataset_id,
        dataset_version=workspace.dataset_version,
        method_id=workspace.method_id,
        workspace_dir=workspace.workspace_dir,
        document_count=len(documents),
    )
