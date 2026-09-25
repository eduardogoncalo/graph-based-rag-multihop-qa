from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.core.schemas import RetrievalResult

RETRIEVAL_ITEMS_COLUMNS = [
    "run_id",
    "question_id",
    "document_id",
    "chunk_id",
    "source_chunk_id_strategy",
    "source_dataset_name",
    "source_cognee_chunk_id",
    "rank",
    "score",
    "similarity",
    "distance",
    "text",
    "metadata_json",
]

NODES_COLUMNS = [
    "run_id",
    "question_id",
    "node_id",
    "label",
    "properties_json",
    "score",
    "source_document_id",
    "source_chunk_id",
]

RELATIONSHIPS_COLUMNS = [
    "run_id",
    "question_id",
    "source",
    "target",
    "type",
    "properties_json",
    "score",
    "source_document_id",
    "source_chunk_id",
]

GRAPH_TRACE_COLUMNS = [
    "run_id",
    "question_id",
    "trace_stage",
    "entities_json",
    "facts_json",
    "memories_json",
    "context_json",
    "subgraph_json",
    "reasoning_inputs_json",
    "raw_trace_json",
]


@dataclass(frozen=True)
class CogneeTracePaths:
    retrieval_items: Path
    nodes: Path
    relationships: Path
    graph_trace: Path

    @classmethod
    def from_tables_dir(cls, tables_dir: str | Path, *, prefix: str = "cognee") -> "CogneeTracePaths":
        root = Path(tables_dir)
        return cls(
            retrieval_items=root / f"{prefix}_retrieval_items.csv",
            nodes=root / f"{prefix}_nodes.csv",
            relationships=root / f"{prefix}_relationships.csv",
            graph_trace=root / f"{prefix}_graph_trace.csv",
        )


class CogneeTraceWriter:
    def __init__(self, paths: CogneeTracePaths) -> None:
        self.paths = paths

    def write_result(
        self,
        *,
        run_id: str,
        question_id: str,
        result: RetrievalResult,
    ) -> None:
        self.write_retrieval_items(run_id=run_id, question_id=question_id, result=result)
        self.write_nodes(run_id=run_id, question_id=question_id, result=result)
        self.write_relationships(run_id=run_id, question_id=question_id, result=result)
        self.write_graph_trace(run_id=run_id, question_id=question_id, result=result)

    def write_retrieval_items(
        self,
        *,
        run_id: str,
        question_id: str,
        result: RetrievalResult,
    ) -> None:
        rows = []
        for rank, item in enumerate(result.items, start=1):
            rows.append(
                {
                    "run_id": run_id,
                    "question_id": question_id,
                    "document_id": item.source_document_id or "",
                    "chunk_id": item.source_chunk_id or "",
                    "source_chunk_id_strategy": item.metadata.get("source_chunk_id_strategy", ""),
                    "source_dataset_name": item.metadata.get("source_dataset_name", ""),
                    "source_cognee_chunk_id": item.metadata.get("source_cognee_chunk_id", ""),
                    "rank": rank,
                    "score": _num(item.score),
                    "similarity": _num(_metadata_num(item.metadata, "similarity")),
                    "distance": _num(_metadata_num(item.metadata, "distance")),
                    "text": item.text,
                    "metadata_json": _json(item.metadata),
                }
            )
        _append_rows(self.paths.retrieval_items, RETRIEVAL_ITEMS_COLUMNS, rows)

    def write_nodes(
        self,
        *,
        run_id: str,
        question_id: str,
        result: RetrievalResult,
    ) -> None:
        rows = []
        for node in _metadata_list(result.metadata, "cognee_nodes"):
            rows.append(
                {
                    "run_id": run_id,
                    "question_id": question_id,
                    "node_id": _first(node, "node_id", "id", "name"),
                    "label": _first(node, "label", "type", "node_label"),
                    "properties_json": _json(node.get("properties", node)),
                    "score": _num(_metadata_num(node, "score")),
                    "source_document_id": _first(node, "source_document_id", "document_id", "doc_id"),
                    "source_chunk_id": _first(node, "source_chunk_id", "chunk_id"),
                }
            )
        _append_rows(self.paths.nodes, NODES_COLUMNS, rows)

    def write_relationships(
        self,
        *,
        run_id: str,
        question_id: str,
        result: RetrievalResult,
    ) -> None:
        rows = []
        for rel in _metadata_list(result.metadata, "cognee_relationships"):
            rows.append(
                {
                    "run_id": run_id,
                    "question_id": question_id,
                    "source": _first(rel, "source", "source_id", "from", "start"),
                    "target": _first(rel, "target", "target_id", "to", "end"),
                    "type": _first(rel, "type", "label", "relationship_type"),
                    "properties_json": _json(rel.get("properties", rel)),
                    "score": _num(_metadata_num(rel, "score")),
                    "source_document_id": _first(rel, "source_document_id", "document_id", "doc_id"),
                    "source_chunk_id": _first(rel, "source_chunk_id", "chunk_id"),
                }
            )
        _append_rows(self.paths.relationships, RELATIONSHIPS_COLUMNS, rows)

    def write_graph_trace(
        self,
        *,
        run_id: str,
        question_id: str,
        result: RetrievalResult,
    ) -> None:
        raw_trace = result.metadata.get("raw_trace")
        row = {
            "run_id": run_id,
            "question_id": question_id,
            "trace_stage": "retrieval",
            "entities_json": _json(_trace_field(raw_trace, "entities")),
            "facts_json": _json(_trace_field(raw_trace, "facts")),
            "memories_json": _json(_trace_field(raw_trace, "memories")),
            "context_json": _json([item.model_dump(mode="json") for item in result.items]),
            "subgraph_json": _json(
                {
                    "nodes": result.metadata.get("cognee_nodes", []),
                    "relationships": result.metadata.get("cognee_relationships", []),
                }
            ),
            "reasoning_inputs_json": _json(_trace_field(raw_trace, "reasoning_inputs")),
            "raw_trace_json": _json(raw_trace),
        }
        _append_rows(self.paths.graph_trace, GRAPH_TRACE_COLUMNS, [row])


def _append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _metadata_list(metadata: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = metadata.get(key)
    return value if isinstance(value, list) else []


def _metadata_num(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _trace_field(raw_trace: Any, key: str) -> Any:
    if isinstance(raw_trace, dict):
        return raw_trace.get(key)
    if isinstance(raw_trace, list):
        values = []
        for item in raw_trace:
            if isinstance(item, dict) and key in item:
                values.append(item[key])
        return values
    return None


def _num(value: float | None) -> str:
    return "" if value is None else str(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
