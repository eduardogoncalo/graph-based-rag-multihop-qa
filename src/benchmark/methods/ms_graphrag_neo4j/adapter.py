from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from benchmark.core.schemas import Document
from benchmark.methods.ms_graphrag_neo4j.config_builder import (
    MS_GRAPHRAG_NEO4J_METHOD_ID,
    GraphRAGNeo4jConfig,
    GraphRAGNeo4jWorkspace,
    build_config_file,
    ensure_workspace,
    validate_neo4j_env_names,
)


@dataclass(frozen=True)
class GraphRAGNeo4jIndexResult:
    document_count: int
    raw_response: dict[str, Any] | str | None = None


@dataclass(frozen=True)
class GraphRAGNeo4jQueryResult:
    query: str
    raw_response: dict[str, Any] | list[Any] | str | None


class GraphRAGNeo4jClient(Protocol):
    def index(
        self,
        *,
        workspace: GraphRAGNeo4jWorkspace,
        config: GraphRAGNeo4jConfig,
        documents: Sequence[Document],
    ) -> GraphRAGNeo4jIndexResult:
        """Index documents with a Neo4j-backed Microsoft GraphRAG-style implementation."""

    def query(
        self,
        *,
        workspace: GraphRAGNeo4jWorkspace,
        config: GraphRAGNeo4jConfig,
        query: str,
        top_k: int,
    ) -> GraphRAGNeo4jQueryResult:
        """Query the Neo4j-backed Microsoft GraphRAG-style implementation."""


class MissingGraphRAGNeo4jClient:
    def index(
        self,
        *,
        workspace: GraphRAGNeo4jWorkspace,
        config: GraphRAGNeo4jConfig,
        documents: Sequence[Document],
    ) -> GraphRAGNeo4jIndexResult:
        raise _missing_dependency_error()

    def query(
        self,
        *,
        workspace: GraphRAGNeo4jWorkspace,
        config: GraphRAGNeo4jConfig,
        query: str,
        top_k: int,
    ) -> GraphRAGNeo4jQueryResult:
        raise _missing_dependency_error()


class GraphRAGNeo4jAdapter:
    def __init__(
        self,
        *,
        workspace: GraphRAGNeo4jWorkspace,
        config: GraphRAGNeo4jConfig | None = None,
        client: GraphRAGNeo4jClient | None = None,
    ) -> None:
        if workspace.method_id != MS_GRAPHRAG_NEO4J_METHOD_ID:
            raise ValueError("GraphRAG Neo4j adapter only supports method_id='ms_graphrag_neo4j'")
        self.workspace = workspace
        self.config = config or GraphRAGNeo4jConfig()
        validate_neo4j_env_names(self.config)
        self.client = client or MissingGraphRAGNeo4jClient()

    def prepare_inputs(self, documents: Sequence[Document]) -> list[Path]:
        ensure_workspace(self.workspace)
        paths: list[Path] = []
        manifest: list[dict[str, str]] = []
        for document in documents:
            path = self.workspace.input_dir / f"{document.document_id}.txt"
            path.write_text(document.text, encoding="utf-8")
            paths.append(path)
            manifest.append(
                {
                    "document_id": document.document_id,
                    "dataset_id": document.dataset_id,
                    "dataset_version": document.dataset_version,
                    "path": str(path),
                }
            )
        (self.workspace.input_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return paths

    def index(self, *, documents: Sequence[Document]) -> GraphRAGNeo4jIndexResult:
        self.prepare_inputs(documents)
        build_config_file(self.workspace, config=self.config)
        result = self.client.index(
            workspace=self.workspace,
            config=self.config,
            documents=documents,
        )
        self._preserve_raw("index", result.raw_response)
        if result.document_count != len(documents):
            raise RuntimeError("GraphRAG Neo4j indexing reported an unexpected document count")
        return result

    def query(self, *, query: str, top_k: int = 5) -> GraphRAGNeo4jQueryResult:
        ensure_workspace(self.workspace)
        build_config_file(self.workspace, config=self.config)
        result = self.client.query(
            workspace=self.workspace,
            config=self.config,
            query=query,
            top_k=top_k,
        )
        self._preserve_raw("query", result.raw_response)
        return result

    def _preserve_raw(self, stem: str, raw_response: object) -> None:
        ensure_workspace(self.workspace)
        path = self.workspace.raw_dir / f"{stem}_result.json"
        path.write_text(json.dumps(raw_response, sort_keys=True, default=str), encoding="utf-8")


def _missing_dependency_error() -> RuntimeError:
    return RuntimeError(
        "ms_graphrag_neo4j live execution is not configured. This phase provides only the "
        "offline adapter boundary. Install and validate neo4j-contrib/ms-graphrag-neo4j or an "
        "approved isolated runtime before running real Neo4j-backed indexing or querying."
    )
