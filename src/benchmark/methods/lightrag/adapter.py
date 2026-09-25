from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from benchmark.core.schemas import Document
from benchmark.ingestion.text_normalization import clean_rag_text
from benchmark.methods.lightrag.config_builder import (
    LIGHTRAG_METHOD_ID,
    LightRAGWorkspace,
    build_config_file,
    ensure_workspace,
)


@dataclass(frozen=True)
class LightRAGIngestResult:
    document_count: int
    raw_response: dict[str, object] | str | None = None


@dataclass(frozen=True)
class LightRAGQueryResult:
    query: str
    mode: str
    raw_response: dict[str, object] | str


class LightRAGRunner(Protocol):
    def ingest(
        self,
        *,
        workspace: LightRAGWorkspace,
        documents: Sequence[Document],
    ) -> LightRAGIngestResult:
        """Use the official LightRAG implementation to ingest documents."""

    def query(
        self,
        *,
        workspace: LightRAGWorkspace,
        query: str,
        mode: str,
        top_k: int,
    ) -> LightRAGQueryResult:
        """Use the official LightRAG implementation to query."""


class LightRAGPythonRunner:
    def ingest(
        self,
        *,
        workspace: LightRAGWorkspace,
        documents: Sequence[Document],
    ) -> LightRAGIngestResult:
        LightRAG, QueryParam = _import_lightrag()
        rag = LightRAG(working_dir=str(workspace.working_dir))
        texts = [document.text for document in documents]
        ids = [document.document_id for document in documents]
        insert_result = rag.insert(texts, ids=ids)
        return LightRAGIngestResult(
            document_count=len(documents),
            raw_response=insert_result if insert_result is not None else {"status": "ok"},
        )

    def query(
        self,
        *,
        workspace: LightRAGWorkspace,
        query: str,
        mode: str,
        top_k: int,
    ) -> LightRAGQueryResult:
        LightRAG, QueryParam = _import_lightrag()
        rag = LightRAG(working_dir=str(workspace.working_dir))
        param = QueryParam(mode=mode, top_k=top_k)
        return LightRAGQueryResult(
            query=query,
            mode=mode,
            raw_response=rag.query(query, param=param),
        )


class LightRAGAdapter:
    def __init__(
        self,
        *,
        workspace: LightRAGWorkspace,
        runner: LightRAGRunner | None = None,
    ) -> None:
        if workspace.method_id != LIGHTRAG_METHOD_ID:
            raise ValueError("LightRAG adapter only supports method_id='lightrag'")
        self.workspace = workspace
        self.runner = runner or LightRAGPythonRunner()

    def prepare_inputs(self, documents: Sequence[Document]) -> list[Path]:
        ensure_workspace(self.workspace)
        paths: list[Path] = []
        for document in documents:
            path = self.workspace.input_dir / f"{document.document_id}.txt"
            path.write_text(clean_rag_text(document.text), encoding="utf-8")
            paths.append(path)
        return paths

    def ingest(
        self,
        *,
        documents: Sequence[Document],
        query_mode: str = "mix",
        top_k: int = 5,
    ) -> LightRAGIngestResult:
        cleaned_documents = _clean_documents_for_rag(documents)
        self.prepare_inputs(cleaned_documents)
        build_config_file(self.workspace, query_mode=query_mode, top_k=top_k)
        result = self.runner.ingest(workspace=self.workspace, documents=cleaned_documents)
        self._preserve_raw("ingest", result.raw_response)
        if result.document_count != len(documents):
            raise RuntimeError("LightRAG ingestion reported an unexpected document count")
        return result

    def query(self, *, query: str, mode: str = "mix", top_k: int = 5) -> LightRAGQueryResult:
        ensure_workspace(self.workspace)
        result = self.runner.query(
            workspace=self.workspace,
            query=query,
            mode=mode,
            top_k=top_k,
        )
        self._preserve_raw(f"query_{mode}", result.raw_response)
        return result

    def _preserve_raw(self, stem: str, raw_response: object) -> None:
        ensure_workspace(self.workspace)
        path = self.workspace.raw_dir / f"{stem}_result.json"
        path.write_text(json.dumps(raw_response, sort_keys=True, default=str), encoding="utf-8")


def _import_lightrag():
    try:
        from lightrag import LightRAG, QueryParam  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "LightRAG is not installed. Install lightrag-hku in a compatible runtime "
            "before running live LightRAG ingestion or queries."
        ) from exc
    return LightRAG, QueryParam


def _clean_documents_for_rag(documents: Sequence[Document]) -> list[Document]:
    return [
        document.model_copy(update={"text": clean_rag_text(document.text)})
        for document in documents
    ]
