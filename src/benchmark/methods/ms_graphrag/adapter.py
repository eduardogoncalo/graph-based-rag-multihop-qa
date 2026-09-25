from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from benchmark.core.schemas import Document
from benchmark.methods.ms_graphrag.config_builder import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MS_GRAPHRAG_METHOD_ID,
    GraphRAGWorkspace,
    build_config_files,
    ensure_workspace,
)


@dataclass(frozen=True)
class GraphRAGCommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


class GraphRAGCommandRunner(Protocol):
    def run(self, args: list[str], *, cwd: Path | None = None) -> GraphRAGCommandResult:
        """Run an official GraphRAG command."""


class GraphRAGSubprocessRunner:
    def run(self, args: list[str], *, cwd: Path | None = None) -> GraphRAGCommandResult:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        return GraphRAGCommandResult(
            args=args,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@dataclass(frozen=True)
class GraphRAGIndexSummary:
    dataset_id: str
    dataset_version: str
    method_id: str
    workspace_dir: Path
    document_count: int


class MicrosoftGraphRAGAdapter:
    def __init__(
        self,
        *,
        workspace: GraphRAGWorkspace,
        runner: GraphRAGCommandRunner | None = None,
    ) -> None:
        if workspace.method_id != MS_GRAPHRAG_METHOD_ID:
            raise ValueError("Microsoft GraphRAG adapter only supports method_id='ms_graphrag'")
        self.workspace = workspace
        self.runner = runner or GraphRAGSubprocessRunner()

    def prepare_inputs(self, documents: list[Document]) -> list[Path]:
        """Write the corpus as ONE CSV with an explicit ``id`` column.

        GraphRAG uses the CSV ``id`` column as the document id, so the
        benchmark's ``doc_*`` namespace survives into ``documents.parquet`` and
        ``text_units.document_id`` — the traceability that document-level
        evidence recall requires. ``title`` mirrors the id (belt and braces).
        """
        ensure_workspace(self.workspace)
        path = self.workspace.input_dir / "documents.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(["id", "title", "text"])
            for document in documents:
                writer.writerow([document.document_id, document.document_id, document.text])
        return [path]

    def initialize_workspace(self, *, force: bool = True) -> GraphRAGCommandResult:
        ensure_workspace(self.workspace)
        # --model/--embedding skip the interactive prompts (3.1.0) AND bind the
        # benchmark models at init time; build_config_files then only patches
        # provider/api_key/CSV input on top of the generated defaults.
        args = [
            "graphrag", "init",
            "--root", str(self.workspace.workspace_dir),
            "--model", DEFAULT_CHAT_MODEL,
            "--embedding", DEFAULT_EMBEDDING_MODEL,
        ]
        if force:
            args.append("--force")
        result = self.runner.run(args, cwd=self.workspace.workspace_dir)
        self._preserve_result(result, "init")
        if result.returncode != 0:
            raise RuntimeError(f"GraphRAG init failed: {result.stderr}")
        build_config_files(self.workspace)
        return result

    def index(self, *, method: str = "standard") -> GraphRAGCommandResult:
        args = [
            "graphrag",
            "index",
            "--root",
            str(self.workspace.workspace_dir),
            "--method",
            method,
        ]
        result = self.runner.run(args, cwd=self.workspace.workspace_dir)
        self._preserve_result(result, "index")
        if result.returncode != 0:
            raise RuntimeError(f"GraphRAG index failed: {result.stderr}")
        return result

    def query_context(self, *, query: str, query_method: str = "local") -> GraphRAGCommandResult:
        """Retrieval-only query: build the search context WITHOUT generating.

        Runs ``scripts/graphrag_query_runner.py`` inside the isolated graphrag
        venv (embedding calls only — no chat completion). Stdout is a JSON
        document with ``context_text`` + per-source ``items`` carrying doc ids.
        """
        # .absolute() e NÃO .resolve(): o python do venv é um symlink para o
        # interpretador-base do uv; derreferenciá-lo perde o pyvenv.cfg (e o
        # site-packages do venv) — o runner cairia no python sem graphrag.
        python_bin = os.environ.get(
            "GRAPHRAG_VENV_PYTHON", str(Path(".venvs/graphrag/bin/python").absolute())
        )
        runner_script = os.environ.get(
            "GRAPHRAG_QUERY_RUNNER", str(Path("scripts/graphrag_query_runner.py").absolute())
        )
        args = [
            python_bin,
            runner_script,
            "--root",
            str(self.workspace.workspace_dir),
            "--method",
            query_method,
            "--query",
            query,
        ]
        result = self.runner.run(args, cwd=self.workspace.workspace_dir)
        self._preserve_result(result, f"context_{query_method}")
        if result.returncode != 0:
            raise RuntimeError(f"GraphRAG context query failed: {result.stderr}")
        return result

    def query_native(self, *, query: str, query_method: str = "local") -> GraphRAGCommandResult:
        """Geração NATIVA (Braço B, as-deployed): roda o runner isolado com --generate
        (engine.search() → resposta própria do GraphRAG). Stdout = JSON {"answer", ...}.

        Reusa o MESMO python do venv isolado do query_context (não o bare `graphrag`),
        para não depender do PATH resolver o binário certo.
        """
        python_bin = os.environ.get(
            "GRAPHRAG_VENV_PYTHON", str(Path(".venvs/graphrag/bin/python").absolute())
        )
        runner_script = os.environ.get(
            "GRAPHRAG_QUERY_RUNNER", str(Path("scripts/graphrag_query_runner.py").absolute())
        )
        args = [
            python_bin, runner_script,
            "--root", str(self.workspace.workspace_dir),
            "--method", query_method,
            "--query", query,
            "--generate",
        ]
        result = self.runner.run(args, cwd=self.workspace.workspace_dir)
        self._preserve_result(result, f"native_{query_method}")
        if result.returncode != 0:
            raise RuntimeError(f"GraphRAG native query failed: {result.stderr}")
        return result

    def query(self, *, query: str, query_method: str = "local") -> GraphRAGCommandResult:
        args = [
            "graphrag",
            "query",
            "--root",
            str(self.workspace.workspace_dir),
            "--method",
            query_method,
            query,
        ]
        result = self.runner.run(args, cwd=self.workspace.workspace_dir)
        self._preserve_result(result, f"query_{query_method}")
        if result.returncode != 0:
            raise RuntimeError(f"GraphRAG query failed: {result.stderr}")
        return result

    def _preserve_result(self, result: GraphRAGCommandResult, stem: str) -> None:
        ensure_workspace(self.workspace)
        (self.workspace.raw_dir / f"{stem}_stdout.txt").write_text(result.stdout, encoding="utf-8")
        (self.workspace.raw_dir / f"{stem}_stderr.txt").write_text(result.stderr, encoding="utf-8")
        (self.workspace.raw_dir / f"{stem}_command.json").write_text(
            json.dumps({"args": result.args, "returncode": result.returncode}, sort_keys=True),
            encoding="utf-8",
        )
