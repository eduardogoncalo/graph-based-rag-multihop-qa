from __future__ import annotations

import json
import os
import select
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from benchmark.methods.hipporag2.config_builder import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    HIPPORAG2_METHOD_ID,
    Hipporag2Workspace,
    ensure_workspace,
)


class Hipporag2QueryClient(Protocol):
    def query(self, *, query: str, top_k: int, mode: str = "retrieve") -> dict:
        """Return the runner payload for one query.

        mode="retrieve" (default) devolve só contexto (Braço A);
        mode="qa" roda a geração nativa do pacote (rag_qa, Braço B).
        """


def _default_python_bin() -> str:
    # .absolute() e NÃO .resolve(): o python do venv é um symlink para o
    # interpretador-base do uv; derreferenciá-lo perde o pyvenv.cfg (e o
    # site-packages do venv) — o runner cairia num python sem hipporag.
    return os.environ.get(
        "HIPPORAG2_VENV_PYTHON", str(Path(".venvs/hipporag2/bin/python").absolute())
    )


def _default_runner_script() -> str:
    return os.environ.get(
        "HIPPORAG2_QUERY_RUNNER", str(Path("scripts/hipporag2_query_runner.py").absolute())
    )


class Hipporag2ServerClient:
    """Persistent runner subprocess speaking JSONL over stdin/stdout.

    HippoRAG loads the whole index (igraph + embedding store) at start-up, so a
    per-query subprocess would reload everything per question. This client keeps
    ONE runner alive (`--serve`) and self-heals: on EOF/timeout it respawns the
    process and retries the query once.
    """

    def __init__(
        self,
        *,
        save_dir: Path,
        llm_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        python_bin: str | None = None,
        runner_script: str | None = None,
        query_timeout_s: float = 300.0,
    ) -> None:
        self.save_dir = save_dir
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.python_bin = python_bin or _default_python_bin()
        self.runner_script = runner_script or _default_runner_script()
        self.query_timeout_s = query_timeout_s
        self._proc: subprocess.Popen | None = None

    # -- lifecycle -----------------------------------------------------------
    def _spawn(self) -> subprocess.Popen:
        args = [
            self.python_bin, self.runner_script,
            "--save-dir", str(self.save_dir),
            "--llm-model", self.llm_model,
            "--embedding-model", self.embedding_model,
            "--serve",
        ]
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        ready = self._read_line(proc, timeout_s=600.0)  # index load pode demorar
        payload = json.loads(ready) if ready else {}
        if not payload.get("ready"):
            proc.kill()
            raise RuntimeError(f"hipporag2 runner failed to become ready: {ready!r}")
        return proc

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._spawn()
        return self._proc

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        self._proc = None

    # -- protocol ------------------------------------------------------------
    def _read_line(self, proc: subprocess.Popen, *, timeout_s: float) -> str:
        assert proc.stdout is not None
        readable, _, _ = select.select([proc.stdout], [], [], timeout_s)
        if not readable:
            raise TimeoutError(f"hipporag2 runner silent for {timeout_s}s")
        return proc.stdout.readline()

    def _query_once(self, *, query: str, top_k: int, mode: str) -> dict:
        proc = self._ensure_proc()
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"query": query, "top_k": top_k, "mode": mode}) + "\n")
        proc.stdin.flush()
        line = self._read_line(proc, timeout_s=self.query_timeout_s)
        if not line:
            raise RuntimeError("hipporag2 runner EOF")
        return json.loads(line)

    def query(self, *, query: str, top_k: int, mode: str = "retrieve") -> dict:
        try:
            return self._query_once(query=query, top_k=top_k, mode=mode)
        except (TimeoutError, RuntimeError, json.JSONDecodeError, BrokenPipeError, OSError):
            # self-heal: respawn once and retry
            self.close()
            return self._query_once(query=query, top_k=top_k, mode=mode)


@dataclass(frozen=True)
class Hipporag2IndexSummary:
    dataset_id: str
    dataset_version: str
    method_id: str
    save_dir: Path
    document_count: int


class Hipporag2Adapter:
    def __init__(
        self,
        *,
        workspace: Hipporag2Workspace,
        client: Hipporag2QueryClient | None = None,
    ) -> None:
        if workspace.method_id != HIPPORAG2_METHOD_ID:
            raise ValueError("HippoRAG 2 adapter only supports method_id='hipporag2'")
        self.workspace = workspace
        self._client = client

    @property
    def client(self) -> Hipporag2QueryClient:
        if self._client is None:
            self._client = Hipporag2ServerClient(save_dir=self.workspace.save_dir)
        return self._client

    def query_context(self, *, query: str, top_k: int = 5) -> dict:
        """Retrieval-only (PPR sobre o KG): devolve o payload do runner.

        Custo por query = 1 embedding call + 1 chamada de triple-filter
        (gpt-4o-mini); nenhuma geração de resposta — o reader fixo do benchmark
        faz a geração (Braço A).
        """
        ensure_workspace(self.workspace)
        payload = self.client.query(query=query, top_k=top_k)
        return self._validate_payload(payload)

    def query_native(self, *, query: str) -> dict:
        """QA nativo do HippoRAG 2 (Braço B, "as deployed"): rag_qa do pacote.

        Retrieval interno as-shipped (retrieval_top_k=200 via PPR) + reader do
        próprio pacote sobre os top qa_top_k=5 docs. Payload inclui
        {"answer","raw_response","items","stats"}.
        """
        ensure_workspace(self.workspace)
        payload = self.client.query(query=query, top_k=5, mode="qa")
        payload = self._validate_payload(payload)
        if "answer" not in payload:
            raise RuntimeError("hipporag2 runner qa payload sem campo 'answer'")
        return payload

    @staticmethod
    def _validate_payload(payload: object) -> dict:
        if not isinstance(payload, dict):
            raise RuntimeError(f"hipporag2 runner returned non-dict payload: {type(payload)}")
        if payload.get("error"):
            raise RuntimeError(f"hipporag2 runner error: {payload['error']}")
        return payload
