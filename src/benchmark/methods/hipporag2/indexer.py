from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from benchmark.methods.hipporag2.adapter import Hipporag2IndexSummary, _default_python_bin
from benchmark.methods.hipporag2.config_builder import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    Hipporag2Workspace,
    ensure_workspace,
)


def index(
    *,
    workspace: Hipporag2Workspace,
    canonical_dir: str | Path,
    limit: int | None = None,
    skip: int = 0,
    llm_model: str = DEFAULT_CHAT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Hipporag2IndexSummary:
    """Indexa o corpus canônico no venv isolado do HippoRAG 2.

    Delega para scripts/hipporag2_index_runner.py (subprocess no
    .venvs/hipporag2), que lê documents.jsonl, chama HippoRAG.index(docs=...)
    **uma única vez** com a janela inteira, e mantém passage_map.json
    (sha1(text) -> document_id).

    **Não retoma, e o índice não é incremental.** Cada chamada a index() volta
    a derivar as arestas e volta a acrescentá-las — a deduplicação por hash de
    conteúdo é do embedding store e não cobre esse passo. Medido a 2026-08-09:
    19.392.908 arestas para 319.817 pares distintos. Por isso o runner recusa
    um `workspace.save_dir` que já tenha conteúdo, e `limit`/`skip` exigem um
    destino próprio por janela.
    """
    ensure_workspace(workspace)
    runner = os.environ.get(
        "HIPPORAG2_INDEX_RUNNER", str(Path("scripts/hipporag2_index_runner.py").absolute())
    )
    args = [
        _default_python_bin(), runner,
        "--save-dir", str(workspace.save_dir),
        "--canonical-dir", str(canonical_dir),
        "--passage-map", str(workspace.passage_map_path),
        "--llm-model", llm_model,
        "--embedding-model", embedding_model,
        "--skip", str(skip),
    ]
    if limit is not None:
        args += ["--limit", str(limit)]
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    (workspace.raw_dir / "index_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (workspace.raw_dir / "index_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (workspace.raw_dir / "index_command.json").write_text(
        json.dumps({"args": args, "returncode": completed.returncode}, sort_keys=True),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"hipporag2 index failed: {completed.stderr[-2000:]}")
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    return Hipporag2IndexSummary(
        dataset_id=workspace.dataset_id,
        dataset_version=workspace.dataset_version,
        method_id=workspace.method_id,
        save_dir=workspace.save_dir,
        document_count=int(summary.get("indexed_total", 0)),
    )
