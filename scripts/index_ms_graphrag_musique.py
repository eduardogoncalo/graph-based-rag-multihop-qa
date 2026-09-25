"""Index the MuSiQue corpus with Microsoft GraphRAG (factory defaults).

Runs in the MAIN venv; the `graphrag` CLI is resolved from the isolated
.venvs/graphrag via PATH prefix. Factory-defaults protocol: settings.yaml is
the `graphrag init` output patched ONLY with model bindings + CSV input.

Usage:
  # shakeout (100 docs, separate workspace — never pollutes the full index):
  .venv/bin/python scripts/index_ms_graphrag_musique.py --limit 100 --shakeout
  # full corpus:
  .venv/bin/python scripts/index_ms_graphrag_musique.py
  # another dataset (e.g. 2Wiki):
  .venv/bin/python scripts/index_ms_graphrag_musique.py --dataset-id twowiki
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from benchmark.core.settings import load_settings
from benchmark.methods.ms_graphrag.adapter import MicrosoftGraphRAGAdapter
from benchmark.methods.ms_graphrag.config_builder import build_workspace
from benchmark.methods.ms_graphrag.indexer import load_canonical_documents

GRAPHRAG_BIN_DIR = Path(".venvs/graphrag/bin").resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="musique")
    parser.add_argument("--dataset-version", default="ans_v1.0_eval1k")
    parser.add_argument("--limit", type=int, default=None, help="index only the first N docs")
    parser.add_argument("--shakeout", action="store_true", help="use an isolated shakeout workspace")
    parser.add_argument("--skip-init", action="store_true", help="reuse existing workspace config")
    args = parser.parse_args()

    canonical = f"data/canonical/{args.dataset_id}_{args.dataset_version}"
    if not Path(canonical).is_dir():
        raise SystemExit(f"canonical dataset not found: {canonical}")

    if not (GRAPHRAG_BIN_DIR / "graphrag").exists():
        raise SystemExit(f"graphrag CLI not found at {GRAPHRAG_BIN_DIR}; create .venvs/graphrag first")
    os.environ["PATH"] = f"{GRAPHRAG_BIN_DIR}:{os.environ['PATH']}"
    os.environ.setdefault("GRAPHRAG_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not os.environ["GRAPHRAG_API_KEY"]:
        raise SystemExit("OPENAI_API_KEY/GRAPHRAG_API_KEY not set (source .env first)")

    settings = load_settings()
    dataset_version = (
        f"{args.dataset_version}.shakeout" if args.shakeout else args.dataset_version
    )
    workspace = build_workspace(
        artifacts_dir=settings.artifacts_dir,
        dataset_id=args.dataset_id,
        dataset_version=dataset_version,
    )
    adapter = MicrosoftGraphRAGAdapter(workspace=workspace)

    documents = load_canonical_documents(canonical)
    if args.limit:
        documents = documents[: args.limit]
    print(f"[index] docs={len(documents)} workspace={workspace.workspace_dir}", flush=True)

    started = time.time()
    adapter.prepare_inputs(documents)
    if not args.skip_init:
        adapter.initialize_workspace(force=True)
        print("[index] init ok (settings patched: models + csv input)", flush=True)
    result = adapter.index(method="standard")
    elapsed = time.time() - started

    summary = {
        "docs": len(documents),
        "elapsed_s": round(elapsed, 1),
        "workspace": str(workspace.workspace_dir),
        "returncode": result.returncode,
    }
    (workspace.raw_dir / "index_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[index] DONE {summary}", flush=True)


if __name__ == "__main__":
    main()
