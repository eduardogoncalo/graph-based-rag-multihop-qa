"""F1.2 do plano 2Wiki — índice LightRAG do twowiki (corpus HippoRAG, 6.119 docs).

Option C: container Neo4j isolado `neo4j_lightrag_twowiki_ans_v1_0_eval1k`
(portas do registry: http 17477 / bolt 17690), volume próprio, cap de memória
4 GiB. Índice retomável (doc_status do LightRAG); relançar é seguro.

Uso: uv run python scripts/twowiki_lightrag_index.py
Env de concorrência (mesmos valores do full do MuSiQue):
  MAX_PARALLEL_INSERT=4 MAX_ASYNC=8 EMBEDDING_FUNC_MAX_ASYNC=2
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from benchmark.core.settings import load_settings
from benchmark.infra.neo4j_containers import (
    container_exists,
    start_neo4j_container,
    wait_for_ready,
)
from benchmark.methods.lightrag_neo4j.adapter import LightRAGNeo4jAdapter
from benchmark.methods.lightrag_neo4j.config_builder import build_workspace
from benchmark.methods.lightrag_neo4j.indexer import index, load_canonical_documents
from benchmark.methods.lightrag_neo4j.option_c import resolve_lightrag_option_c

DATASET_ID = "twowiki"
DATASET_VERSION = "ans_v1.0_eval1k"
CANONICAL = "data/canonical/twowiki_ans_v1.0_eval1k"
STATUS = Path("artifacts/twowiki/reports/lightrag_full_index_status.json")
MEMORY_CAP = "4g"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    binding = resolve_lightrag_option_c(
        dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
    )
    log(f"binding: container={binding.container_name} bolt={binding.bolt_uri}")

    if container_exists(binding.container_name):
        subprocess.run(["podman", "start", binding.container_name], check=True)
        log("container existente reiniciado")
    else:
        start_neo4j_container(binding.container_spec(), wait=True)
        log("container criado")
    subprocess.run(
        ["podman", "update", f"--memory={MEMORY_CAP}", f"--memory-swap={MEMORY_CAP}",
         binding.container_name],
        check=False,
    )
    wait_for_ready(binding.bolt_uri, binding.username, binding.password, timeout=120)
    log("neo4j pronto")

    settings = load_settings()
    workspace = build_workspace(
        artifacts_dir=settings.artifacts_dir,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
    )
    config = replace(binding.to_config(), max_documents=None)
    documents = load_canonical_documents(CANONICAL, max_documents=None)
    log(f"{len(documents)} documentos canônicos; iniciando index (retomável)")

    started = time.monotonic()
    adapter = LightRAGNeo4jAdapter(workspace=workspace, config=config)
    summary = index(adapter=adapter, documents=documents)

    payload = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "container": binding.container_name,
        "bolt_uri": binding.bolt_uri,
        "document_count_input": len(documents),
        "document_count_indexed": summary.document_count,
        "elapsed_s": round(time.monotonic() - started, 1),
        "state": "completed",
        "verdict": "FULL_INDEX_PASS" if summary.document_count == len(documents) else "PARTIAL",
        "provenance": "hipporag_reproduce_dataset",
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2))
    log(f"DONE: {payload['verdict']} {summary.document_count}/{len(documents)} "
        f"em {payload['elapsed_s']}s")
    return 0 if payload["verdict"] == "FULL_INDEX_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
