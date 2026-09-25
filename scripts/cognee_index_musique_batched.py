#!/usr/bin/env python
"""BATCHED cognee indexer for MuSiQue — runs in the ISOLATED venv (.venvs/cognee).

Why this exists
---------------
The original ``cognee_index_musique.py`` calls ``add([1 doc]) + cognify()`` once
PER document. That never uses cognee's internal concurrency (a semaphore over
documents, default 20), so it is slow and gets slower as the graph grows
(observed: ~12s/doc early -> 242s/doc by doc 3728). An isolated probe
(``cognee_probe_batch.py``, 300 docs on a throwaway Neo4j) proved the fix:

  * add a BATCH of docs in ONE ``add([...])`` then ONE ``cognify()`` -> cognee
    processes ~``data_per_batch`` docs concurrently. Effective **0.51-0.60 s/doc**.
  * re-running ``cognify()`` with no new adds is ~0.2s and **0 LLM calls** ->
    cognee's incremental loading skips already-COMPLETED docs. This is what makes
    RESUMING the production index (already 3728/11515 done) safe and cheap.
  * graph verification: 300 distinct DOCUMENT_ID markers, **0 duplicated**.

Resume / no-duplication guarantees (same as the per-doc worker)
---------------------------------------------------------------
  * The durable per-``document_id`` ledger (``checkpoint.json``, shared with the
    per-doc worker) is the source of truth for "what is done". The worklist is
    every doc whose ledger status != "completed", so a resume never re-processes
    a finished doc.
  * ``doc_payload`` is byte-for-byte identical to the per-doc worker, so cognee's
    content-hash dedup recognizes the 3728 already-indexed docs and skips them.
  * Node IDs in cognee are content-derived + written with MERGE, so even if a doc
    is cognified twice it converges to the same nodes (no duplicate entities).

Stores: graph -> Neo4j (Option C, via env); relational+vector -> cognee local
defaults (sqlite + lancedb) under the workspace. Generation + embeddings -> the
shared OpenAI models (gpt-4o-mini / text-embedding-3-small).

Launch detached (``nohup setsid ... &``) so it outlives the shell; hibernation
just pauses/resumes it and the retry handles dropped LLM/Neo4j connections.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Cost tracking via litellm (cognee routes all LLM/embedding calls through it)
# This is the PROVEN callback from cognee_index_musique.py (it tracked the $2.02
# of the production run). The probe used a stripped-down version that recorded
# calls=0; do NOT simplify this.
# --------------------------------------------------------------------------- #
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}}
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


def _record_usage(kwargs, response_obj) -> None:
    try:
        model = (kwargs.get("model") or "?").split("/")[-1]
        usage = getattr(response_obj, "usage", None)
        if usage is None and isinstance(response_obj, dict):
            usage = response_obj.get("usage")
        pt = ct = 0
        if usage is not None:
            pt = int(getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0) or 0)
            ct = int(getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0) or 0)
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += pt
        USAGE["completion_tokens"] += ct
        m = USAGE["by_model"].setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        m["calls"] += 1
        m["prompt_tokens"] += pt
        m["completion_tokens"] += ct
    except Exception as exc:  # never let cost tracking break indexing
        _log(f"WARN: cost record error: {exc!r}")


def _install_cost_callback() -> None:
    try:
        import litellm
        from litellm.integrations.custom_logger import CustomLogger
    except Exception as exc:  # pragma: no cover
        _log(f"WARN: litellm not importable for cost tracking: {exc!r}")
        return

    class _CostLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            _record_usage(kwargs, response_obj)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            _record_usage(kwargs, response_obj)

    litellm.callbacks = list(getattr(litellm, "callbacks", []) or []) + [_CostLogger()]
    litellm.success_callback = list(getattr(litellm, "success_callback", []) or []) + [
        lambda kwargs, response, start, end: _record_usage(kwargs, response)
    ]
    _log("cost callback installed (CustomLogger + success_callback)")


def estimated_cost_usd() -> float:
    total = 0.0
    for model, m in USAGE["by_model"].items():
        pin, pout = PRICES.get(model, (0.0, 0.0))
        total += m["prompt_tokens"] / 1e6 * pin + m["completion_tokens"] / 1e6 * pout
    return round(total, 4)


# --------------------------------------------------------------------------- #
# Checkpoint ledger (durable, resumable, dedup-by-document_id) — SAME file/format
# as the per-doc worker, so the 3728 already-completed docs are recognized.
# --------------------------------------------------------------------------- #
def load_ledger(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"docs": {}, "started_at": _now(), "dataset_name": None}


def save_ledger(path: Path, ledger: dict) -> None:
    ledger["updated_at"] = _now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic on POSIX


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def load_documents(path: Path, n: int | None) -> list[dict]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
            if n is not None and len(docs) >= n:
                break
    return docs


def doc_payload(doc: dict) -> str:
    """IDENTICAL to cognee_index_musique.py.doc_payload — must stay byte-for-byte
    the same so cognee's content-hash recognizes already-indexed docs on resume."""
    doc_id = doc["document_id"]
    title = doc.get("title") or ""
    text = doc.get("text") or doc.get("metadata", {}).get("paragraph_text") or ""
    return f"[DOCUMENT_ID: {doc_id}] [TITLE: {title}]\n{text}"


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
def _is_rate_limit(exc: Exception) -> bool:
    s = repr(exc).lower()
    return "rate limit" in s or "429" in s or "ratelimit" in s


async def with_retry(factory, *, label: str, max_retries: int, cooldown: float):
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            return await factory()
        except Exception as exc:  # noqa: BLE001 - resilience layer
            last = exc
            if attempt >= max_retries:
                break
            wait = cooldown * (4 if _is_rate_limit(exc) else 1)
            _log(f"RETRY {label} attempt={attempt}/{max_retries} after {wait:.0f}s due to {type(exc).__name__}: {str(exc)[:160]}")
            await asyncio.sleep(wait)
    raise last  # type: ignore[misc]


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def run(args) -> int:
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "checkpoint.json"
    usage_path = out_dir / "usage.json"

    # --- cognee env (set BEFORE importing cognee) -------------------------- #
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        _log("FATAL: OPENAI_API_KEY not in environment")
        return 2
    env = {
        "DATA_ROOT_DIRECTORY": str(workspace / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(workspace / "system"),
        "CACHE_ROOT_DIRECTORY": str(workspace / "cache"),
        "COGNEE_LOGS_DIR": str(workspace / "logs"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "CACHING": "false",
        "TELEMETRY_DISABLED": "true",
        "GRAPH_DATABASE_PROVIDER": "neo4j",
        "GRAPH_DATABASE_URL": args.bolt,
        "GRAPH_DATABASE_USERNAME": "neo4j",
        "GRAPH_DATABASE_PASSWORD": args.password,
        "GRAPH_DATABASE_NAME": "neo4j",
        "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
        "DB_PROVIDER": "sqlite",
        "VECTOR_DB_PROVIDER": "lancedb",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": args.llm_model,
        "LLM_API_KEY": openai_key,
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": args.embedding_model,
        "EMBEDDING_DIMENSIONS": str(args.embedding_dimensions),
        "EMBEDDING_API_KEY": openai_key,
        "LITELLM_API_KEY": openai_key,
    }
    os.environ.update(env)
    _log(f"cognee env set: graph={args.bolt} llm={args.llm_model} emb={args.embedding_model} workspace={workspace}")

    _install_cost_callback()

    import cognee  # noqa: E402  - must come after env is set
    _install_cost_callback()  # re-assert after cognee import (cognee may reset litellm callbacks)
    _log(f"cognee {getattr(cognee, '__version__', '?')} imported")

    docs = load_documents(Path(args.documents), args.n)
    _log(f"loaded {len(docs)} documents from {args.documents}")

    ledger = load_ledger(ledger_path)
    ledger["dataset_name"] = args.dataset_name
    save_ledger(ledger_path, ledger)

    completed_before = sum(1 for d in ledger["docs"].values() if d.get("status") == "completed")
    _log(f"resume: {completed_before} docs already completed in ledger")

    # Seed cumulative usage from disk so cost survives the per-batch process restarts
    # (each fresh process would otherwise overwrite usage.json with only its own batch).
    if usage_path.exists():
        try:
            prev = json.loads(usage_path.read_text(encoding="utf-8"))
            USAGE["calls"] = int(prev.get("calls", 0) or 0)
            USAGE["prompt_tokens"] = int(prev.get("prompt_tokens", 0) or 0)
            USAGE["completion_tokens"] = int(prev.get("completion_tokens", 0) or 0)
            USAGE["by_model"] = prev.get("by_model", {}) or {}
            _log(f"seeded cumulative usage from disk: calls={USAGE['calls']} cost=${estimated_cost_usd():.4f}")
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN: could not seed usage.json: {exc!r}")

    # Worklist = docs not yet completed, in file order.
    worklist = [d for d in docs if (ledger["docs"].get(d["document_id"]) or {}).get("status") != "completed"]
    _log(f"worklist: {len(worklist)} docs to process (batch_size={args.batch_size}, data_per_batch={args.data_per_batch})")

    t0 = time.monotonic()
    n_done = 0
    n_batches = 0
    n_batch_failures = 0
    batches_this_proc = 0

    for batch in _chunks(worklist, args.batch_size):
        # Memory hygiene: cognee does NOT release memory between batches inside one
        # long-lived process (observed RSS climbing ~7 -> 15 GB across batches -> OOM).
        # So we process a bounded number of batches, then exit CLEANLY; the supervisor
        # relaunches a FRESH process (RSS resets to baseline). Resume is by the ledger,
        # so exiting here loses NO work: any doc not marked "completed" is re-added by
        # the next process and re-cognified idempotently (MERGE on content-derived ids
        # -> no duplicate nodes). Same dataset/graph -> entity resolution unchanged.
        if args.batches_per_process and batches_this_proc >= args.batches_per_process:
            _log(f"BATCH_LIMIT {args.batches_per_process} reached -> clean exit for fresh process (memory reset); supervisor relaunches with --resume")
            return 0
        n_batches += 1
        batches_this_proc += 1
        batch_ids = [d["document_id"] for d in batch]
        for did in batch_ids:
            prev = ledger["docs"].get(did) or {}
            ledger["docs"][did] = {"status": "running", "attempts": prev.get("attempts", 0) + 1, "started_at": _now()}
        save_ledger(ledger_path, ledger)

        b_t0 = time.monotonic()
        calls_before = USAGE["calls"]
        try:
            payloads = [doc_payload(d) for d in batch]

            async def _add():
                return await cognee.add(payloads, dataset_name=args.dataset_name)

            async def _cognify():
                # data_per_batch = how many docs cognee processes concurrently.
                return await cognee.cognify(datasets=[args.dataset_name], data_per_batch=args.data_per_batch)

            await with_retry(_add, label=f"add[batch{n_batches}]", max_retries=args.max_retries, cooldown=args.cooldown)
            await with_retry(_cognify, label=f"cognify[batch{n_batches}]", max_retries=args.max_retries, cooldown=args.cooldown)
        except Exception as exc:  # noqa: BLE001
            n_batch_failures += 1
            for did in batch_ids:
                att = (ledger["docs"].get(did) or {}).get("attempts", 1)
                ledger["docs"][did] = {"status": "failed", "attempts": att, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "failed_at": _now()}
            save_ledger(ledger_path, ledger)
            usage_path.write_text(json.dumps({**USAGE, "estimated_cost_usd": estimated_cost_usd()}, indent=2), encoding="utf-8")
            _log(f"FAIL batch {n_batches} ({len(batch)} docs): {type(exc).__name__}: {str(exc)[:200]}")
            if args.stop_after_failures and n_batch_failures >= args.stop_after_failures:
                _log("STOP: reached stop_after_failures threshold")
                return 1
            continue  # leave them 'failed' -> a later --resume run retries them

        b_elapsed = time.monotonic() - b_t0
        for did in batch_ids:
            att = (ledger["docs"].get(did) or {}).get("attempts", 1)
            ledger["docs"][did] = {
                "status": "completed",
                "attempts": att,
                "completed_at": _now(),
                "batch": n_batches,
            }
        n_done += len(batch)
        save_ledger(ledger_path, ledger)
        usage_path.write_text(json.dumps({**USAGE, "estimated_cost_usd": estimated_cost_usd()}, indent=2), encoding="utf-8")
        completed_total = completed_before + n_done
        _log(
            f"OK batch {n_batches} docs={len(batch)} [{completed_total}/{len(docs)}] "
            f"add+cognify={b_elapsed:.1f}s eff={b_elapsed/max(len(batch),1):.2f}s/doc "
            f"calls={USAGE['calls'] - calls_before} cum_cost=${estimated_cost_usd():.4f}"
        )

    total_elapsed = time.monotonic() - t0
    completed_total = sum(1 for d in ledger["docs"].values() if d.get("status") == "completed")
    summary = {
        "dataset_name": args.dataset_name,
        "mode": "batched",
        "batch_size": args.batch_size,
        "data_per_batch": args.data_per_batch,
        "docs_input": len(docs),
        "docs_completed_total": completed_total,
        "docs_done_this_run": n_done,
        "batches_this_run": n_batches,
        "batch_failures": n_batch_failures,
        "total_elapsed_s_this_run": round(total_elapsed, 1),
        "avg_s_per_doc": round(total_elapsed / max(n_done, 1), 2),
        "usage": USAGE,
        "estimated_cost_usd": estimated_cost_usd(),
        "cost_per_doc_usd": round(estimated_cost_usd() / max(completed_total, 1), 5),
        "finished_at": _now(),
    }
    (out_dir / "index_summary_batched.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(
        f"INDEX_DONE completed={completed_total}/{len(docs)} this_run={n_done} "
        f"elapsed={total_elapsed:.0f}s cost=${estimated_cost_usd():.4f} cost/doc=${summary['cost_per_doc_usd']:.5f}"
    )

    # --- non-duplication reconciliation against the real graph ------------- #
    if args.verify_graph:
        await reconcile_graph(args, out_dir, completed_total)

    return 0


async def reconcile_graph(args, out_dir: Path, completed_total: int) -> None:
    """Count distinct DOCUMENT_ID markers in Neo4j and check for duplicates —
    the same non-duplication proof used at shakeout."""
    try:
        from neo4j import GraphDatabase
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN: neo4j driver not available for reconciliation: {exc!r}")
        return
    drv = GraphDatabase.driver(args.bolt, auth=("neo4j", args.password))
    try:
        with drv.session() as s:
            nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rows = s.run("MATCH (n) WHERE n.text CONTAINS '[DOCUMENT_ID:' RETURN n.text AS t").data()
    finally:
        drv.close()
    ids = []
    for r in rows:
        m = re.search(r"\[DOCUMENT_ID:\s*(doc_[0-9a-f]+)\]", r["t"] or "")
        if m:
            ids.append(m.group(1))
    from collections import Counter

    c = Counter(ids)
    recon = {
        "graph_nodes": nodes,
        "chunks_with_marker": len(ids),
        "distinct_doc_ids": len(set(ids)),
        "ledger_completed": completed_total,
        "duplicated_doc_ids": {k: v for k, v in c.items() if v > 1},
    }
    (out_dir / "graph_reconciliation.json").write_text(json.dumps(recon, indent=2), encoding="utf-8")
    dup = recon["duplicated_doc_ids"]
    _log(f"RECONCILE nodes={nodes} distinct_doc_ids={recon['distinct_doc_ids']} ledger_completed={completed_total} duplicates={len(dup)}")


def exigir_bolt_permitido(bolt: str) -> str:
    """Recusa um --bolt apontado ao Neo4j da dissertação.

    O `--bolt` já não tem valor por omissão, mas obrigatório não é o mesmo que
    validado: nada impedia que alguém lhe passasse à mão a URI antiga e
    indexasse por cima do grafo que sustenta a dissertação.

    Import tardio e com `sys.path` próprio de propósito: este script corre
    dentro de `.venvs/cognee`, que não tem o pacote `benchmark` instalado. O
    travão só depende da biblioteca padrão, portanto importa em qualquer venv.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from benchmark.infra.guard import exigir_alvo_permitido

    return exigir_alvo_permitido(bolt, origem="--bolt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", required=True)
    ap.add_argument("--n", type=int, default=None, help="limit docs (None=all 11515)")
    ap.add_argument("--workspace", required=True, help="cognee local stores (MUST be the production workspace to resume)")
    ap.add_argument("--out-dir", required=True, help="durable dir for ledger/usage/report (MUST be cognee_full to resume)")
    ap.add_argument("--dataset-name", default="musique_ans_v1_0_eval1k")
    ap.add_argument("--bolt", required=True, help="URI Bolt do Neo4j alvo. Sem valor por omissao de proposito: o antigo apontava ao grafo do ambiente original.")
    ap.add_argument("--password", default="benchmark_cognee")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--embedding-dimensions", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=300, help="docs per add+cognify cycle")
    ap.add_argument("--data-per-batch", type=int, default=20, help="cognee internal doc concurrency")
    ap.add_argument("--batches-per-process", type=int, default=1, help="exit after N batches so the supervisor relaunches a fresh process (memory reset); 0=unlimited (old behavior)")
    ap.add_argument("--resume", action="store_true", help="skip docs already completed in the ledger")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=20.0)
    ap.add_argument("--stop-after-failures", type=int, default=3, help="stop after N failed BATCHES")
    ap.add_argument("--verify-graph", action="store_true", help="reconcile distinct doc_ids vs ledger at the end")
    args = ap.parse_args()
    exigir_bolt_permitido(args.bolt)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
