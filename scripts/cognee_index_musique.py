#!/usr/bin/env python
"""Resilient cognee indexer for MuSiQue — runs in the ISOLATED venv (.venvs/cognee).

Self-contained on purpose: imports only stdlib + cognee (+ litellm for cost, neo4j
for verification). Does NOT import the benchmark package, so it is immune to the
core/isolated dependency split.

Resilience (mirrors how lightrag_neo4j survived a suspend):
  * Durable per-document checkpoint ledger keyed by canonical ``document_id``.
  * ``--resume`` skips docs already ``completed`` → re-running after a hibernate /
    closed terminal / kill never re-indexes a doc (no duplication).
  * Per-doc retry with cooldown; longer cooldown on rate-limit.
  * Ledger is written atomically (tmp + rename) after every doc, so a crash at any
    point leaves a consistent ledger.
Launch detached (``nohup setsid ... &``) so it outlives the shell; hibernation just
pauses/resumes it and the retry handles any dropped LLM/Neo4j connections.

Stores: graph -> Neo4j (Option C, via env); relational+vector -> cognee local
defaults (sqlite + lancedb) under the workspace (D2 local-first). Generation +
embeddings -> the shared OpenAI models (gpt-4o-mini / text-embedding-3-small).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
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
# --------------------------------------------------------------------------- #
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}}
# OpenAI prices (USD per 1M tokens) for the shared models.
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

    # CustomLogger fires for BOTH sync and async litellm calls (cognee is async).
    litellm.callbacks = list(getattr(litellm, "callbacks", []) or []) + [_CostLogger()]
    # function-style fallback for any sync path
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
# Checkpoint ledger (durable, resumable, dedup-by-document_id)
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
# Documents / questions
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


def load_questions(path: Path, n: int) -> list[dict]:
    qs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
            if len(qs) >= n:
                break
    return qs


def doc_payload(doc: dict) -> str:
    """Embed the canonical document_id as a marker so provenance is recoverable
    from cognee's chunks/graph without relying on cognee internals."""
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
        # contain all local state under the throwaway workspace
        "DATA_ROOT_DIRECTORY": str(workspace / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(workspace / "system"),
        "CACHE_ROOT_DIRECTORY": str(workspace / "cache"),
        "COGNEE_LOGS_DIR": str(workspace / "logs"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "CACHING": "false",
        "TELEMETRY_DISABLED": "true",
        # graph -> Neo4j Option C
        "GRAPH_DATABASE_PROVIDER": "neo4j",
        "GRAPH_DATABASE_URL": args.bolt,
        "GRAPH_DATABASE_USERNAME": "neo4j",
        "GRAPH_DATABASE_PASSWORD": args.password,
        "GRAPH_DATABASE_NAME": "neo4j",
        "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
        # relational + vector -> cognee local defaults (sqlite + lancedb)
        "DB_PROVIDER": "sqlite",
        "VECTOR_DB_PROVIDER": "lancedb",
        # models (shared with the other methods)
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

    t0 = time.monotonic()
    n_done = 0
    n_skipped = 0
    for i, doc in enumerate(docs, 1):
        doc_id = doc["document_id"]
        rec = ledger["docs"].get(doc_id)
        if args.resume and rec and rec.get("status") == "completed":
            n_skipped += 1
            continue

        ledger["docs"][doc_id] = {"status": "running", "attempts": (rec or {}).get("attempts", 0) + 1, "started_at": _now()}
        save_ledger(ledger_path, ledger)

        d_t0 = time.monotonic()
        calls_before = USAGE["calls"]
        try:
            payload = doc_payload(doc)

            async def _add():
                return await cognee.add([payload], dataset_name=args.dataset_name)

            async def _cognify():
                return await cognee.cognify(datasets=[args.dataset_name])

            await with_retry(_add, label=f"add[{doc_id}]", max_retries=args.max_retries, cooldown=args.cooldown)
            await with_retry(_cognify, label=f"cognify[{doc_id}]", max_retries=args.max_retries, cooldown=args.cooldown)
        except Exception as exc:  # noqa: BLE001
            ledger["docs"][doc_id] = {"status": "failed", "attempts": ledger["docs"][doc_id]["attempts"], "error": f"{type(exc).__name__}: {str(exc)[:300]}", "failed_at": _now()}
            save_ledger(ledger_path, ledger)
            usage_path.write_text(json.dumps({**USAGE, "estimated_cost_usd": estimated_cost_usd()}, indent=2), encoding="utf-8")
            _log(f"FAIL [{i}/{len(docs)}] {doc_id}: {type(exc).__name__}: {str(exc)[:200]}")
            if args.stop_after_failures and (sum(1 for v in ledger['docs'].values() if v.get('status') == 'failed') >= args.stop_after_failures):
                _log("STOP: reached stop_after_failures threshold")
                return 1
            continue

        elapsed = time.monotonic() - d_t0
        ledger["docs"][doc_id] = {
            "status": "completed",
            "attempts": ledger["docs"][doc_id]["attempts"],
            "elapsed_s": round(elapsed, 2),
            "llm_calls": USAGE["calls"] - calls_before,
            "completed_at": _now(),
        }
        save_ledger(ledger_path, ledger)
        usage_path.write_text(json.dumps({**USAGE, "estimated_cost_usd": estimated_cost_usd()}, indent=2), encoding="utf-8")
        n_done += 1
        _log(f"OK [{i}/{len(docs)}] {doc_id} elapsed={elapsed:.1f}s calls={USAGE['calls'] - calls_before} cum_cost=${estimated_cost_usd():.4f}")

    total_elapsed = time.monotonic() - t0
    completed_total = sum(1 for d in ledger["docs"].values() if d.get("status") == "completed")
    summary = {
        "dataset_name": args.dataset_name,
        "docs_input": len(docs),
        "docs_completed_total": completed_total,
        "docs_done_this_run": n_done,
        "docs_skipped_resume": n_skipped,
        "docs_failed": sum(1 for d in ledger["docs"].values() if d.get("status") == "failed"),
        "total_elapsed_s_this_run": round(total_elapsed, 1),
        "avg_s_per_doc": round(total_elapsed / max(n_done, 1), 1),
        "usage": USAGE,
        "estimated_cost_usd": estimated_cost_usd(),
        "cost_per_doc_usd": round(estimated_cost_usd() / max(completed_total, 1), 5),
        "finished_at": _now(),
    }
    (out_dir / "index_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"INDEX_DONE completed={completed_total}/{len(docs)} this_run={n_done} skipped={n_skipped} "
         f"elapsed={total_elapsed:.0f}s cost=${estimated_cost_usd():.4f} cost/doc=${summary['cost_per_doc_usd']:.5f}")

    # --- retrieval smoke (only_context) ----------------------------------- #
    if args.questions and args.n_questions > 0:
        await retrieval_smoke(cognee, args, out_dir)

    return 0


async def retrieval_smoke(cognee, args, out_dir: Path) -> None:
    qs = load_questions(Path(args.questions), args.n_questions)
    _log(f"retrieval smoke on {len(qs)} questions")
    results = []
    SearchType = None
    try:
        from cognee.modules.search.types import SearchType as _ST
        SearchType = _ST
    except Exception:
        try:
            SearchType = cognee.SearchType  # type: ignore[attr-defined]
        except Exception:
            SearchType = None

    for q in qs:
        qtext = q["question"]
        entry = {"question_id": q.get("question_id"), "question": qtext, "gold_answer": q.get("gold_answer")}
        # try recall(scope=graph) then search(CHUNKS)
        for label, factory in [
            ("recall", lambda: cognee.recall(qtext, datasets=[args.dataset_name], top_k=args.top_k, only_context=True, auto_route=False, scope="graph")),
            ("search_chunks", (lambda: cognee.search(qtext, query_type=getattr(SearchType, "CHUNKS", None), datasets=[args.dataset_name], top_k=args.top_k, only_context=True)) if SearchType is not None else None),
        ]:
            if factory is None:
                continue
            try:
                res = await factory()
                entry[label] = _jsonable(res)
                entry[f"{label}_ok"] = True
            except Exception as exc:  # noqa: BLE001
                entry[label] = f"{type(exc).__name__}: {str(exc)[:200]}"
                entry[f"{label}_ok"] = False
        results.append(entry)
        _log(f"  q={q.get('question_id')} recall_ok={entry.get('recall_ok')} search_ok={entry.get('search_chunks_ok')}")

    (out_dir / "retrieval_smoke.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


def _jsonable(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


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
    ap.add_argument("--questions", default=None)
    ap.add_argument("--n", type=int, default=None, help="limit docs (None=all)")
    ap.add_argument("--n-questions", type=int, default=0)
    ap.add_argument("--workspace", required=True, help="throwaway dir for cognee local stores")
    ap.add_argument("--out-dir", required=True, help="durable dir for ledger/usage/report")
    ap.add_argument("--dataset-name", default="musique_shakeout")
    ap.add_argument("--bolt", required=True, help="Bolt URI of the target Neo4j. No default on purpose: the old one pointed at the original environment's graph.")
    ap.add_argument("--password", default="benchmark_cognee")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--embedding-dimensions", type=int, default=1536)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=20.0)
    ap.add_argument("--stop-after-failures", type=int, default=3)
    args = ap.parse_args()
    exigir_bolt_permitido(args.bolt)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
