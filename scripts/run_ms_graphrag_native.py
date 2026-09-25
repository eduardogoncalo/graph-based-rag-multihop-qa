"""NATIVE GraphRAG (Arm B, "as deployed") — MS GraphRAG answers on its own.

Runs MS GraphRAG's OWN generation (`engine.search()` via
`graphrag_query_runner.py --generate`) over the 1000 MuSiQue eval1k questions and
materializes it as `musique_eval1k_ms_graphrag_native`. Analogous to run_cognee_native.py.

Difference vs Arm A (controlled): there GraphRAG only supplies CONTEXT and our
reader generates; here it is GraphRAG's own NATIVE generation (local/global search with an LLM).
Internal model = gpt-4o-mini (index config). `--method local` = default (fact-centric,
right for multi-hop QA; global = thematic).

Modes:
  --smoke [N]  hop-stratified sample (8/5/2 = 15); generates native answers, writes
               JSONL + gate; does NOT write to the benchmark tables.
  --full       all 1000; persists to musique_eval1k_ms_graphrag_native (idempotent,
               resumable by question_id). The judge is a separate step (run_llm_judge.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmark.cli.app import _connect_postgres, _ms_graphrag_adapter

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "musique"
DATASET_VERSION = "ans_v1.0_eval1k"
QUESTIONS = ROOT / "data/canonical/musique_ans_v1.0_eval1k/questions.jsonl"
REPORTS = ROOT / "artifacts/musique/reports"

TARGET_EXP = "musique_eval1k_ms_graphrag_native"
METHOD = "ms_graphrag"
AGENT_MODE = "single_agent"

SMOKE_STRATA = {"2": 8, "3": 5, "4": 2}


def _bind_dataset(dataset_id: str, dataset_version: str) -> None:
    """Rebinda os globals p/ outro dataset (default musique = retrocompatível)."""
    global DATASET_ID, DATASET_VERSION, QUESTIONS, REPORTS, TARGET_EXP
    DATASET_ID, DATASET_VERSION = dataset_id, dataset_version
    QUESTIONS = ROOT / f"data/canonical/{dataset_id}_{dataset_version}/questions.jsonl"
    REPORTS = ROOT / f"artifacts/{dataset_id}/reports"
    TARGET_EXP = f"{dataset_id}_eval1k_ms_graphrag_native"


def _hop_of(q: dict[str, Any]) -> str:
    dec = (q.get("metadata") or {}).get("decomposition")
    return str(len(dec)) if isinstance(dec, list) else "?"


def _gold_str(q: dict[str, Any]) -> str:
    g = q.get("gold_answer")
    return " | ".join(map(str, g)) if isinstance(g, list) else ("" if g is None else str(g))


def _load_questions() -> list[dict[str, Any]]:
    rows = []
    with QUESTIONS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _smoke_sample(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hop: dict[str, list] = {}
    for q in sorted(questions, key=lambda r: r["question_id"]):
        by_hop.setdefault(_hop_of(q), []).append(q)
    picked = []
    for hop, k in SMOKE_STRATA.items():
        picked.extend(by_hop.get(hop, [])[:k])
    if not picked:
        # dataset sem metadata.decomposition (ex.: twowiki): primeiras N por
        # question_id, mesmo critério do smoke do braço A
        picked = sorted(questions, key=lambda r: r["question_id"])[: sum(SMOKE_STRATA.values())]
    return picked


def _generate_one(adapter, question: dict[str, Any], method: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = adapter.query_native(query=question["question"], query_method=method)
        data = json.loads(result.stdout) if result.stdout.strip() else {}
        answer = str(data.get("answer") or "").strip()
        return {
            "question_id": question["question_id"], "hop": _hop_of(question),
            "question": question["question"], "gold_answer": _gold_str(question),
            "native_answer": answer, "answer_len": len(answer),
            "query_method": method, "latency_ms": (time.perf_counter() - started) * 1000,
            "prompt_tokens": (data.get("stats") or {}).get("prompt_tokens"),
            "status": "completed" if answer else "empty", "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "question_id": question["question_id"], "hop": _hop_of(question),
            "question": question["question"], "gold_answer": _gold_str(question),
            "native_answer": "", "answer_len": 0, "query_method": method,
            "latency_ms": (time.perf_counter() - started) * 1000, "prompt_tokens": None,
            "status": "error", "error": f"{type(exc).__name__}: {exc}",
        }


def _gold_hit(row: dict[str, Any]) -> bool:
    ans = row["native_answer"].lower()
    return bool(ans) and any(g.strip().lower() in ans for g in row["gold_answer"].split(" | ") if g.strip())


# ----------------------------- smoke -----------------------------

def run_smoke(n: int, method: str) -> bool:
    REPORTS.mkdir(parents=True, exist_ok=True)
    sample = _smoke_sample(_load_questions())
    if n != sum(SMOKE_STRATA.values()):
        sample = sample[:n]
    print(f"[smoke] {len(sample)} questions (method={method}); "
          f"hops={ {h: sum(1 for q in sample if _hop_of(q)==h) for h in SMOKE_STRATA} }", flush=True)
    adapter = _ms_graphrag_adapter(DATASET_ID, DATASET_VERSION)
    rows = []
    for i, q in enumerate(sample, 1):
        row = _generate_one(adapter, q, method)
        rows.append(row)
        flag = "OK " if row["status"] == "completed" else "!! "
        hit = "✓gold" if _gold_hit(row) else "     "
        print(f"[smoke] {flag}{hit} {i:2d}/{len(sample)} {row['hop']}hop len={row['answer_len']:4d} "
              f"{row['latency_ms']/1000:5.1f}s :: {row['native_answer'][:80]!r}", flush=True)

    tech_fail = sum(1 for r in rows if r["status"] == "error")
    non_empty = sum(1 for r in rows if r["answer_len"] > 0)
    gold_hits = sum(1 for r in rows if _gold_hit(r))
    gates = {
        "0 technical failures": tech_fail == 0,
        "all answers non-empty": non_empty == len(rows),
    }
    passed = all(gates.values())
    (REPORTS / "ms_graphrag_native_smoke.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print()
    for name, ok in gates.items():
        print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}")
    print(f"[smoke] rough gold-hit (informational): {gold_hits}/{len(rows)}")
    print(f"[smoke] mean latency: {sum(r['latency_ms'] for r in rows)/max(len(rows),1)/1000:.1f}s")
    print(f"[smoke] GATE {'PASS' if passed else 'FAIL'}")
    return passed


# ----------------------------- full -----------------------------

def run_full(method: str, limit: int | None = None) -> None:
    from benchmark.core.ids import deterministic_id
    from benchmark.storage.experiment_store import ExperimentStore

    questions = _load_questions()
    conn = _connect_postgres()
    store = ExperimentStore(conn)
    store.create_experiment(
        experiment_id=TARGET_EXP, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
        metadata={"branch": "as_deployed", "framework": "ms_graphrag", "native_qa": True,
                  "query_method": method, "note": "QA nativo do MS GraphRAG (engine.search); braço B"},
    )
    with conn.cursor() as cur:
        cur.execute("SELECT a.question_id FROM answers a JOIN runs r ON a.run_id=r.run_id "
                    "WHERE r.experiment_id=%s AND r.method_id=%s", (TARGET_EXP, METHOD))
        done = {row[0] for row in cur.fetchall()}
    todo = sorted(
        (q for q in questions if q["question_id"] not in done),
        key=lambda q: q["question_id"],
    )
    if limit:
        todo = todo[:limit]
    print(f"[full] {len(questions)} questions · {len(done)} already done · {len(todo)} this session "
          f"(method={method}{f', limit={limit}' if limit else ''})", flush=True)
    if not todo:
        print("[full] nothing to do", flush=True); conn.close(); return

    adapter = _ms_graphrag_adapter(DATASET_ID, DATASET_VERSION)
    written = empties = 0
    for i, q in enumerate(todo, 1):
        row = _generate_one(adapter, q, method)
        if row["status"] == "error":
            print(f"[full] ERROR {q['question_id']}: {row['error']} — aborting", flush=True)
            raise SystemExit(1)
        if row["answer_len"] == 0:
            empties += 1
        qid = q["question_id"]
        run_id = deterministic_id("run", [TARGET_EXP, DATASET_ID, DATASET_VERSION, METHOD, AGENT_MODE, qid])
        prov = {"native_qa": True, "branch": "as_deployed", "framework": "ms_graphrag",
                "query_method": method, "hop": row["hop"]}
        store.create_run(experiment_id=TARGET_EXP, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
                         method_id=METHOD, agent_mode=AGENT_MODE, run_id=run_id, status="materialized",
                         metadata={"native_qa": True, "branch": "as_deployed", "framework": "ms_graphrag"})
        store.persist_answer(run_id=run_id, method_id=METHOD, agent_mode=AGENT_MODE,
                             answer_text=row["native_answer"], question_id=qid,
                             latency_ms=row["latency_ms"], metadata=prov)
        conn.commit()
        written += 1
        if i % 25 == 0 or i == len(todo):
            print(f"[full] {i}/{len(todo)} written ({empties} empty) last: {row['latency_ms']/1000:.1f}s", flush=True)
    print(f"[full] done: {written} written to {TARGET_EXP} ({empties} empty)", flush=True)
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", nargs="?", type=int, const=sum(SMOKE_STRATA.values()))
    g.add_argument("--full", action="store_true")
    ap.add_argument("--method", default="local", choices=["local", "global"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    args = ap.parse_args()
    _bind_dataset(args.dataset_id, args.dataset_version)
    if args.full:
        run_full(args.method, limit=args.limit)
    else:
        raise SystemExit(0 if run_smoke(args.smoke, args.method) else 1)


if __name__ == "__main__":
    main()
