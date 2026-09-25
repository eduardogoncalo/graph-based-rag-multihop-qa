"""Recompute evaluation_results from already-persisted answers + retrieval_items.

No retrieval, no LLM, no indexing. Deterministic re-evaluation only.

Why a dedicated script (not run_experiment_batch.py): that runner SKIPS any
run_id already in `evaluated`, so it would never recompute. And a naive
recompute would DUPLICATE rows, because `evaluation_result_id` is derived from
the metric *value* (experiment_store.persist_evaluation_result) -- a changed
value (0.0 -> 0.6) yields a new id and INSERTs instead of UPDATEs. So we
DELETE the affected evaluation_results first, then recompute.

Usage:
    .venv/bin/python scripts/recompute_eval.py \
        --experiment-id musique_eval1k_first_results --method vector_rag
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time

from benchmark.cli.app import _connect_postgres
from benchmark.evaluation.evaluator import evaluate_run
from benchmark.storage.experiment_store import ExperimentStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--method", default="vector_rag")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats but DELETE/persist nothing.",
    )
    args = ap.parse_args()

    conn = _connect_postgres()
    store = ExperimentStore(conn)

    # Only runs that actually have an answer can be (re)evaluated.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT r.run_id
            FROM runs r
            JOIN answers a ON a.run_id = r.run_id
            WHERE r.experiment_id = %s AND r.method_id = %s
            ORDER BY r.run_id
            """,
            (args.experiment_id, args.method),
        )
        run_ids = [row[0] for row in cur.fetchall()]

    print(
        f"recompute exp={args.experiment_id} method={args.method} "
        f"runs_with_answers={len(run_ids)} k={args.k} dry_run={args.dry_run}",
        flush=True,
    )
    if not run_ids:
        print("nothing to do")
        return 0

    if not args.dry_run:
        # Delete-first (scoped) so changed metric values UPDATE in place rather
        # than create duplicate rows that would pollute compare/report means.
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM evaluation_results WHERE run_id = ANY(%s)",
                (run_ids,),
            )
            deleted = cur.rowcount
        conn.commit()
        print(f"deleted {deleted} stale evaluation_results rows", flush=True)

    ok = fail = 0
    recalls: list[float] = []
    precs: list[float] = []
    t0 = time.monotonic()
    for i, rid in enumerate(run_ids, 1):
        try:
            if args.dry_run:
                payload = store.load_evaluation_input(run_id=rid, k=args.k)
                from benchmark.evaluation.evaluator import evaluate_input

                summary = evaluate_input(payload)  # no persist
            else:
                summary = evaluate_run(run_id=rid, store=store, k=args.k)
            recalls.append(summary.metrics[f"evidence_recall_at_{args.k}"])
            precs.append(summary.metrics[f"precision_at_{args.k}"])
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            print(f"FAIL run_id={rid} ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
        if i % args.progress_every == 0:
            print(f"[{i}/{len(run_ids)}] ok={ok} fail={fail}", flush=True)

    elapsed = time.monotonic() - t0
    print(
        f"DONE ok={ok} fail={fail} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    if recalls:
        print(
            f"recall@{args.k}: mean={st.mean(recalls):.4f} "
            f">0={sum(1 for x in recalls if x > 0)}/{len(recalls)} max={max(recalls):.3f}",
            flush=True,
        )
        print(
            f"precision@{args.k}: mean={st.mean(precs):.4f} "
            f">0={sum(1 for x in precs if x > 0)}/{len(precs)}",
            flush=True,
        )
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
