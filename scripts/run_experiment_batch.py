"""Batch runner: execute a single-agent method over all eval1k questions.

For each question: run_agent_question (retrieve + generate + persist) then
evaluate_run (deterministic metrics). Resumable: skips questions already
evaluated. Robust: per-question retry + rollback so one failure never aborts
the batch.

Dataset é parametrizável (--dataset-id/--dataset-version, default musique —
retrocompatível); a seleção de perguntas SEMPRE filtra pelo dataset para não
varrer perguntas de outros datasets persistidos no mesmo Postgres.

Usage:
  uv run python scripts/run_experiment_batch.py --method vector_rag \
      --experiment-id musique_eval1k_first_results [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark.cli.app import _connect_postgres
from benchmark.core.ids import deterministic_id
from benchmark.core.naming import slugify_dataset_version
from benchmark.evaluation import evaluate_run
from benchmark.experiments.run_single import load_question, run_agent_question
from benchmark.storage.experiment_store import ExperimentStore

AGENT_MODE = "single_agent"


def _make_run_id(dataset_id: str, dataset_version: str):
    def _run_id(experiment_id: str, method: str, qid: str) -> str:
        return deterministic_id(
            "run",
            [experiment_id, dataset_id, dataset_version, method, AGENT_MODE, qid],
        )

    return _run_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--dataset-id", default="musique")
    ap.add_argument("--dataset-version", default="ans_v1.0_eval1k")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    method = args.method
    exp = args.experiment_id
    dataset_id = args.dataset_id
    dataset_version = args.dataset_version
    _run_id = _make_run_id(dataset_id, dataset_version)
    # O checkpoint acompanha o dataset, e o directório é criado.
    #
    # Era `artifacts/musique/reports/batch_{method}_status.json`, fixo. Duas
    # coisas erradas nisso, e as duas só se veem a correr: o caminho nomeia um
    # dataset que pode não ser o que está a correr, e o directório só existia
    # no ambiente onde a fase experimental correu — numa cópia limpa o batch
    # completa as perguntas todas e **morre a escrever o recibo**, com
    # FileNotFoundError, depois de já ter gasto as chamadas.
    slug = slugify_dataset_version(args.dataset_id, args.dataset_version)
    checkpoint = args.checkpoint or f"artifacts/{slug}/reports/batch_{method}_status.json"
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)

    conn = _connect_postgres()
    store = ExperimentStore(conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT question_id FROM questions "
            "WHERE dataset_id = %s AND dataset_version = %s ORDER BY question_id",
            (dataset_id, dataset_version),
        )
        qids = [r[0] for r in cur.fetchall()]
    if args.limit:
        qids = qids[: args.limit]
    if args.shard_count > 1:
        qids = [q for i, q in enumerate(qids) if i % args.shard_count == args.shard_index]
        print(f"SHARD {args.shard_index}/{args.shard_count}: {len(qids)} questions", flush=True)

    # resume sets: already-evaluated run_ids, and answered-but-not-evaluated
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT run_id FROM evaluation_results")
        evaluated = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT DISTINCT run_id FROM answers")
        answered = {r[0] for r in cur.fetchall()}

    print(
        f"START method={method} exp={exp} questions={len(qids)} "
        f"already_evaluated={sum(1 for q in qids if _run_id(exp, method, q) in evaluated)}",
        flush=True,
    )

    ok = skipped = failed = 0
    sum_em = sum_f1 = sum_rec = 0.0
    t0 = time.monotonic()

    def write_ckpt(i: int, state: str) -> None:
        n = max(ok, 1)
        payload = {
            "method": method,
            "experiment_id": exp,
            "state": state,
            "total": len(qids),
            "processed_index": i,
            "ok": ok,
            "skipped": skipped,
            "failed": failed,
            "running_avg": {
                "exact_match": round(sum_em / n, 4),
                "answer_f1": round(sum_f1 / n, 4),
                f"evidence_recall_at_{args.top_k}": round(sum_rec / n, 4),
            },
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        with open(checkpoint, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    for i, qid in enumerate(qids, 1):
        rid = _run_id(exp, method, qid)
        if rid in evaluated:
            skipped += 1
            continue

        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                if rid not in answered:
                    question = load_question(conn, qid)
                    run_agent_question(
                        connection=conn,
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        method_id=method,
                        experiment_id=exp,
                        question_id=qid,
                        question=question,
                        agent_mode=AGENT_MODE,
                        top_k=args.top_k,
                    )
                summary = evaluate_run(run_id=rid, store=store, k=args.top_k)
                m = summary.metrics
                sum_em += float(m.get("exact_match", 0.0))
                sum_f1 += float(m.get("answer_f1", 0.0))
                sum_rec += float(m.get(f"evidence_recall_at_{args.top_k}", 0.0))
                evaluated.add(rid)
                ok += 1
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
                if attempt < args.retries:
                    time.sleep(2 * attempt)

        if last_err is not None:
            failed += 1
            print(f"FAIL q={qid} ({type(last_err).__name__}: {str(last_err)[:120]})", flush=True)

        if i % args.progress_every == 0:
            rate = i / max(time.monotonic() - t0, 1e-9) * 60
            print(
                f"[{i}/{len(qids)}] ok={ok} skip={skipped} fail={failed} "
                f"~{rate:.1f} q/min  avg_em={sum_em/max(ok,1):.3f} "
                f"avg_f1={sum_f1/max(ok,1):.3f} avg_rec={sum_rec/max(ok,1):.3f}",
                flush=True,
            )
            write_ckpt(i, "running")

    write_ckpt(len(qids), "done")
    print(
        f"DONE method={method} ok={ok} skipped={skipped} failed={failed} "
        f"elapsed_s={round(time.monotonic()-t0,1)} "
        f"avg_em={sum_em/max(ok,1):.4f} avg_f1={sum_f1/max(ok,1):.4f} "
        f"avg_rec={sum_rec/max(ok,1):.4f}",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
