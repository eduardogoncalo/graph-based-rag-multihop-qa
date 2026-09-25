"""Corre a recuperação do HippoRAG 2 sobre uma variante de grafo e compara com o
baseline original, pergunta a pergunta.

Fala com scripts/hipporag2_query_runner.py em modo --serve (JSONL por stdin), que
NÃO escreve no Postgres. Exploratório: as variantes de grafo deduplicado não
entraram na tese.

A comparação é da LISTA ORDENADA dos cinco documentos, não do conjunto: qualquer
alteração de ordem obriga a regenerar a resposta daquela pergunta.

Uso:
    uv run python scripts/hipporag2_variant_run.py \\
        --dataset musique --variant sum --limit 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("artifacts/hipporag2_dedup_work")
VARIANT_ROOT = Path("artifacts_variant")
SLUG = {"musique": "musique_ans_v1_0_eval1k", "twowiki": "twowiki_ans_v1_0_eval1k"}
RUNNER = "scripts/hipporag2_query_runner.py"
VENV = ".venvs/hipporag2/bin/python"


def recall_at_k(topk: list, gold: list) -> float:
    if not gold:
        return float("nan")
    return len(set(topk) & set(gold)) / len(gold)


def all_gold_at_k(topk: list, gold: list) -> float:
    if not gold:
        return float("nan")
    return 1.0 if set(gold) <= set(topk) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("musique", "twowiki"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = todas")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    baseline = json.loads((WORK / f"baseline_{args.dataset}.json").read_text(encoding="utf-8"))
    records = baseline["records"]
    if args.limit:
        records = records[: args.limit]

    workspace = VARIANT_ROOT / SLUG[args.dataset] / "hipporag2" / "workspace"
    save_dir = workspace / "gpt-4o-mini_text-embedding-3-small"
    active = save_dir / "graph.pickle"
    variant_file = save_dir / f"graph_{args.variant}.pickle"
    if not variant_file.exists():
        raise SystemExit(f"ERRO: variante ausente: {variant_file}")

    # Troca só o graph.pickle. Os embeddings e a cache ficam onde estão.
    if active.exists() or active.is_symlink():
        active.unlink()
    active.symlink_to(variant_file.name)
    import hashlib

    digest = hashlib.md5(variant_file.read_bytes()).hexdigest()[:12]
    print(f"grafo activo -> {variant_file.name} md5={digest}", file=sys.stderr)

    process = subprocess.Popen(
        [VENV, RUNNER, "--save-dir", str(workspace), "--serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    load_started = time.time()
    ready = process.stdout.readline()
    if not ready or "ready" not in ready:
        raise SystemExit(f"ERRO: runner não ficou pronto: {ready!r}")
    load_elapsed = time.time() - load_started
    print(f"índice carregado em {load_elapsed:.1f}s", file=sys.stderr)

    out_rows, changed, latencies = [], 0, []
    base_recall, new_recall = [], []
    base_allgold, new_allgold, pool_recall = [], [], []
    started_all = time.time()

    for index, record in enumerate(records, start=1):
        started = time.time()
        process.stdin.write(json.dumps({"query": record["question"], "top_k": args.top_k}) + "\n")
        process.stdin.flush()
        payload = json.loads(process.stdout.readline())
        latency = time.time() - started
        latencies.append(latency)

        if "error" in payload:
            print(f"  FALHA q={record['question_id']}: {payload['error']}", file=sys.stderr)
            continue

        # O pool é tudo o que a execução devolveu. O corte fica sempre em 5,
        # para que a comparação com o baseline seja da mesma unidade.
        pool = [i["document_id"] for i in payload["items"]]
        new_top5 = pool[:5]
        same = new_top5 == record["top5_document_ids"]
        changed += 0 if same else 1
        gold = record["gold_document_ids"]
        if gold:
            base_recall.append(recall_at_k(record["top5_document_ids"], gold))
            new_recall.append(recall_at_k(new_top5, gold))
            base_allgold.append(all_gold_at_k(record["top5_document_ids"], gold))
            new_allgold.append(all_gold_at_k(new_top5, gold))
            pool_recall.append(recall_at_k(pool, gold))

        out_rows.append(
            {
                "question_id": record["question_id"],
                "baseline_top5": record["top5_document_ids"],
                "variant_top5": new_top5,
                "ordered_list_same": same,
                "n_pool": len(pool),
                "pool_document_ids": pool if len(pool) > 5 else None,
                "latency_s": round(latency, 3),
                "variant_scores": [i["score"] for i in payload["items"][:5]],
                "baseline_scores": record["scores"],
            }
        )
        if index % 10 == 0 or index == len(records):
            print(
                f"  [{index}/{len(records)}] mudaram={changed} "
                f"~{sum(latencies)/len(latencies):.2f}s/q",
                file=sys.stderr,
            )

    process.stdin.close()
    process.wait(timeout=60)
    total = time.time() - started_all

    summary = {
        "dataset": args.dataset,
        "variant": args.variant,
        "n": len(out_rows),
        "index_load_s": round(load_elapsed, 1),
        "total_s": round(total, 1),
        "mean_latency_s": round(sum(latencies) / max(len(latencies), 1), 3),
        "requested_top_k": args.top_k,
        "mean_pool_size": round(sum(r["n_pool"] for r in out_rows) / max(len(out_rows), 1), 2),
        "ordered_list_changed": changed,
        "ordered_list_changed_pct": round(100 * changed / max(len(out_rows), 1), 2),
        "baseline_recall_at_5": round(sum(base_recall) / max(len(base_recall), 1), 4),
        "variant_recall_at_5": round(sum(new_recall) / max(len(new_recall), 1), 4),
        "baseline_all_gold_at_5": round(sum(base_allgold) / max(len(base_allgold), 1), 4),
        "variant_all_gold_at_5": round(sum(new_allgold) / max(len(new_allgold), 1), 4),
        "variant_recall_at_pool": round(sum(pool_recall) / max(len(pool_recall), 1), 4),
    }
    tag = f"_k{args.top_k}" if args.top_k != 5 else ""
    out = WORK / f"variant_{args.dataset}_{args.variant}{tag}{'_n' + str(args.limit) if args.limit else ''}.json"
    out.write_text(json.dumps({"summary": summary, "rows": out_rows}, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
