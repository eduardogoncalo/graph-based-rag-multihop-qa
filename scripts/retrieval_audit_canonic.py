"""Canonical retrieval audit: the retrieval metrics the thesis reports.

recall@5, all_gold@5, recall@pool and docs/q in Tables 2 and 3 of the thesis
come from this computation, not from the `evidence_recall@k` that
`run_experiment_batch.py` stores. That one is computed at the run's own top_k,
which is 40 for LightRAG and 20 for Microsoft GraphRAG, so it is not
comparable across methods.

It recomputes everything straight from the raw tables (retrieval_items against
gold_evidence), for every cell of the controlled arm and the oracle.

Per question, over the list of DOCUMENTS deduplicated in rank order:
  - recall@{2,5,10,inf}  = |gold ∩ first k docs| / |gold|   (inf = the whole pool)
  - all_gold@{2,5,10,inf} = 1 if every gold document is in the first k
  - first/last gold rank, and the number of distinct documents (docs/q)

How each method's document id is extracted:
  - vector_rag, ms_graphrag, hipporag2, oracle: source_document_id
  - lightrag_neo4j: the prefix of the chunk id (doc_X-chunk-NNN), else source_document_id
  - cognee: source_document_id plus every doc_* id found in the returned blob, in
    order of first appearance. Cognee does not return a ranked list, so @k is a
    truncation of a set, not a ranking; the output flags it as ranked=false.

Report-only: it writes no table. Output: <reports-dir>/retrieval_audit_canonic_2026-07.json
(and a _per_question.json and a .md next to it). The file name is kept from the
experimental phase, because the export scripts in scripts/provenance/ read it.

Usage:
    python scripts/retrieval_audit_canonic.py --dataset-id musique \\
        --dataset-version ans_v1.0_eval1k --reports-dir artifacts/musique/reports
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STEM = "retrieval_audit_canonic_2026-07"
KS = (2, 5, 10)
CHUNK_SUFFIX_RE = re.compile(r"^(doc_[0-9a-f]+)-chunk-\d+$")
DOC_ID_RE = re.compile(r"(doc_[0-9a-f]{6,})", re.IGNORECASE)

# Published anchors for the dense sanity gate (HippoRAG 2, ICML 2025, Table 3,
# MuSiQue passage recall@5, same corpus protocol; the retrievers differ).
ANCHORS = {
    "BM25": 0.435,
    "Contriever": 0.466,
    "GTR (T5-base)": 0.491,
    "NV-Embed-v2": 0.697,
    "HippoRAG 2 (gpt-4o-mini)": 0.742,
}
GATE_MIN = 0.40


def default_prefix(dataset_id: str) -> str:
    return f"{dataset_id}_eval1k_"


def cells(prefix: str) -> list[tuple[str, str, str, bool]]:
    """(experiment_id, method_id, label, ranked) for every audited cell."""
    return [
        (prefix + "vector_v1free", "vector_rag", "dense baseline", True),
        (prefix + "lightrag_v1free", "lightrag_neo4j", "LightRAG (chunks)", True),
        (prefix + "ms_graphrag_v1free", "ms_graphrag", "Microsoft GraphRAG (local search)", True),
        (prefix + "hipporag2_v1free", "hipporag2", "HippoRAG 2", True),
        (prefix + "cognee_v1free_k5", "cognee", "Cognee (not ranked)", False),
        (prefix + "oracle_gold_v1free", "single_document_context", "oracle (sanity: 1.0)", True),
    ]


def docs_for_method(method_id: str, rows: list[tuple]) -> list[str]:
    """rows = (rank, source_document_id, source_chunk_id, text), in rank order.
    Returns the document ids, deduplicated, preserving rank order."""
    seen: list[str] = []

    def add(doc: str | None) -> None:
        if doc and doc.startswith("doc_") and doc not in seen:
            seen.append(doc)

    for _rank, source_document_id, source_chunk_id, text in rows:
        if method_id == "lightrag_neo4j":
            match = CHUNK_SUFFIX_RE.match(source_chunk_id or "")
            add(match.group(1) if match else source_document_id)
        elif method_id == "cognee":
            add(source_document_id)
            for found in DOC_ID_RE.findall(text or ""):
                add(found)
        else:
            add(source_document_id)
    return seen


def per_question_metrics(docs: list[str], gold: set[str]) -> dict[str, Any]:
    n_gold = len(gold)
    gold_positions = [i + 1 for i, doc in enumerate(docs) if doc in gold]
    out: dict[str, Any] = {
        "retrieved_docs": len(docs),
        "first_gold_rank": gold_positions[0] if gold_positions else None,
        "last_gold_rank": gold_positions[-1] if gold_positions else None,
    }
    for k in (*KS, None):
        subset = docs if k is None else docs[:k]
        hits = len(gold & set(subset))
        tag = "inf" if k is None else str(k)
        out[f"recall@{tag}"] = hits / n_gold
        out[f"all_gold@{tag}"] = 1.0 if hits == n_gold else 0.0
    return out


def _mean(entries: list[dict], key: str) -> float | None:
    values = [entry[key] for entry in entries if entry.get(key) is not None]
    return round(statistics.fmean(values), 4) if values else None


def aggregate(entries: list[dict], *, label: str, ranked: bool) -> dict[str, Any]:
    by_hop = {}
    for hop in (2, 3, 4):
        pool = [entry for entry in entries if entry.get("n_hops") == hop]
        if pool:
            by_hop[hop] = {
                "n": len(pool),
                "recall@5": _mean(pool, "recall@5"),
                "all_gold@5": _mean(pool, "all_gold@5"),
                "recall@inf": _mean(pool, "recall@inf"),
            }
    found = [entry for entry in entries if entry["first_gold_rank"] is not None]
    summary: dict[str, Any] = {
        "label": label,
        "ranked": ranked,
        "n_questions": len(entries),
        "mean_retrieved_docs": _mean(entries, "retrieved_docs"),
        "first_gold_rank_mean(found)": (
            round(statistics.fmean(e["first_gold_rank"] for e in found), 2) if found else None
        ),
        "last_gold_rank_mean(found)": (
            round(statistics.fmean(e["last_gold_rank"] for e in found), 2) if found else None
        ),
        "zero_gold_questions": len(entries) - len(found),
        "by_hop": by_hop,
    }
    for k in ("2", "5", "10", "inf"):
        summary[f"recall@{k}"] = _mean(entries, f"recall@{k}")
        summary[f"all_gold@{k}"] = _mean(entries, f"all_gold@{k}")
    return summary


def audit(cursor: Any, *, dataset_id: str, dataset_version: str, prefix: str) -> tuple[dict, dict]:
    cursor.execute(
        """SELECT ge.question_id, ge.document_id
           FROM gold_evidence ge JOIN questions q ON q.question_id = ge.question_id
           WHERE q.dataset_id = %s AND q.dataset_version = %s""",
        (dataset_id, dataset_version),
    )
    gold: dict[str, set[str]] = defaultdict(set)
    for question_id, document_id in cursor.fetchall():
        gold[question_id].add(document_id)
    # Chain depth: MuSiQue records a decomposition per question; 2Wiki does not,
    # and there the per-hop breakdown is simply empty.
    cursor.execute(
        """SELECT question_id, jsonb_array_length(metadata->'decomposition')
           FROM questions WHERE dataset_id = %s AND dataset_version = %s
             AND jsonb_typeof(metadata->'decomposition') = 'array'""",
        (dataset_id, dataset_version),
    )
    hops = dict(cursor.fetchall())

    summary: dict[str, dict] = {}
    per_question: dict[str, list[dict]] = {}
    for experiment_id, method_id, label, ranked in cells(prefix):
        cursor.execute(
            """SELECT a.question_id, ri.rank, ri.source_document_id, ri.source_chunk_id, ri.text
               FROM retrieval_items ri
               JOIN retrieval_results rr ON rr.retrieval_result_id = ri.retrieval_result_id
               JOIN runs r ON r.run_id = rr.run_id
               JOIN answers a ON a.run_id = r.run_id
               WHERE r.experiment_id = %s AND rr.method_id = %s
               ORDER BY a.question_id, ri.rank""",
            (experiment_id, method_id),
        )
        by_question: dict[str, list[tuple]] = defaultdict(list)
        for question_id, rank, doc, chunk, text in cursor.fetchall():
            by_question[question_id].append((rank, doc, chunk, text))
        key = f"{experiment_id}::{method_id}"
        if not by_question:
            summary[key] = {"label": label, "n_questions": 0, "note": "no retrieval_items"}
            continue
        entries = []
        for question_id, rows in by_question.items():
            question_gold = gold.get(question_id)
            if not question_gold:
                continue
            entry = per_question_metrics(docs_for_method(method_id, rows), question_gold)
            entry["question_id"] = question_id
            entry["n_hops"] = hops.get(question_id)
            entries.append(entry)
        summary[key] = aggregate(entries, label=label, ranked=ranked)
        per_question[key] = entries
    return summary, per_question


def dense_gate(summary: dict[str, dict], prefix: str) -> dict[str, Any]:
    value = summary.get(f"{prefix}vector_v1free::vector_rag", {}).get("recall@5")
    return {
        "metric": "dense baseline recall@5",
        "value": value,
        "threshold_min": GATE_MIN,
        "anchors_published_recall@5": ANCHORS,
        "verdict": "PASS" if (value or 0) >= GATE_MIN else "FAIL",
        "note": "Protocol check against the HippoRAG 2 anchors (MuSiQue). A FAIL means "
        "investigate the protocol before reading any result. Not meaningful on a "
        "20-question smoke sample.",
    }


def render_markdown(payload: dict) -> str:
    lines = [
        f"# Canonical retrieval audit ({payload['dataset']}, {payload['generated_at'][:16]}Z)",
        "",
        f"Dense gate: **{payload['gate']['verdict']}** (recall@5 = {payload['gate']['value']})",
        "",
        "| cell | method | n | r@2 | r@5 | r@10 | r@pool | all_gold@5 | all_gold@pool | 1st gold | docs/q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, cell in payload["summary"].items():
        method = key.split("::")[1]
        if not cell.get("n_questions"):
            lines.append(f"| {cell['label']} | {method} | 0 | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {cell['label']} | {method} | {cell['n_questions']} | {cell['recall@2']} | "
            f"{cell['recall@5']} | {cell['recall@10']} | {cell['recall@inf']} | "
            f"{cell['all_gold@5']} | {cell['all_gold@inf']} | "
            f"{cell['first_gold_rank_mean(found)']} | {cell['mean_retrieved_docs']} |"
        )
    lines += [
        "",
        "- Cognee is not ranked: @k truncates a set.",
        "- LightRAG: document id taken from the chunk-id prefix.",
        "- The oracle must be 1.0 by construction (a sanity check of the audit itself).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--exp-prefix", default=None, help="default: <dataset-id>_eval1k_")
    parser.add_argument("--reports-dir", default=None, help="default: artifacts/<dataset-id>/reports")
    args = parser.parse_args()
    prefix = args.exp_prefix if args.exp_prefix is not None else default_prefix(args.dataset_id)
    reports_dir = Path(args.reports_dir or f"artifacts/{args.dataset_id}/reports")

    from benchmark.core.settings import load_settings
    from benchmark.storage.postgres import connect_postgres

    connection = connect_postgres(load_settings())
    try:
        with connection.cursor() as cursor:
            summary, per_question = audit(
                cursor,
                dataset_id=args.dataset_id,
                dataset_version=args.dataset_version,
                prefix=prefix,
            )
    finally:
        connection.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": f"{args.dataset_id} {args.dataset_version}",
        "gate": dense_gate(summary, prefix),
        "summary": summary,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{STEM}.json").write_text(json.dumps(payload, indent=2))
    (reports_dir / f"{STEM}_per_question.json").write_text(json.dumps(per_question))
    (reports_dir / f"{STEM}.md").write_text(render_markdown(payload))
    for cell in summary.values():
        if cell.get("n_questions"):
            print(
                f"{cell['label']:36s} r@5={cell['recall@5']}  all_gold@5={cell['all_gold@5']}  "
                f"r@pool={cell['recall@inf']}  docs/q={cell['mean_retrieved_docs']}"
            )


if __name__ == "__main__":
    main()
