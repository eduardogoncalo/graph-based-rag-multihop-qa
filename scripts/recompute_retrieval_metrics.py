"""Fase 1.3 — recomputo do evidence_recall em nível de DOCUMENTO sobre dados já persistidos.

Corrige dois defeitos de instrumentação do run original (musique_eval1k_first_results):
- lightrag_neo4j: itens gravados como ``doc_X-chunk-NNN`` (match exato contra ``doc_X`` falha) →
  deriva o doc-id do prefixo do chunk-id;
- cognee: 1 blob/query com todos os doc-ids embutidos no texto, mas só o primeiro era creditado →
  extrai TODOS os ``doc_*`` do texto do item;
- vector_rag: já canônico — serve de CONTROLE (deve reproduzir o valor armazenado 0,0954).

Report-only: não escreve em nenhuma tabela; sai em JSON + Markdown em artifacts/musique/reports/.
Nota de semântica: para o cognee o conjunto recuperado é o do BLOB inteiro (o que o leitor viu);
o tamanho do conjunto é reportado para transparência (não é um top-5 ranqueado).
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

DATASET_ID = "musique"
DATASET_VERSION = "ans_v1.0_eval1k"
# O run canônico de cada método vive em experiment_ids diferentes:
# vector/lightrag no first_results; o cognee completo (1000) no _cognee_fixed.
EXPERIMENTS = {
    "vector_rag": "musique_eval1k_first_results",
    "lightrag_neo4j": "musique_eval1k_first_results",
    "cognee": "musique_eval1k_cognee_fixed",
}
REPORTS_DIR = Path("artifacts/musique/reports")

CHUNK_SUFFIX_RE = re.compile(r"^(doc_[0-9a-f]+)-chunk-\d+$")
DOC_ID_RE = re.compile(r"(doc_[0-9a-f]{6,})", re.IGNORECASE)


def _docs_for_method(method_id: str, rows: list[tuple]) -> list[str]:
    """rows = (rank, source_document_id, source_chunk_id, text) ordenadas por rank."""
    seen: list[str] = []

    def add(doc: str | None) -> None:
        if doc and doc not in seen:
            seen.append(doc)

    for _rank, source_document_id, source_chunk_id, text in rows:
        if method_id == "vector_rag":
            add(source_document_id)
        elif method_id == "lightrag_neo4j":
            match = CHUNK_SUFFIX_RE.match(source_chunk_id or "")
            add(match.group(1) if match else source_document_id)
        elif method_id == "cognee":
            for found in DOC_ID_RE.findall(text or ""):
                add(found)
    return seen


def main() -> None:
    from benchmark.core.settings import load_settings

    connection = psycopg2.connect(load_settings().database_url)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT ge.question_id, ge.document_id
            FROM gold_evidence ge JOIN questions q ON q.question_id = ge.question_id
            WHERE q.dataset_id = %s AND q.dataset_version = %s
            """,
            (DATASET_ID, DATASET_VERSION),
        )
        gold: dict[str, set[str]] = defaultdict(set)
        for question_id, document_id in cursor.fetchall():
            gold[question_id].add(document_id)

        cursor.execute(
            """
            SELECT question_id, jsonb_array_length(metadata->'decomposition')
            FROM questions WHERE dataset_id = %s AND dataset_version = %s
            """,
            (DATASET_ID, DATASET_VERSION),
        )
        hops = dict(cursor.fetchall())

        by_method_question: dict[tuple[str, str], list[tuple]] = defaultdict(list)
        for method_id, experiment_id in EXPERIMENTS.items():
            cursor.execute(
                """
                SELECT a.question_id, ri.rank,
                       ri.source_document_id, ri.source_chunk_id, ri.text
                FROM retrieval_items ri
                JOIN retrieval_results rr ON rr.retrieval_result_id = ri.retrieval_result_id
                JOIN runs r ON r.run_id = rr.run_id
                JOIN answers a ON a.run_id = r.run_id
                WHERE r.experiment_id = %s AND rr.method_id = %s
                ORDER BY a.question_id, ri.rank
                """,
                (experiment_id, method_id),
            )
            for question_id, rank, doc_id, chunk_id, text in cursor.fetchall():
                by_method_question[(method_id, question_id)].append((rank, doc_id, chunk_id, text))

    per_method: dict[str, list[dict]] = defaultdict(list)
    for (method_id, question_id), rows in by_method_question.items():
        gold_docs = gold.get(question_id)
        if not gold_docs:
            continue
        docs = _docs_for_method(method_id, rows)
        per_method[method_id].append(
            {
                "question_id": question_id,
                "n_hops": hops.get(question_id),
                "retrieved_set_size": len(docs),
                "recall": len(gold_docs & set(docs)) / len(gold_docs),
            }
        )

    summary: dict[str, dict] = {}
    for method_id, entries in sorted(per_method.items()):
        by_hop = {
            h: round(statistics.fmean(e["recall"] for e in entries if e["n_hops"] == h), 3)
            for h in (2, 3, 4)
        }
        summary[method_id] = {
            "n_questions": len(entries),
            "evidence_recall_doc_level": round(statistics.fmean(e["recall"] for e in entries), 4),
            "nonzero": sum(1 for e in entries if e["recall"] > 0),
            "mean_retrieved_set_size": round(
                statistics.fmean(e["retrieved_set_size"] for e in entries), 2
            ),
            "recall_by_hop": by_hop,
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": EXPERIMENTS,
        "note": (
            "Recomputo doc-level sobre dados persistidos (sem re-run). vector_rag = controle "
            "(esperado ≈0,0954, índice quebrado à época). cognee = conjunto do blob inteiro "
            "(o que o leitor viu), não top-5 ranqueado."
        ),
        "summary": summary,
        "per_question": {m: entries for m, entries in per_method.items()},
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = REPORTS_DIR / "retrieval_metrics_corrected_2026-07-02"
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    lines = ["# evidence_recall corrigido (doc-level, recomputo sem re-run)", ""]
    for method_id, stats in summary.items():
        lines.append(
            f"- **{method_id}**: recall {stats['evidence_recall_doc_level']} · "
            f"não-zero {stats['nonzero']}/{stats['n_questions']} · "
            f"docs/set {stats['mean_retrieved_set_size']} · por hop {stats['recall_by_hop']}"
        )
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
