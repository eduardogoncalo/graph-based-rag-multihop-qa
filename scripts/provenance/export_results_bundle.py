# PROVENANCE RECORD — one of the two scripts that produced results/, kept verbatim as
# it ran in the experimental environment. It does NOT run on another machine: it
# reads that environment's Postgres (through psycopg2, not a dependency here) and
# the canonical audit files it wrote. It is here so anyone can see exactly how
# every number in results/ was derived. tests/test_results_match_thesis.py checks
# the CSVs against the thesis tables on any machine.
"""Exporta para `results/` o conjunto de dados que sustenta os números da tese.

Três níveis, do agregado ao cru, todos gerados da mesma fonte e cruzáveis pelas
mesmas chaves (`celula`, `question_id`, `document_id`):

  aggregate/     uma linha por célula, com as tabelas do capítulo de resultados
  per_question/  uma linha por pergunta em cada célula, com a resposta literal,
                 o veredicto do juiz e os documentos entregues ao leitor
  retrieval/     uma linha por documento recuperado, só no braço controlado
  corpus/        o texto de cada documento, uma vez só

O texto dos documentos vive apenas no `corpus/`. Repeti-lo em cada item
recuperado multiplicava 8,5 MB por cerca de oito, sem acrescentar informação.
A excepção é o Cognee, que não devolve lista de documentos mas um bloco de
texto de onde os identificadores são extraídos: esse bloco é o único registo
fiel do que chegou ao leitor e vai à parte, em `retrieval/cognee_blocks.csv`.

A regra de extração de identificadores é importada do
`retrieval_audit_canonic.py`, para que estes ficheiros e as métricas da tese
digam a mesma coisa sobre o que cada adaptador entregou.

Só entram as células que a tese usa; as versões descartadas ficam de fora
(ver o cabeçalho de `export_thesis_results_csv.py`).

Read-only. Uso: PYTHONPATH=src .venv/bin/python scripts/export_results_bundle.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_audit_canonic import docs_for_method  # noqa: E402

PREFIX = {"musique": "musique_eval1k_", "twowiki": "twowiki_eval1k_"}

# sufixo -> (framework, papel, method_id)
ARM_A = [
    ("closed_book", "—", "piso", "zero_shot_no_context"),
    ("vector_v1free", "denso", "baseline", "vector_rag"),
    ("hipporag2_v1free_official", "HippoRAG 2", "grafo", "hipporag2"),
    ("lightrag_v1free", "LightRAG", "grafo", "lightrag_neo4j"),
    ("ms_graphrag_v1free", "Microsoft GraphRAG", "grafo", "ms_graphrag"),
    ("cognee_v1free_k5", "Cognee", "grafo", "cognee"),
    ("oracle_gold_v1free", "—", "teto", "single_document_context"),
]
ARM_B = [
    ("lightrag_native", "LightRAG", "lightrag_neo4j"),
    ("ms_graphrag_native", "Microsoft GraphRAG", "ms_graphrag"),
    ("hipporag2_native_official", "HippoRAG 2", "hipporag2"),
    ("cognee_native", "Cognee", "cognee"),
]


def fetch_questions(cursor, dataset_id: str) -> dict:
    cursor.execute(
        """SELECT question_id, question, gold_answer, metadata
             FROM questions WHERE dataset_id = %s""",
        (dataset_id,),
    )
    out = {}
    for question_id, question, gold_answer, metadata in cursor.fetchall():
        metadata = metadata or {}
        decomposition = metadata.get("decomposition")
        out[question_id] = {
            "pergunta": question,
            "resposta_ouro": gold_answer if isinstance(gold_answer, str) else json.dumps(gold_answer, ensure_ascii=False),
            "aliases": "|".join(metadata.get("answer_aliases") or []),
            "hops": len(decomposition) if decomposition else "",
            "tipo": metadata.get("type", ""),
        }
    cursor.execute(
        """SELECT ge.question_id, ge.document_id
             FROM gold_evidence ge JOIN questions q ON q.question_id = ge.question_id
            WHERE q.dataset_id = %s""",
        (dataset_id,),
    )
    gold = defaultdict(set)
    for question_id, document_id in cursor.fetchall():
        gold[question_id].add(document_id)
    for question_id, entry in out.items():
        entry["gold_docs"] = sorted(gold.get(question_id, ()))
    return out


def fetch_answers(cursor, experiment_id: str) -> dict:
    cursor.execute(
        """SELECT a.question_id, a.answer_text, a.latency_ms, a.prompt_tokens,
                  a.completion_tokens, a.total_tokens, a.estimated_cost
             FROM answers a JOIN runs r ON r.run_id = a.run_id
            WHERE r.experiment_id = %s""",
        (experiment_id,),
    )
    return {row[0]: row[1:] for row in cursor.fetchall()}


def fetch_judge(cursor, experiment_id: str) -> dict:
    cursor.execute(
        """SELECT a.question_id, er.metric_value, er.metadata
             FROM evaluation_results er
             JOIN runs r ON r.run_id = er.run_id
             JOIN answers a ON a.run_id = r.run_id
            WHERE r.experiment_id = %s AND er.metric_name = 'answer_correctness_judge'""",
        (experiment_id,),
    )
    return {question_id: (value, metadata or {}) for question_id, value, metadata in cursor.fetchall()}


def fetch_retrieval(cursor, experiment_id: str) -> dict:
    """question_id -> lista de (rank, source_document_id, source_chunk_id, score, text)."""
    cursor.execute(
        """SELECT a.question_id, ri.rank, ri.source_document_id, ri.source_chunk_id,
                  ri.score, ri.text
             FROM runs r
             JOIN answers a ON a.run_id = r.run_id
             JOIN retrieval_results rr ON rr.run_id = r.run_id
             JOIN retrieval_items ri ON ri.retrieval_result_id = rr.retrieval_result_id
            WHERE r.experiment_id = %s
            ORDER BY a.question_id, ri.rank""",
        (experiment_id,),
    )
    out = defaultdict(list)
    for question_id, rank, doc, chunk, score, text in cursor.fetchall():
        out[question_id].append((rank, doc, chunk, score, text))
    return out


def judge_columns(judged: tuple | None) -> dict:
    if judged is None:
        # `gold_suspect`: o juiz marcou a referência como suspeita e o item ficou
        # fora da base pontuável, por isso não tem linha em evaluation_results.
        return {"label_juiz": "gold_suspect", "estrita": "", "valor_juiz": "",
                "pass_a": "", "pass_b": "", "acordo_passes": "", "baixa_confianca": "",
                "ouro_correspondente": "", "justificacao_juiz": ""}
    value, meta = judged
    return {
        "label_juiz": meta.get("label", ""),
        "estrita": 1 if value == 1.0 else 0,
        "valor_juiz": value,
        "pass_a": meta.get("pass_a", ""),
        "pass_b": meta.get("pass_b", ""),
        "acordo_passes": meta.get("pass_agreement", ""),
        "baixa_confianca": meta.get("low_confidence", ""),
        "ouro_correspondente": meta.get("matched_gold") or "",
        "justificacao_juiz": meta.get("rationale", ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    size = path.stat().st_size
    unidade = f"{size/1e6:.1f} MB" if size >= 1e6 else f"{size/1e3:.0f} kB"
    print(f"-> {path}  {len(rows)} linhas, {unidade}")


def main() -> None:
    from benchmark.core.settings import load_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    out = Path(args.out_dir)

    connection = psycopg2.connect(load_settings().database_url)
    cursor = connection.cursor()

    rows_a, rows_b, rows_long, rows_blob, rows_corpus = [], [], [], [], []

    for dataset in ("musique", "twowiki"):
        prefix = PREFIX[dataset]
        questions = fetch_questions(cursor, dataset)

        cursor.execute(
            "SELECT document_id, title, text FROM documents WHERE dataset_id = %s ORDER BY document_id",
            (dataset,),
        )
        for document_id, title, text in cursor.fetchall():
            rows_corpus.append({"dataset": dataset, "document_id": document_id,
                                "titulo": title or "", "texto": text or ""})

        for arm, cells in (("A", ARM_A), ("B", ARM_B)):
            for entry in cells:
                suffix, framework, method_id = (entry[0], entry[1], entry[3]) if arm == "A" else entry
                papel = entry[2] if arm == "A" else "grafo"
                experiment_id = prefix + suffix
                answers = fetch_answers(cursor, experiment_id)
                judge = fetch_judge(cursor, experiment_id)
                retrieval = fetch_retrieval(cursor, experiment_id) if arm == "A" else {}

                for question_id in sorted(answers):
                    meta = questions[question_id]
                    answer_text, latency, p_tok, c_tok, t_tok, cost = answers[question_id]
                    gold_docs = set(meta["gold_docs"])
                    row = {
                        "dataset": dataset, "braco": "controlado" if arm == "A" else "nativo",
                        "celula": suffix, "framework": framework, "papel": papel,
                        "experiment_id": experiment_id, "question_id": question_id,
                        "hops": meta["hops"], "tipo": meta["tipo"],
                        "pergunta": meta["pergunta"], "resposta_ouro": meta["resposta_ouro"],
                        "aliases_ouro": meta["aliases"], "resposta": answer_text or "",
                        **judge_columns(judge.get(question_id)),
                    }
                    if arm == "A":
                        items = retrieval.get(question_id, [])
                        docs = docs_for_method(method_id, [(r, d, c, t) for r, d, c, _s, t in items])
                        top5 = set(docs[:5])
                        row.update({
                            "docs_recuperados": "|".join(docs),
                            "n_docs_recuperados": len(docs),
                            "docs_ouro": "|".join(meta["gold_docs"]),
                            "n_docs_ouro": len(gold_docs),
                            "recall_5": round(len(top5 & gold_docs) / len(gold_docs), 4) if gold_docs else "",
                            "all_gold_5": (1 if gold_docs and gold_docs <= top5 else 0) if gold_docs else "",
                            "recall_pool": round(len(set(docs) & gold_docs) / len(gold_docs), 4) if gold_docs else "",
                        })
                        for rank, doc, chunk, score, text in items:
                            rows_long.append({
                                "dataset": dataset, "celula": suffix, "framework": framework,
                                "question_id": question_id, "rank": rank,
                                "document_id": doc or "", "chunk_id": chunk or "",
                                "score": score if score is not None else "",
                                "e_ouro": 1 if doc in gold_docs else 0,
                            })
                            if method_id == "cognee":
                                rows_blob.append({
                                    "dataset": dataset, "celula": suffix,
                                    "question_id": question_id, "rank": rank,
                                    "texto": text or "",
                                })
                    else:
                        row["nota"] = "pipeline nativo não persiste recuperação"
                    (rows_a if arm == "A" else rows_b).append(row)

    write_csv(out / "per_question" / "arm_a_controlled.csv", rows_a)
    write_csv(out / "per_question" / "arm_b_native.csv", rows_b)
    write_csv(out / "retrieval" / "arm_a_documents.csv", rows_long)
    write_csv(out / "retrieval" / "cognee_blocks.csv", rows_blob)
    write_csv(out / "corpus" / "documents.csv", rows_corpus)
    connection.close()


if __name__ == "__main__":
    main()
