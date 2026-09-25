# PROVENANCE RECORD — one of the two scripts that produced results/, kept verbatim as
# it ran in the experimental environment. It does NOT run on another machine: it
# reads that environment's Postgres (through psycopg2, not a dependency here) and
# the canonical audit files it wrote. It is here so anyone can see exactly how
# every number in results/ was derived. tests/test_results_match_thesis.py checks
# the CSVs against the thesis tables on any machine.
"""Exporta em CSV as células que a tese reporta, um ficheiro por braço.

Fonte da resposta: Postgres (`evaluation_results.answer_correctness_judge`,
judge gpt-4o, prompt judge_v1), agregado por `experiment_id`. Fonte da
recuperação no braço A: a auditoria canónica (`retrieval_audit_canonic_*.json`),
excepto o HippoRAG 2, cujo índice oficial é posterior a essa auditoria e por
isso é recalculado aqui a partir de `retrieval_items` × `gold_evidence`. Fonte
dos deltas e dos `p`: os McNemar oficiais (`mcnemar_*_2026-07official.json`).

Só entram as células que a tese usa. Ficam de fora as versões descartadas:
leitor grounded (`_v2`), `first_results`, `cognee_fixed`, o HippoRAG 2 em lotes
(`hipporag2_v1free` e `hipporag2_native` sem `_official`), os `_rephrased`, os
`_smoke`, os gates e o `sumctl`.

Definições, iguais às do juiz: base pontuável exclui `gold_suspect`, estrita =
`correct` / pontuáveis, lenient = média do `metric_value` (parcial vale 0,5),
recusa sobre as 1.000 respostas da célula, cobertura = 1 − recusa, seletiva =
estrita / cobertura.

Read-only. Uso: PYTHONPATH=src .venv/bin/python scripts/export_thesis_results_csv.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import psycopg2

REPORTS = {"musique": Path("artifacts/musique/reports"), "twowiki": Path("artifacts/twowiki/reports")}
DATASET_VERSION = {"musique": "ans_v1.0_eval1k", "twowiki": "ans_v1.0_eval1k"}
PREFIX = {"musique": "musique_eval1k_", "twowiki": "twowiki_eval1k_"}

# sufixo -> (framework, papel, chave da auditoria canónica de recuperação)
ARM_A = [
    ("closed_book", "—", "piso", None),
    ("vector_v1free", "denso", "baseline", "vector"),
    ("hipporag2_v1free_official", "HippoRAG 2", "grafo", "RECALC"),
    ("lightrag_v1free", "LightRAG", "grafo", "lightrag"),
    ("ms_graphrag_v1free", "Microsoft GraphRAG", "grafo", "ms_graphrag"),
    ("cognee_v1free_k5", "Cognee", "grafo", "cognee"),
    ("oracle_gold_v1free", "—", "teto", "oracle"),
]
ARM_B = [
    ("lightrag_native", "LightRAG", "lightrag_v1free"),
    ("ms_graphrag_native", "Microsoft GraphRAG", "ms_graphrag_v1free"),
    ("hipporag2_native_official", "HippoRAG 2", "hipporag2_v1free_official"),
    ("cognee_native", "Cognee", "cognee_v1free_k5"),
]

# chave curta -> chave "experiment::method" na auditoria canónica de cada dataset
AUDIT_KEY = {
    "musique": {
        # o retrieval do denso foi auditado na célula v2: os dois runs partilham o
        # mesmo retriever e só o leitor difere
        "vector": "musique_eval1k_v2_grounded::vector_rag",
        "lightrag": "musique_eval1k_lightrag_v1free::lightrag_neo4j",
        "ms_graphrag": "musique_eval1k_ms_graphrag_v1free::ms_graphrag",
        "cognee": "musique_eval1k_cognee_v1free_k5::cognee",
        "oracle": "musique_eval1k_oracle_gold_v1free::single_document_context",
    },
    "twowiki": {
        "vector": "twowiki_eval1k_vector_v1free::vector_rag",
        "lightrag": "twowiki_eval1k_lightrag_v1free::lightrag_neo4j",
        "ms_graphrag": "twowiki_eval1k_ms_graphrag_v1free::ms_graphrag",
        "cognee": "twowiki_eval1k_cognee_v1free_k5::cognee",
        "oracle": "twowiki_eval1k_oracle_gold_v1free::single_document_context",
    },
}
NOTA = {
    "cognee_v1free_k5": "cognee não devolve lista ranqueada; @k reportado com essa ressalva",
    "hipporag2_v1free_official": "recuperação recalculada do cru (índice oficial, posterior à auditoria)",
    "closed_book": "sem recuperação",
}
# Notas que só valem num dataset, por (dataset, sufixo).
NOTA_DATASET = {
    ("musique", "vector_v1free"):
        "recuperação medida na célula gémea v2 (mesmo retriever, leitor diferente), como na tese; "
        "a própria célula v1free dá recall@5 0.5528 e all_gold@5 0.217, porque as duas execuções "
        "diferem no top-5 de 8 das 1.000 perguntas e numa delas isso muda o all_gold@5",
}


def q4(value: float | None) -> str:
    if value is None:
        return ""
    return str(Decimal(repr(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def pp1(value: float | None) -> str:
    """Delta em pontos percentuais, uma casa, arredondado do valor exacto."""
    if value is None:
        return ""
    return str((Decimal(repr(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def p_value(value: float) -> str:
    """Os McNemar guardam `p_holm` com seis casas: abaixo disso sai um zero que
    não é zero. Reporta-se como limite superior."""
    return "<1e-06" if value == 0.0 else repr(value)


def judge_cell(cursor, experiment_id: str) -> dict | None:
    cursor.execute(
        """SELECT er.metric_value, er.metadata->>'label', (er.metadata->>'pass_agreement')::bool
             FROM evaluation_results er
             JOIN runs r ON r.run_id = er.run_id
            WHERE r.experiment_id = %s AND er.metric_name = 'answer_correctness_judge'""",
        (experiment_id,),
    )
    rows = cursor.fetchall()
    if not rows:
        return None
    cursor.execute("SELECT count(*) FROM runs WHERE experiment_id = %s", (experiment_id,))
    n_answers = cursor.fetchone()[0]
    labels = defaultdict(int)
    for _, label, _ in rows:
        labels[label] += 1
    scoreable = len(rows)
    correct = labels["correct"]
    refusal = labels["refusal"]
    coverage = 1.0 - refusal / n_answers
    return {
        "n": n_answers,
        "n_pontuaveis": scoreable,
        "gold_suspect": n_answers - scoreable,
        "correct": correct,
        "partial": labels["partial"],
        "incorrect": labels["incorrect"],
        "refusal": refusal,
        "estrita": correct / scoreable,
        "lenient": sum(v for v, _, _ in rows) / scoreable,
        "recusa": refusal / n_answers,
        "cobertura": coverage,
        "seletiva": (correct / scoreable) / coverage if coverage else None,
        "acordo_passes": sum(1 for _, _, agree in rows if agree) / scoreable,
    }


def retrieval_from_raw(cursor, experiment_id: str, dataset_id: str) -> dict:
    """Recalcula docs/q, recall@5, all_gold@5 e recall@pool do cru.

    Mesma regra da auditoria canónica para métodos que expõem `source_document_id`:
    lista de documentos deduplicada, na ordem do rank do item.
    """
    cursor.execute(
        """SELECT ge.question_id, ge.document_id
             FROM gold_evidence ge
             JOIN questions q ON q.question_id = ge.question_id
            WHERE q.dataset_id = %s""",
        (dataset_id,),
    )
    gold = defaultdict(set)
    for question_id, document_id in cursor.fetchall():
        gold[question_id].add(document_id)

    cursor.execute(
        """SELECT a.question_id, ri.source_document_id
             FROM runs r
             JOIN answers a ON a.run_id = r.run_id
             JOIN retrieval_results rr ON rr.run_id = r.run_id
             JOIN retrieval_items ri ON ri.retrieval_result_id = rr.retrieval_result_id
            WHERE r.experiment_id = %s
            ORDER BY a.question_id, ri.rank""",
        (experiment_id,),
    )
    ranked = defaultdict(list)
    for question_id, document_id in cursor.fetchall():
        if document_id and document_id not in ranked[question_id]:
            ranked[question_id].append(document_id)

    n = recall5 = all_gold5 = recall_pool = docs = 0
    for question_id, docs_ranked in ranked.items():
        gold_docs = gold.get(question_id)
        if not gold_docs:
            continue
        n += 1
        docs += len(docs_ranked)
        top5 = set(docs_ranked[:5])
        recall5 += len(top5 & gold_docs) / len(gold_docs)
        all_gold5 += 1 if gold_docs <= top5 else 0
        recall_pool += len(set(docs_ranked) & gold_docs) / len(gold_docs)
    return {
        "docs_q": docs / n,
        "recall_5": recall5 / n,
        "all_gold_5": all_gold5 / n,
        "recall_pool": recall_pool / n,
        "n_recuperacao": n,
    }


def mcnemar_lookup(path: Path, family: str) -> dict:
    payload = json.loads(path.read_text())
    return {(row["A"], row["B"]): row for row in payload[family]}


def cross_check(dataset: str, suffix: str, cell: dict) -> float:
    """Confronta o agregado do Postgres com o relatório do juiz e devolve o acordo.

    O acordo entre passes vem do relatório porque a tese cita esse valor: ele
    corre sobre as 1.000 respostas, incluindo os `gold_suspect`, que não têm
    linha em `evaluation_results` e por isso não são recalculáveis daqui.
    """
    path = REPORTS[dataset] / f"llm_judge_full_{suffix}_summary.json"
    summary = next(iter(json.loads(path.read_text())["by_method"].values()))
    for ours, theirs in (("estrita", "strict_accuracy"), ("lenient", "lenient_accuracy"),
                         ("recusa", "refusal_rate"), ("n_pontuaveis", "n_scoreable")):
        if abs(cell[ours] - summary[theirs]) > 0.0006:
            raise SystemExit(
                f"divergência em {dataset}/{suffix}, {ours}: "
                f"Postgres {cell[ours]} contra juiz {summary[theirs]}"
            )
    return summary["inter_pass_agreement"]


def main() -> None:
    from benchmark.core.settings import load_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/aggregate")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    connection = psycopg2.connect(load_settings().database_url)
    cursor = connection.cursor()

    audit = {ds: json.loads((REPORTS[ds] / "retrieval_audit_canonic_2026-07.json").read_text())["summary"]
             for ds in REPORTS}
    family_a = {ds: mcnemar_lookup(REPORTS[ds] / "mcnemar_v1_arm_2026-07official.json", "family_primary")
                for ds in REPORTS}
    family_ab = {ds: mcnemar_lookup(REPORTS[ds] / "mcnemar_native_arm_2026-07official.json", "family_AB_deployment_gap")
                 for ds in REPORTS}

    rows_a, rows_b = [], []
    strict_by_cell: dict[tuple[str, str], float] = {}

    for dataset in ("musique", "twowiki"):
        prefix = PREFIX[dataset]
        for suffix, framework, papel, audit_key in ARM_A:
            experiment_id = prefix + suffix
            cell = judge_cell(cursor, experiment_id)
            if cell is None:
                raise SystemExit(f"célula sem juiz no Postgres: {experiment_id}")
            strict_by_cell[(dataset, suffix)] = cell["estrita"]
            acordo = cross_check(dataset, suffix, cell)

            retrieval: dict = {}
            if audit_key == "RECALC":
                retrieval = retrieval_from_raw(cursor, experiment_id, dataset)
            elif audit_key:
                entry = audit[dataset][AUDIT_KEY[dataset][audit_key]]
                retrieval = {
                    "docs_q": entry["mean_retrieved_docs"],
                    "recall_5": entry["recall@5"],
                    "all_gold_5": entry["all_gold@5"],
                    "recall_pool": entry["recall@inf"],
                    "n_recuperacao": entry["n_questions"],
                }

            test = family_a[dataset].get((suffix, "vector_v1free"))
            rows_a.append({
                "dataset": dataset,
                "celula": suffix,
                "framework": framework,
                "papel": papel,
                "experiment_id": experiment_id,
                "n": cell["n"],
                "n_pontuaveis": cell["n_pontuaveis"],
                "gold_suspect": cell["gold_suspect"],
                "correct": cell["correct"],
                "partial": cell["partial"],
                "incorrect": cell["incorrect"],
                "refusal": cell["refusal"],
                "estrita": q4(cell["estrita"]),
                "lenient": q4(cell["lenient"]),
                "recusa": q4(cell["recusa"]),
                "cobertura": q4(cell["cobertura"]),
                "seletiva": q4(cell["seletiva"]),
                "acordo_passes": q4(acordo),
                "docs_q": q4(retrieval.get("docs_q")),
                "recall_5": q4(retrieval.get("recall_5")),
                "recall_pool": q4(retrieval.get("recall_pool")),
                "all_gold_5": q4(retrieval.get("all_gold_5")),
                "n_recuperacao": retrieval.get("n_recuperacao", ""),
                "delta_vs_denso_pp": pp1(test["delta"]) if test else "",
                "p_holm": p_value(test["p_holm"]) if test else "",
                "significativo": ("sim" if test["significant_0.05_holm"] else "nao") if test else "",
                "n_pareado": test["n_paired"] if test else "",
                "nota": NOTA_DATASET.get((dataset, suffix)) or NOTA.get(suffix, ""),
            })

        for suffix, framework, controlada in ARM_B:
            experiment_id = prefix + suffix
            cell = judge_cell(cursor, experiment_id)
            if cell is None:
                raise SystemExit(f"célula sem juiz no Postgres: {experiment_id}")
            acordo = cross_check(dataset, suffix, cell)
            test = family_ab[dataset][(suffix, controlada)]
            rows_b.append({
                "dataset": dataset,
                "celula": suffix,
                "framework": framework,
                "experiment_id": experiment_id,
                "n": cell["n"],
                "n_pontuaveis": cell["n_pontuaveis"],
                "gold_suspect": cell["gold_suspect"],
                "correct": cell["correct"],
                "partial": cell["partial"],
                "incorrect": cell["incorrect"],
                "refusal": cell["refusal"],
                "estrita": q4(cell["estrita"]),
                "lenient": q4(cell["lenient"]),
                "recusa": q4(cell["recusa"]),
                "cobertura": q4(cell["cobertura"]),
                "seletiva": q4(cell["seletiva"]),
                "acordo_passes": q4(acordo),
                "celula_controlada": controlada,
                "estrita_controlada": q4(strict_by_cell[(dataset, controlada)]),
                "delta_vs_controlado_pp": pp1(test["delta"]),
                "p_holm": p_value(test["p_holm"]),
                "significativo": "sim" if test["significant_0.05_holm"] else "nao",
                "n_pareado": test["n_paired"],
                "nota": "pipeline nativo não persiste recuperação comparável",
            })

    for rows, name in ((rows_a, "arm_a_controlled.csv"),
                       (rows_b, "arm_b_native.csv")):
        path = out_dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"-> {path}  ({len(rows)} linhas)")

    connection.close()


if __name__ == "__main__":
    main()
