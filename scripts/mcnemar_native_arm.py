"""P5 (parcial) — McNemar pareado no Braço B nativo (strict, judge), com Holm.

Famílias PRÉ-REGISTRADAS (dossiê dois braços §1.6, aprovado 2026-07-04):
  B  = pares nativos: os 4 frameworks "as deployed" comparados par a par
       (6 pares, Holm em conjunto). Não há baseline denso nativo — o denso
       "as deployed" É o pipeline controlado.
  A↔B = deployment gap por framework: nativo vs controlado v1free do MESMO
       framework (4 pares, Holm em família própria).

Teste: McNemar EXATO (binomial bicaudal sobre os pares discordantes b,c) —
Dror et al. (ACL 2018); pareamento por question_id (mesmas 1000 perguntas;
gold_suspect fica fora por não ter metric_value, igual ao braço A).
strict = answer_correctness_judge == 1.0. Report-only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


# Sufixos; o prefixo do dataset (--exp-prefix) monta os experiment_ids.
NATIVE_SUFFIX = [
    "lightrag_native",
    "ms_graphrag_native",
    "hipporag2_native",
    "cognee_native",
]
FAMILY_B_SUFFIX = list(combinations(NATIVE_SUFFIX, 2))
FAMILY_AB_SUFFIX = [
    ("lightrag_native", "lightrag_v1free"),
    ("ms_graphrag_native", "ms_graphrag_v1free"),
    ("hipporag2_native", "hipporag2_v1free"),
    ("cognee_native", "cognee_v1free_k5"),
]


def strict_map(cursor, experiment_id: str) -> dict[str, bool]:
    cursor.execute(
        """SELECT a.question_id, er.metric_value
           FROM evaluation_results er
           JOIN runs r ON r.run_id = er.run_id
           JOIN answers a ON a.run_id = r.run_id
           WHERE r.experiment_id = %s AND er.metric_name = 'answer_correctness_judge'""",
        (experiment_id,),
    )
    return {qid: value == 1.0 for qid, value in cursor.fetchall()}


def mcnemar_exact(b: int, c: int) -> float:
    """p bicaudal exato: 2 * P(X <= min(b,c)), X ~ Binomial(b+c, 0.5); capado em 1."""
    import math

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def compare(cursor, exp_a: str, exp_b: str, prefix: str) -> dict:
    a_map, b_map = strict_map(cursor, exp_a), strict_map(cursor, exp_b)
    common = sorted(set(a_map) & set(b_map))
    if not common:
        # Every pair of this family is required: a missing cell is an error, not
        # a row to skip (it would also divide by zero below).
        raise SystemExit(
            f"no paired judged questions between {exp_a} and {exp_b}. "
            "Did both cells run and get judged?"
        )
    both = sum(1 for q in common if a_map[q] and b_map[q])
    only_a = sum(1 for q in common if a_map[q] and not b_map[q])   # b (A ganha)
    only_b = sum(1 for q in common if not a_map[q] and b_map[q])   # c (B ganha)
    return {
        "A": exp_a.replace(prefix, ""),
        "B": exp_b.replace(prefix, ""),
        "n_paired": len(common),
        "strict_A": round((both + only_a) / len(common), 4),
        "strict_B": round((both + only_b) / len(common), 4),
        "delta": round((only_a - only_b) / len(common), 4),
        "discordant_A_wins(b)": only_a,
        "discordant_B_wins(c)": only_b,
        "p_mcnemar_exact": mcnemar_exact(only_a, only_b),
    }


def holm(results: list[dict]) -> None:
    indexed = sorted(range(len(results)), key=lambda i: results[i]["p_mcnemar_exact"])
    m = len(results)
    prev = 0.0
    for rank, i in enumerate(indexed):
        adj = min(1.0, (m - rank) * results[i]["p_mcnemar_exact"])
        adj = max(adj, prev)  # monotonicidade
        results[i]["p_holm"] = round(adj, 6)
        results[i]["significant_0.05_holm"] = adj < 0.05
        prev = adj


def main() -> None:
    import argparse

    from benchmark.core.settings import load_settings
    from benchmark.storage.postgres import connect_postgres

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-prefix", default="musique_eval1k_")
    ap.add_argument("--reports-dir", default="artifacts/musique/reports")
    ap.add_argument(
        "--suffix-override",
        action="append",
        default=[],
        metavar="ANTIGO=NOVO",
        help="troca um sufixo de célula, por exemplo "
        "hipporag2_native=hipporag2_native_official. Repetível. Só muda QUAIS "
        "células entram, nunca como o teste é calculado.",
    )
    ap.add_argument("--out-tag", default="", help="sufixo do ficheiro de saída")
    args = ap.parse_args()
    prefix = args.exp_prefix
    reports_dir = Path(args.reports_dir)

    if args.suffix_override:
        mapping = dict(pair.split("=", 1) for pair in args.suffix_override)
        globals()["NATIVE_SUFFIX"] = [mapping.get(s, s) for s in NATIVE_SUFFIX]
        globals()["FAMILY_B_SUFFIX"] = [
            (mapping.get(a, a), mapping.get(b, b)) for a, b in FAMILY_B_SUFFIX
        ]
        globals()["FAMILY_AB_SUFFIX"] = [
            (mapping.get(a, a), mapping.get(b, b)) for a, b in FAMILY_AB_SUFFIX
        ]
        print(f"sufixos trocados: {mapping}")

    connection = connect_postgres(load_settings())
    cursor = connection.cursor()

    family_b = [compare(cursor, prefix + a, prefix + b, prefix) for a, b in FAMILY_B_SUFFIX]
    holm(family_b)
    family_ab = [compare(cursor, prefix + a, prefix + b, prefix) for a, b in FAMILY_AB_SUFFIX]
    holm(family_ab)
    for row in family_ab:
        row["note"] = "deployment gap: nativo vs controlado v1free (mesmo framework)"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exp_prefix": prefix,
        "test": "McNemar exato (binomial bicaudal, pares discordantes), pareado por question_id",
        "metric": "answer_correctness_judge == 1.0 (strict)",
        "family_B_native_pairs": family_b,
        "family_AB_deployment_gap": family_ab,
        "refs": "Dror et al. ACL 2018; Holm 1979; dossiê dois braços §1.6",
    }
    for row in [*family_b, *family_ab]:
        row["p_mcnemar_exact"] = round(row["p_mcnemar_exact"], 8)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"mcnemar_native_arm_2026-07{args.out_tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
