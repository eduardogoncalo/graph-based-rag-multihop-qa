"""S13 (parcial) — McNemar pareado no Braço A-v1 (strict, judge), com Holm.

Família PRIMÁRIA pré-definida (metodologia_v1 §Bloco 5): cada substrato de grafo
vs o baseline denso, no braço A-v1:
  H1: lightrag_v1free  vs vector_v1free
  H2: ms_graphrag_v1free vs vector_v1free
Secundária (exploratória, reportada sem Holm): lightrag vs ms_graphrag.

Teste: McNemar EXATO (binomial bicaudal sobre os pares discordantes b,c) —
Dror et al. (ACL 2018); pareamento por question_id (mesmas 1000 perguntas).
strict = answer_correctness_judge == 1.0. Report-only.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


# Pares por SUFIXO; o prefixo do dataset (--exp-prefix) monta os experiment_ids.
PRIMARY_SUFFIX = [
    ("lightrag_v1free", "vector_v1free"),
    ("ms_graphrag_v1free", "vector_v1free"),
    ("hipporag2_v1free", "vector_v1free"),
    ("cognee_v1free_k5", "vector_v1free"),
]
SECONDARY_SUFFIX = [
    ("lightrag_v1free", "ms_graphrag_v1free"),
    ("lightrag_v1free_native_context", "lightrag_v1free"),
    ("hipporag2_v1free", "cognee_v1free_k5"),
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
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def compare(cursor, exp_a: str, exp_b: str, prefix: str) -> dict | None:
    a_map, b_map = strict_map(cursor, exp_a), strict_map(cursor, exp_b)
    common = sorted(set(a_map) & set(b_map))
    if not common:
        return None  # par sem dados neste dataset (ex.: célula que só existe no musique)
    both = sum(1 for q in common if a_map[q] and b_map[q])
    neither = sum(1 for q in common if not a_map[q] and not b_map[q])
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
        "hipporag2_v1free=hipporag2_v1free_official. Repetível. Só muda QUAIS "
        "células entram, nunca como o teste é calculado.",
    )
    ap.add_argument("--out-tag", default="", help="sufixo do ficheiro de saída")
    args = ap.parse_args()
    prefix = args.exp_prefix
    reports_dir = Path(args.reports_dir)

    if args.suffix_override:
        mapping = dict(pair.split("=", 1) for pair in args.suffix_override)
        globals()["PRIMARY_SUFFIX"] = [
            (mapping.get(a, a), mapping.get(b, b)) for a, b in PRIMARY_SUFFIX
        ]
        globals()["SECONDARY_SUFFIX"] = [
            (mapping.get(a, a), mapping.get(b, b)) for a, b in SECONDARY_SUFFIX
        ]
        print(f"sufixos trocados: {mapping}")

    connection = connect_postgres(load_settings())
    cursor = connection.cursor()

    primary = [compare(cursor, prefix + a, prefix + b, prefix) for a, b in PRIMARY_SUFFIX]
    missing_primary = [PRIMARY_SUFFIX[i] for i, r in enumerate(primary) if r is None]
    primary = [r for r in primary if r is not None]
    holm(primary)
    secondary = []
    skipped = [f"{a} vs {b}" for a, b in missing_primary]
    for a, b in SECONDARY_SUFFIX:
        row = compare(cursor, prefix + a, prefix + b, prefix)
        if row is None:
            skipped.append(f"{a} vs {b}")
            continue
        row["note"] = "exploratória (fora da família Holm)"
        secondary.append(row)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exp_prefix": prefix,
        "test": "McNemar exato (binomial bicaudal, pares discordantes), pareado por question_id",
        "metric": "answer_correctness_judge == 1.0 (strict)",
        "family_primary": primary,
        "secondary_exploratory": secondary,
        "skipped_pairs_no_data": skipped,
        "refs": "Dror et al. ACL 2018; Holm 1979",
    }
    for row in [*primary, *secondary]:
        row["p_mcnemar_exact"] = round(row["p_mcnemar_exact"], 8)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"mcnemar_v1_arm_2026-07{args.out_tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
