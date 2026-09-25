"""Fase 2 — suíte estatística do MuSiQue eval1k (receita do relatório §4.3).

Pré-registro: a família confirmatória são os pares entre métodos v2 (leitor
grounded) + ms_graphrag quando disponível, com correção de Holm. Testes:
McNemar exato (binário pareado, judge strict), bootstrap BCa (ICs de células e
diferenças), TOST (equivalência de retrieval, δ=2pp) e estratos por hop com
IC + tamanho de efeito + n + MDE aproximado (SEM "poder observado" — decisão
do autor 2026-07-04).

Read-only sobre o Postgres; saída em artifacts/musique/reports/. Enquanto
`musique_eval1k_ms_graphrag` não tiver judge, a saída é marcada PRELIMINAR.

Uso: .venv/bin/python scripts/run_stats.py [--reps 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

from benchmark.cli.app import _connect_postgres

R = Path("artifacts/musique/reports")
RETRIEVAL_JSON = R / "retrieval_metrics_corrected_2026-07-02.json"
# ATENÇÃO: no JSON "corrected", a entrada vector é o run QUEBRADO (IVFFlat, 0.0954);
# o vector consertado (busca exata, 0.555) vive no after_fix — usar este p/ o TOST.
VECTOR_FIXED_JSON = R / "vector_recall_after_fix_2026-07-01.json"

# rótulo -> (experiment_id, method_id)
EXPERIMENTS = {
    "closed_book": ("musique_eval1k_closed_book", "zero_shot_no_context"),
    "vector_v1": ("musique_eval1k_first_results", "vector_rag"),
    "lightrag_v1": ("musique_eval1k_first_results", "lightrag_neo4j"),
    "cognee_v1": ("musique_eval1k_cognee_fixed", "cognee"),
    "vector_v2": ("musique_eval1k_v2_grounded", "vector_rag"),
    "lightrag_v2": ("musique_eval1k_lightrag_v2", "lightrag_neo4j"),
    "cognee_v2": ("musique_eval1k_cognee_v2", "cognee"),
    "oracle_v2": ("musique_eval1k_oracle_gold", "single_document_context"),
    "oracle_v1": ("musique_eval1k_oracle_gold_v1free", "single_document_context"),
    # v1-livre LIMPO (mesmo retrieval do lightrag_v2; só READER_GROUNDING=v1) — 2026-07-04.
    "lightrag_v1free": ("musique_eval1k_lightrag_v1free", "lightrag_neo4j"),
    "ms_graphrag": ("musique_eval1k_ms_graphrag", "ms_graphrag"),
}
# família confirmatória (Holm aplicado em conjunto); graphrag entra quando judged
PRIMARY_METHODS = ["vector_v2", "lightrag_v2", "cognee_v2", "ms_graphrag"]
# contrastes secundários descritivos (Holm próprio, família separada)
SECONDARY_PAIRS = [
    ("lightrag_v1free", "lightrag_v2"),  # grounding LIMPO: mesmo retrieval, só a instrução muda (2026-07-04)
    ("lightrag_v1", "lightrag_v2"),  # efeito do grounding (v1 legado = first_results)
    ("cognee_v1", "cognee_v2"),
    ("vector_v1", "vector_v2"),
    ("oracle_v1", "oracle_v2"),      # custo do grounding no teto
]


def load_judge(conn, experiment_id: str, method_id: str) -> dict[str, dict]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT a.question_id, er.metric_value, er.metadata->>'label',
                      jsonb_array_length(q.metadata->'decomposition')
               FROM evaluation_results er
               JOIN runs r ON r.run_id = er.run_id
               JOIN answers a ON a.run_id = r.run_id
               JOIN questions q ON q.question_id = a.question_id
               WHERE r.experiment_id = %s AND r.method_id = %s
                 AND er.metric_name = 'answer_correctness_judge'""",
            (experiment_id, method_id),
        )
        return {
            qid: {"strict": 1.0 if float(v) == 1.0 else 0.0, "label": lab, "hops": int(h)}
            for qid, v, lab, h in cur.fetchall()
        }


def bca_ci(x: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    try:
        res = stats.bootstrap((x,), np.mean, n_resamples=reps, method="BCa",
                              confidence_level=0.95, rng=rng)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        res = stats.bootstrap((x,), np.mean, n_resamples=reps, method="percentile",
                              confidence_level=0.95, rng=rng)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_diff_ci(x: np.ndarray, y: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    diffs = np.empty(reps)
    for i in range(reps):
        sample = rng.choice(idx, size=len(idx), replace=True)
        diffs[i] = x[sample].mean() - y[sample].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> dict:
    b = int(np.sum((x == 1) & (y == 0)))  # A acerta, B erra
    c = int(np.sum((x == 0) & (y == 1)))
    n_disc = b + c
    p = float(stats.binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue) if n_disc else 1.0
    return {"b": b, "c": c, "discordantes": n_disc, "odds_ratio_bc": (b / c) if c else None, "p_exato": p}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def mde_mcnemar(n: int, p_disc: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """MDE aproximado (pp de diferença) p/ McNemar dado n e taxa de discordância."""
    if n == 0 or p_disc == 0:
        return float("nan")
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((z_a + z_b) * np.sqrt(p_disc / n))


def tost_paired(x: np.ndarray, y: np.ndarray, delta: float) -> dict:
    d = x - y
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t_low = (d.mean() + delta) / se   # H0: diff <= -delta
    t_up = (d.mean() - delta) / se    # H0: diff >= +delta
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_up = stats.t.cdf(t_up, n - 1)
    p = float(max(p_low, p_up))
    return {"diff_media": float(d.mean()), "delta": delta, "p_tost": p, "equivalente_5pct": p < 0.05}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    conn = _connect_postgres()
    data: dict[str, dict[str, dict]] = {}
    for label, (exp, method) in EXPERIMENTS.items():
        rows = load_judge(conn, exp, method)
        if rows:
            data[label] = rows
    # ---- AUDITORIA (pré-análise): exclusões, pareamento, labels, binarização ----
    with conn.cursor() as cur:
        cur.execute("SELECT question_id FROM questions ORDER BY question_id")
        all_qids = {row[0] for row in cur.fetchall()}
    audit: dict = {"n_total_questions": len(all_qids), "por_experimento": {}}
    for label, rows in data.items():
        qids = set(rows)
        labels_count: dict[str, int] = {}
        for r in rows.values():
            labels_count[r["label"]] = labels_count.get(r["label"], 0) + 1
        n_correct_bin = sum(1 for r in rows.values() if r["strict"] == 1.0)
        audit["por_experimento"][label] = {
            "n_julgados": len(qids),
            "excluidos_gold_suspect": sorted(all_qids - qids),
            "n_excluidos": len(all_qids - qids),
            "label_dist": labels_count,
            "binarizacao_ok": n_correct_bin == labels_count.get("correct", 0),
            "regra": "strict=1 sse label=correct; partial/incorrect/refusal=0",
        }
    # pareamento: interseções entre todos os experimentos carregados
    labels_loaded = sorted(data)
    base = set(data[labels_loaded[0]])
    inter = set(base)
    for lbl in labels_loaded[1:]:
        inter &= set(data[lbl])
    audit["pareamento"] = {
        "intersecao_global": len(inter),
        "identicos_todos": all(set(data[lbl]) == set(data[labels_loaded[0]]) for lbl in labels_loaded),
        "detalhe_pares_diferentes": {
            lbl: sorted(set(data[labels_loaded[0]]) ^ set(data[lbl]))[:10]
            for lbl in labels_loaded[1:]
            if set(data[lbl]) != set(data[labels_loaded[0]])
        },
    }
    conn.close()
    preliminary = "ms_graphrag" not in data

    out: dict = {
        "generated": date.today().isoformat(),
        "status": "PRELIMINAR (ms_graphrag pendente)" if preliminary else "FINAL",
        "seed": args.seed, "bootstrap_reps": args.reps,
        "judge": "answer_correctness_judge (strict: correct=1, resto=0)",
        "experimentos": {k: EXPERIMENTS[k][0] for k in data},
        "auditoria": audit,
    }

    # ---- 1. células da régua: acurácia + IC BCa ----
    cells = {}
    for label, rows in data.items():
        x = np.array([r["strict"] for r in rows.values()])
        low, high = bca_ci(x, args.reps, args.seed)
        cells[label] = {"n": len(x), "strict": float(x.mean()), "ic95": [low, high]}
    out["celulas"] = cells

    # ---- 2. McNemar + Holm (família primária) ----
    def paired_vectors(a: str, b: str):
        common = sorted(set(data[a]) & set(data[b]))
        xa = np.array([data[a][q]["strict"] for q in common])
        xb = np.array([data[b][q]["strict"] for q in common])
        hops = np.array([data[a][q]["hops"] for q in common])
        return common, xa, xb, hops

    primary = [m for m in PRIMARY_METHODS if m in data]
    prim_results, prim_p = [], []
    for i in range(len(primary)):
        for j in range(i + 1, len(primary)):
            a, b = primary[i], primary[j]
            _, xa, xb, _ = paired_vectors(a, b)
            mc = mcnemar_exact(xa, xb)
            ci = paired_diff_ci(xa, xb, args.reps, args.seed)
            prim_results.append({
                "par": f"{a} vs {b}", "n": len(xa),
                "acc_a": float(xa.mean()), "acc_b": float(xb.mean()),
                "diff_pp": float((xa.mean() - xb.mean()) * 100), "ic95_diff": ci, **mc,
            })
            prim_p.append(mc["p_exato"])
    for res, p_adj in zip(prim_results, holm(prim_p)):
        res["p_holm"] = p_adj
        res["significativo_5pct"] = p_adj < 0.05
    out["familia_primaria_v2"] = prim_results

    # ---- 3. contrastes secundários (grounding v1 vs v2; oracle) ----
    sec_results, sec_p = [], []
    for a, b in SECONDARY_PAIRS:
        if a not in data or b not in data:
            continue
        _, xa, xb, _ = paired_vectors(a, b)
        mc = mcnemar_exact(xa, xb)
        ci = paired_diff_ci(xa, xb, args.reps, args.seed)
        sec_results.append({
            "par": f"{a} vs {b}", "n": len(xa),
            "acc_a": float(xa.mean()), "acc_b": float(xb.mean()),
            "diff_pp": float((xa.mean() - xb.mean()) * 100), "ic95_diff": ci, **mc,
        })
        sec_p.append(mc["p_exato"])
    for res, p_adj in zip(sec_results, holm(sec_p)):
        res["p_holm"] = p_adj
        res["significativo_5pct"] = p_adj < 0.05
    out["familia_secundaria_grounding"] = sec_results

    # ---- 4. estratos por hop (pares primários): IC, efeito, n, MDE ----
    strata = []
    for i in range(len(primary)):
        for j in range(i + 1, len(primary)):
            a, b = primary[i], primary[j]
            _, xa, xb, hops = paired_vectors(a, b)
            for h in (2, 3, 4):
                mask = hops == h
                if not mask.any():
                    continue
                xah, xbh = xa[mask], xb[mask]
                mc = mcnemar_exact(xah, xbh)
                ci = paired_diff_ci(xah, xbh, args.reps, args.seed + h)
                n = int(mask.sum())
                strata.append({
                    "par": f"{a} vs {b}", "hops": h, "n": n,
                    "acc_a": float(xah.mean()), "acc_b": float(xbh.mean()),
                    "diff_pp": float((xah.mean() - xbh.mean()) * 100),
                    "ic95_diff": ci, "b": mc["b"], "c": mc["c"],
                    "mde_80pct_pp": mde_mcnemar(n, mc["discordantes"] / n) * 100,
                })
    out["estratos_por_hop"] = strata

    # ---- 5. TOST retrieval (vector vs lightrag, δ=2pp) — independe do graphrag ----
    if RETRIEVAL_JSON.exists() and VECTOR_FIXED_JSON.exists():
        rj = json.loads(RETRIEVAL_JSON.read_text())
        vf = json.loads(VECTOR_FIXED_JSON.read_text())
        rec_l = {e["question_id"]: e["recall"] for e in rj["per_question"]["lightrag_neo4j"]}
        # comparável ao doc-level do lightrag (conjunto recuperado ~5 docs): recall_at_5 do denso consertado
        rec_v = {e["question_id"]: e["recall_at_5"] for e in vf["per_question"]}
        common = sorted(set(rec_v) & set(rec_l))
        xv = np.array([rec_v[q] for q in common])
        xl = np.array([rec_l[q] for q in common])
        tost = tost_paired(xv, xl, delta=0.02)
        tost["n"] = len(common)
        tost["fontes"] = {"vector": "after_fix recall_at_5 (busca exata)",
                          "lightrag": "corrected doc-level (parser fixado)"}
        tost["ic95_diff"] = paired_diff_ci(xv, xl, args.reps, args.seed)
        out["tost_retrieval_vector_vs_lightrag"] = tost

    json_path = R / f"stats_{out['generated']}.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- markdown ----
    lines = [f"# Estatística MuSiQue eval1k — {out['generated']} ({out['status']})", ""]
    lines.append(f"Judge strict (correct=1). Bootstrap {args.reps} réplicas, seed {args.seed}. "
                 "McNemar exato; correção de Holm por família (pré-registrada).")
    lines.append("\n## Auditoria (pré-análise)\n")
    aud = out["auditoria"]
    par = aud["pareamento"]
    lines.append(f"- Perguntas no dataset: {aud['n_total_questions']} · interseção global julgada: "
                 f"{par['intersecao_global']} · conjuntos idênticos em todos os experimentos: "
                 f"{'SIM' if par['identicos_todos'] else 'NÃO — ver JSON'}")
    lines.append("- Regra de binarização: strict=1 ⟺ label='correct'; partial, incorrect e "
                 "**refusal contam como não-correto** no McNemar.")
    lines.append("\n| experimento | julgados | excluídos (gold_suspect) | correct | partial | incorrect | refusal | binarização confere |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in ["closed_book","vector_v1","vector_v2","lightrag_v1","lightrag_v2",
              "cognee_v1","cognee_v2","ms_graphrag","oracle_v2","oracle_v1"]:
        if k in aud["por_experimento"]:
            a = aud["por_experimento"][k]; ld = a["label_dist"]
            lines.append(f"| {k} | {a['n_julgados']} | {a['n_excluidos']} | "
                         f"{ld.get('correct',0)} | {ld.get('partial',0)} | {ld.get('incorrect',0)} | "
                         f"{ld.get('refusal',0)} | {'✓' if a['binarizacao_ok'] else 'ERRO'} |")
    lines.append("\n## Células (acurácia strict, IC95 BCa)\n")
    lines.append("| experimento | n | strict | IC95 |")
    lines.append("|---|---|---|---|")
    order = ["closed_book", "vector_v1", "vector_v2", "lightrag_v1", "lightrag_v2",
             "cognee_v1", "cognee_v2", "ms_graphrag", "oracle_v2", "oracle_v1"]
    for k in order:
        if k in cells:
            c = cells[k]
            lines.append(f"| {k} | {c['n']} | {c['strict']:.4f} | [{c['ic95'][0]:.4f}, {c['ic95'][1]:.4f}] |")
    for title, results in (("Família primária (v2 + graphrag) — McNemar + Holm", prim_results),
                           ("Família secundária (efeito do grounding)", sec_results)):
        lines.append(f"\n## {title}\n")
        lines.append("| par | n | A✓/B✗ (b) | A✗/B✓ (c) | b+c | Δ acertos | Δpp | IC95 Δ | p bruto | p Holm | sig. |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in results:
            lines.append(
                f"| {r['par']} | {r['n']} | {r['b']} | {r['c']} | {r['discordantes']} | "
                f"{r['b']-r['c']:+d} | {r['diff_pp']:+.2f} | "
                f"[{r['ic95_diff'][0]*100:+.2f}, {r['ic95_diff'][1]*100:+.2f}] | "
                f"{r['p_exato']:.4g} | {r['p_holm']:.4g} | "
                f"{'SIM' if r['significativo_5pct'] else 'não'} |")
    lines.append("\n## Estratos por hop (IC, efeito, n, MDE≈80% — sem 'poder observado')\n")
    lines.append("| par | hops | n | Δpp | IC95 Δ | b/c | MDE (pp) |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in strata:
        lines.append(
            f"| {s['par']} | {s['hops']} | {s['n']} | {s['diff_pp']:+.2f} | "
            f"[{s['ic95_diff'][0]*100:+.2f}, {s['ic95_diff'][1]*100:+.2f}] | "
            f"{s['b']}/{s['c']} | {s['mde_80pct_pp']:.1f} |")
    if "tost_retrieval_vector_vs_lightrag" in out:
        t = out["tost_retrieval_vector_vs_lightrag"]
        lines.append("\n## TOST equivalência de retrieval (vector vs lightrag, δ=2pp)\n")
        lines.append(f"n={t['n']} · Δ média={t['diff_media']*100:+.2f}pp · "
                     f"IC95 [{t['ic95_diff'][0]*100:+.2f}, {t['ic95_diff'][1]*100:+.2f}] · "
                     f"p_TOST={t['p_tost']:.4g} → "
                     f"{'EQUIVALENTES na margem de 2pp' if t['equivalente_5pct'] else 'equivalência NÃO demonstrada na margem de 2pp'}")
    md_path = R / f"stats_{out['generated']}.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[stats] {out['status']}")
    print(f"[stats] json -> {json_path}")
    print(f"[stats] md   -> {md_path}")


if __name__ == "__main__":
    main()
