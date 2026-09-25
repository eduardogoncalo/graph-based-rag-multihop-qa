"""P5 — consolidado FINAL dos 2 braços × 4 frameworks (dossiê dois braços §1.6).

Lê os summaries do judge (gpt-4o, judge_v1) + os 2 JSONs de McNemar já gerados
(mcnemar_v1_arm_2026-07.json = família A; mcnemar_native_arm_2026-07.json =
famílias B e A↔B) e materializa:
  - artifacts/musique/reports/consolidated_p5_2026-07-08.json (todas as células)
  - artifacts/musique/reports/figures/fig_p5_*.png (3 figuras)

Métricas seletivas (mesma definição do quadro de 2026-07-04):
  cobertura = 1 - refusal_rate; seletiva = strict / cobertura.
Read-only sobre os artefatos; nada de Postgres nem API.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("artifacts/musique/reports")
FIG = R / "figures"

# rótulo -> (summary json, chave em by_method)
CELLS = {
    ("lightrag", "controlado"): ("llm_judge_full_lightrag_v1free_summary.json", "lightrag_neo4j"),
    ("ms_graphrag", "controlado"): ("llm_judge_full_ms_graphrag_v1free_summary.json", "ms_graphrag"),
    ("hipporag2", "controlado"): ("llm_judge_full_hipporag2_v1free_summary.json", "hipporag2"),
    ("cognee", "controlado"): ("llm_judge_full_cognee_v1free_k5_summary.json", "cognee"),
    ("lightrag", "nativo"): ("llm_judge_full_lightrag_native_summary.json", "lightrag_neo4j"),
    ("ms_graphrag", "nativo"): ("llm_judge_full_ms_graphrag_native_summary.json", "ms_graphrag"),
    ("hipporag2", "nativo"): ("llm_judge_full_hipporag2_native_summary.json", "hipporag2"),
    ("cognee", "nativo"): ("llm_judge_full_cognee_native_summary.json", "cognee"),
}
BASELINES = {
    "vector_denso (controlado)": ("llm_judge_full_vector_v1free_summary.json", "vector_rag"),
    "closed_book (piso)": ("llm_judge_full_closed_book_summary.json", None),
    "oracle_gold_v1 (teto)": ("llm_judge_full_oracle_gold_v1free_summary.json", None),
}
EXPLORATORY = {
    "lightrag nctx (contexto nativo serializado, reader controlado)":
        ("llm_judge_full_lightrag_nctx_summary.json", None),
}

FRAMEWORKS = ["lightrag", "ms_graphrag", "hipporag2", "cognee"]

# paleta validada (dataviz defaults, light): slot1 azul=controlado, slot2 aqua=nativo
BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e7e5e0", "#fcfcfb"


def cell(path: str, method: str | None) -> dict:
    s = json.loads((R / path).read_text())
    m = s["by_method"][method] if method else next(iter(s["by_method"].values()))
    cobertura = round(1.0 - m["refusal_rate"], 4)
    return {
        "strict": m["strict_accuracy"], "lenient": m["lenient_accuracy"],
        "refusal": m["refusal_rate"], "cobertura": cobertura,
        "seletiva": round(m["strict_accuracy"] / cobertura, 4) if cobertura else None,
        "n_scoreable": m["n_scoreable"], "inter_pass_agreement": m["inter_pass_agreement"],
        "source": path,
    }


def main() -> None:
    import argparse

    global R, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default=str(R))
    ap.add_argument("--out-name", default="consolidated_p5_2026-07-08.json")
    ap.add_argument(
        "--summary-override",
        action="append",
        default=[],
        metavar="OLD.json=NEW.json",
        help="swap a cell's summary file, for example "
        "llm_judge_full_hipporag2_v1free_summary.json="
        "llm_judge_full_hipporag2_v1free_official_summary.json. Repeatable. Only "
        "changes WHICH summary is read, never how the metrics are computed.",
    )
    args = ap.parse_args()
    R = Path(args.reports_dir)
    FIG = R / "figures"
    out_name = args.out_name

    if args.summary_override:
        mapping = dict(pair.split("=", 1) for pair in args.summary_override)
        for table in (CELLS, BASELINES, EXPLORATORY):
            for key, (path, method) in list(table.items()):
                if path in mapping:
                    table[key] = (mapping[path], method)
        print(f"summaries swapped: {mapping}")

    FIG.mkdir(parents=True, exist_ok=True)
    cells = {f"{fw}|{arm}": cell(*CELLS[(fw, arm)]) for (fw, arm) in CELLS}
    baselines = {k: cell(*v) for k, v in BASELINES.items()}
    # exploratórias são específicas do dataset (ex.: nctx só existe no musique)
    exploratory = {}
    for k, v in EXPLORATORY.items():
        if (R / v[0]).exists():
            exploratory[k] = cell(*v)
    mcnemar_a = json.loads((R / "mcnemar_v1_arm_2026-07.json").read_text())
    mcnemar_b = json.loads((R / "mcnemar_native_arm_2026-07.json").read_text())

    out = {
        "spec": "P5 consolidado — dossiê dois braços §1.6; judge gpt-4o judge_v1; "
                "cobertura=1-refusal; seletiva=strict/cobertura",
        "cells": cells, "baselines": baselines, "exploratory": exploratory,
        "mcnemar_family_A_controlled": mcnemar_a,
        "mcnemar_families_B_and_AB_native": mcnemar_b,
    }
    (R / out_name).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    })

    # ---- Fig 1: strict por framework × braço (barras pareadas horizontais) ----
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = list(range(len(FRAMEWORKS)))[::-1]
    h = 0.34
    ctrl = [cells[f"{fw}|controlado"]["strict"] for fw in FRAMEWORKS]
    natv = [cells[f"{fw}|nativo"]["strict"] for fw in FRAMEWORKS]
    ax.barh([i + h / 2 + 0.02 for i in y], ctrl, height=h, color=BLUE, label="controlled (fixed reader v1free)")
    ax.barh([i - h / 2 - 0.02 for i in y], natv, height=h, color=AQUA, label="native (as deployed)")
    for i, fw in zip(y, FRAMEWORKS):
        ax.text(cells[f"{fw}|controlado"]["strict"] + 0.008, i + h / 2 + 0.02,
                f"{cells[f'{fw}|controlado']['strict']:.3f}", va="center", fontsize=9, color=INK)
        ax.text(cells[f"{fw}|nativo"]["strict"] + 0.008, i - h / 2 - 0.02,
                f"{cells[f'{fw}|nativo']['strict']:.3f}", va="center", fontsize=9, color=INK)
    import matplotlib.transforms as mtransforms

    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for label, key, style in [("closed-book (floor)", "closed_book (piso)", ":"),
                              ("controlled dense", "vector_denso (controlado)", "--"),
                              ("oracle v1 (ceiling)", "oracle_gold_v1 (teto)", "-.")]:
        v = baselines[key]["strict"]
        ax.axvline(v, color=MUTED, linestyle=style, linewidth=1)
        ax.text(v, 1.01, f"{label}\n{v:.3f}", transform=blend,
                ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_yticks(y, FRAMEWORKS)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("strict accuracy (judge gpt-4o, 1000q MuSiQue eval1k)")
    ax.set_title("Arm A (controlled) × Arm B (native) — strict by framework",
                 fontsize=11, color=INK, pad=36)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_p5_strict_by_arm.png", dpi=180)
    plt.close(fig)

    # ---- Fig 2: deployment gap (nativo − controlado), diverging ----
    gaps = {r["A"].replace("_native", ""): r for r in mcnemar_b["family_AB_deployment_gap"]}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = list(range(len(FRAMEWORKS)))[::-1]
    for i, fw in zip(y, FRAMEWORKS):
        r = gaps[fw]
        d = r["delta"]
        sig = r["significant_0.05_holm"]
        ax.barh(i, d, height=0.5, color=(BLUE if d > 0 else RED))
        ax.text(d + (0.004 if d > 0 else -0.004), i,
                f"{d:+.3f} ({'sig. Holm' if sig else 'n.s.'})",
                va="center", ha="left" if d > 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, FRAMEWORKS)
    ax.set_xlim(-0.15, 0.15)
    ax.set_xlabel("Δ strict (native − controlled), paired McNemar; p_holm in the report table")
    ax.set_title("Family A↔B — deployment gap by framework", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_p5_deployment_gap.png", dpi=180)
    plt.close(fig)

    # ---- Fig 3: recusa por framework × braço ----
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = list(range(len(FRAMEWORKS)))[::-1]
    for i, fw in zip(y, FRAMEWORKS):
        c, n = cells[f"{fw}|controlado"], cells[f"{fw}|nativo"]
        ax.barh(i + h / 2 + 0.02, c["refusal"], height=h, color=BLUE,
                label="controlled (fixed reader v1free)" if fw == FRAMEWORKS[0] else None)
        ax.barh(i - h / 2 - 0.02, n["refusal"], height=h, color=AQUA,
                label="native (as deployed)" if fw == FRAMEWORKS[0] else None)
        ax.text(c["refusal"] + 0.003, i + h / 2 + 0.02, f"{c['refusal']:.3f}", va="center", fontsize=9, color=INK)
        ax.text(n["refusal"] + 0.003, i - h / 2 - 0.02, f"{n['refusal']:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, FRAMEWORKS)
    ax.set_xlim(0, 0.26)
    ax.set_xlabel("refusal rate (judge: label=refusal)")
    ax.set_title("Refusal by framework × arm", fontsize=11, color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_p5_refusal.png", dpi=180)
    plt.close(fig)

    # ---- tabela no stdout ----
    print(f"{'framework':<12} {'arm':<11} {'strict':>7} {'lenient':>8} {'refusal':>7} {'coverage':>8} {'selective':>9}")
    for fw in FRAMEWORKS:
        for arm in ("controlado", "nativo"):
            c = cells[f"{fw}|{arm}"]
            print(f"{fw:<12} {arm:<11} {c['strict']:>7.3f} {c['lenient']:>8.3f} "
                  f"{c['refusal']:>7.3f} {c['cobertura']:>8.3f} {c['seletiva']:>9.3f}")
    for k, c in {**baselines, **exploratory}.items():
        print(f"{k:<24} {c['strict']:>7.3f} {c['lenient']:>8.3f} {c['refusal']:>7.3f} "
              f"{c['cobertura']:>8.3f} {c['seletiva']:>9.3f}")
    print(f"\n-> {R/out_name}\n-> {FIG}/fig_p5_*.png")


if __name__ == "__main__":
    main()
