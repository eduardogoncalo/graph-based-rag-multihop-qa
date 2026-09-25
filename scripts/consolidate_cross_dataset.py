"""F4.3 — Consolidado CROSS-dataset (MuSiQue × 2Wiki): a peça nova da expansão.

Lê os consolidated_p5 e mcnemar_* dos dois datasets e produz:
  - artifacts/cross_dataset/cross_dataset_consolidated.json
  - artifacts/cross_dataset/fig_cross_dataset_arms.png (2 painéis: braço A e B)
  - tabela no stdout (células lado a lado + veredictos das 4 leituras + ordenações)

Uso: PYTHONPATH=src .venv/bin/python scripts/consolidate_cross_dataset.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MUS = Path("artifacts/musique/reports")
TWK = Path("artifacts/twowiki/reports")
OUT = Path("artifacts/cross_dataset")

FRAMEWORKS = ["lightrag", "ms_graphrag", "hipporag2", "cognee"]
BLUE, GOLD = "#1f77b4", "#b8860b"  # paleta validada (datasets)
INK, MUT, SURFACE = "#1f1f1e", "#6b6a66", "#fcfcfb"


def load(reports: Path, name_hint: str) -> dict:
    cands = sorted(reports.glob("consolidated_p5_*.json"))
    if not cands:
        raise SystemExit(f"consolidated_p5 not found in {reports}")
    return json.loads(cands[-1].read_text())


def mc(reports: Path, name: str) -> dict:
    return json.loads((reports / name).read_text())


def sig_delta(mcnemar: dict, family: str, a: str, b: str):
    for row in mcnemar[family]:
        if row["A"] == a and row["B"] == b:
            return row["delta"], bool(row.get("significant_0.05_holm"))
    return None, None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p5 = {"musique": load(MUS, "musique"), "twowiki": load(TWK, "twowiki")}
    mc_a = {"musique": mc(MUS, "mcnemar_v1_arm_2026-07.json"),
            "twowiki": mc(TWK, "mcnemar_v1_arm_2026-07.json")}
    mc_b = {"musique": mc(MUS, "mcnemar_native_arm_2026-07.json"),
            "twowiki": mc(TWK, "mcnemar_native_arm_2026-07.json")}

    cells = {}
    for ds in ("musique", "twowiki"):
        cells[ds] = {
            "dense": p5[ds]["baselines"]["vector_denso (controlado)"]["strict"],
            "floor": p5[ds]["baselines"]["closed_book (piso)"]["strict"],
            "ceiling": p5[ds]["baselines"]["oracle_gold_v1 (teto)"]["strict"],
        }
        for fw in FRAMEWORKS:
            for arm in ("controlado", "nativo"):
                cells[ds][f"{fw}|{arm}"] = p5[ds]["cells"][f"{fw}|{arm}"]["strict"]

    # ---- as 4 leituras pré-registradas (família A: substrato vs denso) ----
    readings = []
    for fw in FRAMEWORKS:
        row = {"pair": f"{fw} vs denso (braço A)"}
        for ds in ("musique", "twowiki"):
            d, s = sig_delta(mc_a[ds], "family_primary",
                             f"{fw}_v1free" if fw != "cognee" else "cognee_v1free_k5",
                             "vector_v1free")
            row[ds] = {"delta": d, "sig": s}
        m, t = row["musique"], row["twowiki"]
        if m["sig"] and t["sig"] and (m["delta"] > 0) == (t["delta"] > 0):
            verdict = "REPLICA"
        elif m["sig"] and not t["sig"]:
            verdict = "ATENUA (perde significância no 2Wiki)"
        elif not m["sig"] and t["sig"]:
            verdict = "EMERGE (ganha significância no 2Wiki)"
        else:
            verdict = "n.s. nos dois"
        row["verdict"] = verdict
        readings.append(row)

    # ---- deployment gap (família A↔B) ----
    gaps = []
    for fw in FRAMEWORKS:
        row = {"pair": f"{fw}: nativo vs controlado"}
        for ds in ("musique", "twowiki"):
            ctrl = f"{fw}_v1free" if fw != "cognee" else "cognee_v1free_k5"
            d, s = sig_delta(mc_b[ds], "family_AB_deployment_gap", f"{fw}_native", ctrl)
            row[ds] = {"delta": d, "sig": s}
        gaps.append(row)

    # ---- ordenações (numéricas; empates estatísticos anotados via família B) ----
    orders = {}
    for ds in ("musique", "twowiki"):
        for arm in ("controlado", "nativo"):
            ranked = sorted(FRAMEWORKS, key=lambda f: -cells[ds][f"{f}|{arm}"])
            orders[f"{ds}|{arm}"] = ranked

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cells_strict": cells,
        "readings_family_A": readings,
        "deployment_gap_family_AB": gaps,
        "orderings": orders,
        "sources": {
            "musique": str(sorted(MUS.glob('consolidated_p5_*.json'))[-1]),
            "twowiki": str(sorted(TWK.glob('consolidated_p5_*.json'))[-1]),
        },
    }
    (OUT / "cross_dataset_consolidated.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    # ---- figura: 2 painéis (braço A / braço B), barras agrupadas por dataset ----
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), dpi=170, sharey=True)
    import numpy as np
    y = np.arange(len(FRAMEWORKS))[::-1]
    h = 0.36
    for ax, arm, title in zip(
        axes, ("controlado", "nativo"),
        ("Arm A (controlled, fixed reader)", "Arm B (native, as deployed)"),
    ):
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.xaxis.grid(True, color="#e7e5e0", linewidth=1.0)
        ax.set_axisbelow(True)
        mus_v = [cells["musique"][f"{f}|{arm}"] for f in FRAMEWORKS]
        twk_v = [cells["twowiki"][f"{f}|{arm}"] for f in FRAMEWORKS]
        b1 = ax.barh(y + h / 2 + 0.02, mus_v, height=h, color=BLUE, label="MuSiQue")
        b2 = ax.barh(y - h / 2 - 0.02, twk_v, height=h, color=GOLD, label="2Wiki")
        for bars in (b1, b2):
            for r in bars:
                ax.text(r.get_width() + 0.006, r.get_y() + r.get_height() / 2,
                        f"{r.get_width():.3f}", va="center", fontsize=7.5, color=INK)
        if arm == "controlado":
            ax.axvline(cells["musique"]["dense"], color=BLUE, ls="--", lw=1, alpha=0.75)
            ax.axvline(cells["twowiki"]["dense"], color=GOLD, ls="--", lw=1, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(FRAMEWORKS, fontsize=9)
        ax.set_xlim(0, 0.85)
        ax.set_title(title, fontsize=10, color=INK)
        ax.tick_params(colors=MUT, labelcolor=INK)
    axes[0].set_xlabel("strict (judge gpt-4o); dashed = dataset's dense baseline",
                       fontsize=8, color=MUT)
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_dataset_arms.png", bbox_inches="tight")
    plt.close(fig)

    # ---- stdout ----
    print("=== 4 readings (family A, substrate vs dense) ===")
    for r in readings:
        m, t = r["musique"], r["twowiki"]
        print(f"  {r['pair']:28} musique Δ={m['delta']:+.3f} sig={m['sig']} | "
              f"2wiki Δ={t['delta']:+.3f} sig={t['sig']} -> {r['verdict']}")
    print("=== deployment gap (A↔B) ===")
    for r in gaps:
        m, t = r["musique"], r["twowiki"]
        print(f"  {r['pair']:32} musique Δ={m['delta']:+.3f} sig={m['sig']} | "
              f"2wiki Δ={t['delta']:+.3f} sig={t['sig']}")
    print("=== orderings (strict) ===")
    for k, v in orders.items():
        print(f"  {k:20} {' > '.join(v)}")
    print(f"\n-> {OUT}/cross_dataset_consolidated.json\n-> {OUT}/fig_cross_dataset_arms.png")


if __name__ == "__main__":
    main()
