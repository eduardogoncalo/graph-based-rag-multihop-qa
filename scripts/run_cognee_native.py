"""P2 (Braço B) — cognee NATIVO ("as deployed").

Roda a geração PRÓPRIA do cognee (GRAPH_COMPLETION com only_context=False) sobre as
1000 perguntas MuSiQue eval1k e materializa como experimento
`musique_eval1k_cognee_native`. É o segundo ponto do Braço B (o primeiro foi o
lightrag nativo / P1).

Diferença crucial vs P1: o cognee rodou o braço controlado com only_context=True e
NUNCA gerou uma resposta nativa (0 chamadas LLM na consulta). Então aqui NÃO é ETL —
é geração real: 1 chamada gpt-4o-mini por pergunta, feita DENTRO do cognee (o
completion step do GraphCompletionRetriever). A resposta sai em
`retrieval.metadata["generated_answer"]` (já extraída pelo parser).

top_k = 10 = DEFAULT da API `recall()` do cognee 1.1.2 ("as deployed"), NÃO o k=5
imposto ao braço controlado para parear com vector/lightrag. Os demais knobs
(wide_search_top_k=100, neighborhood_seed_top_k, global_context_index_top_k...) ficam
nos defaults nativos do cognee — o adapter só passa top_k, os outros o cognee resolve.

Infra: lê o índice local Option C (Neo4j bolt 17689 + sqlite + lancedb sob o
workspace), via benchmark.methods.cognee.option_c.resolve_cognee_config. Modelo interno
= gpt-4o-mini (mesmo do braço A). Rode a partir do .venv (mesmo caminho da avaliação
controlada), com .env exportado e os envs Neo4j de lightrag/graphrag limpos.

Modos:
  --smoke [N]   amostra estratificada por hop (seed 42, default 15); gera as nativas,
                escreve JSONL + relatório de gate; NÃO escreve nas tabelas do benchmark.
  --full        todas as 1000; persiste em musique_eval1k_cognee_native (idempotente,
                resumível por question_id). O judge é passo separado
                (scripts/run_llm_judge.py --full --experiment-id musique_eval1k_cognee_native).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from benchmark.methods.cognee.adapter import CogneeAdapter
from benchmark.methods.cognee.config_builder import build_workspace
from benchmark.methods.cognee.option_c import resolve_cognee_config

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "musique"
DATASET_VERSION = "ans_v1.0_eval1k"
QUESTIONS = ROOT / "data/canonical/musique_ans_v1.0_eval1k/questions.jsonl"
REPORTS = ROOT / "artifacts/musique/reports"

TARGET_EXP = "musique_eval1k_cognee_native"
METHOD = "cognee"
AGENT_MODE = "single_agent"


def _bind_dataset(dataset_id: str, dataset_version: str) -> None:
    """Rebinda os globals p/ outro dataset (default musique = retrocompatível)."""
    global DATASET_ID, DATASET_VERSION, QUESTIONS, REPORTS, TARGET_EXP
    DATASET_ID, DATASET_VERSION = dataset_id, dataset_version
    QUESTIONS = ROOT / f"data/canonical/{dataset_id}_{dataset_version}/questions.jsonl"
    REPORTS = ROOT / f"artifacts/{dataset_id}/reports"
    TARGET_EXP = f"{dataset_id}_eval1k_cognee_native"

# cognee recall() API default (1.1.2) — the "as deployed" retrieval breadth.
COGNEE_NATIVE_TOP_K = 10

# Smoke stratification: proporcional a 535/318/147 → 8/5/2 = 15.
SMOKE_STRATA = {"2": 8, "3": 5, "4": 2}
SMOKE_SEED = 42


# ----------------------------- dados -----------------------------

def _hop_of(question: dict[str, Any]) -> str:
    meta = question.get("metadata") or {}
    src = meta.get("source_question_id") or ""
    if "hop" in src:
        return src.split("hop", 1)[0]
    dec = meta.get("decomposition")
    return str(len(dec)) if isinstance(dec, list) else "?"


def _gold_str(question: dict[str, Any]) -> str:
    gold = question.get("gold_answer")
    if isinstance(gold, list):
        return " | ".join(str(g) for g in gold)
    return "" if gold is None else str(gold)


def _load_questions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with QUESTIONS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _smoke_sample(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Amostra estratificada determinística (ordena por question_id, pega os
    primeiros k de cada estrato de hop). Seed fixa via ordenação, sem RNG."""
    by_hop: dict[str, list[dict[str, Any]]] = {}
    for q in sorted(questions, key=lambda r: r["question_id"]):
        by_hop.setdefault(_hop_of(q), []).append(q)
    picked: list[dict[str, Any]] = []
    for hop, k in SMOKE_STRATA.items():
        picked.extend(by_hop.get(hop, [])[:k])
    if not picked:
        # dataset sem metadata.decomposition (ex.: twowiki): primeiras N por
        # question_id, mesmo critério do smoke do braço A
        picked = sorted(questions, key=lambda r: r["question_id"])[: sum(SMOKE_STRATA.values())]
    return picked


# ----------------------------- neo4j invariância -----------------------------

def _cognee_node_count() -> int:
    import os

    from neo4j import GraphDatabase

    from benchmark.methods.cognee.option_c import resolve_cognee_option_c

    config = resolve_cognee_option_c(
        dataset_id=DATASET_ID, dataset_version=DATASET_VERSION, preflight=False
    )
    password = os.environ.get("COGNEE_NEO4J_PASSWORD", "benchmark_cognee")
    driver = GraphDatabase.driver(config.local_index.graph_url, auth=("neo4j", password))
    try:
        with driver.session() as session:
            return int(session.run("MATCH (n) RETURN count(n) AS c").single()["c"])
    finally:
        driver.close()


# ----------------------------- execução -----------------------------

def _build_adapter(top_k: int) -> CogneeAdapter:
    workspace = build_workspace(
        artifacts_dir=ROOT / "artifacts",
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
    )
    config = resolve_cognee_config(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        top_k=top_k,
    )
    # Braço B: geração nativa do cognee (o completion step roda 1 chamada gpt-4o-mini).
    config = dataclasses.replace(config, only_context=False)
    return CogneeAdapter(workspace=workspace, config=config)


async def _generate_one(adapter: CogneeAdapter, question: dict[str, Any], top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await adapter.retrieve_async(query=question["question"], top_k=top_k)
        # only_context=False → recall força GRAPH_COMPLETION e a resposta GERADA sai como
        # o único item (result[0].text). O parser não distingue answer de blob de contexto
        # pela forma, então extraímos direto do item (aqui SABEMOS que é geração nativa).
        answer = ""
        if result.items:
            answer = (result.items[0].text or "").strip()
        if not answer:
            answer = str(result.metadata.get("generated_answer") or "").strip()
        raw = result.raw_response if isinstance(result.raw_response, dict) else {}
        return {
            "question_id": question["question_id"],
            "hop": _hop_of(question),
            "question": question["question"],
            "gold_answer": _gold_str(question),
            "native_answer": answer,
            "answer_len": len(answer),
            "items_count": len(result.items),
            "only_context": bool(raw.get("only_context", True)),
            "primary_api": str(raw.get("primary_api") or ""),
            "retriever_purity": str(result.metadata.get("retriever_purity") or ""),
            "latency_ms": result.latency_ms,
            "status": "completed",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 — queremos registrar qualquer falha técnica
        return {
            "question_id": question["question_id"],
            "hop": _hop_of(question),
            "question": question["question"],
            "gold_answer": _gold_str(question),
            "native_answer": "",
            "answer_len": 0,
            "items_count": 0,
            "only_context": True,
            "primary_api": "",
            "retriever_purity": "",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _gold_hit(row: dict[str, Any]) -> bool:
    """Sinal grosseiro (NÃO é o judge): a gold aparece literal na resposta?"""
    ans = row["native_answer"].lower()
    if not ans:
        return False
    for gold in row["gold_answer"].split(" | "):
        g = gold.strip().lower()
        if g and g in ans:
            return True
    return False


# ----------------------------- smoke -----------------------------

async def run_smoke(n: int) -> bool:
    REPORTS.mkdir(parents=True, exist_ok=True)
    questions = _load_questions()
    sample = _smoke_sample(questions)
    if n != sum(SMOKE_STRATA.values()):
        sample = sample[:n]
    print(f"[smoke] {len(sample)} perguntas (hops: "
          f"{ {h: sum(1 for q in sample if _hop_of(q) == h) for h in SMOKE_STRATA} }); "
          f"top_k={COGNEE_NATIVE_TOP_K} (default nativo do cognee)")

    nodes_before = _cognee_node_count()
    adapter = _build_adapter(COGNEE_NATIVE_TOP_K)

    rows: list[dict[str, Any]] = []
    for i, q in enumerate(sample, start=1):
        row = await _generate_one(adapter, q, COGNEE_NATIVE_TOP_K)
        rows.append(row)
        flag = "OK " if row["status"] == "completed" and row["answer_len"] > 0 else "!! "
        hit = "✓gold" if _gold_hit(row) else "     "
        print(f"[smoke] {flag}{hit} {i:2d}/{len(sample)} {row['hop']}hop {row['question_id']} "
              f"len={row['answer_len']:4d} {row['latency_ms']/1000:5.1f}s :: {row['native_answer'][:80]!r}")

    nodes_after = _cognee_node_count()

    technical_failures = sum(1 for r in rows if r["status"] != "completed")
    non_empty = sum(1 for r in rows if r["answer_len"] > 0)
    native_confirmed = sum(1 for r in rows if r["only_context"] is False)
    # Prova de que houve GERAÇÃO (não devolução do blob de contexto): a resposta é curta.
    # Contexto do cognee ~7.7k chars (deep-dive); uma resposta gerada é << 3000.
    answer_like = sum(1 for r in rows if 0 < r["answer_len"] < 3000)
    gold_hits = sum(1 for r in rows if _gold_hit(r))
    graph_unchanged = nodes_before == nodes_after

    gates = {
        "0 falhas técnicas": technical_failures == 0,
        "todas as respostas não-vazias": non_empty == len(rows),
        "geração nativa (only_context=False + resposta curta, não blob)":
            native_confirmed == len(rows) and answer_like == len(rows),
        "grafo cognee inalterado (retrieval-only)": graph_unchanged,
    }
    passed = all(gates.values())

    # artefatos
    (REPORTS / "cognee_native_smoke.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    _write_smoke_report(rows, gates, passed, nodes_before, nodes_after, gold_hits)

    print()
    for name, ok in gates.items():
        print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}")
    print(f"[smoke] sinal grosseiro gold-hit (informativo, não-gate): {gold_hits}/{len(rows)}")
    print(f"[smoke] GATE {'PASS' if passed else 'FAIL'} — relatório: "
          f"{REPORTS / 'cognee_native_smoke_report.md'}")
    return passed


def _write_smoke_report(
    rows: list[dict[str, Any]],
    gates: dict[str, bool],
    passed: bool,
    nodes_before: int,
    nodes_after: int,
    gold_hits: int,
) -> None:
    lines = [
        "# Smoke — cognee NATIVO (Braço B, P2)",
        "",
        f"Perguntas: {len(rows)} (estratificado por hop, seed {SMOKE_SEED}). "
        f"`top_k={COGNEE_NATIVE_TOP_K}` = default da API `recall()` do cognee 1.1.2 "
        f"(\"as deployed\"), NÃO o k=5 do braço controlado.",
        "",
        f"Geração nativa: `GRAPH_COMPLETION`, `only_context=False` → 1 chamada gpt-4o-mini/pergunta "
        f"dentro do cognee. Índice: Option C Neo4j (bolt 17689) + sqlite + lancedb.",
        "",
        "## Gate",
        "",
        "| critério | resultado |",
        "|---|---|",
    ]
    for name, ok in gates.items():
        lines.append(f"| {name} | {'✅ PASS' if ok else '❌ FAIL'} |")
    lines.append(f"| **GATE global** | {'✅ **PASS**' if passed else '❌ **FAIL**'} |")
    lines += [
        "",
        f"- grafo cognee (nós) antes/depois: `{nodes_before}` / `{nodes_after}`",
        f"- gold-hit grosseiro (informativo, NÃO é o judge): `{gold_hits}/{len(rows)}`",
        f"- latência média: `{sum(r['latency_ms'] for r in rows) / max(len(rows), 1) / 1000:.1f}s`",
        "",
        "## Perguntas",
        "",
        "| # | hop | question_id | status | len | only_ctx | api | lat(s) | gold-hit | resposta (prévia) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        preview = r["native_answer"][:90].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | {r['hop']} | `{r['question_id']}` | {r['status']} | {r['answer_len']} | "
            f"{r['only_context']} | {r['primary_api']} | {r['latency_ms']/1000:.1f} | "
            f"{'✓' if _gold_hit(r) else ''} | {preview} |"
        )
    lines += [
        "",
        "## Próximo passo (se PASS)",
        "",
        "Full 1000 + judge (aprovação do autor):",
        "```",
        "python scripts/run_cognee_native.py --full",
        "python scripts/run_llm_judge.py --full --experiment-id musique_eval1k_cognee_native --out-tag cognee_native",
        "```",
    ]
    (REPORTS / "cognee_native_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------- full -----------------------------

async def run_full(limit: int | None = None) -> None:
    from benchmark.cli.app import _connect_postgres
    from benchmark.core.ids import deterministic_id
    from benchmark.storage.experiment_store import ExperimentStore

    questions = _load_questions()
    conn = _connect_postgres()
    store = ExperimentStore(conn)

    store.create_experiment(
        experiment_id=TARGET_EXP,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        metadata={
            "branch": "as_deployed", "framework": "cognee", "native_qa": True,
            "retrieval_mode": "GRAPH_COMPLETION", "only_context": False,
            "top_k": COGNEE_NATIVE_TOP_K, "top_k_note": "cognee recall() API default (as deployed)",
            "note": "QA nativo do cognee (geração própria, gpt-4o-mini interno); braço B / P2",
        },
    )

    # resume: pula question_ids já gravados neste experimento
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.question_id FROM answers a JOIN runs r ON a.run_id = r.run_id "
            "WHERE r.experiment_id = %s AND r.method_id = %s",
            (TARGET_EXP, METHOD),
        )
        done = {row[0] for row in cur.fetchall()}
    todo = [q for q in questions if q["question_id"] not in done]
    # Reinício em lotes: cognee vaza memória por chamada de recall() (conexões
    # LanceDB/subprocess não liberadas), então cada invocação processa no máximo
    # `limit` perguntas e sai; o supervisor relança um processo NOVO (memória zerada)
    # até done == len(questions). Idempotente via os question_ids já em `done`.
    if limit is not None and limit > 0:
        todo = todo[:limit]
    print(f"[full] {len(questions)} perguntas · {len(done)} já feitas · "
          f"{len(todo)} nesta sessão"
          f"{f' (limite={limit})' if limit else ''} · top_k={COGNEE_NATIVE_TOP_K}", flush=True)
    if not todo:
        print("[full] nada a fazer nesta sessão (tudo já gravado)", flush=True)
        conn.close()
        return

    adapter = _build_adapter(COGNEE_NATIVE_TOP_K)
    written = 0
    empties = 0
    for i, q in enumerate(todo, start=1):
        row = await _generate_one(adapter, q, COGNEE_NATIVE_TOP_K)
        if row["status"] != "completed":
            print(f"[full] ERRO {q['question_id']}: {row['error']} — abortando para inspeção", flush=True)
            raise SystemExit(1)
        if row["answer_len"] == 0:
            empties += 1
        question_id = q["question_id"]
        run_id = deterministic_id(
            "run", [TARGET_EXP, DATASET_ID, DATASET_VERSION, METHOD, AGENT_MODE, question_id]
        )
        provenance = {
            "native_qa": True, "branch": "as_deployed", "framework": "cognee",
            "retrieval_mode": "GRAPH_COMPLETION", "only_context": False,
            "top_k": COGNEE_NATIVE_TOP_K, "retriever_purity": row["retriever_purity"],
            "primary_api": row["primary_api"], "hop": row["hop"],
        }
        store.create_run(
            experiment_id=TARGET_EXP, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
            method_id=METHOD, agent_mode=AGENT_MODE, run_id=run_id, status="materialized",
            metadata={k: provenance[k] for k in ("native_qa", "branch", "framework", "top_k")},
        )
        store.persist_answer(
            run_id=run_id, method_id=METHOD, agent_mode=AGENT_MODE,
            answer_text=row["native_answer"], question_id=question_id,
            latency_ms=row["latency_ms"], metadata=provenance,
        )
        conn.commit()
        written += 1
        if i % 25 == 0 or i == len(todo):
            print(f"[full] {i}/{len(todo)} gravadas ({written} nesta sessão, {empties} vazias) "
                  f"últ.: {row['latency_ms']/1000:.1f}s", flush=True)
    print(f"[full] concluído: {written} gravadas em {TARGET_EXP} ({empties} respostas vazias)", flush=True)
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", nargs="?", type=int, const=sum(SMOKE_STRATA.values()),
                       help="roda o smoke (default 15 perguntas estratificadas)")
    group.add_argument("--full", action="store_true", help="roda as 1000 e persiste")
    ap.add_argument("--limit", type=int, default=None,
                    help="máx. de perguntas nesta invocação; p/ reinício em lotes (só com --full)")
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    args = ap.parse_args()
    _bind_dataset(args.dataset_id, args.dataset_version)
    if args.full:
        asyncio.run(run_full(limit=args.limit))
    else:
        passed = asyncio.run(run_smoke(args.smoke))
        raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
