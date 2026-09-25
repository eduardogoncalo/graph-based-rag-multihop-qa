"""HippoRAG 2 NATIVO (Braço B, "as deployed") — o pacote responde por si.

Roda a geração PRÓPRIA do HippoRAG 2 (`rag.rag_qa()` via
`hipporag2_query_runner.py` em mode="qa") sobre as 1000 MuSiQue eval1k e
materializa como `musique_eval1k_hipporag2_native`. Análogo ao
run_ms_graphrag_native.py.

Diferença vs o Braço A (controlado): lá o HippoRAG 2 só entrega CONTEXTO e o
nosso reader gera; aqui é a geração NATIVA do pacote (prompt CoT do próprio
framework, short answer parseada de "Answer:"). As-shipped: retrieval interno
retrieval_top_k=200 (PPR), reader lê qa_top_k=5 docs — os mesmos 5 do braço
controlado, comparação nativo×controlado mais limpa da matriz.

Modos:
  --smoke [N]  amostra estratificada por hop (8/5/2 = 15); gera as nativas,
               escreve JSONL + gate; NÃO escreve nas tabelas do benchmark.
  --full       todas as 1000; persiste em musique_eval1k_hipporag2_native
               (idempotente, resumível por question_id). Judge é passo
               separado (run_llm_judge.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmark.core.settings import load_settings
from benchmark.methods.hipporag2 import Hipporag2Adapter, build_workspace

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "musique"
DATASET_VERSION = "ans_v1.0_eval1k"
QUESTIONS = ROOT / "data/canonical/musique_ans_v1.0_eval1k/questions.jsonl"
REPORTS = ROOT / "artifacts/musique/reports"

TARGET_EXP = "musique_eval1k_hipporag2_native"


def _bind_dataset(dataset_id: str, dataset_version: str, exp_suffix: str = "") -> None:
    """Rebinda os globals p/ outro dataset (default musique = retrocompatível).

    `exp_suffix` desvia a escrita para um experimento novo. Existe para as
    variantes de grafo deduplicado (exploratórias, fora da tese), que não podem
    sobrescrever a execução original: com sufixo "first" o destino passa a
    `<ds>_eval1k_hipporag2_native_first`. Sem sufixo o comportamento é o antigo.
    """
    global DATASET_ID, DATASET_VERSION, QUESTIONS, REPORTS, TARGET_EXP
    DATASET_ID, DATASET_VERSION = dataset_id, dataset_version
    QUESTIONS = ROOT / f"data/canonical/{dataset_id}_{dataset_version}/questions.jsonl"
    REPORTS = ROOT / f"artifacts/{dataset_id}/reports"
    TARGET_EXP = f"{dataset_id}_eval1k_hipporag2_native"
    if exp_suffix:
        TARGET_EXP = f"{TARGET_EXP}_{exp_suffix}"
METHOD = "hipporag2"
AGENT_MODE = "single_agent"

SMOKE_STRATA = {"2": 8, "3": 5, "4": 2}


def _hop_of(q: dict[str, Any]) -> str:
    dec = (q.get("metadata") or {}).get("decomposition")
    return str(len(dec)) if isinstance(dec, list) else "?"


def _gold_str(q: dict[str, Any]) -> str:
    g = q.get("gold_answer")
    return " | ".join(map(str, g)) if isinstance(g, list) else ("" if g is None else str(g))


def _load_questions() -> list[dict[str, Any]]:
    rows = []
    with QUESTIONS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _smoke_sample(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hop: dict[str, list] = {}
    for q in sorted(questions, key=lambda r: r["question_id"]):
        by_hop.setdefault(_hop_of(q), []).append(q)
    picked = []
    for hop, k in SMOKE_STRATA.items():
        picked.extend(by_hop.get(hop, [])[:k])
    if not picked:
        # dataset sem metadata.decomposition (ex.: twowiki): primeiras N por
        # question_id, mesmo critério do smoke do braço A
        picked = sorted(questions, key=lambda r: r["question_id"])[: sum(SMOKE_STRATA.values())]
    return picked


def _build_adapter() -> Hipporag2Adapter:
    settings = load_settings()
    return Hipporag2Adapter(
        workspace=build_workspace(
            artifacts_dir=settings.artifacts_dir,
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
        )
    )


def _generate_one(adapter: Hipporag2Adapter, question: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = adapter.query_native(query=question["question"])
        answer = str(payload.get("answer") or "").strip()
        doc_ids = [it.get("document_id") for it in payload.get("items") or []]
        stats = payload.get("stats") or {}
        return {
            "question_id": question["question_id"], "hop": _hop_of(question),
            "question": question["question"], "gold_answer": _gold_str(question),
            "native_answer": answer, "answer_len": len(answer),
            "raw_response": str(payload.get("raw_response") or ""),
            "doc_ids": doc_ids, "latency_ms": (time.perf_counter() - started) * 1000,
            "prompt_tokens": stats.get("prompt_tokens"),
            "completion_tokens": stats.get("completion_tokens"),
            "status": "completed" if answer else "empty", "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "question_id": question["question_id"], "hop": _hop_of(question),
            "question": question["question"], "gold_answer": _gold_str(question),
            "native_answer": "", "answer_len": 0, "raw_response": "", "doc_ids": [],
            "latency_ms": (time.perf_counter() - started) * 1000,
            "prompt_tokens": None, "completion_tokens": None,
            "status": "error", "error": f"{type(exc).__name__}: {exc}",
        }


def _gold_hit(row: dict[str, Any]) -> bool:
    ans = row["native_answer"].lower()
    return bool(ans) and any(g.strip().lower() in ans for g in row["gold_answer"].split(" | ") if g.strip())


# ----------------------------- smoke -----------------------------

def run_smoke(n: int) -> bool:
    REPORTS.mkdir(parents=True, exist_ok=True)
    sample = _smoke_sample(_load_questions())
    if n != sum(SMOKE_STRATA.values()):
        sample = sample[:n]
    print(f"[smoke] {len(sample)} perguntas; "
          f"hops={ {h: sum(1 for q in sample if _hop_of(q)==h) for h in SMOKE_STRATA} }", flush=True)
    adapter = _build_adapter()
    rows = []
    for i, q in enumerate(sample, 1):
        row = _generate_one(adapter, q)
        rows.append(row)
        flag = "OK " if row["status"] == "completed" else "!! "
        hit = "✓gold" if _gold_hit(row) else "     "
        print(f"[smoke] {flag}{hit} {i:2d}/{len(sample)} {row['hop']}hop len={row['answer_len']:4d} "
              f"{row['latency_ms']/1000:5.1f}s :: {row['native_answer'][:80]!r}", flush=True)

    tech_fail = sum(1 for r in rows if r["status"] == "error")
    non_empty = sum(1 for r in rows if r["answer_len"] > 0)
    gold_hits = sum(1 for r in rows if _gold_hit(r))
    gates = {
        "0 falhas técnicas": tech_fail == 0,
        "todas as respostas não-vazias": non_empty == len(rows),
    }
    passed = all(gates.values())
    (REPORTS / "hipporag2_native_smoke.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print()
    for name, ok in gates.items():
        print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}")
    print(f"[smoke] gold-hit grosseiro (informativo): {gold_hits}/{len(rows)}")
    print(f"[smoke] latência média: {sum(r['latency_ms'] for r in rows)/max(len(rows),1)/1000:.1f}s")
    print(f"[smoke] GATE {'PASS' if passed else 'FAIL'}")
    return passed


# ----------------------------- full -----------------------------

def run_full(limit: int | None = None) -> None:
    from benchmark.cli.app import _connect_postgres
    from benchmark.core.ids import deterministic_id
    from benchmark.storage.experiment_store import ExperimentStore

    questions = _load_questions()
    conn = _connect_postgres()
    store = ExperimentStore(conn)
    store.create_experiment(
        experiment_id=TARGET_EXP, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
        metadata={"branch": "as_deployed", "framework": "hipporag2", "native_qa": True,
                  "retrieval_top_k": 200, "qa_top_k": 5,
                  "note": "QA nativo do HippoRAG 2 (rag_qa, prompt CoT do pacote); braço B"},
    )
    with conn.cursor() as cur:
        cur.execute("SELECT a.question_id FROM answers a JOIN runs r ON a.run_id=r.run_id "
                    "WHERE r.experiment_id=%s AND r.method_id=%s", (TARGET_EXP, METHOD))
        done = {row[0] for row in cur.fetchall()}
    todo = sorted(
        (q for q in questions if q["question_id"] not in done),
        key=lambda q: q["question_id"],
    )
    if limit:
        todo = todo[:limit]
    print(f"[full] {len(questions)} perguntas · {len(done)} já feitas · {len(todo)} nesta sessão"
          f"{f' (limite={limit})' if limit else ''}", flush=True)
    if not todo:
        print("[full] nada a fazer", flush=True); conn.close(); return

    adapter = _build_adapter()
    written = empties = 0
    for i, q in enumerate(todo, 1):
        row = _generate_one(adapter, q)
        if row["status"] == "error":
            print(f"[full] ERRO {q['question_id']}: {row['error']} — abortando", flush=True)
            raise SystemExit(1)
        if row["answer_len"] == 0:
            empties += 1
        qid = q["question_id"]
        run_id = deterministic_id("run", [TARGET_EXP, DATASET_ID, DATASET_VERSION, METHOD, AGENT_MODE, qid])
        prov = {"native_qa": True, "branch": "as_deployed", "framework": "hipporag2",
                "hop": row["hop"], "doc_ids": row["doc_ids"],
                "raw_response": row["raw_response"],
                "prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"]}
        store.create_run(experiment_id=TARGET_EXP, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
                         method_id=METHOD, agent_mode=AGENT_MODE, run_id=run_id, status="materialized",
                         metadata={"native_qa": True, "branch": "as_deployed", "framework": "hipporag2"})
        store.persist_answer(run_id=run_id, method_id=METHOD, agent_mode=AGENT_MODE,
                             answer_text=row["native_answer"], question_id=qid,
                             latency_ms=row["latency_ms"], metadata=prov)
        conn.commit()
        written += 1
        if i % 25 == 0 or i == len(todo):
            print(f"[full] {i}/{len(todo)} gravadas ({empties} vazias) últ.: {row['latency_ms']/1000:.1f}s", flush=True)
    print(f"[full] concluído: {written} gravadas em {TARGET_EXP} ({empties} vazias)", flush=True)
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", nargs="?", type=int, const=sum(SMOKE_STRATA.values()))
    g.add_argument("--full", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    ap.add_argument(
        "--exp-suffix",
        default="",
        help="sufixo do experiment_id de destino. Obrigatório ao correr sobre "
        "um grafo variante, para não sobrescrever a execução original.",
    )
    args = ap.parse_args()
    _bind_dataset(args.dataset_id, args.dataset_version, args.exp_suffix)
    if args.full:
        run_full(limit=args.limit)
    else:
        raise SystemExit(0 if run_smoke(args.smoke) else 1)


if __name__ == "__main__":
    main()
