"""answer_correctness_judge — automated LLM-as-judge. NO human validation.

Reference-based pointwise judge: given (question, gold + aliases, prediction) decide if the
prediction ASSERTS the gold answer (or a clear equivalent), ignoring verbosity/markdown.
Two passes with controlled variation (reversed few-shot order + emphasis line), temperature=0,
structured JSON output. Persists ADDITIVELY to metric `answer_correctness_judge` (NEVER touches
baseline metrics; gold_suspect excluded because metric_value is NOT NULL). Saves raw incrementally
and resumes; captures token usage for real cost.

Modes:
  --dry-run         : load items, print go/no-go report (source, composition, already-done,
                      remaining, EXACT input tokens via tiktoken, cost estimate). ZERO API calls.
  (pilot, default)  : stratified ~N from answer_quality_sample_500.jsonl.
  --full            : judge ALL answers (1000x2) loaded from Postgres -> population number.

Usage:
  python scripts/run_llm_judge.py --full --dry-run
  python scripts/run_llm_judge.py --full
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

from benchmark.cli.app import _connect_postgres
from benchmark.core.ids import deterministic_id

EXP = "musique_eval1k_first_results"
SAMPLE = "artifacts/musique/reports/answer_quality_sample_500.jsonl"
METRIC = "answer_correctness_judge"
BASELINE = ["exact_match", "answer_f1"]
PROMPT_VERSION = "judge_v1"
PASSES = 2  # two controlled passes per item (inter-pass agreement)
LABELS = ["correct", "partial", "incorrect", "refusal", "gold_suspect"]
SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "refusal": 0.0, "gold_suspect": None}

# gpt-4o reference pricing (USD per token) — VERIFY before trusting the $ figure.
PRICE_IN = 2.50 / 1_000_000
PRICE_OUT = 10.0 / 1_000_000

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "matched_gold": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
    "required": ["label", "matched_gold", "rationale"],
    "additionalProperties": False,
}

DEFINITIONS = """You grade SHORT-ANSWER multi-hop QA (MuSiQue). You are given a question, the gold
answer plus aliases, and a model prediction. Decide whether the prediction ASSERTS the gold answer
(or a clear equivalent: synonym, abbreviation, alternate name, different date/number format, same
entity), IGNORING verbosity, markdown, restating the question, or extra explanation.

Labels:
- correct: the prediction asserts the gold answer (or a valid equivalent) as THE answer.
- partial: a multi-hop chain partly resolved — right intermediate but wrong/incomplete final; or it
  hedges between candidates including the right one; or a correct-but-less-specific form.
- incorrect: it commits to a WRONG final answer (wrong entity/value, or answered the wrong sub-question).
- refusal: it declines / says not enough information / denies the premise / is empty.
- gold_suspect: the gold/aliases look wrong, ambiguous or broken, so the item cannot be judged fairly.

Decision rules:
- Verbosity/markdown/preambles NEVER penalize.
- Require ASSERTION, not mere mention: if the gold appears only as context or as one option among
  several, it is NOT correct.
- When unsure between partial and correct, choose partial.
- Output ONLY the structured JSON (label, matched_gold, rationale<=1 sentence)."""

FEWSHOTS = [
    {"q": "Who sings Home Alone Tonight with the singer of Light It Up?",
     "gold": "Karen Fairchild", "aliases": [],
     "pred": '**Luke Bryan sings "Home Alone Tonight" with Karen Fairchild.**',
     "label": "correct", "why": "asserts the gold; only wrapped in markdown/sentence"},
    {"q": "When was the painter of Femme nue couchée born?",
     "gold": "10 June 1819", "aliases": [],
     "pred": "Gustave Courbet, the painter, was born on June 10, 1819.",
     "label": "correct", "why": "same date, different format"},
    {"q": "What team is the highest goal scorer in the EPL a member of?",
     "gold": "Egypt national football team", "aliases": [],
     "pred": "The highest scorer is Mohamed Salah, who is a member of Liverpool FC.",
     "label": "partial", "why": "right player, gives the club not the requested national team"},
    {"q": "Who was honored with the award Dhondo Keshav Karve received prior to becoming president of India?",
     "gold": "A.P.J. Abdul Kalam", "aliases": ["Abdul Kalam", "Kalam"],
     "pred": "The award is the **Bharat Ratna**.",
     "label": "incorrect", "why": "answered the award, not the person honored with it"},
    {"q": "When did the town WIZE is licensed in become capitol of the state where Ward Township is located?",
     "gold": "1839", "aliases": [],
     "pred": "Springfield has never been the capital of Ohio; the capital is Columbus. There is no relevant answer.",
     "label": "refusal", "why": "denies the premise, gives no answer"},
    {"q": "What is the meaning of the majority religion's word in the Arabic dictionary, in the area that became India?",
     "gold": "the country of India", "aliases": ["India", "Hindustan"],
     "pred": 'The word is "Islam", which in Arabic means "submission" or "surrender".',
     "label": "gold_suspect", "why": "gold 'the country of India' does not answer a 'meaning of the word' question"},
]


def _fewshot_text(shots) -> str:
    out = []
    for s in shots:
        out.append(
            f"Q: {s['q']}\nGOLD: {s['gold']}\nALIASES: {s['aliases']}\nPREDICTION: {s['pred']}\n"
            f'JSON: {{"label": "{s["label"]}", "matched_gold": '
            f'{json.dumps(s["gold"]) if s["label"] in ("correct", "partial") else "null"}, '
            f'"rationale": "{s["why"]}"}}'
        )
    return "\n\n".join(out)


def build_messages(item: dict, pass_id: str) -> list[dict]:
    shots = FEWSHOTS if pass_id == "A" else list(reversed(FEWSHOTS))
    emphasis = "" if pass_id == "A" else (
        "\nReminder: a verbose answer whose ASSERTED answer is the gold is `correct`; an answer that "
        "only mentions the gold in passing or as one option is NOT `correct`."
    )
    system = f"{DEFINITIONS}{emphasis}\n\nExamples:\n{_fewshot_text(shots)}"
    user = (
        f"Q: {item['question']}\nGOLD: {item['gold']}\nALIASES: {item.get('aliases', [])}\n"
        f"PREDICTION: {item['prediction']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------- item loading ----------
def _gold_value(raw):
    if isinstance(raw, str) and raw[:1] in '"[{':
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _aliases(metadata) -> list[str]:
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return []
    if isinstance(metadata, dict) and isinstance(metadata.get("answer_aliases"), list):
        return [str(a) for a in metadata["answer_aliases"]]
    return []


def load_items_full(conn, exp: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.run_id, r.method_id, a.question_id, q.question, q.gold_answer, q.metadata, a.answer_text
            FROM answers a
            JOIN runs r ON r.run_id = a.run_id
            JOIN questions q ON q.question_id = a.question_id
            WHERE r.experiment_id = %s
            ORDER BY r.method_id, a.question_id
            """,
            (exp,),
        )
        rows = cur.fetchall()
    items = []
    for run_id, method, qid, question, gold_raw, metadata, pred in rows:
        items.append({
            "run_id": run_id, "method": method, "question_id": qid, "question": question,
            "gold": _as_text(_gold_value(gold_raw)), "aliases": _aliases(metadata),
            "prediction": pred or "", "bucket": "full",
        })
    return items


def stratified_pilot(rows: list[dict], size: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["bucket"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["question_id"])
    quota = max(1, -(-size // len(groups)))
    taken: dict[tuple, int] = {}
    picked: list[dict] = []
    for key in sorted(groups):
        k = min(quota, len(groups[key]))
        picked.extend(groups[key][:k])
        taken[key] = k
    while len(picked) < size:
        progressed = False
        for key in sorted(groups):
            if len(picked) >= size:
                break
            if taken[key] < len(groups[key]):
                picked.append(groups[key][taken[key]])
                taken[key] += 1
                progressed = True
        if not progressed:
            break
    return picked[:size]


# ---------- judging ----------
def judge_once(client, model: str, item: dict, pass_id: str) -> dict:
    resp = client.chat.completions.create(
        model=model, temperature=0, messages=build_messages(item, pass_id),
        response_format={"type": "json_schema",
                         "json_schema": {"name": "answer_correctness_judge", "strict": True, "schema": SCHEMA}},
    )
    parsed = json.loads(resp.choices[0].message.content)
    u = resp.usage
    return {"pass_id": pass_id, "model": resp.model, "raw": resp.choices[0].message.content,
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, **parsed}


def persist_from_records(conn, records: list[dict]) -> tuple:
    scored = [r for r in records if r["label"] != "error"]
    run_ids = sorted({r["run_id"] for r in scored if r["run_id"]})
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), COALESCE(sum(metric_value),0) FROM evaluation_results "
                    "WHERE run_id = ANY(%s) AND metric_name = ANY(%s)", (run_ids, BASELINE))
        base_before = tuple(cur.fetchone())
        cur.execute("DELETE FROM evaluation_results WHERE run_id = ANY(%s) AND metric_name = %s",
                    (run_ids, METRIC))
        inserted = 0
        for r in scored:
            if not r["run_id"] or SCORE[r["label"]] is None:  # gold_suspect excluded (NOT NULL col)
                continue
            meta = {"label": r["label"], "matched_gold": r.get("matched_gold"),
                    "rationale": r.get("rationale"), "judge_model": r["judge_model"],
                    "prompt_version": PROMPT_VERSION, "temperature": 0,
                    "pass_a": r["label_a"], "pass_b": r["label_b"],
                    "pass_agreement": r["pass_agreement"], "low_confidence": r["low_confidence"]}
            score = SCORE[r["label"]]
            rid = deterministic_id("eval", [r["run_id"], METRIC, score, meta])
            cur.execute("INSERT INTO evaluation_results (evaluation_result_id, run_id, metric_name, metric_value, metadata) "
                        "VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT (evaluation_result_id) "
                        "DO UPDATE SET metric_value = EXCLUDED.metric_value, metadata = EXCLUDED.metadata",
                        (rid, r["run_id"], METRIC, score, json.dumps(meta)))
            inserted += 1
        cur.execute("SELECT count(*), COALESCE(sum(metric_value),0) FROM evaluation_results "
                    "WHERE run_id = ANY(%s) AND metric_name = ANY(%s)", (run_ids, BASELINE))
        base_after = tuple(cur.fetchone())
    # Count deve bater EXATO (linhas baseline não podem sumir/surgir); a soma tolera
    # ruído de float — o SUM do Postgres sobre floats não é bit-estável entre chamadas
    # (agregação paralela), então uma oscilação ~1e-13 é ruído, não corrupção.
    if base_after[0] != base_before[0] or abs(float(base_after[1]) - float(base_before[1])) > 1e-6:
        conn.rollback()
        raise SystemExit(f"ABORT: baseline changed {base_before} -> {base_after}; rolled back.")
    conn.commit()
    return inserted, base_before, base_after


def summarize(records: list[dict], extra: dict) -> dict:
    scored = [r for r in records if r["label"] != "error"]
    summary = {"judged": len(scored), "errors": len(records) - len(scored),
               "prompt_version": PROMPT_VERSION, **extra, "by_method": {}}
    for method in sorted({r["method"] for r in scored}):
        mr = [r for r in scored if r["method"] == method]
        n = len(mr)
        dist = Counter(r["label"] for r in mr)
        sc = [r for r in mr if SCORE[r["label"]] is not None]
        summary["by_method"][method] = {
            "n": n, "label_dist": dict(dist),
            "strict_accuracy": round(sum(1 for r in sc if r["label"] == "correct") / max(len(sc), 1), 4),
            "lenient_accuracy": round(sum(SCORE[r["label"]] for r in sc) / max(len(sc), 1), 4),
            "refusal_rate": round(dist.get("refusal", 0) / n, 4),
            "gold_suspect_rate": round(dist.get("gold_suspect", 0) / n, 4),
            "inter_pass_agreement": round(sum(1 for r in mr if r["pass_agreement"]) / n, 4),
            "low_confidence_rate": round(sum(1 for r in mr if r["low_confidence"]) / n, 4),
            "n_scoreable": len(sc),
        }
    return summary


# ---------- resume + offline token estimation (no API calls) ----------
def load_done(raw_path: str) -> set:
    """Items already judged (non-error) in the raw JSONL, keyed by (method, question_id)."""
    done: set = set()
    if os.path.exists(raw_path):
        for line in open(raw_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("label") != "error":
                done.add((r.get("method"), r.get("question_id")))
    return done


def raw_token_totals(raw_path: str) -> tuple[int, int]:
    """Sum prompt/completion tokens recorded in the raw file -> resume-safe real cost."""
    tin = tout = 0
    if os.path.exists(raw_path):
        for line in open(raw_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            tin += int(r.get("prompt_tokens") or 0)
            tout += int(r.get("completion_tokens") or 0)
    return tin, tout


def estimate_input_tokens(items: list[dict]) -> int:
    """Exact prompt-token count for BOTH passes of each item via tiktoken. ZERO API calls.
    Returns -1 if tiktoken/model encoding is unavailable."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:  # noqa: BLE001
            enc = tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001
        return -1
    total = 0
    for it in items:
        for pid in ("A", "B"):
            for m in build_messages(it, pid):
                total += len(enc.encode(m["content"])) + 4  # ~per-message chat overhead
            total += 3  # ~per-call priming
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", default=EXP)
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--full", action="store_true", help="judge ALL answers from DB (population)")
    ap.add_argument("--pilot-size", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out-tag",
        default=None,
        help="suffix for output artifacts (e.g. v2_grounded) so re-runs of other "
        "experiments never overwrite the canonical llm_judge_full_* files",
    )
    ap.add_argument(
        "--out-dir",
        default="artifacts/musique/reports",
        help="directory for raw/summary artifacts (use artifacts/<dataset>/reports "
        "for non-musique experiments)",
    )
    args = ap.parse_args()

    mode = "full" if args.full else "pilot"
    tag = f"_{args.out_tag}" if args.out_tag else ""
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = f"{args.out_dir}/llm_judge_{mode}{tag}_raw.jsonl"
    summary_path = f"{args.out_dir}/llm_judge_{mode}{tag}_summary.json"

    conn = _connect_postgres()
    if args.full:
        items = load_items_full(conn, args.experiment_id)
    else:
        rows = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]
        items = stratified_pilot(rows, args.pilot_size)
    if args.limit:
        items = items[: args.limit]

    comp = Counter(i["method"] for i in items)
    n_items = len(items)
    src = (f"Postgres experiment_id={args.experiment_id!r}  (NOT the {SAMPLE} 500-sample)"
           if args.full else f"{SAMPLE} stratified pilot")
    print(f"mode={mode} judge={args.judge_model} prompt={PROMPT_VERSION} items={n_items} by_method={dict(comp)}")

    # resume bookkeeping (used by BOTH dry-run and the real run)
    done = load_done(raw_path)
    todo = [i for i in items if (i["method"], i["question_id"]) not in done]

    if args.dry_run:
        calls_total = n_items * PASSES
        calls_remaining = len(todo) * PASSES
        out_real, out_ceil = 50, 120  # per-call output tokens: realistic / ceiling
        in_exact = estimate_input_tokens(todo)  # exact prompt tokens, 0 API calls
        if in_exact >= 0:
            in_tok, in_str = in_exact, f"{in_exact/1e6:.3f}M (exact, tiktoken)"
            cost_real = in_tok * PRICE_IN + calls_remaining * out_real * PRICE_OUT
            cost_ceil = in_tok * PRICE_IN + calls_remaining * out_ceil * PRICE_OUT
        else:  # tiktoken unavailable -> coarse estimate/ceiling
            in_tok, in_str = calls_remaining * 450, f"~{calls_remaining*450/1e6:.3f}M (estimate, no tiktoken)"
            cost_real = in_tok * PRICE_IN + calls_remaining * out_real * PRICE_OUT
            cost_ceil = calls_remaining * 500 * PRICE_IN + calls_remaining * out_ceil * PRICE_OUT
        exp_ok = (args.full and comp.get("vector_rag") == 1000
                  and comp.get("lightrag_neo4j") == 1000 and n_items == 2000)
        print("\n========== DRY-RUN (0 API calls) ==========")
        print(f"INPUT SOURCE      : {src}")
        print(f"composition       : " + ", ".join(f"{m}={c}" for m, c in sorted(comp.items())))
        print(f"scope check       : {'OK  vector_rag=1000 + lightrag_neo4j=1000 = 2000' if exp_ok else 'WARNING  expected vector_rag=1000, lightrag_neo4j=1000, total=2000'}")
        print(f"items total       : {n_items}")
        print(f"passes per item   : {PASSES}")
        print(f"expected calls    : {calls_total}")
        print(f"already in raw    : {len(done)} items   ({raw_path})")
        print(f"remaining items   : {len(todo)}")
        print(f"remaining calls   : {calls_remaining}")
        print(f"est. input tokens : {in_str}")
        print(f"est. output tokens: {calls_remaining*out_real/1e6:.3f}M (realistic ~{out_real}/call) .. {calls_remaining*out_ceil/1e6:.3f}M (ceiling ~{out_ceil}/call)")
        print(f"price (gpt-4o)    : ${PRICE_IN*1e6:.2f}/1M in, ${PRICE_OUT*1e6:.2f}/1M out   [VERIFY current rates]")
        print(f"COST ESTIMATE     : ~${cost_real:.2f} (realistic)  ..  ~${cost_ceil:.2f} (ceiling)")
        print("NOTE              : prompt caching on the static system block lowers input cost further.")
        print("===========================================")
        return 0

    print(f"resume: {len(done)} already judged, {len(todo)} to judge")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    tok_in = tok_out = 0
    t0 = time.monotonic()
    raw_f = open(raw_path, "a", encoding="utf-8")
    for idx, item in enumerate(todo, 1):
        passes = {}
        err = None
        for pid in ("A", "B"):
            for attempt in range(1, 4):
                try:
                    passes[pid] = judge_once(client, args.judge_model, item, pid)
                    break
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {str(exc)[:160]}"
                    time.sleep(2 * attempt)
        if "A" not in passes or "B" not in passes:
            print(f"  ERROR {item['question_id']}: {err}", flush=True)
            rec = {"question_id": item["question_id"], "method": item["method"],
                   "run_id": item["run_id"], "label": "error", "error": err}
            raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); raw_f.flush()
            continue
        item_in = passes["A"]["prompt_tokens"] + passes["B"]["prompt_tokens"]
        item_out = passes["A"]["completion_tokens"] + passes["B"]["completion_tokens"]
        tok_in += item_in; tok_out += item_out
        la, lb = passes["A"]["label"], passes["B"]["label"]
        rec = {
            "question_id": item["question_id"], "method": item["method"], "run_id": item["run_id"],
            "bucket": item.get("bucket"), "gold": item["gold"], "prediction": item["prediction"],
            "label": la, "score": SCORE[la], "pass_agreement": la == lb, "low_confidence": la != lb,
            "label_a": la, "label_b": lb, "matched_gold": passes["A"].get("matched_gold"),
            "rationale": passes["A"].get("rationale"), "judge_model": passes["A"]["model"],
            "prompt_version": PROMPT_VERSION, "temperature": 0,
            "prompt_tokens": item_in, "completion_tokens": item_out,
            "raw_a": passes["A"]["raw"], "raw_b": passes["B"]["raw"],
        }
        raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); raw_f.flush()
        if idx % 50 == 0:
            print(f"  [{idx}/{len(todo)}] {time.monotonic()-t0:.0f}s tok_in={tok_in} tok_out={tok_out}", flush=True)
    raw_f.close()

    # persist EVERYTHING in the raw file (resumed + new)
    records = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
    inserted, base_before, base_after = persist_from_records(conn, records)
    total_in, total_out = raw_token_totals(raw_path)         # resume-safe: ALL calls in raw
    cost_new = tok_in * PRICE_IN + tok_out * PRICE_OUT         # this session only
    cost_total = total_in * PRICE_IN + total_out * PRICE_OUT   # full run incl. resumed items
    summary = summarize(records, {
        "mode": mode, "judge_model": args.judge_model, "items": n_items,
        "tokens_in_new_calls": tok_in, "tokens_out_new_calls": tok_out,
        "estimated_cost_usd_new_calls": round(cost_new, 4),
        "prompt_tokens_total": total_in, "completion_tokens_total": total_out,
        "total_tokens": total_in + total_out,
        "estimated_cost_usd_total": round(cost_total, 4),
        "price_per_1m_in": PRICE_IN * 1e6, "price_per_1m_out": PRICE_OUT * 1e6,
        "inserted_rows": inserted,
        "baseline_fingerprint_before": list(base_before),
        "baseline_fingerprint_after": list(base_after),
        "baseline_unchanged": base_after[0] == base_before[0]
        and abs(float(base_after[1]) - float(base_before[1])) <= 1e-6,
    })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nraw -> {raw_path}\nsummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
