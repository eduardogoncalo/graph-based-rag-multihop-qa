# Graph-Based RAG for Multi-hop Question Answering: A Two-Arm Evaluation of Four Frameworks

Code and data for my MSc dissertation in Artificial Intelligence (National
College of Ireland, 2026). **The dissertation is [`thesis.pdf`](thesis.pdf).**

The study compares four graph-based RAG frameworks — **LightRAG**, **Microsoft
GraphRAG**, **HippoRAG 2** and **Cognee** — on multi-hop question answering, over
1,000 questions of **MuSiQue** and 1,000 of **2WikiMultiHopQA**.

Published evaluations of these frameworks are end-to-end: the same system
retrieves and answers, so the result does not separate what the index found
from what the pipeline did with it. This study runs every framework in **two
arms over the same index**:

- **controlled arm** — the framework only retrieves, and a fixed reader
  (`gpt-4o-mini`) answers from what it returned;
- **native arm** — the framework answers with its own pipeline, the way someone
  who installed it would use it.

Answers are graded by an LLM judge (`gpt-4o`) validated against human labels,
and compared with exact McNemar tests under Holm correction.

## Main results

**Controlled arm.** Strict accuracy with the fixed reader, and the paired
difference against the dense baseline. `docs/q` is how many documents each
framework actually hands to the reader.

| | docs/q | recall@5 | strict accuracy | vs. dense (pp) |
|---|---:|---:|---:|---:|
| **MuSiQue** | | | | |
| Closed-book floor | — | — | 0.209 | |
| Dense baseline | 4.92 | 0.553 | 0.463 | |
| HippoRAG 2 | 5.00 | 0.603 | 0.502 | +3.8 \* |
| LightRAG | 20.00 | 0.506 | **0.519** | +5.7 \*\*\* |
| Microsoft GraphRAG | 16.57 | 0.447 | 0.495 | +3.2 \* |
| Cognee | 2.76 | 0.434 | 0.401 | −6.2 \*\*\* |
| Oracle ceiling | — | — | 0.770 | |
| **2WikiMultiHopQA** | | | | |
| Closed-book floor | — | — | 0.335 | |
| Dense baseline | 4.82 | 0.704 | 0.678 | |
| HippoRAG 2 | 5.00 | 0.864 | **0.801** | +12.2 \*\*\* |
| LightRAG | 20.00 | 0.693 | 0.731 | +5.3 \*\*\* |
| Microsoft GraphRAG | 12.95 | 0.608 | 0.694 | +1.6 |
| Cognee | 2.46 | 0.625 | 0.615 | −6.3 \*\*\* |
| Oracle ceiling | — | — | 0.890 | |

Exact McNemar, Holm-corrected: \*\*\* p < 0.001, \* p < 0.05, no mark = not
significant.

**Native arm against its own controlled arm**, over the same index. A negative
delta means the framework's own pipeline answers worse than the fixed reader.

| | controlled | native | Δ (pp) | Holm p | native refusal |
|---|---:|---:|---:|---:|---:|
| **MuSiQue** | | | | | |
| LightRAG | 0.519 | 0.548 | +2.8 | 0.142 | 11.9% |
| Microsoft GraphRAG | 0.495 | 0.474 | −2.1 | 0.168 | 9.2% |
| HippoRAG 2 | 0.502 | 0.466 | −3.5 | 0.024 | 15.1% |
| Cognee | 0.401 | 0.349 | −5.2 | < 0.001 | 14.8% |
| **2WikiMultiHopQA** | | | | | |
| LightRAG | 0.731 | 0.615 | −11.6 | < 0.001 | 21.8% |
| Microsoft GraphRAG | 0.694 | 0.518 | −17.6 | < 0.001 | 26.0% |
| HippoRAG 2 | 0.801 | 0.723 | −7.7 | < 0.001 | 16.0% |
| Cognee | 0.615 | 0.534 | −8.0 | < 0.001 | 20.9% |

What the tables support:

- With the fixed reader, LightRAG and HippoRAG 2 beat the dense baseline on
  both datasets, and Cognee falls below it on both.
- **These gaps do not isolate the effect of the graph.** The frameworks deliver
  very different amounts of context — from 2.5 to 20 documents per question —
  and the design does not control for it.
- Retrieving better is not the same as answering better: on MuSiQue, HippoRAG 2
  has the best recall@5 and LightRAG the best accuracy.
- On 2WikiMultiHopQA, all four native pipelines score below their controlled
  counterparts, by 7.7 to 17.6 points. Part of the error turns into refusals:
  native refusal is 16.0% to 26.0%, against 1.6% to 4.1% in the controlled arm.

These are Tables 3 and 4 of the dissertation. They are rebuilt from
[`results/aggregate/`](results/aggregate/), and
`tests/test_results_match_thesis.py` checks the two against each other.

---

## What is in this repository

| | |
|---|---|
| `src/benchmark/` | the harness: ingestion, method adapters, reader, evaluation, CLI |
| `scripts/` | indexers, native runners, judge, retrieval audit, statistics, and `reproduce.sh` |
| `configs/` | datasets, methods, experiments, the Neo4j port registry |
| `requirements/` | manifests for the three isolated framework environments |
| `tests/` | the test suite |
| `results/` | **the data behind the dissertation**: 22 cells, per question and per retrieved document |
| `validation/s10_judge/` | the human validation of the judge, annotated by hand |
| `docs/` | design notes and one decision record |

`src/`, `scripts/` and `configs/` are code you run. `results/` and
`validation/` are **evidence from the original run**: the pipeline neither
reads nor regenerates them, and only the checks in
[Checking the dissertation](#checking-the-dissertation-against-its-data) use
them. Your own run produces your own numbers, which will differ, because
indexing, reading and judging all use language models.

Most of what a user reads (output, `--help`, reports) is in English. Some code
comments, internal identifiers, a few messages and the decision record in
`docs/decisions/` are still in Portuguese.

---

## Getting started

Requirements: [`uv`](https://docs.astral.sh/uv/) (it installs its own Python),
`podman` or `docker` with compose, and an OpenAI API key. No GPU.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # 1. uv

./scripts/bootstrap_envs.sh                         # 2. the main environment
./scripts/bootstrap_envs.sh --full                  #    plus the three isolated ones

cp .env.example .env && $EDITOR .env                # 3. fill in OPENAI_API_KEY

./scripts/reproduce.sh --dry-run                    # 4. checks everything, spends nothing
./scripts/reproduce.sh                              # 5. the smoke run
```

`--dry-run` checks the environments, the key, the configuration, the container
runtime and whether the data files are present, without a single paid call. It
does not start any service. Run it before any paid run. The first real run
downloads the raw datasets itself, through `scripts/fetch_datasets.py`, and
checks their sha256.

`reproduce.sh` has independent axes, and confusing them costs money:

| option | chooses |
|---|---|
| `--mode smoke\|full` | **which methods** run: `smoke` is the dense baseline and LightRAG; `full` adds Microsoft GraphRAG, Cognee and HippoRAG 2 |
| `--dataset`, `--version` | **how many questions**: a 20-question sample by default, or a complete 1,000-question dataset |
| `--protocol thesis` | **what runs**: without it, only the controlled arm; with it, the whole protocol of the dissertation (implies `--mode full`) |

The two smoke samples, `musique_smoke_20` (399 documents) and
`twowiki_smoke_20` (191 documents), have 20 questions each. Indexing is paid by
documents, not questions: shrink `num_questions` in the dataset config and the
corpus shrinks with it.

Disk: about 3 GB for smoke mode and 12 GB for full mode, most of it the
HippoRAG 2 environment (torch).

---

## Reproducing the protocol

```bash
./scripts/reproduce.sh --protocol thesis                           # on the 20-question sample
./scripts/reproduce.sh --dataset musique --version ans_v1.0_eval1k --protocol thesis
./scripts/reproduce.sh --dataset twowiki --version ans_v1.0_eval1k --protocol thesis
```

`--protocol thesis` runs, in order:

1. ingestion, with the sample checked against the fingerprints in
   `configs/datasets/fingerprints/`;
2. indexing with all five methods;
3. the **controlled arm**: one experiment per cell, named
   `<dataset>_eval1k_<cell>`, with the retrieval parameters the dissertation used;
4. the **controls**: closed-book floor and oracle ceiling;
5. the **native arm**: HippoRAG 2, Microsoft GraphRAG and Cognee through their
   own runners, and LightRAG from the native answers it produced during the
   controlled run (`scripts/etl_lightrag_native.py`);
6. the **canonical retrieval audit** behind recall@5, all_gold@5 and docs/q;
7. the **judge**, two passes per answer;
8. **McNemar with Holm** for both arms, and the consolidation.

The cells and their parameters:

| cell | method | top-k | notes |
|---|---|---:|---|
| `vector_v1free` | `vector_rag` | 5 | dense baseline |
| `lightrag_v1free` | `lightrag_neo4j` | 40 | `CHUNK_TOP_K=20`, which gives the 20 documents per question |
| `ms_graphrag_v1free` | `ms_graphrag` | 20 | local search; the cut is on the assembled context |
| `hipporag2_v1free` | `hipporag2` | 5 | in `results/` this cell and the native one carry the suffix `_official`: the same method over the single-pass index |
| `cognee_v1free_k5` | `cognee` | 5 | Cognee returns an unranked set, so @k truncates a set |
| `closed_book` | `zero_shot_no_context` | — | no documents |
| `oracle_gold_v1free` | `single_document_context` | — | the gold documents |
| `*_native` | each framework's own pipeline | — | Cognee in batches of 150, one process each, because its recall leaks memory |

The fixed reader is the **free reader** (`READER_GROUNDING=v1`), the prompt in
Appendix B: it uses the retrieved chunks when they help, and otherwise answers
from its own knowledge. `--protocol thesis` refuses any other reader.

### Cost and time

**The 20-question sample**, measured on 2026-09-25 with `--protocol thesis`:
every stage ran end to end, all 11 cells answered 20 of 20, and the judge cost
$1.23 (about $0.11 per cell). Cognee indexing took 51 minutes and $0.23, and
LightRAG indexing about 16 minutes. The cost of the other indexers and of the
readers was not recorded separately.

**The complete datasets** are a different order of magnitude:

| | |
|---|---|
| **the judge** | **roughly $120** for the 22,000 answers of both datasets, extrapolated from the sample |
| Cognee indexing, MuSiQue | about 38 hours and $3.88 in the original run |

The other indexers and the readers were not costed separately; the readers use
`gpt-4o-mini`, which is much cheaper than the judge's `gpt-4o`.

Budget days, not an afternoon. `--limit N` cuts the questions the readers
handle without touching indexing, which is useful for checking the chain on a
complete dataset before letting it run.

### What reproduction means here

The pipeline is not deterministic, so a new run reproduces the **protocol**, not
the numbers. The 20-question sample shows that the chain works. It says
nothing about which method is better: at 20 questions, one item is five points.

---

## Checking the dissertation against its data

No API key and no paid call needed:

```bash
.venv/bin/python -m pytest tests/test_results_match_thesis.py   # Tables 3 and 4 ← results/aggregate/
.venv/bin/python scripts/compute_judge_agreement.py            # Table B.4 ← the human labels
```

The judge validation reproduces the published agreement from the raw labels:

- **primary protocol**, 120 items: 100% agreement, Cohen's κ = 1.000. The
  annotator saw the judge's label and verified or corrected it, so anchoring may
  inflate this figure;
- **blind supplement**, 30 items: 93.3% agreement (28/30), 16/16 on native-arm
  refusals. κ is not computed, because the supplement is unbalanced by design.

[`results/README.md`](results/README.md) describes the data column by column,
and [`validation/s10_judge/README.md`](validation/s10_judge/README.md) the
annotation protocol.

---

## Things that look like faults and are not

- **Low EM and F1** (EM ≈ 0, F1 ≈ 0.1). The reader answers in full sentences,
  and token-overlap metrics penalise that. The metric the dissertation reports
  is the judge's strict accuracy.
- **Native refusals.** Several native pipelines decline to answer when their
  context is thin. That is their designed behaviour, and the dissertation
  reports it next to accuracy.
- **LightRAG's native cell on MuSiQue comes from an experiment named `_v2`.**
  LightRAG generates its native answer inside the library, before the harness's
  reader runs, and its adapter never reads `READER_GROUNDING`. The reader
  instruction does not reach the native answer.
- **`evidence_recall@5` in the batch runner is not the dissertation's
  recall@5.** It uses each run's own top-k (40 for LightRAG, 20 for Microsoft
  GraphRAG). The dissertation's figures come from the canonical audit
  (`scripts/retrieval_audit_canonic.py`), which cuts every method at five. An
  earlier identifier bug that zeroed this metric is fixed; see
  `docs/decisions/0001-retrieval-evaluation-granularity.md`.

## Traps that cost real time

[`scripts/README.md`](scripts/README.md) has all of them. These bite first:

- **The `.env` has to be exported, not just read.** Several adapters read
  `os.environ` directly. `reproduce.sh` does `set -a; . ./.env; set +a`; calling
  the CLI by hand means doing the same, or LightRAG fails with
  `embedding_func is required for vector storage`.
- **`LIGHTRAG_REUSE_RAG=1` is mandatory.** Without it LightRAG builds a RAG
  object per question and runs out of memory.
- **HippoRAG 2 indexes in a single pass.** Indexing in batches corrupts the
  graph: 19,392,908 edges over 319,817 distinct pairs, against 493,714 in the
  healthy one. The runner refuses a used workspace.
- **Neo4j containers are named after *(method, dataset)*** and are global to
  podman, not to the folder.
- **Compose prefixes volumes with the directory name.** Bringing the stack up
  from a folder with a different name gives you an empty database.
- **Under rootless podman**, Neo4j port forwarding only comes back after `stop`
  followed by `start`.

The ports are Postgres 15432 and Cognee's Postgres 15433
(`docker-compose.yml`), and Neo4j in the 18xxx band (`docker-compose.yml` and
`configs/infra/neo4j_ports.yaml`). The harness refuses the ports of the
original experimental environment (`benchmark.infra.guard`): 17474–17478,
17687–17691 and 7689 always, and the default Postgres and Neo4j ports only when
it finds that environment on the same machine.

---

## Citation

```bibtex
@mastersthesis{pereira2026graphrag,
  author = {Pereira, Eduardo Gon{\c{c}}alo},
  title  = {Graph-Based Retrieval-Augmented Generation for Multi-hop Question
            Answering: A Two-Arm Evaluation of Four Frameworks},
  school = {National College of Ireland},
  type   = {MSc Research Project},
  year   = {2026}
}
```

## License

The code is under the [MIT](LICENSE) license. The datasets are not in this
repository: `scripts/fetch_datasets.py` downloads them, MuSiQue from Hugging
Face mirrors and 2WikiMultiHopQA from the HippoRAG repository, and checks their
sha256. Question text, gold answers and document text from both datasets appear
in `results/` and `validation/`, and remain under the datasets' own licenses.
