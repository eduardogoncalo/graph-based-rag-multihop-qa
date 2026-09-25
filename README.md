# graph_based_rag

A comparative benchmark of graph-based RAG for multi-hop question answering,
over **MuSiQue** and **2WikiMultiHopQA**.

It compares five retrieval substrates — dense vector RAG, LightRAG, Microsoft
GraphRAG, Cognee and HippoRAG 2 — across **two arms**:

- **controlled**, where a fixed reader answers from the retrieved context, under
  the same conditions for every method;
- **native**, where each framework answers through its own pipeline, the way
  someone who installed it would use it.

The only thing that changes between the two reader arms is one instruction. It
lives in an environment variable, `READER_GROUNDING`, and the security section
below explains why it is the most dangerous variable in this repository.

---

## What runs, and what is evidence

Two things in this package look alike and are not.

**The code runs.** `src/`, `scripts/`, `configs/` and `tests/` are the pipeline.
Point them at a dataset and they produce indexes, answers and a table.

**`results/` and `validation/` are evidence from the original study.** They came
out of the experimental run that the thesis reports, on the machine where that
run happened. Nothing here regenerates them, no code reads them, and running the
pipeline will not reproduce them — it produces *your* numbers, which will be
different. They are here so the thesis can be checked against its own data.

The human annotation in `validation/s10_judge/` was **filled in by hand, by the
author**. There is no code path that could produce it.

---

## Getting started

Four steps. The first needs one binary; the rest need nothing that is not
already in the repository.

```bash
# 1. uv, which installs its own Python (no pyenv, no system Python
#    at a particular version)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. the environments
./scripts/bootstrap_envs.sh          # the main one; enough for smoke mode
./scripts/bootstrap_envs.sh --full   # plus the three isolated ones, for full mode

# 3. configuration
cp .env.example .env
$EDITOR .env                          # fill in OPENAI_API_KEY

# 4. run it
./scripts/reproduce.sh --dry-run      # checks everything, spends nothing
./scripts/reproduce.sh                # the whole chain
```

`--dry-run` checks the environment, the key, `MODEL_PROVIDER`, compose, the
ports and the data files **without making a single paid call**. It is always
worth doing before the first run.

### Requirements

| | what for |
|---|---|
| `uv` | builds the environments and installs Python |
| `podman` or `docker`, with compose | Postgres and Neo4j |
| an OpenAI key | indexing, reading and the judge |
| ~3 GB of disk | `smoke` mode |
| ~12 GB of disk | `full` mode |

Measured on 2026-08-09, with the smoke sample indexed by all five methods:
`.venv` 1.2 GB · `.venvs/` 8.8 GB (7 GB of which is HippoRAG 2, because of
torch) · `data/` 358 MB · `artifacts/` 457 MB. Podman volumes on top of that.

No GPU needed. The embedder is OpenAI's `text-embedding-3-small`, by declared
choice — the HippoRAG 2 paper uses NV-Embed-v2, and the thesis owns that
divergence as a controlled comparison.

---

## Two axes, and confusing them costs money

`reproduce.sh` has **two independent axes**, and the first one's name misleads:

| axis | option | chooses |
|---|---|---|
| mode | `--mode smoke\|full` | **which methods** run |
| dataset | `--dataset` / `--version` | **how many questions** |

`--mode full` over the smoke sample costs cents. The same `--mode full` over the
complete dataset costs tens of dollars and around 38 hours in Cognee indexing
alone. Same option.

### `smoke` mode — the default

`vector_rag` and `lightrag_neo4j`, the two that run with the main environment
and nothing else. The first exercises the dense path and Postgres with pgvector;
the second exercises the graph path, Option C, and the per-*(method, dataset)*
Neo4j container.

### `full` mode

Adds **`ms_graphrag`**, `cognee` and `hipporag2`. Needs the three isolated
environments and Cognee's Postgres.

> That is `ms_graphrag` — Microsoft's official file/parquet implementation,
> indexed through the CLI in `.venvs/graphrag` — and **not**
> `ms_graphrag_neo4j`. That variant is in the CLI but **has no live client**:
> the adapter raises `ms_graphrag_neo4j live execution is not configured`. It is
> an offline boundary that never got an implementation, and the Microsoft
> GraphRAG in the thesis is the other one.

### The smoke samples

Two of them, one per dataset, **20 questions** each:

| dataset | questions | documents | why that document count |
|---|---:|---:|---|
| `musique_smoke_20` | 20 | 399 | each MuSiQue question drags ~20 paragraphs along — 2 or 3 supporting, the rest distractors, and all of them part of the corpus |
| `twowiki_smoke_20` | 20 | 191 | the 2Wiki corpus is a separately published file of 6,119 passages; the loader narrows it to the passages belonging to the sampled questions |

**Indexing is paid for by documents, not by questions.** To shrink this further,
shrink `num_questions` in the dataset config and the corpus shrinks with it.

> Narrowing the 2Wiki corpus is not cosmetic. Without it, 20 questions dragged
> in all 6,119 documents from the published file — roughly **thirteen hours** of
> LightRAG indexing, against about 50 minutes for MuSiQue's 399. It would have
> stopped being a smoke test. The full-dataset path does none of this: it loads
> the 1,000 questions against the whole corpus, as it always has.

```bash
./scripts/reproduce.sh                              # MuSiQue, 20 questions
./scripts/reproduce.sh --dataset twowiki_smoke_20   # 2Wiki, 20 questions
./scripts/reproduce.sh --mode full                  # all five methods
```

---

## What the pipeline does

```
fetch_datasets.py     downloads the raw files and checks their sha256
       ↓
benchmark ingest      raw → canonical, and REFUSES a divergent sample
       ↓
benchmark persist     canonical → Postgres
       ↓
benchmark index       one index per (method, dataset)
       ↓
run_experiment_batch  the reader, question by question, resumable
       ↓
run_llm_judge         strict accuracy, via gpt-4o
       ↓
benchmark report      the table
```

### The checks that run on their own

- **The sample.** `configs/datasets/fingerprints/` records the identifiers of
  each dataset. `ingest` compares them and **refuses to write** a sample that
  does not match, before exporting anything at all. A raw file from a different
  release would produce a different 1,000 questions without anything complaining.
- **The raw files.** Each has its sha256 pinned in `fetch_datasets.py`.
- **The original environment.** See the security section.

---

## Costs and timings

Measured on a real `smoke` run on 2026-08-09 — 20 questions, 399 documents, 414
chunks — with `gpt-4o-mini` as the reader, `text-embedding-3-small` for
embeddings and `gpt-4o` as the judge:

| step | time | cost |
|---|---|---|
| ingest + persist | seconds | $0 |
| `vector_rag` indexing | seconds | < $0.01 |
| `lightrag_neo4j` indexing | ~50 min | cents |
| `vector_rag` reader (20 questions) | ~1 min | $0.0034 |
| `lightrag_neo4j` reader (20 questions) | ~2 min | $0.0035 |
| `cognee` indexing (399 documents) | ~1 h | $0.2178 |
| judge (53 answers across five methods) | ~2 min | **$0.2897** |

**The judge is the expensive step** among those that scale with the number of
questions: it uses `gpt-4o` and grades every answer twice. The readers use
`gpt-4o-mini` and cost fractions of a cent.

> An earlier version of this table put **$0.218** against the judge, over 40
> answers. That was wrong: $0.2178 is the cost of **Cognee indexing**, and the
> judge came to $0.2897 — taken from `estimated_cost_usd_total` in the run
> summary, which graded the 53 answers from all five methods (20+20+5+5+3), not
> just the 40 from `smoke` mode. Corrected on 2026-08-10.

LightRAG indexing is the **slow** step — about 50 minutes for 399 documents,
because it extracts entities and relations document by document. It scales with
documents, not with questions.

### What comes out

```
| method_id      | exact_match | answer_f1 | evidence_recall@5 | latency_ms |
| lightrag_neo4j |      0.0000 |    0.1082 |            0.7542 |    5649.60 |
| vector_rag     |      0.0000 |    0.1024 |            0.7292 |    2325.06 |
```

And from the judge, the metric the thesis reports.

**MuSiQue** (`musique_smoke_20`), `full` mode, all five methods:

| method | n | recall@5 | strict accuracy | refusal |
|---|---:|---:|---:|---:|
| `vector_rag` | 20 | 0.729 | 0.450 | 0.350 |
| `lightrag_neo4j` | 20 | 0.754 | 0.400 | 0.300 |
| `hipporag2` | 5 | 1.000 | 0.800 | 0.200 |
| `ms_graphrag` | 5 | 0.433 | 0.200 | 0.600 |
| `cognee` | 3 | 0.278 | 0.333 | 0.000 |

**2Wiki** (`twowiki_smoke_20`), `smoke` mode, measured 2026-08-10 — 191
documents, about 20 minutes end to end, **$0.23** in total, $0.22 of which was
the judge:

| method | n | recall@5 | strict accuracy | refusal |
|---|---:|---:|---:|---:|
| `vector_rag` | 20 | 0.750 | 0.450 | 0.250 |
| `lightrag_neo4j` | 20 | 0.788 | 0.450 | 0.350 |

> Note the `exact_match = 0.0000` next to `answer_f1 ≈ 0.10`. **Not a fault** —
> it is the artefact described below, and it is why the reference metric is the
> judge's strict accuracy rather than EM.
>
> **And do not draw conclusions from these tables.** They hold 3 to 20 questions
> per method against the protocol's 1,000. At 20 questions the gap between 0.450
> and 0.400 is *one item*; at 5, every item is worth 0.20. What they show is that
> the paths are alive — nothing about which method is better, or which dataset
> is harder.

`full` mode over the complete 1,000 × 2 protocol is a different order of
magnitude: four indexing runs across two datasets, 4,000 generations and 8,000
judge calls.

---

## `results/` — the data behind the thesis

**`results/` holds the data from the original experimental run**, at three
levels of detail — 44 MB, seven CSV files across four subfolders:

| | what it is |
|---|---|
| `aggregate/` | the study's 22 cells, one per row: two arms × two datasets × four frameworks, plus the three controls |
| `per_question/` | 22,000 rows — one question answered by one cell, with the answer in full and the judge's verdict |
| `retrieval/` | 98,599 rows — one retrieved document per row, with its rank |
| `corpus/` | the 17,634 source documents, once each. The only place the text appears |

**`results/README.md` explains it column by column**, including three things
worth knowing before opening the files: the native arm has no retrieval to show,
Cognee does not return a ranked list, and the dense baseline on MuSiQue was run
twice.

> Column names are in Portuguese, the language of the thesis.

**This is evidence, not output.** Three things follow from that:

1. **No code in this repository reads that folder.** Not a script, not a config,
   not a test. Delete it and nothing breaks; the pipeline does not know it
   exists.
2. **Running the pipeline does not regenerate it.** These files came from a run
   over both complete datasets, on the machine where the experimental phase
   happened. Run this package and you get your own results, which will differ —
   see «What this package guarantees» below.
3. **They are here to be read and checked**, not to be fed into anything. This
   is the material that lets someone compare what the thesis reports against
   what the pipeline produces.

---

## Reindexing everything — the full protocol

This section is for running **the thesis protocol** rather than the sample:
1,000 questions on each dataset, with all five methods. The commands are ready;
what changes against the smoke run is scale, money and time.

> **Read this before running it.** What comes out are not the thesis's numbers —
> nothing here is deterministic, and the next section explains why. What gets
> reproduced is the **protocol**.

### What it costs, and how long it takes

Measured during the experimental phase, on MuSiQue. 2Wiki has fewer documents
(6,119 against 11,515) and is proportionally cheaper to index.

| step | time | cost |
|---|---|---|
| `fetch_datasets.py` (both) | minutes | $0 |
| ingest + persist | minutes | $0 |
| `vector_rag` indexing | minutes | cents |
| `lightrag_neo4j` indexing | hours | a few dollars |
| `ms_graphrag` indexing | hours | a few dollars |
| **`cognee` indexing** | **~38 sequential hours** | **$3.88** |
| `hipporag2` indexing | hours | a few dollars |
| readers (5 methods × 1,000 questions) | hours | a few dollars (`gpt-4o-mini`) |
| **the judge** (2 passes per answer) | hours | **the expensive step** — `gpt-4o` |

**Budget days, not an afternoon.** And Cognee indexing is the step most worth
running on its own: it leaks memory on every recall call, and it needs batches
with `--limit` and a supervisor that resumes.

### The commands

```bash
# 1. both complete datasets (350 MB; 2Wiki pulls a 259 MB zip into cache)
python scripts/fetch_datasets.py

# 2. the environments, including the three isolated ones
./scripts/bootstrap_envs.sh --full

# 3. the .env — see the table below for what is required
cp .env.example .env && $EDITOR .env

# 4. check everything without spending. ALWAYS before the first paid run.
./scripts/reproduce.sh --dataset musique --version ans_v1.0_eval1k --mode full --dry-run

# 5. MuSiQue, complete
./scripts/reproduce.sh --dataset musique --version ans_v1.0_eval1k --mode full

# 6. 2Wiki, complete
./scripts/reproduce.sh --dataset twowiki --version ans_v1.0_eval1k --mode full
```

Each of those last two is a run that lasts days. `--limit N` cuts the number of
questions the readers handle without touching indexing, which is useful for
checking the chain before letting it loose:

```bash
./scripts/reproduce.sh --dataset musique --version ans_v1.0_eval1k --mode full --limit 5
```

### The `.env` for a complete run

`.env.example` carries everything, commented. What **must** be filled in:

| variable | value | why |
|---|---|---|
| `OPENAI_API_KEY` | your key | without it the judge fails immediately |
| `MODEL_PROVIDER` | `openai` | the code's default is `fake`, which returns synthetic answers and exists for the test suite |
| `COGNEE_NEO4J_PASSWORD` | any password | `full` mode only; checked at startup, so you do not find it missing after 38 hours |

What **should not** change unless you know what it does:

| variable | value | why |
|---|---|---|
| `READER_GROUNDING` | `v2` | the switch between the thesis's two arms — see the security section |
| `LIGHTRAG_REUSE_RAG` | `1` | without it LightRAG builds a RAG object per question and runs out of memory |

Everything else has a default that works. The ports — Postgres 15432, Cognee
Postgres 15433, Neo4j in the 18xxx band — already match `docker-compose.yml`.

> **The `.env` has to be exported, not just read.** `reproduce.sh` does
> `set -a; . ./.env; set +a` for you. Calling the CLI by hand means doing the
> same — see the security section.

### Something that only happens on the author's machine

Run this **on the machine where the experimental phase happened** and the
complete datasets are refused:

```
--dataset of reproduce.sh names dataset 'musique_ans_v1_0_eval1k', one of the
datasets from the original experimental phase. […] DO NOT remove that container.
```

Not a fault. Neo4j containers are named after *(method, dataset)* and podman's
namespace is global to the machine, so indexing the complete dataset there would
write over the indexes the thesis rests on. **On a machine that never had that
environment the brake sits inert** and the complete datasets run normally —
which is exactly what the detection is for. See the security section.

---

## What this package guarantees, and what it does not

**It guarantees** that the chain runs end to end from a clean copy, and that the
ingested sample is the same one the thesis used. That is verifiable, and it is
verified.

**It does not guarantee the numbers.** Indexing, reading and judging all use
language models, and none of that is deterministic. The proof is inside the
project: re-indexing HippoRAG 2 with the same code on the same machine changed
the ordering between methods.

Running this reproduces the **experimental protocol**. `smoke` mode shows the
pipeline works; it does **not** reproduce the thesis's conclusions.

### Three things that look like faults and are not

1. **Low F1 and EM.** Values around F1 ≈ 0.09 and EM ≈ 0 are an artefact of
   verbosity and of measurement, diagnosed on 2026-06-23. The metric the thesis
   uses is the **judge's strict accuracy**. Anyone who "fixes" the reader over
   this is chasing a problem that does not exist.
2. **`evidence_recall@5` of zero for `vector_rag` on MuSiQue.** There was an
   identifier namespace problem (`ev_` against `doc_`), it is documented, and the
   fix was **deliberately not applied**. Understand what was left alone, and why,
   before touching retrieval metrics.
3. **Reader refusals.** In the grounded arm (`v2`) the reader is instructed to
   abstain when the context is not enough. A non-zero refusal rate is the
   designed behaviour, not a failure.

---

## The human validation of the judge

`validation/s10_judge/` holds the judge's labels, **the hand annotation done by
the author**, and the protocol that produced it. Like `results/`, it is evidence
from the original study: nothing in this package regenerates the human column.

Reproduce the agreement and Cohen's κ:

```bash
python scripts/compute_judge_agreement.py
```

It reproduces the published values from the raw labels: agreement of 1.0 across
the 120 items of the main protocol, and κ of **0.877** (MuSiQue) and **0.8855**
(2Wiki) over the 30 supplementary items.

> The main protocol was **verification and correction** with the judge's label
> pre-filled, not blind labelling, so there is an anchoring risk that tends to
> inflate agreement. The 30-item supplement was blind. The thesis says so, and
> so does this.

---

## The file that is a security matter

This repository was built next to the environment where the experimental phase
ran, and **that environment still exists on that machine**, holding the indexes
and the 4,000 judgements the thesis rests on.

A `.env` with the old URIs connects to those databases and **can write to them**.
As far as the program is concerned that is a legitimate connection, so nothing
blocks it.

There is a brake (`benchmark.infra.guard`) that refuses the original
environment's ports — the 17xxx band and 7689 always, and the default Postgres
(5432/5433) and Neo4j (7474/7687) ports only while it detects the original
repository alongside. **If you received this package on a machine that never had
that environment, there is nothing to worry about**: the detection finds nothing
and those ports are yours.

This repository's own ports live in `docker-compose.yml`:

| service | port |
|---|---|
| benchmark Postgres | 15432 |
| Cognee Postgres | 15433 |
| Neo4j | 18xxx band |

### `READER_GROUNDING`

This is the **single-variable** switch between the thesis's two arms, and the
only thing that changes between them is the instruction given to the reader.

| value | reader |
|---|---|
| `v2` (default) | **grounded** — answers only from the retrieved context, and abstains when it is not enough |
| `v1`, `off`, `free`, `0`, `no`, `none` | **free** — may fall back on parametric knowledge and never declines |

Set it wrong and you produce, **silently**, a different experiment under the
same name: the output files do not distinguish one from the other.
`reproduce.sh` always says out loud which one is about to run.

### The `.env` has to be exported, not just read

Settings go through `Settings` (pydantic), which reads the `.env`. But **half the
adapters read `os.environ` directly** — `MODEL_PROVIDER`, `READER_GROUNDING`,
`LIGHTRAG_REUSE_RAG` and others — and those variables never arrive that way.

`reproduce.sh` does `set -a; . ./.env; set +a`. Calling the CLI by hand means
doing the same, or LightRAG fails with `embedding_func is required for vector
storage`, a message that points nowhere near the cause.

---

## Traps that cost real time

`scripts/README.md` has all of them, with the detail. These bite first:

- **`LIGHTRAG_REUSE_RAG=1` is mandatory.** Without it LightRAG builds a RAG
  object per question and runs out of memory.
- **HippoRAG 2 indexing is a single pass.** Indexing in batches corrupts the
  graph: 19,392,908 edges over 319,817 distinct pairs, against roughly 1.58M for
  a healthy one. The runner refuses a used workspace, and `--batch-size` does not
  exist.
- **Neo4j containers are named after *(method, dataset)*** and are global to
  podman, not to the folder. That is why the smoke samples have identifiers of
  their own.
- **Compose prefixes volumes with the directory name.** Bringing this up from a
  folder with a different name creates an empty database.
- **Under rootless podman**, Neo4j port forwarding only comes back after `stop`
  followed by `start`.

---

## Layout

```
src/benchmark/      the core: ingestion, methods, agents, evaluation, CLI
scripts/            indexers, native readers, judge, statistics
configs/            datasets, methods, experiments, the Neo4j port registry
requirements/       manifests for the three isolated environments
tests/              the proof that the core works

data/               the datasets: raw files and canonical samples
results/            the data behind the thesis — evidence, not output
validation/         the human validation of the judge — hand annotation
```

The bottom three are **data**. `src/`, `scripts/` and `configs/` are what you
run; `data/` is what gets read; `results/` and `validation/` are what the
original study produced, kept here so it can be checked.

## Documentation

- **`scripts/README.md`** — architecture and traps: why there are four Python
  environments, the names that have to line up between indexer and reader, the
  design of the per-*(method, dataset)* Neo4j containers, the three brakes, and
  what does not get changed and why. Read it before touching the code or calling
  a script by hand.
- **`results/README.md`** — the results files, column by column.
- **`validation/s10_judge/README.md`** — the human annotation protocol.
