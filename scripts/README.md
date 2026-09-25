# `scripts/` — architecture, and where it bites

The README at the root tells you **what to run**. This one tells you **how the
thing is built** and **where it hurts**. It complements that document rather
than repeating it.

If all you want is to run the pipeline, you do not need any of this —
`./scripts/reproduce.sh` is enough. Read on if you are going to change the code,
call a script by hand, or work out why something is shaped the way it is.

---

## 1. Why there are four Python environments

Not tidiness. The frameworks' dependencies genuinely conflict, and no single
resolution puts them all in one place.

| environment | holds | who uses it |
|---|---|---|
| `.venv` (main) | `lightrag`, `cognee` | `vector_rag`, `lightrag_neo4j`, the judge, the statistics, **and Cognee retrieval** |
| `.venvs/cognee` | `cognee` pinned to `neo4j==5.28.4` | only the Cognee **indexing** scripts |
| `.venvs/graphrag` | the official Microsoft GraphRAG CLI | only `ms_graphrag` indexing and querying |
| `.venvs/hipporag2` | `hipporag` 2.0.0a3 and torch | only HippoRAG 2, over a subprocess |

**The conflict belongs to indexing, not to the method.** The main environment's
`neo4j-driver` 6.x fights the `neo4j<6` that Cognee requires — but only Cognee's
*indexing* touches that driver. *Retrieval* runs in the main environment
(`methods/cognee/adapter.py` imports `cognee` directly) and coexists with
`neo4j` 6.2.0 without trouble. That is why `pyproject.toml` could declare
`cognee` without breaking `lightrag_neo4j`.

**How the dispatch finds the venv.** The adapters read an environment variable
and, when it is absent, resolve a path relative to the working directory:

```
HIPPORAG2_VENV_PYTHON   →  .venvs/hipporag2/bin/python
GRAPHRAG_VENV_PYTHON    →  .venvs/graphrag/bin/python
```

The venvs themselves are **not relocatable** — absolute paths live in their
`pyvenv.cfg` and in every shebang. Moving the folder breaks them;
`bootstrap_envs.sh` builds them again.

---

## 2. The names that have to line up, and nothing checks them

**The most expensive trap in this repository.** The indexing scripts take their
destination **as an argument**, but whatever *reads* the index **computes** it.
If the two do not match character for character, indexing runs fine, spends
money, and the reader finds nothing.

| script | argument | has to be |
|---|---|---|
| `hipporag2_index_runner.py` | `--save-dir` | `artifacts/<slug>/hipporag2/`**`workspace`** — what `hipporag2/config_builder.py` computes |
| `cognee_index_musique.py` | `--workspace` | `artifacts/<slug>/cognee` — **without** `/workspace` |
| `cognee_index_musique.py` | `--dataset-name` | **the slug**, e.g. `musique_smoke_20_v1` — what `cognee/option_c.py` looks for |

The `<slug>` is `slugify_dataset_version(dataset_id, dataset_version)`, with dots
turned into underscores: `musique` plus `ans_v1.0_eval1k` gives
`musique_ans_v1_0_eval1k`.

**Neither symptom points anywhere near the cause:**

- HippoRAG 2 starts with `passages_mapped=0` and dies on
  `shapes (0,) and (1536,) not aligned`;
- Cognee raises `DatasetNotFoundError: No datasets found`, which looks like a
  missing index and is a mistyped name.

**`reproduce.sh` passes the right values.** If you call these scripts by hand,
this table is the only thing standing between you and a wasted indexing run.

### The names lie, and they do not get renamed

`index_ms_graphrag_musique.py`, `cognee_index_musique.py`,
`twowiki_lightrag_index.py` — the names say one dataset, but **the scripts are
parameterised**. The first takes `--dataset-id twowiki`; the second takes
`--documents`, `--workspace` and `--out-dir`.

Renaming would be cosmetic and would risk breaking references scattered across
scripts, logs and documents. `reproduce.sh` hides the old names, which is what
it is for.

---

## 3. Option C: one Neo4j container per *(method, dataset)*

Every *(method, dataset)* pair gets its own container, volume and ports. The
names come from the pair:

```
neo4j_lightrag_musique_smoke_20_v1          the container
neo4j_lightrag_musique_smoke_20_v1_data     the volume
```

Ports come from `configs/infra/neo4j_ports.yaml` and are **never hardcoded**.
`resolve_ports` runs a preflight: it refuses a reserved or occupied port before
trying to start anything, rather than letting the container fail to bind.

**There are two sources of truth for Neo4j, and both are load-bearing.**
`docker-compose.yml` declares fixed `neo4j_graphrag` and `neo4j_lightrag`
services, *and* Option C creates per-dataset containers. This is not leftover
duplication:

- the **fixed services** are what the **CLI** uses (`benchmark index` resolves
  through `LIGHTRAG_NEO4J_*` / `GRAPHRAG_NEO4J_*`);
- **Option C** is what the **scripts** use — `twowiki_lightrag_index.py`,
  `run_cognee_native.py`, and `run_single.py` for Cognee.

Delete either one and half the package stops working.

**Two operational notes:**

- containers are **global to podman**, not to the folder. A container named for
  another dataset will still be found;
- under rootless podman, Neo4j port forwarding only comes back after `stop`
  followed by `start`. A `restart` is not enough.

---

## 4. The three brakes against the original environment

This package was built alongside the machine where the experimental phase ran.
If you received it on a machine that never had that environment, **none of this
applies** and the brakes sit inert. That is deliberate.

`benchmark.infra.guard` has three checks, one per route in:

| brake | closes | what it catches |
|---|---|---|
| `exigir_alvo_permitido` | **URIs** | `bolt://localhost:17688` in a `.env` |
| `exigir_caminho_permitido` | **disk** | HippoRAG 2's `--save-dir` pointed at the original `artifacts_variant/` |
| `exigir_dataset_permitido` | **container names** | `--dataset musique --version ans_v1.0_eval1k`, whose container is the thesis's own |

There are also **two levels of refusal**, and that distinction is what makes the
package usable by anyone else:

- **always refused** — the 17xxx band and 7689. Those were chosen here, they
  mean nothing on another machine, and refusing them costs nobody anything;
- **refused only while the original is present** — Postgres 5432/5433, Neo4j
  7474/7687, and the slugs `musique_ans_v1_0_eval1k` and
  `twowiki_ans_v1_0_eval1k`. Whoever receives this package has every right to
  those ports and those datasets.

The original environment is located either through `BENCHMARK_AMBIENTE_ORIGINAL`
or by finding a sibling of this repository carrying a signature of four
directories: `artifacts/`, `artifacts_variant/`, `status/` and `writing/`. Four,
not two, because `artifacts/` and `status/` on their own are common enough names
that the detection once picked the wrong tree.

---

## 5. HippoRAG 2 indexing is a single pass

**Indexing in batches corrupts the graph.** Measured: 19,392,908 edges over
319,817 distinct pairs — a multiplicity around 60, against roughly 1.58M for a
healthy graph.

The cause is OpenIE re-deriving the edges on every call to `index()`. The
embedding store's hash deduplication covers **passages**, not **edges**, and
that confusion is exactly what the old runner's docstring taught.

What `hipporag2_index_runner.py` does now:

- **`--batch-size` no longer exists.** One call to `index()`, over the whole
  window.
- **A used workspace is refused**, with an explicit and destructive `--reset` as
  the only way out. The check looks at the **contents**, not at a receipt: a run
  that dies halfway leaves a partial index and no receipt at all, and indexing on
  top of that is the same duplication.
- **`--skip` and `--limit` remain**, because the small sample needs them, but
  each window needs its own destination. Indexing `[0:500]` and then `[500:1000]`
  into the same place is two calls, exactly like two batches.
- **The summary reports the graph**: vertices, edges, distinct pairs and maximum
  multiplicity. It reports and does not block, because HippoRAG links the same
  pair for more than one reason (*relation*, *synonym*, *context*) and a
  multiplicity of 2 or 3 can be legitimate.

Do not restore `--batch-size`, and do not trade the used-workspace refusal for a
resume.

---

## 6. Sample fingerprints

`configs/datasets/fingerprints/` records, for each dataset, the number of
questions and documents, the sha256 of each identifier set, and the full list of
ids. `benchmark ingest` compares **between loading and exporting** and refuses to
write a sample that does not match.

The ordering is the whole point. A wrong sample written to disk is
indistinguishable from a good one as far as indexing, the readers and the judge
are concerned — it would only ever show up in the numbers.

**Three cases where it does not check, none of them a mistake:** an explicit
`--sample-size` (the sample diverges by construction, and it says so out loud),
a dataset with no recorded fingerprint, and `--allow-divergent-sample`, which is
the deliberate way out. That escape hatch exists so nobody is ever pushed into
**deleting the fingerprint file**, which would remove the check for everyone,
permanently.

**Do not update the sha256 values in `fetch_datasets.py` to silence a download
that fails verification.** If the file changed, the sample changed with it.

---

## 7. Resetting Cognee when indexing dies halfway

Cognee is the method most likely to die halfway: a full index takes around 38
sequential hours and leaks memory on every recall call. A partial index cannot be
recovered by indexing over it — `cognify` reprocesses whatever is pending across
the whole dataset, not just the document you asked for, and the result is
duplication that is hard to audit.

**Resetting costs less than pressing on.** Cognee owns three stores, and only
these. Nothing else is touched:

| store | where | what it holds |
|---|---|---|
| relational + vector | Postgres from the `postgres_cognee` service, port **15433**, database `cognee_benchmark`, schema `cognee` | checkpoints, payloads, embeddings |
| graph | container `neo4j_cognee_<slug>`, port from the registry | nodes and relationships |
| files | `artifacts/<slug>/cognee/` | workspace, local cache, outputs |

`<slug>` is the dataset's — `musique_smoke_20_v1`, `twowiki_ans_v1_0_eval1k`. The
bolt ports come from `configs/infra/neo4j_ports.yaml`: 18692 (MuSiQue smoke),
18695 (MuSiQue full), 18698 (2Wiki full), 18701 (2Wiki smoke).

```bash
SLUG=musique_smoke_20_v1
BOLT=18692

# 1. the graph
podman rm -f "neo4j_cognee_${SLUG}" && podman volume rm "neo4j_cognee_${SLUG}_data"

# 2. the files — only THIS dataset's Cognee subtree
rm -rf "artifacts/${SLUG}/cognee"

# 3. Postgres. --dry-run is the DEFAULT: without --execute it prints the SQL and
#    does nothing. Run it that way first and read what it was going to do.
.venv/bin/python scripts/bootstrap_cognee_postgres.py \
  --create-database --create-schema --create-vector-extension

# 4. and then for real
.venv/bin/python scripts/bootstrap_cognee_postgres.py \
  --create-database --create-schema --create-vector-extension --execute

# 5. bring the graph back up, then reindex
.venv/bin/python scripts/cognee_start_neo4j.py \
  --name "neo4j_cognee_${SLUG}" --http-port 18478 --bolt-port "$BOLT" \
  --password "$COGNEE_NEO4J_PASSWORD"
```

**Three things you do not do here:**

- **`podman volume prune`** — it removes volumes for anything that happens to be
  stopped, and takes the LightRAG and GraphRAG indexes with it;
- **touch the `benchmark` database** on port 15432. Different database,
  different server, and it holds the answers, the judgements and the
  `vector_rag` embeddings. A Cognee reset goes nowhere near it;
- **delete `data/canonical/`.** Resetting Cognee does not require re-ingesting
  anything.

> This replaces `plan_cognee_reset.py`, removed on 2026-08-10. It generated a
> static document with the date frozen at 2026-06-17, a body describing an
> incident from the CUAD era, and commands pointing at a
> `docker-compose.cognee.yml` that does not exist, a `neo4j_cognee` service that
> does not exist, and an `artifacts/cognee` path that is not this repository's
> layout. The test shipped alongside it pinned exactly those wrong values.

---

## 8. What each script is for

**Indexing**
`hipporag2_index_runner.py` · `index_ms_graphrag_musique.py` ·
`cognee_index_musique.py` and `cognee_index_musique_batched.py` ·
`twowiki_lightrag_index.py` · `cognee_start_neo4j.py` ·
`bootstrap_cognee_postgres.py`

**Readers**
`run_experiment_batch.py` (the controlled arm, resumable) ·
`run_cognee_native.py` · `run_hipporag2_native.py` · `run_ms_graphrag_native.py` ·
`hipporag2_variant_run.py`

**Called by `src/`, not from the command line** — which is precisely why they
went missing once:
`hipporag2_query_runner.py` · `graphrag_query_runner.py`

> The check that catches them, if they ever go missing again:
> ```bash
> for f in $(grep -rhoE "scripts/[a-z0-9_]+\.py" src/ scripts/ | sort -u); do
>   [ -e "$f" ] || echo "MISSING: $f"
> done
> ```

**Judge and evaluation**
`run_llm_judge.py` · `recompute_eval.py` · `recompute_retrieval_metrics.py` ·
`compute_judge_agreement.py`

**Statistics and consolidation**
`run_stats.py` · `mcnemar_v1_arm.py` · `mcnemar_native_arm.py` ·
`consolidate_p5.py` · `consolidate_cross_dataset.py`

**Setup and environment checks**
`fetch_datasets.py` · `bootstrap_envs.sh` · `reproduce.sh` ·
`validate_cognee_env.py` · `validate_ms_graphrag_neo4j_env.py` ·
`validate_lightrag_neo4j_real_import.py`

---

## 9. Things that do not get changed, and why

- **`run_llm_judge.py` stays as it is.** It has a known fragile key: `load_done`
  is indexed by *(method, question_id)* without the experiment, so two datasets
  sharing an identifier would make the judge skip answers silently. It never
  happened — the corpora do not intersect. This is recorded as a warning rather
  than fixed: trading a known and worked-around problem for a reproducibility
  risk is a bad deal.
- **`--out-tag` has to differ per dataset.** Give both the same one and the
  summary files collapse into a single file. This has happened.
- **`ms_graphrag_neo4j` has no live client.** It is in the CLI, but the adapter
  is an offline boundary and raises `live execution is not configured`. The
  Microsoft GraphRAG that actually runs is `ms_graphrag`.
- **`bm25` was removed** because it left the thesis. The migration numbering has
  a hole at 002 **on purpose** — renumbering `003` would break databases that
  have already been migrated.
- **`evidence_recall@5` of zero for `vector_rag` on MuSiQue** has a documented
  cause (an identifier namespace clash, `ev_` against `doc_`) and the fix was
  **deliberately not applied**. Understand what was left alone, and why, before
  touching retrieval metrics.
- **Low F1 and EM are not a fault** — they are an artefact of verbosity and of
  measurement. The thesis reports the judge's strict accuracy.

---

## 10. Editing a `.sh` while it is running

There is no guarantee at all: the running process keeps reading the old version.
On a run that lasts hours this looks like a bug in the code, and it is the file
being read in pieces.
