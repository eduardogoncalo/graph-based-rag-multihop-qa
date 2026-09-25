"""The native arm of LightRAG: materialise its own answers as a cell.

LightRAG's native answers never had a run of their own. In the controlled arm
the adapter calls `rag.query_llm(query, QueryParam(mode="mix", top_k=...))`,
and the library generates its own answer with its own prompt before the fixed
reader ever sees the context. That answer is persisted with the retrieval, in
`retrieval_results.raw_response.llm_response.content`. This script copies those
answers, one per question, into an experiment of their own
(`<prefix>lightrag_native`), which the judge and `mcnemar_native_arm.py` then
read like any other cell.

Consequences worth knowing:

- The native cell inherits the retrieval of its source run: top_k=40 and
  CHUNK_TOP_K=20 in the thesis configuration. Run the controlled LightRAG cell
  with those values first (reproduce.sh does).
- The answer does not depend on READER_GROUNDING. The LightRAG adapter never
  reads it; the instruction only reaches the fixed reader, which runs after.
  That is why, in the thesis, the MuSiQue native cell could be materialised
  from the `musique_eval1k_lightrag_v2` run (same retrieval, grounded reader)
  and the 2Wiki one from `twowiki_eval1k_lightrag_v1free`.
- It makes no API call and is idempotent: run ids are deterministic and the
  store upserts.

Usage:
    python scripts/etl_lightrag_native.py --dataset-id musique \\
        --dataset-version ans_v1.0_eval1k
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

from benchmark.core.ids import deterministic_id

METHOD = "lightrag_neo4j"
AGENT_MODE = "single_agent"

SOURCE_QUERY = """
    SELECT a.question_id,
           r.run_id AS source_run_id,
           rr.retrieval_result_id AS source_rr_id,
           rr.raw_response::jsonb #>> '{llm_response,content}' AS native
    FROM runs r
    JOIN answers a ON a.run_id = r.run_id
    JOIN retrieval_results rr ON rr.run_id = r.run_id
    WHERE r.experiment_id = %s AND r.method_id = %s
    ORDER BY a.question_id
"""


def default_prefix(dataset_id: str) -> str:
    """The experiment prefix every other script of the protocol uses."""
    return f"{dataset_id}_eval1k_"


def fetch_source_rows(cursor: Any, source_experiment: str) -> list[tuple[str, str, str, str | None]]:
    cursor.execute(SOURCE_QUERY, (source_experiment, METHOD))
    return list(cursor.fetchall())


def materialize(
    store: Any,
    rows: Iterable[tuple[str, str, str, str | None]],
    *,
    source_experiment: str,
    target_experiment: str,
    dataset_id: str,
    dataset_version: str,
) -> int:
    """Write one run and one answer per source row. Returns how many were written.

    Refuses to write anything if any native answer is empty: an empty answer
    would be judged as a refusal and bias the cell without any visible sign.
    """
    rows = list(rows)
    if not rows:
        raise SystemExit(
            f"no LightRAG runs in {source_experiment}. Run the controlled LightRAG cell first."
        )
    empty = [question_id for question_id, _, _, native in rows if not (native or "").strip()]
    if empty:
        raise SystemExit(
            f"{len(empty)} empty native answers in {source_experiment} "
            f"(e.g. {empty[:3]}); nothing was written"
        )

    store.create_experiment(
        experiment_id=target_experiment,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        metadata={
            "branch": "as_deployed",
            "framework": "lightrag",
            "native_qa": True,
            "source_experiment_id": source_experiment,
            "note": "LightRAG's native answers (llm_response.content), native arm",
        },
    )
    written = 0
    for question_id, source_run_id, source_rr_id, native in rows:
        run_id = deterministic_id(
            "run",
            [target_experiment, dataset_id, dataset_version, METHOD, AGENT_MODE, question_id],
        )
        provenance = {
            "native_qa": True,
            "branch": "as_deployed",
            "framework": "lightrag",
            "origin": "retrieval_results.raw_response.llm_response.content",
            "source_experiment_id": source_experiment,
            "source_run_id": source_run_id,
            "source_retrieval_result_id": source_rr_id,
        }
        store.create_run(
            experiment_id=target_experiment,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            method_id=METHOD,
            agent_mode=AGENT_MODE,
            run_id=run_id,
            status="materialized",
            metadata={
                key: provenance[key]
                for key in ("native_qa", "branch", "framework", "source_experiment_id", "source_run_id")
            },
        )
        store.persist_answer(
            run_id=run_id,
            method_id=METHOD,
            agent_mode=AGENT_MODE,
            answer_text=native,
            question_id=question_id,
            metadata=provenance,
        )
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--exp-prefix", default=None, help="default: <dataset-id>_eval1k_")
    parser.add_argument("--source-exp", default=None, help="default: <prefix>lightrag_v1free")
    parser.add_argument("--target-exp", default=None, help="default: <prefix>lightrag_native")
    args = parser.parse_args()

    prefix = args.exp_prefix if args.exp_prefix is not None else default_prefix(args.dataset_id)
    source = args.source_exp or f"{prefix}lightrag_v1free"
    target = args.target_exp or f"{prefix}lightrag_native"

    from benchmark.cli.app import _connect_postgres
    from benchmark.storage.experiment_store import ExperimentStore

    connection = _connect_postgres()
    try:
        with connection.cursor() as cursor:
            rows = fetch_source_rows(cursor, source)
        print(f"[etl] source {source}: {len(rows)} runs")
        written = materialize(
            ExperimentStore(connection),
            rows,
            source_experiment=source,
            target_experiment=target,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
        )
        connection.commit()
        print(f"[etl] wrote {written} answers to {target}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
