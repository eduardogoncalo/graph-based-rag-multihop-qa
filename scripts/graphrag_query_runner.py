"""Retrieval-only query runner for Microsoft GraphRAG 3.1.0.

Runs INSIDE the isolated graphrag venv (.venvs/graphrag). Builds the search
context via the context builder directly — embedding calls only, no chat
completion — and prints a single JSON document to stdout:

    {
      "context_text": str,           # the assembled context (fed to our reader)
      "items": [                     # one per source/text-unit in context
        {"unit_id", "unit_short_id", "document_id", "title", "text", "in_context"}
      ],
      "stats": {...}                 # counts + token estimates for the smoke log
    }

Usage:
    .venvs/graphrag/bin/python scripts/graphrag_query_runner.py \
        --root <workspace_dir> --method local --query "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd


def _fail(message: str) -> None:
    print(json.dumps({"error": message}), file=sys.stderr)
    raise SystemExit(1)


def _load_parquet(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / f"{name}.parquet"
    if not path.exists():
        _fail(f"missing artifact: {path}")
    return pd.read_parquet(path)


def _connect_entity_store(config):
    """Entity-description embedding store — the official API path (3.1.0)."""
    try:
        from graphrag.config.embeddings import entity_description_embedding
        from graphrag.utils.api import get_embedding_store

        return get_embedding_store(config.vector_store, entity_description_embedding)
    except Exception as exc:  # pragma: no cover - environment specific
        _fail(f"could not connect entity embedding store: {exc!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--method", default="local", choices=["local", "global"])
    parser.add_argument("--query", required=True)
    parser.add_argument("--community-level", type=int, default=2,
                        help="-1 = sem filtro de comunidade (todas as entidades)")
    parser.add_argument("--generate", action="store_true",
                        help="geração NATIVA (as-deployed): roda engine.search() e emite "
                             "{\"answer\": ...} em vez do contexto retrieval-only")
    args = parser.parse_args()

    os.environ.setdefault("GRAPHRAG_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    root = Path(args.root).resolve()
    os.chdir(root)  # config db_uri/base_dir are root-relative
    output_dir = root / "output"

    from graphrag.config.load_config import load_config

    config = load_config(root)

    entities_df = _load_parquet(output_dir, "entities")
    communities_df = _load_parquet(output_dir, "communities")
    reports_df = _load_parquet(output_dir, "community_reports")
    text_units_df = _load_parquet(output_dir, "text_units")
    relationships_df = _load_parquet(output_dir, "relationships")
    documents_df = _load_parquet(output_dir, "documents")

    from graphrag.query.indexer_adapters import (
        read_indexer_communities,
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )

    community_level = None if args.community_level < 0 else args.community_level
    entities = read_indexer_entities(entities_df, communities_df, community_level)
    reports = read_indexer_reports(reports_df, communities_df, community_level)
    text_units = read_indexer_text_units(text_units_df)
    relationships = read_indexer_relationships(relationships_df)

    # ---- doc-id traceability maps (text unit -> document -> benchmark doc_*) --
    doc_title: dict[str, str] = {}
    for row in documents_df.itertuples(index=False):
        payload = row._asdict() if hasattr(row, "_asdict") else dict(row._mapping)  # type: ignore[attr-defined]
        doc_title[str(payload.get("id"))] = str(payload.get("title") or payload.get("id"))

    unit_doc: dict[str, str] = {}
    unit_by_short: dict[str, dict] = {}
    for row in text_units_df.itertuples(index=False):
        payload = row._asdict() if hasattr(row, "_asdict") else dict(row._mapping)  # type: ignore[attr-defined]
        unit_uuid = str(payload.get("id"))
        short_id = str(payload.get("human_readable_id"))
        document_id = payload.get("document_id")
        if document_id is None:  # pre-v3 plural column, defensive
            ids = payload.get("document_ids")
            document_id = ids[0] if isinstance(ids, (list, tuple)) and len(ids) else None
        record = {"uuid": unit_uuid, "short_id": short_id, "document_id": str(document_id)}
        unit_doc[unit_uuid] = str(document_id)
        unit_by_short[short_id] = record
        unit_by_short[unit_uuid] = record

    store = _connect_entity_store(config)

    from graphrag.query.factory import get_global_search_engine, get_local_search_engine

    if args.method == "local":
        engine = get_local_search_engine(
            config,
            reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates={},
            response_type="multiple paragraphs",
            description_embedding_store=store,
        )
        # No modo generate, engine.search() já constrói o contexto internamente —
        # pular este build evita retrieval em dobro.
        ctx = None if args.generate else engine.context_builder.build_context(
            query=args.query, **engine.context_builder_params
        )
        if ctx is not None and asyncio.iscoroutine(ctx):
            ctx = asyncio.run(ctx)
    else:
        communities = read_indexer_communities(communities_df, reports_df)
        engine = get_global_search_engine(
            config,
            reports=reports,
            entities=entities,
            communities=communities,
            response_type="multiple paragraphs",
        )
        if args.generate:
            ctx = None
        else:
            maybe = engine.context_builder.build_context(
                query=args.query, **engine.context_builder_params
            )
            ctx = asyncio.run(maybe) if asyncio.iscoroutine(maybe) else maybe

    # Geração NATIVA (Braço B): o próprio GraphRAG responde via engine.search()
    # (contexto + LLM). Emite {"answer": ...}; o output_parser lê a chave "answer".
    if args.generate:
        search_result = engine.search(args.query)
        if asyncio.iscoroutine(search_result):
            search_result = asyncio.run(search_result)
        resp = getattr(search_result, "response", search_result)
        answer = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
        print(json.dumps({
            "answer": answer,
            "method": args.method,
            "stats": {
                "llm_calls": getattr(search_result, "llm_calls", None),
                "prompt_tokens": getattr(search_result, "prompt_tokens", None),
                "answer_chars": len(answer),
            },
        }))
        return

    chunks = ctx.context_chunks
    context_text = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
    records = ctx.context_records or {}

    items: list[dict] = []
    sources = records.get("sources")
    if sources is not None and len(sources):
        for row in sources.itertuples(index=False):
            payload = row._asdict() if hasattr(row, "_asdict") else dict(row._mapping)  # type: ignore[attr-defined]
            raw_id = str(payload.get("id"))
            record = unit_by_short.get(raw_id)
            document_id = record["document_id"] if record else unit_doc.get(raw_id)
            title = doc_title.get(str(document_id), str(document_id))
            in_context = payload.get("in_context", True)
            items.append(
                {
                    "unit_id": record["uuid"] if record else raw_id,
                    "unit_short_id": raw_id,
                    "document_id": document_id,
                    "title": title,
                    "text": str(payload.get("text") or ""),
                    "in_context": bool(in_context),
                }
            )
    else:
        reports_records = records.get("reports")
        if reports_records is not None and len(reports_records):
            for row in reports_records.itertuples(index=False):
                payload = row._asdict() if hasattr(row, "_asdict") else dict(row._mapping)  # type: ignore[attr-defined]
                items.append(
                    {
                        "unit_id": f"report:{payload.get('id')}",
                        "unit_short_id": str(payload.get("id")),
                        "document_id": None,
                        "title": str(payload.get("title") or ""),
                        "text": str(payload.get("content") or payload.get("full_content") or ""),
                        "in_context": bool(payload.get("in_context", True)),
                    }
                )

    stats = {
        "n_items": len(items),
        "n_in_context": sum(1 for item in items if item["in_context"]),
        "record_keys": sorted(records.keys()),
        "records_counts": {key: int(len(value)) for key, value in records.items()},
        "context_builder_llm_calls": getattr(ctx, "llm_calls", None),
        "context_builder_prompt_tokens": getattr(ctx, "prompt_tokens", None),
        "context_text_chars": len(context_text),
    }

    print(json.dumps({"context_text": context_text, "items": items, "stats": stats}))


if __name__ == "__main__":
    main()
