from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.methods.cognee.parser import extract_document_id as parse_document_id


@dataclass(frozen=True)
class StoreCounts:
    postgres_table_count: int | None
    neo4j_node_count: int | None
    neo4j_relationship_count: int | None


def count_cognee_postgres_tables(
    *,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
) -> int:
    import psycopg

    with psycopg.connect(
        host=host,
        port=port,
        dbname=database,
        user=username,
        password=password,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                """
            )
            return int(cursor.fetchone()[0])


def count_neo4j_graph(
    *,
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
) -> tuple[int, int]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            return int(node_count), int(relationship_count)
    finally:
        driver.close()


def count_cognee_postgres_tables_by_name(
    *,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
) -> dict[str, int | str]:
    import psycopg

    counts: dict[str, int | str] = {}
    with psycopg.connect(
        host=host,
        port=port,
        dbname=database,
        user=username,
        password=password,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name
                """
            )
            tables = [(str(schema), str(table)) for schema, table in cursor.fetchall()]
            for schema, table in tables:
                quoted_schema = '"' + schema.replace('"', '""') + '"'
                quoted_table = '"' + table.replace('"', '""') + '"'
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {quoted_schema}.{quoted_table}")
                    counts[f"{schema}.{table}"] = int(cursor.fetchone()[0])
                except Exception as exc:  # pragma: no cover - defensive live audit path
                    counts[f"{schema}.{table}"] = f"unavailable: {exc}"
    return counts


def count_cognee_vector_tables(
    *,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
) -> dict[str, int | str]:
    import psycopg

    counts: dict[str, int | str] = {}
    with psycopg.connect(
        host=host,
        port=port,
        dbname=database,
        user=username,
        password=password,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.columns
                WHERE udt_name = 'vector'
                ORDER BY table_schema, table_name
                """
            )
            for schema, table in [(str(schema), str(table)) for schema, table in cursor.fetchall()]:
                quoted_schema = '"' + schema.replace('"', '""') + '"'
                quoted_table = '"' + table.replace('"', '""') + '"'
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {quoted_schema}.{quoted_table}")
                    counts[f"{schema}.{table}"] = int(cursor.fetchone()[0])
                except Exception as exc:  # pragma: no cover - defensive live audit path
                    counts[f"{schema}.{table}"] = f"unavailable: {exc}"
    return counts


def audit_neo4j_schema(
    *,
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str = "neo4j",
    driver: Any | None = None,
) -> list[dict[str, Any]]:
    close_driver = False
    if driver is None:
        from neo4j import GraphDatabase

        if uri is None or username is None or password is None:
            raise ValueError("uri, username, and password are required when driver is not provided")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        close_driver = True

    try:
        with driver.session(database=database) as session:
            rows: list[dict[str, Any]] = []
            labels = _single_list(session, "CALL db.labels() YIELD label RETURN collect(label) AS values")
            rel_types = _single_list(
                session,
                "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS values",
            )
            rows.append(
                {
                    "section": "schema",
                    "name": "labels",
                    "count": len(labels),
                    "detail": json.dumps(labels, sort_keys=True),
                }
            )
            rows.append(
                {
                    "section": "schema",
                    "name": "relationship_types",
                    "count": len(rel_types),
                    "detail": json.dumps(rel_types, sort_keys=True),
                }
            )
            for record in session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label AS name, count(*) AS count
                ORDER BY count DESC, name
                """
            ):
                rows.append(
                    {
                        "section": "node_label_count",
                        "name": record["name"],
                        "count": int(record["count"]),
                        "detail": "",
                    }
                )
            for record in session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS name, count(*) AS count
                ORDER BY count DESC, name
                """
            ):
                rows.append(
                    {
                        "section": "relationship_type_count",
                        "name": record["name"],
                        "count": int(record["count"]),
                        "detail": "",
                    }
                )
            for record in session.run(
                """
                MATCH (n)
                RETURN labels(n) AS labels, keys(n) AS property_keys, properties(n) AS properties
                LIMIT 20
                """
            ):
                properties = dict(record.get("properties") or {})
                rows.append(
                    {
                        "section": "node_property_sample",
                        "name": ":".join(record.get("labels") or []),
                        "count": "",
                        "detail": json.dumps(
                            {
                                "property_keys": record.get("property_keys") or [],
                                "properties": _sample_properties(properties),
                            },
                            sort_keys=True,
                            ensure_ascii=True,
                        ),
                    }
                )
            mapping_rows = audit_neo4j_document_mapping(session=session)
            rows.extend(mapping_rows)
            return rows
    finally:
        if close_driver:
            driver.close()


def audit_neo4j_document_mapping(*, session: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    property_names: set[str] = set()
    for record in session.run(
        """
        MATCH (n)
        UNWIND keys(n) AS property_key
        WITH DISTINCT property_key
        WHERE toLower(property_key) CONTAINS 'document'
           OR toLower(property_key) CONTAINS 'chunk'
           OR property_key = 'source_content_hash'
        RETURN property_key AS name
        ORDER BY name
        """
    ):
        property_names.add(str(record["name"]))
    rows.append(
        {
            "section": "document_mapping",
            "name": "candidate_properties",
            "count": len(property_names),
            "detail": json.dumps(sorted(property_names), sort_keys=True),
        }
    )

    document_ids = neo4j_indexed_document_ids(session=session)
    rows.append(
        {
            "section": "document_mapping",
            "name": "document_ids_extracted_from_text",
            "count": len(document_ids),
            "detail": json.dumps(sorted(document_ids), sort_keys=True),
        }
    )

    for record in session.run(
        """
        MATCH (n:DocumentChunk)
        RETURN n.source_content_hash AS source_content_hash,
               n.chunk_index AS chunk_index,
               n.text AS text
        ORDER BY source_content_hash, chunk_index
        LIMIT 50
        """
    ):
        text = str(record.get("text") or "")
        rows.append(
            {
                "section": "document_mapping_sample",
                "name": extract_document_id(text) or "NO_DOCUMENT_ID_IN_TEXT",
                "count": "",
                "detail": json.dumps(
                    {
                        "source_content_hash": record.get("source_content_hash"),
                        "chunk_index": record.get("chunk_index"),
                        "text_prefix": text[:160],
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            }
        )
    return rows


def neo4j_indexed_document_ids(
    *,
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str = "neo4j",
    driver: Any | None = None,
    session: Any | None = None,
) -> set[str]:
    if session is not None:
        return _neo4j_indexed_document_ids_from_session(session)

    close_driver = False
    if driver is None:
        from neo4j import GraphDatabase

        if uri is None or username is None or password is None:
            raise ValueError("uri, username, and password are required when driver is not provided")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        close_driver = True
    try:
        with driver.session(database=database) as resolved_session:
            return _neo4j_indexed_document_ids_from_session(resolved_session)
    finally:
        if close_driver:
            driver.close()


def generate_neo4j_schema_audit_report(
    *,
    rows: list[dict[str, Any]],
    csv_path: Path,
    report_path: Path,
) -> None:
    _write_csv(csv_path, rows)
    labels = [row for row in rows if row["section"] == "node_label_count"]
    rels = [row for row in rows if row["section"] == "relationship_type_count"]
    mappings = [row for row in rows if row["section"] == "document_mapping"]
    lines = [
        "# Cognee Neo4j Schema Audit",
        "",
        f"Generated at: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Counts",
        "",
        "| Kind | Name | Count |",
        "|---|---:|---:|",
    ]
    for row in labels:
        lines.append(f"| node label | `{row['name']}` | `{row['count']}` |")
    for row in rels:
        lines.append(f"| relationship type | `{row['name']}` | `{row['count']}` |")
    lines.extend(["", "## Document Mapping", ""])
    for row in mappings:
        lines.append(f"- `{row['name']}`: `{row['count']}`; `{row['detail']}`")
    lines.extend(
        [
            "",
            "The current Cognee Neo4j graph does not expose a dedicated `document_id` or `chunk_id` property on sampled nodes. "
            "Canonical document mapping is recoverable from the `DocumentChunk.text` prefix and from `source_content_hash`.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_document_id(text: str) -> str | None:
    return parse_document_id(text)


def load_cognee_chunk_mappings_from_postgres(
    *,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
) -> list[dict[str, Any]]:
    import psycopg

    with psycopg.connect(
        host=host,
        port=port,
        dbname=database,
        user=username,
        password=password,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT n.slug::text AS source_cognee_chunk_id,
                       n.data_id::text AS data_id,
                       n.dataset_id::text AS dataset_id,
                       d.name AS source_dataset_name,
                       n.attributes AS attributes
                FROM public.nodes n
                LEFT JOIN public.datasets d ON d.id = n.dataset_id
                WHERE n.type = 'DocumentChunk'
                ORDER BY d.name, n.created_at, n.slug::text
                """
            )
            rows = []
            for record in cursor.fetchall():
                attributes = record[4] or {}
                if isinstance(attributes, str):
                    try:
                        attributes = json.loads(attributes)
                    except json.JSONDecodeError:
                        attributes = {}
                text = str(attributes.get("text") or "")
                source_content_hash = _optional_str(attributes.get("source_content_hash"))
                rows.append(
                    {
                        "source_cognee_chunk_id": _optional_str(record[0]) or _optional_str(attributes.get("id")),
                        "data_id": _optional_str(record[1]),
                        "dataset_id": _optional_str(record[2]),
                        "source_dataset_name": _optional_str(record[3]),
                        "source_content_hash": source_content_hash,
                        "chunk_index": attributes.get("chunk_index"),
                        "text": text,
                        "source_document_id": extract_document_id(text),
                    }
                )
    document_by_hash = {
        row["source_content_hash"]: row["source_document_id"]
        for row in rows
        if row.get("source_content_hash") and row.get("source_document_id")
    }
    for row in rows:
        if not row.get("source_document_id"):
            row["source_document_id"] = document_by_hash.get(row.get("source_content_hash"))
    return rows


def audit_cognee_mapping_rows(
    *,
    chunk_mappings: list[dict[str, Any]],
    retrieval_items: list[Any] | None = None,
) -> list[dict[str, Any]]:
    retrieval_items = retrieval_items or []
    total_chunks = len(chunk_mappings)
    doc_recoverable = sum(1 for row in chunk_mappings if row.get("source_document_id"))
    canonical_chunk_ids = sum(1 for row in chunk_mappings if row.get("source_chunk_id"))
    cognee_chunk_ids = sum(1 for row in chunk_mappings if row.get("source_cognee_chunk_id"))
    deterministic_fallbacks = total_chunks - canonical_chunk_ids
    retrieval_items_with_document_id = sum(
        1 for item in retrieval_items if getattr(item, "source_document_id", None)
    )
    retrieval_items_with_chunk_id = sum(1 for item in retrieval_items if getattr(item, "source_chunk_id", None))
    source_document_ready = total_chunks > 0 and doc_recoverable == total_chunks
    retrieval_ready = not retrieval_items or retrieval_items_with_document_id == len(retrieval_items)
    mapping_status = "OK" if source_document_ready and retrieval_ready and deterministic_fallbacks == total_chunks else "FAIL"
    if mapping_status == "FAIL" and source_document_ready:
        mapping_status = "PARTIAL"
    rows = [
        _mapping_row("summary", "mapping_status", mapping_status, "OK requires document ids plus explicit chunk strategy"),
        _mapping_row("chunk_mapping", "document_chunks_total", total_chunks, ""),
        _mapping_row("chunk_mapping", "document_id_recoverable", doc_recoverable, ""),
        _mapping_row("chunk_mapping", "canonical_chunk_id_recoverable", canonical_chunk_ids, ""),
        _mapping_row("chunk_mapping", "source_cognee_chunk_id_used", cognee_chunk_ids, ""),
        _mapping_row("chunk_mapping", "deterministic_fallback_used", deterministic_fallbacks, ""),
        _mapping_row("retrieval_items", "retrieval_items_total", len(retrieval_items), ""),
        _mapping_row("retrieval_items", "source_document_id_populated", retrieval_items_with_document_id, ""),
        _mapping_row("retrieval_items", "source_chunk_id_populated", retrieval_items_with_chunk_id, ""),
        _mapping_row("readiness", "source_document_id_ready_for_benchmark", source_document_ready, ""),
        _mapping_row(
            "readiness",
            "source_chunk_id_strategy",
            "deterministic_fallback",
            "No canonical chunk_id is exposed by Cognee; source_chunk_id is a stable Cognee-derived fallback.",
        ),
        _mapping_row(
            "risk",
            "methodological_risk",
            "medium",
            "Chunk ids are not canonical; document ids are recovered through Cognee chunk text/hash mapping.",
        ),
    ]
    for row in chunk_mappings:
        rows.append(
            _mapping_row(
                "chunk_sample",
                str(row.get("source_cognee_chunk_id") or ""),
                row.get("source_document_id") or "",
                json.dumps(
                    {
                        "source_dataset_name": row.get("source_dataset_name"),
                        "source_content_hash": row.get("source_content_hash"),
                        "chunk_index": row.get("chunk_index"),
                        "text_prefix": str(row.get("text") or "")[:160],
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            )
        )
    return rows


def generate_cognee_mapping_audit_report(
    *,
    rows: list[dict[str, Any]],
    csv_path: Path,
    report_path: Path,
) -> None:
    _write_csv(csv_path, rows)
    lookup = {(row["section"], row["name"]): row for row in rows}

    def value(section: str, name: str) -> Any:
        row = lookup.get((section, name))
        return "" if row is None else row["count"]

    lines = [
        "# Cognee Mapping Audit",
        "",
        f"Generated at: `{datetime.now(UTC).isoformat()}`",
        "",
        f"- `mapping_status`: `{value('summary', 'mapping_status')}`",
        f"- DocumentChunks: `{value('chunk_mapping', 'document_chunks_total')}`",
        f"- DocumentChunks with recoverable `document_id`: `{value('chunk_mapping', 'document_id_recoverable')}`",
        f"- DocumentChunks with canonical `chunk_id`: `{value('chunk_mapping', 'canonical_chunk_id_recoverable')}`",
        f"- DocumentChunks with `source_cognee_chunk_id`: `{value('chunk_mapping', 'source_cognee_chunk_id_used')}`",
        f"- DocumentChunks using deterministic fallback: `{value('chunk_mapping', 'deterministic_fallback_used')}`",
        f"- Retrieval items audited: `{value('retrieval_items', 'retrieval_items_total')}`",
        f"- Retrieval items with `source_document_id`: `{value('retrieval_items', 'source_document_id_populated')}`",
        f"- Retrieval items with `source_chunk_id`: `{value('retrieval_items', 'source_chunk_id_populated')}`",
        "",
        "## Interpretation",
        "",
        "Cognee does not expose canonical `chunk_id` values. The benchmark adapter now keeps `source_cognee_chunk_id` separately and sets `source_chunk_id_strategy=deterministic_fallback` when a canonical chunk id is unavailable.",
        "The fallback `source_chunk_id` has the form `cognee::<dataset_name>::<source_cognee_chunk_id_or_hash>` and must not be treated as a canonical chunk id.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def has_wrong_store_reference(value: Any) -> bool:
    text = str(value)
    return (
        "bolt://localhost:7688" in text
        or "bolt://localhost:7687" in text
        or "vector_rag_chunk_embeddings" in text
        or "/lightrag/" in text
        or "/graphrag/" in text
    )


def _mapping_row(section: str, name: str, count: Any, detail: str) -> dict[str, Any]:
    return {"section": section, "name": name, "count": count, "detail": detail}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _neo4j_indexed_document_ids_from_session(session: Any) -> set[str]:
    document_ids: set[str] = set()
    for record in session.run(
        """
        MATCH (n:DocumentChunk)
        RETURN n.text AS text
        """
    ):
        document_id = extract_document_id(str(record.get("text") or ""))
        if document_id:
            document_ids.add(document_id)
    return document_ids


def _single_list(session: Any, query: str) -> list[Any]:
    record = session.run(query).single()
    if record is None:
        return []
    return list(record.get("values") or [])


def _sample_properties(properties: dict[str, Any]) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for key in sorted(properties)[:12]:
        value = properties[key]
        if isinstance(value, str) and len(value) > 180:
            value = value[:180]
        sampled[key] = value
    return sampled


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "name", "count", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
