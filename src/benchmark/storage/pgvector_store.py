from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from benchmark.core.schemas import RetrievalResult, RetrievedItem

VECTOR_RAG_METHOD_ID = "vector_rag"
VECTOR_TABLE = "vector_rag_chunk_embeddings"

# IVFFlat is approximate: with the default probes=1 the search inspects a single
# list (~1% of the corpus), and the lists are meaningless when the index was
# built before the data was loaded. probes >= lists forces every list to be
# scanned, making index results identical to an exact scan, so both constants
# must stay equal.
IVFFLAT_LISTS = 100
IVFFLAT_PROBES = IVFFLAT_LISTS


@dataclass(frozen=True)
class VectorRecord:
    dataset_id: str
    dataset_version: str
    method_id: str
    chunk_id: str
    document_id: str
    embedding_model: str
    embedding_dim: int
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


class PgVectorStore:
    def __init__(self, connection: Any, *, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.connection = connection
        self.dimension = dimension

    def schema_sql(self) -> list[str]:
        return pgvector_schema_sql(self.dimension)

    def ensure_schema(self) -> None:
        with self.connection.cursor() as cursor:
            for statement in self.schema_sql():
                cursor.execute(statement)
        self.connection.commit()

    def index_sql(self) -> str:
        return pgvector_index_sql()

    def ensure_index(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(self.index_sql())
        self.connection.commit()

    def upsert_embeddings(self, records: list[VectorRecord]) -> int:
        if any(record.method_id != VECTOR_RAG_METHOD_ID for record in records):
            raise ValueError("PgVectorStore only accepts method_id='vector_rag'")

        sql = f"""
        INSERT INTO {VECTOR_TABLE} (
            dataset_id,
            dataset_version,
            method_id,
            chunk_id,
            document_id,
            embedding_model,
            embedding_dim,
            text,
            metadata,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
        ON CONFLICT (dataset_id, dataset_version, method_id, chunk_id)
        DO UPDATE SET
            document_id = EXCLUDED.document_id,
            embedding_model = EXCLUDED.embedding_model,
            embedding_dim = EXCLUDED.embedding_dim,
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding
        """
        with self.connection.cursor() as cursor:
            for record in records:
                cursor.execute(
                    sql,
                    (
                        record.dataset_id,
                        record.dataset_version,
                        record.method_id,
                        record.chunk_id,
                        record.document_id,
                        record.embedding_model,
                        record.embedding_dim,
                        record.text,
                        json.dumps(record.metadata, sort_keys=True),
                        _vector_literal(record.embedding),
                    ),
                )
        self.connection.commit()
        return len(records)

    def search(
        self,
        *,
        dataset_id: str,
        dataset_version: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> RetrievalResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        started = time.perf_counter()
        sql = f"""
        SELECT
            chunk_id,
            document_id,
            text,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM {VECTOR_TABLE}
        WHERE dataset_id = %s
          AND dataset_version = %s
          AND method_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        vector = _vector_literal(query_embedding)
        with self.connection.cursor() as cursor:
            cursor.execute(f"SET ivfflat.probes = {int(IVFFLAT_PROBES)}")
            cursor.execute(
                sql,
                (vector, dataset_id, dataset_version, VECTOR_RAG_METHOD_ID, vector, top_k),
            )
            rows = cursor.fetchall()

        items = [
            RetrievedItem(
                item_id=f"vector_rag:{row[0]}",
                text=row[2],
                score=float(row[4]) if row[4] is not None else None,
                source_document_id=row[1],
                source_chunk_id=row[0],
                metadata=dict(row[3] or {}),
            )
            for row in rows
        ]
        latency_ms = (time.perf_counter() - started) * 1000
        return RetrievalResult(
            method_id=VECTOR_RAG_METHOD_ID,
            query=query,
            items=items,
            latency_ms=latency_ms,
        )


def pgvector_schema_sql(dimension: int) -> list[str]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {VECTOR_TABLE} (
            dataset_id text NOT NULL,
            dataset_version text NOT NULL,
            method_id text NOT NULL CHECK (method_id = 'vector_rag'),
            chunk_id text NOT NULL,
            document_id text NOT NULL,
            embedding_model text NOT NULL,
            embedding_dim integer NOT NULL,
            text text NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({dimension}) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (dataset_id, dataset_version, method_id, chunk_id)
        )
        """,
    ]


def pgvector_index_sql() -> str:
    # The index must only be created AFTER the embeddings are loaded: IVFFlat
    # samples the existing rows to place its list centroids, so an index built
    # on an empty table clusters nothing and degrades search to near-random.
    return f"""
        CREATE INDEX IF NOT EXISTS {VECTOR_TABLE}_cosine_idx
        ON {VECTOR_TABLE}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {int(IVFFLAT_LISTS)})
        """


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"
