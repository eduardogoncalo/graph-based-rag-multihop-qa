from benchmark.storage.pgvector_store import (
    IVFFLAT_LISTS,
    IVFFLAT_PROBES,
    VECTOR_RAG_METHOD_ID,
    VECTOR_TABLE,
    PgVectorStore,
    VectorRecord,
    pgvector_index_sql,
    pgvector_schema_sql,
)


def test_pgvector_schema_is_method_isolated() -> None:
    sql = "\n".join(pgvector_schema_sql(16))

    assert VECTOR_TABLE in sql
    assert "embedding vector(16)" in sql
    assert "CHECK (method_id = 'vector_rag')" in sql
    # The ANN index must NOT be part of the pre-load schema: IVFFlat centroids
    # are sampled from existing rows, so the index is only valid post-load.
    assert "vector_cosine_ops" not in sql


def test_pgvector_index_is_created_separately_with_explicit_lists() -> None:
    sql = pgvector_index_sql()

    assert "vector_cosine_ops" in sql
    assert f"lists = {IVFFLAT_LISTS}" in sql
    assert IVFFLAT_PROBES == IVFFLAT_LISTS


def test_pgvector_store_rejects_non_vector_rag_records() -> None:
    store = PgVectorStore(FakeConnection(), dimension=4)
    record = VectorRecord(
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        method_id="lightrag_neo4j",
        chunk_id="chunk_1",
        document_id="doc_1",
        embedding_model="fake",
        embedding_dim=4,
        text="text",
        metadata={},
        embedding=[0.1, 0.2, 0.3, 0.4],
    )

    try:
        store.upsert_embeddings([record])
    except ValueError as exc:
        assert "vector_rag" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pgvector_store_search_returns_retrieval_result() -> None:
    connection = FakeConnection(
        rows=[
            (
                "chunk_1",
                "doc_1",
                "Governing law is Delaware.",
                {"method_id": VECTOR_RAG_METHOD_ID},
                0.91,
            )
        ]
    )
    store = PgVectorStore(connection, dimension=4)

    result = store.search(
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        query="law",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=1,
    )

    assert result.method_id == VECTOR_RAG_METHOD_ID
    assert result.items[0].source_chunk_id == "chunk_1"
    assert result.items[0].score == 0.91
    # Exact-search guarantee: probes must be raised to cover every IVFFlat list
    # before the similarity query runs.
    executed_sql = [sql for sql, _ in connection.cursor_obj.executed]
    assert any(f"SET ivfflat.probes = {IVFFLAT_PROBES}" in sql for sql in executed_sql)


class FakeConnection:
    def __init__(self, rows: list[tuple] | None = None) -> None:
        self.cursor_obj = FakeCursor(rows or [])
        self.commits = 0

    def cursor(self) -> "FakeCursor":
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1


class FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, object | None]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object | None = None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return self.rows
