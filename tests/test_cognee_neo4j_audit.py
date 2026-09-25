from __future__ import annotations

from benchmark.core.schemas import RetrievedItem
from benchmark.methods.cognee.audit import audit_cognee_mapping_rows, audit_neo4j_schema, extract_document_id


def test_neo4j_audit_extracts_labels_types_and_document_mapping_with_fake_driver() -> None:
    rows = audit_neo4j_schema(driver=FakeDriver())

    label_rows = [row for row in rows if row["section"] == "node_label_count"]
    relationship_rows = [row for row in rows if row["section"] == "relationship_type_count"]
    mapping_rows = [row for row in rows if row["section"] == "document_mapping"]

    assert {"section": "node_label_count", "name": "DocumentChunk", "count": 2, "detail": ""} in label_rows
    assert {
        "section": "relationship_type_count",
        "name": "contains",
        "count": 1,
        "detail": "",
    } in relationship_rows
    assert any(row["name"] == "document_ids_extracted_from_text" and "doc_abc123" in row["detail"] for row in mapping_rows)


def test_extract_document_id_accepts_colon_equals_and_bare_patterns() -> None:
    assert extract_document_id("document_id: doc_abc123") == "doc_abc123"
    assert extract_document_id("document_id=doc_def456") == "doc_def456"
    assert extract_document_id("prefix doc_feed01 suffix") == "doc_feed01"


def test_mapping_audit_ok_with_document_ids_and_explicit_fallback() -> None:
    rows = audit_cognee_mapping_rows(
        chunk_mappings=[
            {
                "source_document_id": "doc_abc123",
                "source_cognee_chunk_id": "chunk_1",
                "source_dataset_name": "dataset_1",
                "source_content_hash": "hash_1",
                "text": "document_id: doc_abc123",
            }
        ],
        retrieval_items=[
            RetrievedItem(
                item_id="chunk_1",
                text="document_id: doc_abc123",
                source_document_id="doc_abc123",
                source_chunk_id="cognee::dataset_1::chunk_1",
                metadata={"source_chunk_id_strategy": "deterministic_fallback"},
            )
        ],
    )

    assert _row(rows, "summary", "mapping_status")["count"] == "OK"
    assert _row(rows, "chunk_mapping", "document_id_recoverable")["count"] == 1


def _row(rows, section, name):
    return next(row for row in rows if row["section"] == section and row["name"] == name)


class FakeDriver:
    def session(self, database: str):
        return FakeSession()

    def close(self) -> None:
        pass


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def run(self, query: str):
        if "CALL db.labels" in query:
            return FakeResult([{"values": ["DocumentChunk", "Entity"]}])
        if "CALL db.relationshipTypes" in query:
            return FakeResult([{"values": ["contains"]}])
        if "UNWIND labels(n)" in query:
            return FakeResult([{"name": "DocumentChunk", "count": 2}, {"name": "Entity", "count": 1}])
        if "RETURN type(r) AS name" in query:
            return FakeResult([{"name": "contains", "count": 1}])
        if "RETURN labels(n) AS labels" in query:
            return FakeResult(
                [
                    {
                        "labels": ["DocumentChunk"],
                        "property_keys": ["text", "source_content_hash"],
                        "properties": {
                            "text": "document_id: doc_abc123\ntext",
                            "source_content_hash": "hash_a",
                        },
                    }
                ]
            )
        if "UNWIND keys(n)" in query:
            return FakeResult([{"name": "source_content_hash"}])
        if "MATCH (n:DocumentChunk)" in query and "source_content_hash" in query:
            return FakeResult(
                [
                    {
                        "source_content_hash": "hash_a",
                        "chunk_index": 0,
                        "text": "document_id: doc_abc123\nbody",
                    }
                ]
            )
        if "MATCH (n:DocumentChunk)" in query and "RETURN n.text AS text" in query:
            return FakeResult([{"text": "document_id: doc_abc123\nbody"}])
        return FakeResult([])


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None
