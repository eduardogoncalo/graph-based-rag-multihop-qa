from __future__ import annotations

METHOD_LABEL = "BenchmarkGraphRAGDocument"
CHUNK_LABEL = "BenchmarkGraphRAGChunk"
ENTITY_LABEL = "BenchmarkGraphRAGEntity"
RELATIONSHIP_TYPE = "BENCHMARK_GRAPHRAG_RELATION"
SOURCE_RELATIONSHIP_TYPE = "HAS_SOURCE_CHUNK"

METHOD_ID_PROPERTY = "method_id"
DATASET_ID_PROPERTY = "dataset_id"
DATASET_VERSION_PROPERTY = "dataset_version"

EXPECTED_METHOD_ID = "ms_graphrag_neo4j"

SCHEMA_DESCRIPTION = {
    "method_id": EXPECTED_METHOD_ID,
    "labels": [METHOD_LABEL, CHUNK_LABEL, ENTITY_LABEL],
    "relationship_types": [RELATIONSHIP_TYPE, SOURCE_RELATIONSHIP_TYPE],
    "required_properties": [
        METHOD_ID_PROPERTY,
        DATASET_ID_PROPERTY,
        DATASET_VERSION_PROPERTY,
    ],
}
