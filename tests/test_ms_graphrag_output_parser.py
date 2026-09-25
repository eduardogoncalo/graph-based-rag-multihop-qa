import json

from benchmark.methods.ms_graphrag.output_parser import parse_query_output


def test_ms_graphrag_output_parser_converts_json_contexts() -> None:
    result = parse_query_output(
        query="governing law",
        query_method="global",
        latency_ms=10.0,
        top_k=1,
        stdout=json.dumps(
            {
                "answer": "Delaware law.",
                "contexts": [
                    {
                        "text": "Governing law is Delaware.",
                        "chunk_id": "chunk_1",
                        "document_id": "doc_1",
                        "score": 0.9,
                    },
                    {"text": "Extra context", "chunk_id": "chunk_2"},
                ],
            }
        ),
    )

    assert result.method_id == "ms_graphrag"
    assert result.query == "governing law"
    assert len(result.items) == 1
    assert result.items[0].source_chunk_id == "chunk_1"
    assert result.items[0].metadata["query_method"] == "global"


def test_ms_graphrag_output_parser_handles_plain_text() -> None:
    result = parse_query_output(
        query="governing law",
        query_method="local",
        latency_ms=5.0,
        top_k=5,
        stdout="Delaware law.",
    )

    assert result.items[0].text == "Delaware law."
    assert result.raw_response == "Delaware law."
