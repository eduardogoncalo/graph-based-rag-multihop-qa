from benchmark.methods.lightrag.output_parser import parse_query_output


def test_lightrag_output_parser_converts_structured_contexts() -> None:
    result = parse_query_output(
        query="governing law",
        query_mode="mix",
        latency_ms=10.0,
        top_k=1,
        raw_response={
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
        },
    )

    assert result.method_id == "lightrag"
    assert result.query == "governing law"
    assert len(result.items) == 1
    assert result.items[0].source_chunk_id == "chunk_1"
    assert result.items[0].metadata["query_mode"] == "mix"


def test_lightrag_output_parser_handles_plain_text() -> None:
    result = parse_query_output(
        query="governing law",
        query_mode="mix",
        latency_ms=5.0,
        top_k=5,
        raw_response="Delaware law.",
    )

    assert result.items[0].text == "Delaware law."
    assert result.raw_response == "Delaware law."
