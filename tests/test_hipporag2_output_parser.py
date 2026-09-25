from benchmark.methods.hipporag2.output_parser import parse_query_payload


def _payload() -> dict:
    return {
        "query": "What county is Erik Hort's birthplace a part of?",
        "items": [
            {"rank": 1, "text": "Erik Hort was born in Montebello.", "score": 0.055,
             "document_id": "doc_0a1b2c3d4e5f6789"},
            {"rank": 2, "text": "Montebello is a village in Rockland County.", "score": 0.024,
             "document_id": "doc_ffee00112233aabb"},
            {"rank": 3, "text": "Passage without mapping.", "score": 0.011,
             "document_id": None},
        ],
        "stats": {"n_docs_returned": 3},
    }


def test_parse_populates_source_document_id() -> None:
    result = parse_query_payload(
        query="q", payload=_payload(), latency_ms=12.5, top_k=5
    )
    assert result.method_id == "hipporag2"
    assert [item.source_document_id for item in result.items] == [
        "doc_0a1b2c3d4e5f6789",
        "doc_ffee00112233aabb",
        None,
    ]
    assert result.items[0].item_id == "hipporag2:ppr:1"
    assert result.items[0].score == 0.055
    # evidence-recall lê também metadata.document_ids
    assert result.items[0].metadata["document_ids"] == ["doc_0a1b2c3d4e5f6789"]
    assert result.metadata["unmapped_items"] == 1
    assert result.metadata["n_docs_returned"] == 3


def test_parse_respects_top_k() -> None:
    result = parse_query_payload(query="q", payload=_payload(), latency_ms=1.0, top_k=2)
    assert len(result.items) == 2


def test_parse_doc_id_regex_fallback() -> None:
    payload = {
        "items": [
            {"rank": 1, "text": "ver doc_00aa11bb22cc33dd para detalhes", "score": 1.0,
             "document_id": None}
        ]
    }
    result = parse_query_payload(query="q", payload=payload, latency_ms=1.0, top_k=5)
    assert result.items[0].source_document_id == "doc_00aa11bb22cc33dd"
    assert result.metadata["unmapped_items"] == 0
