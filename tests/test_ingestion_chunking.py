import pytest

from benchmark.core.schemas import Document
from benchmark.ingestion.chunking import chunk_document


def test_chunk_document_generates_ordered_offsets() -> None:
    document = Document(
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="abcdefghij",
    )

    chunks = chunk_document(document, chunk_size=4, chunk_overlap=1)

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [(0, 4), (3, 7), (6, 10)]
    assert all(chunk.dataset_id == "musique_smoke_20" for chunk in chunks)


def test_chunk_document_rejects_invalid_overlap() -> None:
    document = Document(
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="abcdefghij",
    )

    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(document, chunk_size=4, chunk_overlap=4)
