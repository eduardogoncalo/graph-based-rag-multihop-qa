from __future__ import annotations

from benchmark.core.ids import deterministic_id
from benchmark.core.schemas import Chunk, Document


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = document.text
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunk_id = deterministic_id("chunk", [document.document_id, start, end, chunk_text])
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                dataset_id=document.dataset_id,
                dataset_version=document.dataset_version,
                text=chunk_text,
                start_char=start,
                end_char=end,
                metadata={"chunk_index": len(chunks)},
            )
        )
        if end == len(text):
            break
        start = end - chunk_overlap

    return chunks
