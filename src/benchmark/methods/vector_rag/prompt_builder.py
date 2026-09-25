from __future__ import annotations

from benchmark.core.schemas import RetrievedItem


def build_retrieval_prompt(question: str, retrieved_items: list[RetrievedItem]) -> str:
    context_blocks = []
    for index, item in enumerate(retrieved_items, start=1):
        citation = item.source_chunk_id or item.item_id
        context_blocks.append(f"[{index}] chunk_id={citation}\n{item.text}")

    context = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."
    return (
        "Answer the question using only the retrieved context. "
        "Cite chunk IDs for any factual claims.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )
