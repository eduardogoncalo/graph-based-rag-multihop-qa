"""Reference baselines: closed-book floor and gold-context (oracle) ceiling.

zero_shot_no_context — the reader answers from parametric knowledge alone
(no retrieval). single_document_context — the reader receives the question's
gold documents as context, isolating reading quality from retrieval quality.
"""

from __future__ import annotations

import time
from typing import Any

from benchmark.core.schemas import RetrievalResult, RetrievedItem

ZERO_SHOT_METHOD_ID = "zero_shot_no_context"
ORACLE_METHOD_ID = "single_document_context"


def retrieve_zero_shot(*, query: str) -> RetrievalResult:
    return RetrievalResult(
        method_id=ZERO_SHOT_METHOD_ID,
        query=query,
        items=[],
        latency_ms=0.0,
        metadata={"baseline": "closed_book"},
    )


def retrieve_gold_documents(
    *,
    connection: Any,
    query: str,
    question_id: str,
    dataset_id: str,
    dataset_version: str,
) -> RetrievalResult:
    started = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT d.document_id, d.title, d.text
            FROM gold_evidence g
            JOIN documents d ON d.document_id = g.document_id
            WHERE g.question_id = %s
              AND d.dataset_id = %s
              AND d.dataset_version = %s
            ORDER BY d.document_id
            """,
            (question_id, dataset_id, dataset_version),
        )
        rows = cursor.fetchall()
    if not rows:
        raise ValueError(f"No gold documents found for question: {question_id}")
    items = [
        RetrievedItem(
            item_id=str(document_id),
            text=f"{title}\n{text}" if title else str(text),
            score=1.0,
            source_document_id=str(document_id),
            source_chunk_id=str(document_id),
            metadata={"document_ids": [str(document_id)], "gold": True},
        )
        for document_id, title, text in rows
    ]
    return RetrievalResult(
        method_id=ORACLE_METHOD_ID,
        query=query,
        items=items,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        metadata={"baseline": "oracle_gold_context"},
    )
