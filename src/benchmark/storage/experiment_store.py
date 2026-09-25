from __future__ import annotations

import json
from typing import Any

from benchmark.core.ids import deterministic_id
from benchmark.core.schemas import RetrievalResult, RetrievalTrace, RetrievedItem
from benchmark.evaluation.evaluator import EvaluationInput


class ExperimentStore:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def create_experiment(
        self,
        *,
        experiment_id: str,
        dataset_id: str,
        dataset_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO experiments (experiment_id, dataset_id, dataset_version, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (experiment_id)
                DO UPDATE SET metadata = EXCLUDED.metadata, updated_at = now()
                """,
                (experiment_id, dataset_id, dataset_version, _json(metadata or {})),
            )
        self.connection.commit()
        return experiment_id

    def create_run(
        self,
        *,
        experiment_id: str,
        dataset_id: str,
        dataset_version: str,
        method_id: str,
        agent_mode: str,
        run_id: str | None = None,
        status: str = "created",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        resolved_run_id = run_id or deterministic_id(
            "run",
            [experiment_id, dataset_id, dataset_version, method_id, agent_mode, metadata or {}],
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO runs (
                    run_id,
                    experiment_id,
                    dataset_id,
                    dataset_version,
                    method_id,
                    agent_mode,
                    status,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (run_id)
                DO UPDATE SET status = EXCLUDED.status, metadata = EXCLUDED.metadata
                """,
                (
                    resolved_run_id,
                    experiment_id,
                    dataset_id,
                    dataset_version,
                    method_id,
                    agent_mode,
                    status,
                    _json(metadata or {}),
                ),
            )
        self.connection.commit()
        return resolved_run_id

    def persist_retrieval_result(
        self,
        *,
        result: RetrievalResult,
        dataset_id: str,
        dataset_version: str,
        top_k: int,
        run_id: str | None = None,
        retrieval_result_id: str | None = None,
    ) -> str:
        resolved_id = retrieval_result_id or deterministic_id(
            "retrieval",
            [run_id, dataset_id, dataset_version, result.method_id, result.query, top_k],
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO retrieval_results (
                    retrieval_result_id,
                    run_id,
                    dataset_id,
                    dataset_version,
                    method_id,
                    query,
                    latency_ms,
                    top_k,
                    raw_response
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (retrieval_result_id)
                DO UPDATE SET
                    latency_ms = EXCLUDED.latency_ms,
                    raw_response = EXCLUDED.raw_response
                """,
                (
                    resolved_id,
                    run_id,
                    dataset_id,
                    dataset_version,
                    result.method_id,
                    result.query,
                    result.latency_ms,
                    top_k,
                    _json(result.raw_response),
                ),
            )
            for rank, item in enumerate(result.items, start=1):
                item_id = deterministic_id("retrieval_item", [resolved_id, rank, item.item_id])
                cursor.execute(
                    """
                    INSERT INTO retrieval_items (
                        retrieval_item_id,
                        retrieval_result_id,
                        rank,
                        item_id,
                        source_document_id,
                        source_chunk_id,
                        score,
                        text,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (retrieval_item_id)
                    DO UPDATE SET score = EXCLUDED.score, text = EXCLUDED.text
                    """,
                    (
                        item_id,
                        resolved_id,
                        rank,
                        item.item_id,
                        item.source_document_id,
                        item.source_chunk_id,
                        item.score,
                        item.text,
                        _json(item.metadata),
                    ),
                )
        self.connection.commit()
        return resolved_id

    def persist_retrieval_trace(
        self,
        *,
        trace: RetrievalTrace,
    ) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO retrieval_traces (
                    trace_id,
                    run_id,
                    trace_status,
                    trace,
                    raw_trace,
                    metadata
                )
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    trace_status = EXCLUDED.trace_status,
                    trace = EXCLUDED.trace,
                    raw_trace = EXCLUDED.raw_trace,
                    metadata = EXCLUDED.metadata
                """,
                (
                    trace.trace_id,
                    trace.run_id,
                    trace.trace_status,
                    _json(trace.trace),
                    _json(trace.raw_trace),
                    _json(trace.metadata),
                ),
            )
        self.connection.commit()
        return trace.trace_id

    def persist_answer(
        self,
        *,
        run_id: str,
        method_id: str,
        agent_mode: str,
        answer_text: str,
        question_id: str | None = None,
        citations: list[str] | None = None,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost: float | None = None,
        metadata: dict[str, Any] | None = None,
        answer_id: str | None = None,
    ) -> str:
        resolved_id = answer_id or deterministic_id(
            "answer",
            [run_id, question_id, method_id, agent_mode, answer_text],
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO answers (
                    answer_id,
                    run_id,
                    question_id,
                    method_id,
                    agent_mode,
                    answer_text,
                    citations,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (answer_id)
                DO UPDATE SET answer_text = EXCLUDED.answer_text, metadata = EXCLUDED.metadata
                """,
                (
                    resolved_id,
                    run_id,
                    question_id,
                    method_id,
                    agent_mode,
                    answer_text,
                    _json(citations or []),
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost,
                    _json(metadata or {}),
                ),
            )
        self.connection.commit()
        return resolved_id

    def persist_agent_message(
        self,
        *,
        run_id: str,
        agent_name: str,
        role: str,
        content: str,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> str:
        resolved_id = message_id or deterministic_id("message", [run_id, agent_name, role, content])
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_messages (
                    message_id, run_id, agent_name, role, content, token_count, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (message_id)
                DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata
                """,
                (
                    resolved_id,
                    run_id,
                    agent_name,
                    role,
                    content,
                    token_count,
                    _json(metadata or {}),
                ),
            )
        self.connection.commit()
        return resolved_id

    def persist_evaluation_result(
        self,
        *,
        run_id: str,
        metric_name: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
        evaluation_result_id: str | None = None,
    ) -> str:
        resolved_id = evaluation_result_id or deterministic_id(
            "eval",
            [run_id, metric_name, metric_value, metadata or {}],
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO evaluation_results (
                    evaluation_result_id, run_id, metric_name, metric_value, metadata
                )
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (evaluation_result_id)
                DO UPDATE SET metric_value = EXCLUDED.metric_value, metadata = EXCLUDED.metadata
                """,
                (resolved_id, run_id, metric_name, metric_value, _json(metadata or {})),
            )
        self.connection.commit()
        return resolved_id

    def load_evaluation_input(self, *, run_id: str, k: int = 5) -> EvaluationInput:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    a.answer_text,
                    a.question_id,
                    a.citations,
                    a.latency_ms,
                    a.prompt_tokens,
                    a.completion_tokens,
                    a.total_tokens,
                    a.estimated_cost,
                    q.gold_answer,
                    q.gold_evidence,
                    q.metadata
                FROM answers a
                LEFT JOIN questions q ON q.question_id = a.question_id
                WHERE a.run_id = %s
                ORDER BY a.created_at DESC
                LIMIT 1
                """,
                (run_id,),
            )
            answer_row = cursor.fetchone()
            if not answer_row:
                raise ValueError(f"No answer found for run_id={run_id}")

            # Document-level support set for retrieval recall/precision: the
            # canonical ``doc_*`` ids behind this question's gold evidence.
            # Retrieved items carry ``source_document_id=doc_*`` so this is the
            # namespace that actually intersects. ``q.gold_evidence`` (ev_* ids)
            # is kept only for citation/attribution.
            cursor.execute(
                """
                SELECT DISTINCT document_id
                FROM gold_evidence
                WHERE question_id = %s AND document_id IS NOT NULL
                """,
                (answer_row[1],),
            )
            gold_document_ids = [row[0] for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT retrieval_result_id, latency_ms
                FROM retrieval_results
                WHERE run_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id,),
            )
            retrieval_row = cursor.fetchone()
            retrieval_result_id = retrieval_row[0] if retrieval_row else None
            retrieval_latency = float(retrieval_row[1]) if retrieval_row else 0.0

            items: list[RetrievedItem] = []
            if retrieval_result_id is not None:
                cursor.execute(
                    """
                    SELECT
                        item_id,
                        text,
                        score,
                        source_document_id,
                        source_chunk_id,
                        metadata
                    FROM retrieval_items
                    WHERE retrieval_result_id = %s
                    ORDER BY rank ASC
                    LIMIT %s
                    """,
                    (retrieval_result_id, k),
                )
                items = [
                    RetrievedItem(
                        item_id=row[0],
                        text=row[1],
                        score=row[2],
                        source_document_id=row[3],
                        source_chunk_id=row[4],
                        metadata=row[5] or {},
                    )
                    for row in cursor.fetchall()
                ]

        return EvaluationInput(
            run_id=run_id,
            prediction=answer_row[0],
            gold_answer=answer_row[8],
            retrieved_items=items,
            gold_document_ids=gold_document_ids,
            gold_evidence_ids=_as_str_list(answer_row[9]),
            gold_answer_aliases=_answer_aliases(answer_row[10]),
            citations=_as_str_list(answer_row[2]),
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=answer_row[3],
            prompt_tokens=answer_row[4],
            completion_tokens=answer_row[5],
            total_tokens=answer_row[6],
            estimated_cost=answer_row[7],
            k=k,
        )

    def load_compare_rows(self, *, experiment_id: str) -> list[dict[str, Any]]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    r.method_id,
                    r.agent_mode,
                    q.question_type,
                    er.metric_name,
                    er.metric_value
                FROM evaluation_results er
                JOIN runs r ON r.run_id = er.run_id
                LEFT JOIN answers a ON a.run_id = r.run_id
                LEFT JOIN questions q ON q.question_id = a.question_id
                WHERE r.experiment_id = %s
                """,
                (experiment_id,),
            )
            rows = cursor.fetchall()
        return [
            {
                "method_id": row[0],
                "agent_mode": row[1],
                "question_type": row[2] or "unknown",
                "metric_name": row[3],
                "metric_value": float(row[4]),
            }
            for row in rows
        ]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _as_str_list(parsed)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _answer_aliases(metadata: object) -> list[str]:
    """Extract MuSiQue ``answer_aliases`` from a question's metadata blob."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return []
    if isinstance(metadata, dict):
        return _as_str_list(metadata.get("answer_aliases"))
    return []
