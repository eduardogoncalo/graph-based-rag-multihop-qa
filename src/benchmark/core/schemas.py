from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Metadata = dict[str, Any]
QuestionType = Literal[
    "yes_no",
    "clause_lookup",
    "span_extraction",
    "short_answer",
    "relational_reasoning",
    "multi_hop_reasoning",
]


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Document(BenchmarkModel):
    document_id: str
    dataset_id: str
    dataset_version: str
    title: str | None = None
    text: str
    source_path: str | None = None
    metadata: Metadata = Field(default_factory=dict)


class Chunk(BenchmarkModel):
    chunk_id: str
    document_id: str
    dataset_id: str
    dataset_version: str
    method_id: str | None = None
    text: str
    start_char: int | None = None
    end_char: int | None = None
    metadata: Metadata = Field(default_factory=dict)


class GoldEvidence(BenchmarkModel):
    evidence_id: str
    question_id: str
    document_id: str
    text: str
    start_char: int | None = None
    end_char: int | None = None
    metadata: Metadata = Field(default_factory=dict)


class Question(BenchmarkModel):
    question_id: str
    dataset_id: str
    dataset_version: str
    document_id: str | None = None
    question: str
    question_type: QuestionType
    clause_type: str | None = None
    gold_answer: str | list[str] | bool | None = None
    gold_evidence: list[str] | None = None
    metadata: Metadata = Field(default_factory=dict)


class RetrievedItem(BenchmarkModel):
    item_id: str
    text: str
    score: float | None = None
    source_document_id: str | None = None
    source_chunk_id: str | None = None
    metadata: Metadata = Field(default_factory=dict)


class RetrievalResult(BenchmarkModel):
    method_id: str
    query: str
    items: list[RetrievedItem] = Field(default_factory=list)
    raw_response: dict[str, Any] | list[Any] | str | None = None
    latency_ms: float
    metadata: Metadata = Field(default_factory=dict)


class RetrievalTrace(BenchmarkModel):
    trace_id: str
    run_id: str
    trace_status: Literal["complete", "partial", "missing", "not_supported", "error"]
    trace: Metadata = Field(default_factory=dict)
    raw_trace: Metadata = Field(default_factory=dict)
    metadata: Metadata = Field(default_factory=dict)
