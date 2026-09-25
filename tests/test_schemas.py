import pytest
from pydantic import ValidationError

from benchmark.core.schemas import (
    Chunk,
    Document,
    GoldEvidence,
    Question,
    RetrievalResult,
    RetrievedItem,
)


def test_document_schema_accepts_canonical_fields() -> None:
    document = Document(
        document_id="contract_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Agreement text",
    )

    assert document.metadata == {}


def test_question_schema_rejects_unknown_question_type() -> None:
    with pytest.raises(ValidationError):
        Question(
            question_id="q1",
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            question="What is the governing law?",
            question_type="unsupported",
        )


def test_retrieval_result_defaults_items() -> None:
    result = RetrievalResult(method_id="vector_rag", query="termination", latency_ms=1.5)

    assert result.items == []


def test_retrieval_result_accepts_items() -> None:
    item = RetrievedItem(item_id="item_1", text="Termination clause", score=0.9)
    result = RetrievalResult(
        method_id="vector_rag", query="termination", items=[item], latency_ms=1.5
    )

    assert result.items[0].item_id == "item_1"


def test_canonical_ingestion_schemas_accept_phase_one_fields() -> None:
    document = Document(
        document_id="doc_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        title="Contract",
        text="Termination clause",
    )
    chunk = Chunk(
        chunk_id="chunk_1",
        document_id=document.document_id,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        text="Termination",
        start_char=0,
        end_char=11,
    )
    evidence = GoldEvidence(
        evidence_id="ev_1",
        question_id="q_1",
        document_id=document.document_id,
        text="Termination",
        start_char=0,
        end_char=11,
    )
    question = Question(
        question_id="q_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        document_id=document.document_id,
        question="What clause applies?",
        question_type="span_extraction",
        gold_answer="Termination",
        gold_evidence=[evidence.evidence_id],
    )

    assert chunk.document_id == document.document_id
    assert question.gold_evidence == ["ev_1"]
