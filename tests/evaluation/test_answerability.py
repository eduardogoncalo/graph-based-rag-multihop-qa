from __future__ import annotations

from benchmark.evaluation.answerability import (
    AMBIGUOUS,
    EMPTY_GOLD_NO_EVIDENCE,
    POSITIVE_EVIDENCE_ANSWERABLE,
    classify_answerability,
)


def test_empty_gold_and_no_evidence_is_no_answer() -> None:
    result = classify_answerability(gold_answer="", gold_evidence_count=0)

    assert result.answerability_group == EMPTY_GOLD_NO_EVIDENCE
    assert result.gold_answer_is_empty is True
    assert result.gold_evidence_count == 0


def test_non_empty_gold_and_evidence_is_positive() -> None:
    result = classify_answerability(gold_answer="The clause", gold_evidence_count=2)

    assert result.answerability_group == POSITIVE_EVIDENCE_ANSWERABLE
    assert result.gold_answer_is_empty is False
    assert result.gold_evidence_count == 2


def test_empty_gold_with_evidence_is_ambiguous() -> None:
    result = classify_answerability(gold_answer="", gold_evidence_count=1)

    assert result.answerability_group == AMBIGUOUS
    assert result.classification_reason == "empty_gold_answer_with_positive_gold_evidence"


def test_non_empty_gold_without_evidence_is_ambiguous() -> None:
    result = classify_answerability(gold_answer="The clause", gold_evidence_count=0)

    assert result.answerability_group == AMBIGUOUS
    assert result.classification_reason == "non_empty_gold_answer_without_gold_evidence"


def test_is_impossible_with_empty_gold_is_no_answer() -> None:
    result = classify_answerability(
        gold_answer=None,
        gold_evidence_count=0,
        metadata={"is_impossible": True},
    )

    assert result.answerability_group == EMPTY_GOLD_NO_EVIDENCE
    assert result.metadata_is_impossible is True
    assert result.classification_reason == "metadata_is_impossible_true_with_empty_gold"


def test_is_impossible_with_positive_gold_is_ambiguous() -> None:
    result = classify_answerability(
        gold_answer="The clause",
        gold_evidence_count=1,
        metadata={"is_impossible": True},
    )

    assert result.answerability_group == AMBIGUOUS
    assert result.metadata_is_impossible is True
    assert result.classification_reason == "metadata_is_impossible_true_contradicts_positive_gold"
