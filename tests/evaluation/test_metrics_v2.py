from __future__ import annotations

import math

from benchmark.evaluation.answerability import (
    EMPTY_GOLD_NO_EVIDENCE,
    POSITIVE_EVIDENCE_ANSWERABLE,
)
from benchmark.evaluation.metrics_v2 import (
    build_evaluation_v2_flags,
    predicted_no_answer_normalized,
    predicted_no_answer_strict,
)


def test_strict_abstention_values() -> None:
    assert predicted_no_answer_strict(None)
    assert predicted_no_answer_strict("")
    assert predicted_no_answer_strict(math.nan)
    assert predicted_no_answer_strict([])
    assert predicted_no_answer_strict("[]")


def test_normalized_abstention_phrases() -> None:
    assert predicted_no_answer_normalized("No relevant clause found")
    assert predicted_no_answer_normalized("I do not have enough information")
    assert predicted_no_answer_normalized("Not applicable")
    assert predicted_no_answer_normalized("The contract does not contain this clause")


def test_positive_empty_prediction_is_false_no_answer() -> None:
    flags = build_evaluation_v2_flags(
        answerability_group=POSITIVE_EVIDENCE_ANSWERABLE,
        prediction="",
        evidence_recall_at_5_raw=0.5,
        citation_accuracy_raw=0.25,
    )

    assert flags.false_no_answer is True
    assert flags.false_answer_on_no_evidence is False


def test_positive_non_empty_prediction_is_not_false_no_answer() -> None:
    flags = build_evaluation_v2_flags(
        answerability_group=POSITIVE_EVIDENCE_ANSWERABLE,
        prediction="The clause",
        evidence_recall_at_5_raw=1.0,
        citation_accuracy_raw=1.0,
    )

    assert flags.false_no_answer is False


def test_no_answer_empty_prediction_is_correct_abstention() -> None:
    flags = build_evaluation_v2_flags(
        answerability_group=EMPTY_GOLD_NO_EVIDENCE,
        prediction="",
        evidence_recall_at_5_raw=1.0,
        citation_accuracy_raw=1.0,
    )

    assert flags.predicted_no_answer_strict is True
    assert flags.false_answer_on_no_evidence is False


def test_no_answer_non_empty_prediction_is_false_answer() -> None:
    flags = build_evaluation_v2_flags(
        answerability_group=EMPTY_GOLD_NO_EVIDENCE,
        prediction="The contract has this clause",
        evidence_recall_at_5_raw=1.0,
        citation_accuracy_raw=0.0,
    )

    assert flags.predicted_no_answer_normalized is False
    assert flags.false_answer_on_no_evidence is True


def test_positive_adjusted_evidence_and_citation_are_applicable() -> None:
    flags = build_evaluation_v2_flags(
        answerability_group=POSITIVE_EVIDENCE_ANSWERABLE,
        prediction="The clause",
        evidence_recall_at_5_raw=0.75,
        citation_accuracy_raw=0.5,
    )

    assert flags.evidence_recall_applicable is True
    assert flags.evidence_recall_at_5_adjusted == 0.75
    assert flags.citation_accuracy_adjusted == 0.5


def test_no_answer_adjusted_evidence_and_citation_are_not_applicable() -> None:
    flags = build_evaluation_v2_flags(
        answerability_group=EMPTY_GOLD_NO_EVIDENCE,
        prediction="No answer",
        evidence_recall_at_5_raw=1.0,
        citation_accuracy_raw=1.0,
    )

    assert flags.evidence_recall_applicable is False
    assert flags.evidence_recall_at_5_adjusted is None
    assert flags.citation_accuracy_adjusted is None
