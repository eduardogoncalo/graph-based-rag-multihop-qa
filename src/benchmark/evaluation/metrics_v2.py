from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from typing import Any

from benchmark.evaluation.answerability import (
    AMBIGUOUS,
    EMPTY_GOLD_NO_EVIDENCE,
    POSITIVE_EVIDENCE_ANSWERABLE,
)

NO_ANSWER_PREFIXES = (
    "no answer",
    "no relevant clause found",
    "no related clause",
    "not applicable",
    "i do not have enough information",
    "insufficient information",
    "no clause found",
    "the contract does not contain",
)


@dataclass(frozen=True)
class EvaluationV2Flags:
    predicted_no_answer_strict: bool
    predicted_no_answer_normalized: bool
    evidence_recall_applicable: bool
    evidence_recall_at_5_adjusted: float | None
    citation_accuracy_adjusted: float | None
    false_no_answer: bool
    false_answer_on_no_evidence: bool


def predicted_no_answer_strict(prediction: Any) -> bool:
    if prediction is None or _is_nan(prediction):
        return True
    if isinstance(prediction, list | tuple | set):
        return len(prediction) == 0
    if isinstance(prediction, dict):
        return len(prediction) == 0
    if isinstance(prediction, str):
        stripped = prediction.strip()
        return stripped == "" or stripped == "[]" or stripped.lower() in {"nan", "none", "null"}
    return False


def predicted_no_answer_normalized(prediction: Any) -> bool:
    if predicted_no_answer_strict(prediction):
        return True
    normalized = _normalize_no_answer_text(prediction)
    return any(
        normalized == phrase or normalized.startswith(f"{phrase} ")
        for phrase in NO_ANSWER_PREFIXES
    )


def build_evaluation_v2_flags(
    *,
    answerability_group: str,
    prediction: Any,
    evidence_recall_at_5_raw: float | None,
    citation_accuracy_raw: float | None,
    normalized_abstention: bool = True,
) -> EvaluationV2Flags:
    strict = predicted_no_answer_strict(prediction)
    normalized = predicted_no_answer_normalized(prediction)
    predicted_no_answer = normalized if normalized_abstention else strict

    if answerability_group == POSITIVE_EVIDENCE_ANSWERABLE:
        return EvaluationV2Flags(
            predicted_no_answer_strict=strict,
            predicted_no_answer_normalized=normalized,
            evidence_recall_applicable=True,
            evidence_recall_at_5_adjusted=evidence_recall_at_5_raw,
            citation_accuracy_adjusted=citation_accuracy_raw,
            false_no_answer=predicted_no_answer,
            false_answer_on_no_evidence=False,
        )
    if answerability_group == EMPTY_GOLD_NO_EVIDENCE:
        return EvaluationV2Flags(
            predicted_no_answer_strict=strict,
            predicted_no_answer_normalized=normalized,
            evidence_recall_applicable=False,
            evidence_recall_at_5_adjusted=None,
            citation_accuracy_adjusted=None,
            false_no_answer=False,
            false_answer_on_no_evidence=not predicted_no_answer,
        )
    return EvaluationV2Flags(
        predicted_no_answer_strict=strict,
        predicted_no_answer_normalized=normalized,
        evidence_recall_applicable=False,
        evidence_recall_at_5_adjusted=None,
        citation_accuracy_adjusted=None,
        false_no_answer=False,
        false_answer_on_no_evidence=False,
    )


def is_v2_group(value: str) -> bool:
    return value in {
        POSITIVE_EVIDENCE_ANSWERABLE,
        EMPTY_GOLD_NO_EVIDENCE,
        AMBIGUOUS,
    }


def _normalize_no_answer_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = text.replace("'", "")
    translation = str.maketrans({char: " " for char in string.punctuation if char != "'"})
    text = text.translate(translation)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)
