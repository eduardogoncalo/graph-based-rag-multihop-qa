from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

POSITIVE_EVIDENCE_ANSWERABLE = "positive_evidence_answerable"
EMPTY_GOLD_NO_EVIDENCE = "empty_gold_no_evidence"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class AnswerabilityClassification:
    answerability_group: str
    gold_answer_is_empty: bool
    gold_evidence_count: int
    metadata_is_impossible: bool | None
    classification_reason: str


def normalize_gold_answer(value: Any) -> str:
    if value is None or _is_nan(value):
        return ""
    if isinstance(value, list):
        values = [normalize_gold_answer(item) for item in value]
        return " ".join(value for value in values if value).strip()
    if isinstance(value, tuple):
        values = [normalize_gold_answer(item) for item in value]
        return " ".join(value for value in values if value).strip()
    if isinstance(value, dict):
        if "text" in value:
            return normalize_gold_answer(value.get("text"))
        if "answer" in value:
            return normalize_gold_answer(value.get("answer"))
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "[]"}:
        return ""
    return text


def gold_answer_is_empty(value: Any) -> bool:
    return normalize_gold_answer(value) == ""


def count_gold_evidence(value: Any) -> int:
    if value is None or _is_nan(value):
        return 0
    if isinstance(value, list | tuple | set):
        return sum(1 for item in value if _has_evidence_value(item))
    if isinstance(value, dict):
        if "evidence" in value:
            return count_gold_evidence(value.get("evidence"))
        if "items" in value:
            return count_gold_evidence(value.get("items"))
        return 1 if _has_evidence_value(value) else 0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "[]"} or stripped.lower() in {"nan", "none", "null"}:
            return 0
        return 1
    return 1


def metadata_is_impossible(metadata: Any) -> bool | None:
    if not isinstance(metadata, dict):
        return None
    for key in ("is_impossible", "metadata_is_impossible"):
        if key in metadata:
            return _coerce_bool(metadata.get(key))
    nested = metadata.get("metadata")
    if isinstance(nested, dict) and "is_impossible" in nested:
        return _coerce_bool(nested.get("is_impossible"))
    return None


def classify_answerability(
    *,
    gold_answer: Any,
    gold_evidence: Any = None,
    gold_evidence_count: int | None = None,
    metadata: Any = None,
) -> AnswerabilityClassification:
    answer_empty = gold_answer_is_empty(gold_answer)
    evidence_count = (
        count_gold_evidence(gold_evidence)
        if gold_evidence_count is None
        else max(int(gold_evidence_count), 0)
    )
    impossible = metadata_is_impossible(metadata)
    has_positive_answer = not answer_empty
    has_positive_evidence = evidence_count > 0

    if impossible is True and (has_positive_answer or has_positive_evidence):
        return AnswerabilityClassification(
            answerability_group=AMBIGUOUS,
            gold_answer_is_empty=answer_empty,
            gold_evidence_count=evidence_count,
            metadata_is_impossible=impossible,
            classification_reason="metadata_is_impossible_true_contradicts_positive_gold",
        )
    if has_positive_answer and has_positive_evidence:
        return AnswerabilityClassification(
            answerability_group=POSITIVE_EVIDENCE_ANSWERABLE,
            gold_answer_is_empty=answer_empty,
            gold_evidence_count=evidence_count,
            metadata_is_impossible=impossible,
            classification_reason="non_empty_gold_answer_and_positive_gold_evidence",
        )
    if answer_empty and evidence_count == 0:
        reason = (
            "metadata_is_impossible_true_with_empty_gold"
            if impossible is True
            else "empty_gold_answer_and_no_gold_evidence"
        )
        return AnswerabilityClassification(
            answerability_group=EMPTY_GOLD_NO_EVIDENCE,
            gold_answer_is_empty=answer_empty,
            gold_evidence_count=evidence_count,
            metadata_is_impossible=impossible,
            classification_reason=reason,
        )
    if has_positive_answer and evidence_count == 0:
        reason = "non_empty_gold_answer_without_gold_evidence"
    elif answer_empty and has_positive_evidence:
        reason = "empty_gold_answer_with_positive_gold_evidence"
    else:
        reason = "inconsistent_gold_fields"
    return AnswerabilityClassification(
        answerability_group=AMBIGUOUS,
        gold_answer_is_empty=answer_empty,
        gold_evidence_count=evidence_count,
        metadata_is_impossible=impossible,
        classification_reason=reason,
    )


def _has_evidence_value(value: Any) -> bool:
    if value is None or _is_nan(value):
        return False
    if isinstance(value, dict):
        if not value:
            return False
        return any(_has_evidence_value(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(_has_evidence_value(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and stripped != "[]" and stripped.lower() not in {"nan", "none", "null"}
    return True


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or _is_nan(value):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    if isinstance(value, int | float):
        return bool(value)
    return None


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)
