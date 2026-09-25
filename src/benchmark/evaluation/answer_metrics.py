from __future__ import annotations

import re
import string
from collections import Counter

ARTICLES = {"a", "an", "the"}


def normalize_answer(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: object, gold_answer: object) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def answer_f1(prediction: object, gold_answer: object) -> float:
    prediction_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold_answer)
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _tokens(value: object) -> list[str]:
    normalized = normalize_answer(value)
    return re.findall(r"\S+", normalized)


def _clean_golds(golds: object) -> list[object]:
    """Drop empty/blank gold strings; tolerate a single value or a list."""
    if golds is None:
        return []
    if not isinstance(golds, (list, tuple)):
        golds = [golds]
    return [g for g in golds if str(g).strip()]


def exact_match_aliases(prediction: object, golds: object) -> float:
    """Best exact match over gold + answer aliases (MuSiQue official protocol).

    Distinct from ``exact_match`` (single gold) — reported under
    ``exact_match_aliases`` so the raw baseline is never overwritten.
    """
    candidates = _clean_golds(golds)
    return max((exact_match(prediction, g) for g in candidates), default=0.0)


def answer_f1_aliases(prediction: object, golds: object) -> float:
    """Best token-F1 over gold + answer aliases (MuSiQue official protocol)."""
    candidates = _clean_golds(golds)
    return max((answer_f1(prediction, g) for g in candidates), default=0.0)


def answer_recall_containment(prediction: object, golds: object) -> float:
    """Diagnostic: 1.0 if any gold/alias's tokens are a SUBSET of the prediction
    tokens (i.e. the answer is present, possibly buried in a verbose reply).

    NOT a final metric — over-counts when gold tokens appear incidentally.
    """
    prediction_tokens = set(_tokens(prediction))
    for gold in _clean_golds(golds):
        gold_tokens = _tokens(gold)
        if gold_tokens and set(gold_tokens).issubset(prediction_tokens):
            return 1.0
    return 0.0


def answer_substring_containment(prediction: object, golds: object) -> float:
    """Diagnostic: 1.0 if any normalized gold/alias is a contiguous substring of
    the normalized prediction. Stricter than ``answer_recall_containment``.

    NOT a final metric — see the containment caveat above.
    """
    prediction_norm = normalize_answer(prediction)
    for gold in _clean_golds(golds):
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in prediction_norm:
            return 1.0
    return 0.0
