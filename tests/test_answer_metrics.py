from benchmark.evaluation.answer_metrics import (
    answer_f1,
    answer_f1_aliases,
    answer_recall_containment,
    answer_substring_containment,
    exact_match,
    exact_match_aliases,
    normalize_answer,
)


def test_normalize_answer_removes_articles_punctuation_and_case() -> None:
    assert normalize_answer("The, Delaware Law!") == "delaware law"


def test_exact_match_uses_normalized_answers() -> None:
    assert exact_match("The Delaware law.", "delaware law") == 1.0
    assert exact_match("New York", "Delaware") == 0.0


def test_answer_f1_scores_token_overlap() -> None:
    assert answer_f1("Delaware law", "Delaware law") == 1.0
    assert answer_f1("Delaware law", "Delaware courts") == 0.5
    assert answer_f1("California", "Delaware") == 0.0


def test_exact_match_aliases_takes_best_over_gold_plus_aliases() -> None:
    # primary gold fails, an alias matches -> 1.0
    assert exact_match_aliases("Tracy Mosby", ["Tracy McConnell", "Tracy Mosby"]) == 1.0
    # nothing matches -> 0.0
    assert exact_match_aliases("Ted Mosby", ["Tracy McConnell", "Tracy Mosby"]) == 0.0
    # tolerates a single (non-list) gold and blank entries
    assert exact_match_aliases("Delaware", "Delaware") == 1.0
    assert exact_match_aliases("Delaware", ["", "Delaware"]) == 1.0
    # empty gold set -> 0.0, never raises
    assert exact_match_aliases("anything", []) == 0.0


def test_answer_f1_aliases_takes_best_over_aliases() -> None:
    # best alias overlap wins
    assert answer_f1_aliases("Delaware courts", ["California", "Delaware law"]) == 0.5
    assert answer_f1_aliases("Delaware law", ["Delaware law"]) == 1.0


def test_answer_recall_containment_detects_buried_answer() -> None:
    # gold tokens are a subset of a verbose prediction -> 1.0 (the artifact case)
    assert answer_recall_containment(
        "Luke Bryan sings Home Alone Tonight with Karen Fairchild.",
        ["Karen Fairchild"],
    ) == 1.0
    # answer simply absent -> 0.0
    assert answer_recall_containment("Some unrelated sentence.", ["Karen Fairchild"]) == 0.0
    # article/punctuation-insensitive (SQuAD normalization)
    assert answer_recall_containment("the United States of America", ["United States"]) == 1.0


def test_answer_substring_containment_is_contiguous() -> None:
    assert answer_substring_containment("born on June 10 1819 in town", ["June 10 1819"]) == 1.0
    # tokens present but NOT contiguous -> substring is 0.0 ...
    assert answer_substring_containment("June then later 1819", ["June 1819"]) == 0.0
    # ... while the subset-based recall metric WOULD fire on the same input
    assert answer_recall_containment("June then later 1819", ["June 1819"]) == 1.0
    assert answer_substring_containment("the answer is Delaware law today", ["New York"]) == 0.0
    # empty golds never raise
    assert answer_substring_containment("anything", []) == 0.0
