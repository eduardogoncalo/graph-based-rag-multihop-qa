from benchmark.ingestion.text_normalization import clean_rag_text, normalize_display_text


def test_clean_rag_text_removes_paragraph_mark() -> None:
    assert clean_rag_text("Alpha ¶ Beta") == "Alpha Beta"


def test_clean_rag_text_removes_replacement_character() -> None:
    assert clean_rag_text("Alpha � Beta") == "Alpha Beta"


def test_clean_rag_text_collapses_repeated_whitespace() -> None:
    assert clean_rag_text("Alpha\t\t   Beta\n\n\n\nGamma") == "Alpha Beta\n\nGamma"


def test_clean_rag_text_preserves_redaction_marker() -> None:
    assert clean_rag_text("during [**], market") == "during [**], market"


def test_clean_rag_text_preserves_legal_section_numbering() -> None:
    text = "Sections 1.1, 5.10, and 10.3 survive."
    assert clean_rag_text(text) == text


def test_clean_rag_text_preserves_dates() -> None:
    text = "Effective as of January 1, 2020 and 12/31/2021."
    assert clean_rag_text(text) == text


def test_clean_rag_text_preserves_dollar_amounts() -> None:
    text = "The fee is $1,250.50 and 12.5% of revenue."
    assert clean_rag_text(text) == text


def test_clean_rag_text_preserves_party_names() -> None:
    text = "ACME Holdings, Inc. and Beta Pharma LLC agree."
    assert clean_rag_text(text) == text


def test_clean_rag_text_is_deterministic() -> None:
    text = "ACME\t\tHoldings ¶ Inc. �\n\n\nSection 5.10"
    assert clean_rag_text(text) == clean_rag_text(text)


def test_normalize_display_text_aliases_clean_rag_text() -> None:
    text = "Alpha\t\tBeta"
    assert normalize_display_text(text) == clean_rag_text(text)
