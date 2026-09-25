from __future__ import annotations

import re
import unicodedata

_REPEATED_BLANK_LINES = re.compile(r"\n{3,}")
_REPEATED_INLINE_WHITESPACE = re.compile(r"[ \t]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")


def clean_rag_text(text: str) -> str:
    """Apply light visual-artifact cleanup without preserving offset mappings."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\ufffd", "")
    normalized = normalized.replace("¶", " ")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or not unicodedata.category(character).startswith("C")
    )
    normalized = _REPEATED_INLINE_WHITESPACE.sub(" ", normalized)
    normalized = _SPACE_AROUND_NEWLINE.sub("\n", normalized)
    normalized = _REPEATED_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()


def normalize_display_text(text: str) -> str:
    return clean_rag_text(text)
