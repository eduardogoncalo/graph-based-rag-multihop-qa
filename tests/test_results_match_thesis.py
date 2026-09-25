"""results/ is the evidence behind the thesis. These checks tie it to the
printed tables, so an edited CSV (or a thesis table that drifted from its data)
fails here, on any machine.

Expected values are transcribed from the thesis PDF: Table 3 (controlled arm)
and Table 4 (native arm against its own controlled arm). The thesis prints
three decimals and one decimal for percentage points; the CSVs keep four.
"""

from __future__ import annotations

import csv
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

AGGREGATE = Path(__file__).resolve().parents[1] / "results" / "aggregate"

# Table 3: (dataset, cell) -> (docs/q, recall@5, recall@pool, all_gold@5, strict, delta vs dense)
TABLE_3 = {
    ("musique", "closed_book"): (None, None, None, None, 0.209, None),
    ("musique", "vector_v1free"): (4.92, 0.553, 0.553, 0.218, 0.463, None),
    ("musique", "hipporag2_v1free_official"): (5.00, 0.603, 0.603, 0.303, 0.502, 3.8),
    ("musique", "lightrag_v1free"): (20.00, 0.506, 0.704, 0.181, 0.519, 5.7),
    ("musique", "ms_graphrag_v1free"): (16.57, 0.447, 0.615, 0.167, 0.495, 3.2),
    ("musique", "cognee_v1free_k5"): (2.76, 0.434, 0.434, 0.109, 0.401, -6.2),
    ("musique", "oracle_gold_v1free"): (None, None, None, None, 0.770, None),
    ("twowiki", "closed_book"): (None, None, None, None, 0.335, None),
    ("twowiki", "vector_v1free"): (4.82, 0.704, 0.704, 0.409, 0.678, None),
    ("twowiki", "hipporag2_v1free_official"): (5.00, 0.864, 0.864, 0.672, 0.801, 12.2),
    ("twowiki", "lightrag_v1free"): (20.00, 0.693, 0.777, 0.400, 0.731, 5.3),
    ("twowiki", "ms_graphrag_v1free"): (12.95, 0.608, 0.670, 0.298, 0.694, 1.6),
    ("twowiki", "cognee_v1free_k5"): (2.46, 0.625, 0.625, 0.266, 0.615, -6.3),
    ("twowiki", "oracle_gold_v1free"): (None, None, None, None, 0.890, None),
}

# Table 4: (dataset, native cell) -> (controlled strict, native strict, delta pp, refusal)
TABLE_4 = {
    ("musique", "lightrag_native"): (0.519, 0.548, 2.8, 0.119),
    ("musique", "ms_graphrag_native"): (0.495, 0.474, -2.1, 0.092),
    ("musique", "hipporag2_native_official"): (0.502, 0.466, -3.5, 0.151),
    ("musique", "cognee_native"): (0.401, 0.349, -5.2, 0.148),
    ("twowiki", "lightrag_native"): (0.731, 0.615, -11.6, 0.218),
    ("twowiki", "ms_graphrag_native"): (0.694, 0.518, -17.6, 0.260),
    ("twowiki", "hipporag2_native_official"): (0.801, 0.723, -7.7, 0.160),
    ("twowiki", "cognee_native"): (0.615, 0.534, -8.0, 0.209),
}


def _rows(name: str) -> dict[tuple[str, str], dict[str, str]]:
    with (AGGREGATE / name).open(encoding="utf-8", newline="") as handle:
        return {(row["dataset"], row["celula"]): row for row in csv.DictReader(handle)}


def _close(value: str, expected: float, decimals: int) -> None:
    # The thesis rounds half up (0.5015 prints as 0.502), as the export did;
    # Python's round() would round half to even.
    printed = Decimal(value).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    assert printed == Decimal(str(expected)).quantize(Decimal(1).scaleb(-decimals))


def test_controlled_arm_has_exactly_the_cells_of_table_3() -> None:
    assert set(_rows("arm_a_controlled.csv")) == set(TABLE_3)


@pytest.mark.parametrize(("key", "expected"), TABLE_3.items(), ids=lambda v: str(v))
def test_controlled_arm_matches_table_3(key, expected) -> None:
    row = _rows("arm_a_controlled.csv")[key]
    docs_q, recall_5, recall_pool, all_gold_5, strict, delta = expected
    _close(row["estrita"], strict, 3)
    for column, value, decimals in (
        ("docs_q", docs_q, 2),
        ("recall_5", recall_5, 3),
        ("recall_pool", recall_pool, 3),
        ("all_gold_5", all_gold_5, 3),
        ("delta_vs_denso_pp", delta, 1),
    ):
        if value is not None:
            _close(row[column], value, decimals)


def test_native_arm_has_exactly_the_cells_of_table_4() -> None:
    assert set(_rows("arm_b_native.csv")) == set(TABLE_4)


@pytest.mark.parametrize(("key", "expected"), TABLE_4.items(), ids=lambda v: str(v))
def test_native_arm_matches_table_4(key, expected) -> None:
    row = _rows("arm_b_native.csv")[key]
    controlled, native, delta, refusal = expected
    _close(row["estrita_controlada"], controlled, 3)
    _close(row["estrita"], native, 3)
    _close(row["delta_vs_controlado_pp"], delta, 1)
    _close(row["recusa"], refusal, 3)


def test_every_controlled_cell_ran_the_free_reader() -> None:
    for (_dataset, cell), row in _rows("arm_a_controlled.csv").items():
        if cell != "closed_book":
            assert "v1free" in cell and "v2" not in row["experiment_id"], cell
