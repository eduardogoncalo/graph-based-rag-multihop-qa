from __future__ import annotations

import pytest

from benchmark.core.naming import (
    method_short_tag,
    neo4j_container_name,
    neo4j_volume_name,
    resolve_neo4j_names,
    slugify_dataset_version,
)


def test_slugify_musique_example() -> None:
    assert slugify_dataset_version("musique", "ans_v1.0_eval1k") == "musique_ans_v1_0_eval1k"


@pytest.mark.parametrize(
    "dataset_id,dataset_version,expected",
    [
        ("musique_smoke_20", "v1", "musique_smoke_20_v1"),
        ("MuSiQue", "ANS-v1.0", "musique_ans_v1_0"),
        ("ds", "a..b--c", "ds_a_b_c"),
        ("ds", "  _weird@@name__ ", "ds_weirdname"),
    ],
)
def test_slugify_rules(dataset_id: str, dataset_version: str, expected: str) -> None:
    assert slugify_dataset_version(dataset_id, dataset_version) == expected


def test_slug_has_no_dots_dashes_or_leading_trailing_underscore() -> None:
    slug = slugify_dataset_version("x.y-z", ".-v1.0-.")
    assert "." not in slug and "-" not in slug
    assert not slug.startswith("_") and not slug.endswith("_")
    assert "__" not in slug


def test_method_short_tags() -> None:
    assert method_short_tag("ms_graphrag_neo4j") == "graphrag"
    assert method_short_tag("lightrag_neo4j") == "lightrag"
    assert method_short_tag("cognee") == "cognee"


def test_method_short_tag_passthrough_for_unknown() -> None:
    assert method_short_tag("isodiag_a") == "isodiag_a"


def test_container_and_volume_names() -> None:
    slug = slugify_dataset_version("musique", "ans_v1.0_eval1k")
    assert neo4j_container_name("graphrag", slug) == "neo4j_graphrag_musique_ans_v1_0_eval1k"
    assert (
        neo4j_volume_name("lightrag", slug) == "neo4j_lightrag_musique_ans_v1_0_eval1k_data"
    )


def test_resolve_neo4j_names_for_known_method() -> None:
    container, volume = resolve_neo4j_names("cognee", "musique", "ans_v1.0_eval1k")
    assert container == "neo4j_cognee_musique_ans_v1_0_eval1k"
    assert volume == "neo4j_cognee_musique_ans_v1_0_eval1k_data"
