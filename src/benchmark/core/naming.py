from __future__ import annotations

import re

# Option C (separate Neo4j container + volume per dataset) naming helpers.
# A single slug feeds every Docker object name and artifact dir so they stay
# in lockstep. See artifacts/musique/reports/option_c_infrastructure_plan.md.

_NON_SLUG = re.compile(r"[^a-z0-9_]+")
_MULTI_UNDERSCORE = re.compile(r"_+")

# Docker-friendly short tags for the verbose method_ids.
METHOD_SHORT_TAGS: dict[str, str] = {
    "ms_graphrag_neo4j": "graphrag",
    "lightrag_neo4j": "lightrag",
    "cognee": "cognee",
}


def slugify_dataset_version(dataset_id: str, dataset_version: str) -> str:
    """Deterministic slug for ``(dataset_id, dataset_version)``.

    Rules: lowercase; ``.`` and ``-`` become ``_``; drop any character that is
    not ``[a-z0-9_]``; collapse repeated ``_``; trim leading/trailing ``_``.

    >>> slugify_dataset_version("musique", "ans_v1.0_eval1k")
    'musique_ans_v1_0_eval1k'
    """
    raw = f"{dataset_id}_{dataset_version}".lower()
    raw = raw.replace(".", "_").replace("-", "_")
    raw = _NON_SLUG.sub("", raw)
    raw = _MULTI_UNDERSCORE.sub("_", raw)
    return raw.strip("_")


def method_short_tag(method_id: str) -> str:
    """Resolve the Docker-friendly short tag for a method.

    Known graph methods map via ``METHOD_SHORT_TAGS``. Any other value is
    slugified and passed through (lets tests use synthetic tags such as
    ``isodiag_a`` without registering them).
    """
    if method_id in METHOD_SHORT_TAGS:
        return METHOD_SHORT_TAGS[method_id]
    slug = _MULTI_UNDERSCORE.sub("_", _NON_SLUG.sub("", method_id.lower())).strip("_")
    if not slug:
        raise ValueError(f"method_id {method_id!r} has no usable short tag")
    return slug


def neo4j_container_name(method_short: str, slug: str) -> str:
    """``neo4j_{method_short}_{slug}``."""
    return f"neo4j_{method_short}_{slug}"


def neo4j_volume_name(method_short: str, slug: str) -> str:
    """``neo4j_{method_short}_{slug}_data``."""
    return f"{neo4j_container_name(method_short, slug)}_data"


def resolve_neo4j_names(
    method_id: str,
    dataset_id: str,
    dataset_version: str,
) -> tuple[str, str]:
    """Return ``(container_name, volume_name)`` for a (method, dataset) pair."""
    short = method_short_tag(method_id)
    slug = slugify_dataset_version(dataset_id, dataset_version)
    return neo4j_container_name(short, slug), neo4j_volume_name(short, slug)


def port_registry_key(method_short: str, slug: str) -> str:
    """Key used in ``configs/infra/neo4j_ports.yaml`` (``{method_short}_{slug}``)."""
    return f"{method_short}_{slug}"
