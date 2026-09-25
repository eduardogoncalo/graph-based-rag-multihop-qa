from __future__ import annotations

import os

import pytest

from benchmark.core.naming import (
    neo4j_container_name,
    neo4j_volume_name,
    slugify_dataset_version,
)
from benchmark.infra.neo4j_containers import (
    Neo4jContainerSpec,
    docker_available,
    preflight_ports,
    remove_container,
    start_neo4j_container,
    volume_mountpoint,
)

# Heavy: starts two real Neo4j containers. Gated behind an explicit opt-in flag
# AND docker availability so the default unit suite stays fast and offline.
RUN_FLAG = "BENCHMARK_RUN_DOCKER_INTEGRATION"

pytestmark = pytest.mark.skipif(
    not os.getenv(RUN_FLAG) or not docker_available(),
    reason=f"set {RUN_FLAG}=1 and have docker/podman available to run Option C isolation test",
)

MARKER_A = "OPTION_C_ISOLATION_A_MARKER"
MARKER_B = "OPTION_C_ISOLATION_B_MARKER"
PASSWORD = "benchmark_isodiag"
SLUG = slugify_dataset_version("musique", "ans_v1.0_eval1k")

# Test ports in the 27xxx band, kept distinct from the 17xxx production band
# and from the host Neo4j on 7474/7687.
A_HTTP, A_BOLT = 27474, 27687
B_HTTP, B_BOLT = 27475, 27688


def _write_marker(handle, marker: str) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(handle.bolt_uri, auth=(handle.username, handle.password))
    try:
        with driver.session(database="neo4j") as session:
            session.run("CREATE (:IsolationMark {tag: $tag})", tag=marker).consume()
    finally:
        driver.close()


def _read_markers(handle) -> list[str]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(handle.bolt_uri, auth=(handle.username, handle.password))
    try:
        with driver.session(database="neo4j") as session:
            return sorted(
                r["tag"] for r in session.run("MATCH (m:IsolationMark) RETURN m.tag AS tag").data()
            )
    finally:
        driver.close()


def test_option_c_two_containers_are_isolated() -> None:
    spec_a = Neo4jContainerSpec.for_dataset(
        method_short="isodiag_a", slug=SLUG,
        http_port=A_HTTP, bolt_port=A_BOLT, password=PASSWORD,
    )
    spec_b = Neo4jContainerSpec.for_dataset(
        method_short="isodiag_b", slug=SLUG,
        http_port=B_HTTP, bolt_port=B_BOLT, password=PASSWORD,
    )

    # Deterministic names follow the Option C convention.
    assert spec_a.container_name == neo4j_container_name("isodiag_a", SLUG)
    assert spec_a.volume_name == neo4j_volume_name("isodiag_a", SLUG)
    assert spec_a.container_name != spec_b.container_name
    assert spec_a.volume_name != spec_b.volume_name

    # Preflight: refuse to start on a busy/reserved port.
    preflight_ports(A_HTTP, A_BOLT, B_HTTP, B_BOLT)

    handle_a = handle_b = None
    try:
        handle_a = start_neo4j_container(spec_a)
        handle_b = start_neo4j_container(spec_b)

        _write_marker(handle_a, MARKER_A)
        _write_marker(handle_b, MARKER_B)

        # Each container returns only its own marker — no cross-bleed.
        assert _read_markers(handle_a) == [MARKER_A]
        assert _read_markers(handle_b) == [MARKER_B]

        # Volumes are physically distinct.
        mount_a = volume_mountpoint(spec_a.volume_name)
        mount_b = volume_mountpoint(spec_b.volume_name)
        assert mount_a and mount_b and mount_a != mount_b
    finally:
        # Remove ONLY the temp containers/volumes this test created.
        remove_container(spec_a.container_name, remove_volume=spec_a.volume_name)
        remove_container(spec_b.container_name, remove_volume=spec_b.volume_name)
