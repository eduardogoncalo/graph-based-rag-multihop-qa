from __future__ import annotations

import os

import pytest

from benchmark.infra.neo4j_containers import docker_available, temporary_neo4j, volume_mountpoint
from benchmark.methods.lightrag_neo4j.option_c import resolve_lightrag_option_c

# Infra/connectivity smoke ONLY — no LightRAG indexing or retrieval. Gated
# behind an explicit opt-in flag + docker availability.
RUN_FLAG = "BENCHMARK_RUN_DOCKER_INTEGRATION"

pytestmark = pytest.mark.skipif(
    not os.getenv(RUN_FLAG) or not docker_available(),
    reason=f"set {RUN_FLAG}=1 and have docker/podman available to run the LightRAG Option C smoke",
)

MARKER = "LIGHTRAG_OPTION_C_SMOKE_MARKER"


def test_lightrag_option_c_container_connectivity_and_marker() -> None:
    binding = resolve_lightrag_option_c(
        dataset_id="musique",
        dataset_version="ans_v1.0_eval1k",
        preflight=True,
    )
    assert binding.container_name == "neo4j_lightrag_musique_ans_v1_0_eval1k"
    # 18694 desde 2026-08-10. Com a atribuição antiga (17688) este teste NUNCA
    # poderia ter passado: corre com preflight=True, e a 17688 está em
    # `reserved`, portanto o resolve_lightrag_option_c levantaria
    # PortConflictError antes de chegar ao container. Estava a ser salvo pelo
    # skipif — só corre com BENCHMARK_RUN_DOCKER_INTEGRATION=1.
    assert binding.bolt_uri == "bolt://localhost:18694"
    assert binding.bolt_port != 7687  # never the host Neo4j

    from neo4j import GraphDatabase

    # temporary_neo4j removes ONLY this container + its volume afterwards.
    with temporary_neo4j(binding.container_spec()) as handle:
        assert handle.bolt_uri == binding.bolt_uri
        driver = GraphDatabase.driver(handle.bolt_uri, auth=(handle.username, handle.password))
        try:
            with driver.session(database="neo4j") as session:
                session.run("CREATE (:LightragSmoke {tag: $t})", t=MARKER).consume()
                tags = sorted(
                    r["tag"]
                    for r in session.run(
                        "MATCH (m:LightragSmoke) RETURN m.tag AS tag"
                    ).data()
                )
        finally:
            driver.close()
        assert tags == [MARKER]
        assert volume_mountpoint(binding.volume_name) is not None
