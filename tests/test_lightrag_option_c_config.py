from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.infra.guard import VARIAVEL_DO_AMBIENTE_ORIGINAL
from benchmark.infra.neo4j_containers import PortConflictError
from benchmark.methods.lightrag_neo4j.adapter import scoped_lightrag_neo4j_env
from benchmark.methods.lightrag_neo4j.config_builder import build_workspace
from benchmark.methods.lightrag_neo4j.option_c import resolve_lightrag_option_c

DATASET_ID = "musique"
DATASET_VERSION = "ans_v1.0_eval1k"


@pytest.fixture(autouse=True)
def _maquina_de_quem_recebe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Estes testes são da CONSTRUÇÃO do Option C, não do travão.

    Declarado explicitamente para a suíte dar o mesmo resultado aqui e na
    máquina de quem clona: nesta, o ambiente original está mesmo ao lado, e
    desde 2026-08-10 o `exigir_dataset_permitido` recusa os slugs da
    dissertação enquanto ele existir. O travão tem os testes dele em
    `test_guard_ambiente_original.py`, dos dois lados.
    """
    monkeypatch.setenv(VARIAVEL_DO_AMBIENTE_ORIGINAL, "")


def _binding():
    # preflight=False: resolution must not depend on host port state.
    return resolve_lightrag_option_c(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        preflight=False,
    )


def test_resolves_container_volume_ports_for_musique() -> None:
    b = _binding()
    assert b.slug == "musique_ans_v1_0_eval1k"
    assert b.method_short == "lightrag"
    assert b.container_name == "neo4j_lightrag_musique_ans_v1_0_eval1k"
    assert b.volume_name == "neo4j_lightrag_musique_ans_v1_0_eval1k_data"
    assert (b.http_port, b.bolt_port) == (18480, 18694)
    assert b.bolt_uri == "bolt://localhost:18694"
    # Must never collide with the host-native Neo4j.
    assert b.bolt_port != 7687 and b.http_port != 7474


def test_container_spec_matches_binding() -> None:
    b = _binding()
    spec = b.container_spec()
    assert spec.container_name == b.container_name
    assert spec.volume_name == b.volume_name
    assert (spec.http_port, spec.bolt_port) == (b.http_port, b.bolt_port)
    assert spec.password == b.password


def test_to_config_carries_explicit_connection() -> None:
    b = _binding()
    config = b.to_config()
    assert config.method_id == "lightrag_neo4j"
    assert config.neo4j_uri == "bolt://localhost:18694"
    assert config.neo4j_user == "neo4j"
    assert config.neo4j_password == "benchmark_lightrag"
    assert config.neo4j_database == "neo4j"
    # Env-var NAMES stay the LightRAG-specific guards.
    assert config.neo4j_uri_env == "LIGHTRAG_NEO4J_URI"


def test_password_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTRAG_NEO4J_PASSWORD", "from_env_secret")
    b = resolve_lightrag_option_c(
        dataset_id=DATASET_ID, dataset_version=DATASET_VERSION, preflight=False
    )
    assert b.password == "from_env_secret"


def test_preflight_rejects_reserved_host_port() -> None:
    # graphrag/lightrag bolt 7687 is reserved (host Neo4j); a registry key that
    # maps to a reserved port must be refused. Simulate via direct preflight on
    # the lightrag binding's resolver hitting a reserved port through the
    # registry's reserved set is covered in test_neo4j_port_registry; here we
    # assert the resolver surfaces PortConflictError when preflight is on and a
    # port is already bound. We use the host Neo4j bolt port 7687 explicitly.
    from benchmark.infra.neo4j_containers import preflight_ports

    with pytest.raises(PortConflictError):
        preflight_ports(7687, reserved=frozenset({7687}))


def test_scoped_env_injects_option_c_bolt_over_global(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A stale global LIGHTRAG_NEO4J_URI must NOT win over the Option C override.
    monkeypatch.setenv("LIGHTRAG_NEO4J_URI", "bolt://localhost:7688")
    monkeypatch.setenv("LIGHTRAG_NEO4J_USER", "neo4j")
    monkeypatch.setenv("LIGHTRAG_NEO4J_PASSWORD", "global_pw")
    monkeypatch.setenv("LIGHTRAG_NEO4J_DATABASE", "neo4j")

    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
    )
    # Workspace path is slugified and dataset-scoped.
    assert workspace.artifact_dir == tmp_path / "musique_ans_v1_0_eval1k" / "lightrag_neo4j"

    # Explicit Option C password, distinct from the global env one, to prove
    # the override wins over the global value.
    binding = resolve_lightrag_option_c(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        preflight=False,
        password="option_c_pw",
    )
    config = binding.to_config()
    import os

    with scoped_lightrag_neo4j_env(workspace=workspace, config=config):
        assert os.environ["NEO4J_URI"] == "bolt://localhost:18694"  # not the global 7688
        assert os.environ["NEO4J_PASSWORD"] == "option_c_pw"  # not the global global_pw
    # Scoped mapping (NEO4J_URI) is cleaned up; the global LIGHTRAG_NEO4J_URI is untouched.
    assert "NEO4J_URI" not in os.environ
    assert os.environ["LIGHTRAG_NEO4J_URI"] == "bolt://localhost:7688"
