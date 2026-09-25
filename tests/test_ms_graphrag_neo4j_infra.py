from __future__ import annotations

import sys

import yaml

import scripts.validate_ms_graphrag_neo4j_env as validator


def test_docker_compose_has_isolated_graphrag_neo4j_service() -> None:
    compose = _load_compose()
    service = compose["services"]["neo4j_graphrag"]

    assert service["image"] == "docker.io/library/neo4j:5"
    assert service["environment"]["NEO4J_AUTH"] == "neo4j/benchmark_graphrag"
    assert service["ports"] == ["18474:7474", "18687:7687"]
    assert service["volumes"] == ["neo4j_graphrag_data:/data"]
    assert "neo4j_graphrag_data" in compose["volumes"]


def test_docker_compose_graphrag_neo4j_has_apoc_and_gds_plugins() -> None:
    service = _load_compose()["services"]["neo4j_graphrag"]
    environment = service["environment"]

    assert environment["NEO4J_PLUGINS"] == '["apoc","graph-data-science"]'
    assert environment["NEO4J_dbms_security_procedures_unrestricted"] == "apoc.*,gds.*"
    assert environment["NEO4J_dbms_security_procedures_allowlist"] == "apoc.*,gds.*"


def test_docker_compose_lightrag_neo4j_remains_separate() -> None:
    compose = _load_compose()
    service = compose["services"]["neo4j_lightrag"]

    assert service["environment"]["NEO4J_AUTH"] == "neo4j/benchmark_lightrag"
    assert service["ports"] == ["18475:7474", "18688:7687"]
    assert service["volumes"] == ["neo4j_lightrag_data:/data"]
    assert "neo4j_lightrag_data" in compose["volumes"]


def test_validator_without_check_neo4j_does_not_require_driver(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_ms_graphrag_neo4j_env.py"])

    assert validator.main() == 0


def test_validator_reads_only_graphrag_neo4j_env_for_required_env(monkeypatch) -> None:
    for name in validator.REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIGHTRAG_NEO4J_URI", "bolt://localhost:7688")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:9999")

    result = validator._check_graphrag_env()

    assert not result.ok
    assert "GRAPHRAG_NEO4J_URI" in result.detail
    assert "LIGHTRAG_NEO4J_URI" not in result.detail
    assert "localhost:7688" not in result.detail
    assert "localhost:9999" not in result.detail


def _load_compose() -> dict:
    with open("docker-compose.yml", encoding="utf-8") as file:
        return yaml.safe_load(file)
