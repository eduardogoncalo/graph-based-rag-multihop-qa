from __future__ import annotations

from scripts import validate_cognee_env as validator


def test_validator_reports_postgres_missing_without_creating(monkeypatch) -> None:
    settings = validator.PostgresSettings(
        host="127.0.0.1",
        port=5433,
        username="benchmark",
        password="",
        cognee_database="cognee_benchmark",
        schema="cognee",
        vector_provider="pgvector",
        vector_database="cognee_benchmark",
        vector_schema="cognee",
        vector_password="",
    )
    monkeypatch.setattr(validator, "_port_is_open", lambda host, port: False)

    results = validator._check_postgres_live(settings)

    assert any(result.name == "Postgres live check" and "skipped" in result.detail for result in results)
    assert any(result.name == "Action required before smoke" for result in results)


def test_validator_postgres_env_uses_cognee_password_env(monkeypatch) -> None:
    settings = validator.PostgresSettings(
        host="127.0.0.1",
        port=5433,
        username="benchmark",
        password="secret",
        cognee_database="cognee_benchmark",
        schema="cognee",
        vector_provider="pgvector",
        vector_database="cognee_benchmark",
        vector_schema="cognee",
        vector_password="secret",
    )
    monkeypatch.setenv("COGNEE_POSTGRES_PASSWORD", "secret")

    result = validator._check_cognee_postgres_env(settings)

    assert result.ok is True
    assert "cognee_benchmark" in result.detail


def test_validator_vector_env_reports_pgvector_target(monkeypatch) -> None:
    settings = validator.PostgresSettings(
        host="127.0.0.1",
        port=5433,
        username="benchmark",
        password="secret",
        cognee_database="cognee_benchmark",
        schema="cognee",
        vector_provider="pgvector",
        vector_database="cognee_benchmark",
        vector_schema="cognee",
        vector_password="secret",
    )

    result = validator._check_cognee_vector_env(settings)

    assert result.ok is True
    assert "provider=pgvector" in result.detail
    assert "database=cognee_benchmark" in result.detail


def test_validator_vector_env_rejects_benchmark_database() -> None:
    settings = validator.PostgresSettings(
        host="127.0.0.1",
        port=5433,
        username="benchmark",
        password="secret",
        cognee_database="cognee_benchmark",
        schema="cognee",
        vector_provider="pgvector",
        vector_database="benchmark",
        vector_schema="cognee",
        vector_password="secret",
    )

    result = validator._check_cognee_vector_env(settings)

    assert result.ok is False
    assert "benchmark database" in result.detail


def test_validator_graph_env_does_not_require_hardcoded_password(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "neo4j")
    monkeypatch.setenv("GRAPH_DATABASE_URL", "bolt://localhost:18689")
    monkeypatch.setenv("GRAPH_DATABASE_USERNAME", "neo4j")
    monkeypatch.setenv("GRAPH_DATABASE_NAME", "neo4j")
    monkeypatch.setenv("COGNEE_NEO4J_PASSWORD", "secret")

    result = validator._check_cognee_graph_env()

    assert result.ok is True
