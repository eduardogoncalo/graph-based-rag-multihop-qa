from __future__ import annotations

from pathlib import Path

import pytest

from scripts import bootstrap_cognee_postgres as bootstrap


def test_bootstrap_postgres_defaults_to_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COGNEE_POSTGRES_PASSWORD", raising=False)

    args = bootstrap.parse_args([])
    plan = bootstrap.build_plan(args)

    assert plan.dry_run is True
    assert plan.execute is False
    assert plan.database_name == "cognee_benchmark"
    assert plan.schema_name == "cognee"
    assert "CREATE EXTENSION IF NOT EXISTS vector;" not in plan.sql


def test_bootstrap_plan_prints_create_sql_only_with_explicit_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGNEE_POSTGRES_PASSWORD", "secret")

    args = bootstrap.parse_args(["--create-database", "--create-schema", "--database-name", "cognee_benchmark"])
    plan = bootstrap.build_plan(args)

    assert plan.execute is False
    assert "CREATE DATABASE cognee_benchmark;" in plan.sql
    assert "CREATE SCHEMA IF NOT EXISTS cognee;" in plan.sql
    assert all("benchmark." not in statement for statement in plan.sql)


def test_bootstrap_script_has_no_destructive_sql() -> None:
    source = Path("scripts/bootstrap_cognee_postgres.py").read_text(encoding="utf-8").upper()

    assert "DROP " not in source
    assert "DROP EXTENSION" not in source
    assert "TRUNCATE " not in source
    assert "DELETE " not in source
    assert "VECTOR_RAG_CHUNK_EMBEDDINGS" not in source


def test_bootstrap_rejects_unsafe_identifiers() -> None:
    with pytest.raises(ValueError):
        bootstrap.validate_identifier("benchmark; SELECT 1")


def test_bootstrap_vector_extension_sql_requires_explicit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGNEE_POSTGRES_PASSWORD", "secret")

    without_flag = bootstrap.build_plan(bootstrap.parse_args(["--create-database", "--create-schema"]))
    with_flag = bootstrap.build_plan(bootstrap.parse_args(["--create-vector-extension"]))

    assert "CREATE EXTENSION IF NOT EXISTS vector;" not in without_flag.sql
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in with_flag.sql


def test_bootstrap_rejects_vector_extension_in_benchmark_database() -> None:
    args = bootstrap.parse_args(["--create-vector-extension", "--database-name", "benchmark"])

    with pytest.raises(ValueError, match="benchmark database"):
        bootstrap.build_plan(args)
