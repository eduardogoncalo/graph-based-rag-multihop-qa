from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.methods.cognee.config_builder import (
    CogneeConfig,
    CogneeVectorStoreConfig,
    build_workspace,
    cognee_env,
    reset_workspace_path,
    validate_environment_isolation,
    validate_reset_target,
)


def test_cognee_config_has_no_api_key() -> None:
    config = CogneeConfig()

    assert not hasattr(config, "api_key")
    assert "api" not in repr(config).lower()


def test_cognee_env_uses_only_cognee_graph_vars(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    env = cognee_env(workspace=workspace)

    assert env["GRAPH_DATABASE_URL"] == "bolt://localhost:18689"
    assert env["GRAPH_DATABASE_PASSWORD"] == ""
    assert env["SYSTEM_ROOT_DIRECTORY"].endswith("cognee/system")
    assert env["DATA_ROOT_DIRECTORY"].endswith("cognee/raw")
    assert env["ENABLE_BACKEND_ACCESS_CONTROL"] == "false"
    assert env["VECTOR_DATASET_DATABASE_HANDLER"] == "pgvector"
    assert "LIGHTRAG_NEO4J_URI" not in env
    assert "GRAPHRAG_NEO4J_URI" not in env
    assert "DATABASE_URL" not in env


def test_cognee_env_uses_scoped_password_envs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGNEE_NEO4J_PASSWORD", "neo4j-secret")
    monkeypatch.setenv("COGNEE_POSTGRES_PASSWORD", "postgres-secret")
    monkeypatch.setenv("COGNEE_POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("COGNEE_POSTGRES_PORT", "5433")
    monkeypatch.setenv("COGNEE_POSTGRES_USER", "benchmark")
    monkeypatch.setenv("COGNEE_POSTGRES_DB", "cognee_benchmark")
    monkeypatch.setenv("COGNEE_POSTGRES_SCHEMA", "cognee")
    monkeypatch.setenv("COGNEE_VECTOR_PROVIDER", "pgvector")
    monkeypatch.setenv("COGNEE_VECTOR_DB_NAME", "cognee_benchmark")
    monkeypatch.setenv("COGNEE_VECTOR_DB_SCHEMA", "cognee")
    monkeypatch.setenv("DATABASE_URL", "postgresql://benchmark:benchmark@127.0.0.1:5433/benchmark")
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    env = cognee_env(workspace=workspace)

    assert env["GRAPH_DATABASE_PASSWORD"] == "neo4j-secret"
    assert env["NEO4J_PASSWORD"] == "neo4j-secret"
    assert env["DB_PROVIDER"] == "postgres"
    assert env["DB_HOST"] == "127.0.0.1"
    assert env["DB_PORT"] == "5433"
    assert env["DB_USERNAME"] == "benchmark"
    assert env["DB_PASSWORD"] == "postgres-secret"
    assert env["DB_NAME"] == "cognee_benchmark"
    assert env["DB_SCHEMA"] == "cognee"
    assert env["VECTOR_DB_PROVIDER"] == "pgvector"
    assert env["VECTOR_DB_NAME"] == "cognee_benchmark"
    assert env["VECTOR_DB_SCHEMA"] == "cognee"
    assert env["VECTOR_DB_PASSWORD"] == "postgres-secret"
    assert "DATABASE_URL" not in env


def test_cognee_env_maps_project_openai_models_to_cognee_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://benchmark:benchmark@127.0.0.1:5433/benchmark")
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    env = cognee_env(workspace=workspace)

    assert env["LLM_PROVIDER"] == "openai"
    assert env["LLM_MODEL"] == "gpt-4o-mini"
    assert env["LLM_API_KEY"] == "openai-secret"
    assert env["EMBEDDING_PROVIDER"] == "openai"
    assert env["EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert env["EMBEDDING_DIMENSIONS"] == "1536"
    assert env["EMBEDDING_API_KEY"] == "openai-secret"
    assert env["LITELLM_API_KEY"] == "openai-secret"
    assert "DATABASE_URL" not in env


def test_cognee_config_separates_graph_relational_and_vector_stores() -> None:
    config = CogneeConfig()

    assert config.graph_store.provider == "neo4j"
    assert config.graph_store.url == "bolt://localhost:18689"
    assert config.graph_store.password_env == "COGNEE_NEO4J_PASSWORD"
    assert config.relational_store.provider == "postgres"
    assert config.relational_store.database == "cognee_benchmark"
    assert config.relational_store.schema == "cognee"
    assert config.relational_store.password_env == "COGNEE_POSTGRES_PASSWORD"
    assert config.vector_store.provider == "pgvector"
    assert config.vector_store.database == "cognee_benchmark"
    assert config.vector_store.database != "benchmark"
    assert config.vector_store.schema == "cognee"
    assert config.vector_store.extension == "vector"


def test_cognee_config_rejects_vector_store_on_benchmark_database() -> None:
    config = CogneeConfig(vector_store=CogneeVectorStoreConfig(database="benchmark"))

    with pytest.raises(ValueError, match="benchmark database"):
        cognee_env(
            workspace=build_workspace(
                artifacts_dir=Path("/tmp"),
                dataset_id="musique_smoke_20",
                dataset_version="v1",
            ),
            config=config,
        )


def test_environment_isolation_rejects_lightrag_and_graphrag_envs() -> None:
    with pytest.raises(ValueError, match="LIGHTRAG_NEO4J_URI"):
        validate_environment_isolation({"LIGHTRAG_NEO4J_URI": "bolt://localhost:7688"})

    with pytest.raises(ValueError, match="GRAPHRAG_NEO4J_URI"):
        validate_environment_isolation({"GRAPHRAG_NEO4J_URI": "bolt://localhost:7687"})


def test_cognee_paths_stay_under_cognee_root(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    assert workspace.artifact_dir == tmp_path / "musique_smoke_20_v1" / "cognee"
    for path in (
        workspace.system_root,
        workspace.vector_root,
        workspace.db_root,
        workspace.neo4j_root,
        workspace.raw_dir,
    ):
        assert workspace.artifact_dir.resolve() in path.resolve().parents


def test_reset_safety_rejects_path_outside_cognee_root(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    with pytest.raises(ValueError, match="escapes"):
        validate_reset_target(tmp_path / "musique_smoke_20_v1" / "lightrag", workspace=workspace)


def test_reset_workspace_path_removes_only_valid_cognee_path(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )
    target = workspace.system_root / "tmp.txt"
    target.parent.mkdir(parents=True)
    target.write_text("ok", encoding="utf-8")

    reset_workspace_path(target, workspace=workspace)

    assert not target.exists()


def test_reset_safety_rejects_postgres_like_external_target(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    with pytest.raises(ValueError, match="escapes"):
        validate_reset_target(tmp_path / "postgres", workspace=workspace)
