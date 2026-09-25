from pathlib import Path

import pytest

from benchmark.core.config_loader import load_method_config
from benchmark.methods.ms_graphrag_neo4j.config_builder import (
    MS_GRAPHRAG_NEO4J_METHOD_ID,
    GraphRAGNeo4jConfig,
    build_config_file,
    build_workspace,
)


def test_ms_graphrag_neo4j_method_config_parses() -> None:
    config = load_method_config("configs/methods/ms_graphrag_neo4j.yaml")

    assert config.method_id == MS_GRAPHRAG_NEO4J_METHOD_ID
    assert config.neo4j_uri_env == "GRAPHRAG_NEO4J_URI"
    assert config.neo4j_user_env == "GRAPHRAG_NEO4J_USER"
    assert config.neo4j_password_env == "GRAPHRAG_NEO4J_PASSWORD"
    assert config.neo4j_database_env == "GRAPHRAG_NEO4J_DATABASE"
    assert config.artifact_subdir == MS_GRAPHRAG_NEO4J_METHOD_ID
    assert config.dependency_mode == "optional"
    assert config.index["max_documents"] is None
    assert config.query["top_k"] == 5


def test_workspace_paths_are_method_isolated(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    assert workspace.method_id == MS_GRAPHRAG_NEO4J_METHOD_ID
    assert workspace.artifact_dir == tmp_path / "musique_smoke_20_v1" / "ms_graphrag_neo4j"
    assert workspace.workspace_dir == workspace.artifact_dir / "workspace"
    assert workspace.input_dir == workspace.workspace_dir / "input"
    assert workspace.raw_dir == workspace.artifact_dir / "raw"
    assert workspace.native_dir == workspace.artifact_dir / "native"


def test_config_file_records_graphrag_neo4j_env_names(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    path = build_config_file(workspace)

    text = path.read_text(encoding="utf-8")
    assert '"neo4j_uri_env": "GRAPHRAG_NEO4J_URI"' in text
    assert "LIGHTRAG_NEO4J_URI" not in text
    assert '"method_id": "ms_graphrag_neo4j"' in text


def test_config_rejects_generic_neo4j_uri(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    with pytest.raises(ValueError, match="generic NEO4J_URI"):
        build_config_file(workspace, config=GraphRAGNeo4jConfig(neo4j_uri_env="NEO4J_URI"))


def test_config_rejects_lightrag_neo4j_uri(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    with pytest.raises(ValueError, match="LIGHTRAG_NEO4J_URI"):
        build_config_file(
            workspace,
            config=GraphRAGNeo4jConfig(neo4j_uri_env="LIGHTRAG_NEO4J_URI"),
        )


def test_workspace_rejects_other_method_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ms_graphrag_neo4j"):
        build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            method_id="ms_graphrag",
        )
