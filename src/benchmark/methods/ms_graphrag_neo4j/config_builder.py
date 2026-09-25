from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MS_GRAPHRAG_NEO4J_METHOD_ID = "ms_graphrag_neo4j"
GRAPHRAG_NEO4J_URI_ENV = "GRAPHRAG_NEO4J_URI"
GRAPHRAG_NEO4J_USER_ENV = "GRAPHRAG_NEO4J_USER"
GRAPHRAG_NEO4J_PASSWORD_ENV = "GRAPHRAG_NEO4J_PASSWORD"
GRAPHRAG_NEO4J_DATABASE_ENV = "GRAPHRAG_NEO4J_DATABASE"
FORBIDDEN_NEO4J_URI_ENVS = {"NEO4J_URI", "LIGHTRAG_NEO4J_URI"}


@dataclass(frozen=True)
class GraphRAGNeo4jWorkspace:
    dataset_id: str
    dataset_version: str
    method_id: str
    artifact_dir: Path
    workspace_dir: Path
    input_dir: Path
    raw_dir: Path
    native_dir: Path


@dataclass(frozen=True)
class GraphRAGNeo4jConfig:
    method_id: str = MS_GRAPHRAG_NEO4J_METHOD_ID
    neo4j_uri_env: str = GRAPHRAG_NEO4J_URI_ENV
    neo4j_user_env: str = GRAPHRAG_NEO4J_USER_ENV
    neo4j_password_env: str = GRAPHRAG_NEO4J_PASSWORD_ENV
    neo4j_database_env: str = GRAPHRAG_NEO4J_DATABASE_ENV
    dependency_mode: str = "optional"
    top_k: int = 5
    # None = todos os documentos. Era 5, um valor de desenvolvimento que
    # limitava o índice em silêncio; ver o comentário no YAML do método.
    max_documents: int | None = None


def build_workspace(
    *,
    artifacts_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    method_id: str = MS_GRAPHRAG_NEO4J_METHOD_ID,
) -> GraphRAGNeo4jWorkspace:
    if method_id != MS_GRAPHRAG_NEO4J_METHOD_ID:
        raise ValueError("GraphRAG Neo4j adapter only supports method_id='ms_graphrag_neo4j'")
    artifact_dir = Path(artifacts_dir) / f"{dataset_id}_{dataset_version}" / method_id
    workspace = GraphRAGNeo4jWorkspace(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        method_id=method_id,
        artifact_dir=artifact_dir,
        workspace_dir=artifact_dir / "workspace",
        input_dir=artifact_dir / "workspace" / "input",
        raw_dir=artifact_dir / "raw",
        native_dir=artifact_dir / "native",
    )
    for path in (
        workspace.artifact_dir,
        workspace.workspace_dir,
        workspace.input_dir,
        workspace.raw_dir,
        workspace.native_dir,
    ):
        _assert_within(path, workspace.artifact_dir)
    return workspace


def ensure_workspace(workspace: GraphRAGNeo4jWorkspace) -> None:
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    workspace.raw_dir.mkdir(parents=True, exist_ok=True)
    workspace.native_dir.mkdir(parents=True, exist_ok=True)


def build_config_file(
    workspace: GraphRAGNeo4jWorkspace,
    *,
    config: GraphRAGNeo4jConfig | None = None,
) -> Path:
    ensure_workspace(workspace)
    resolved = config or GraphRAGNeo4jConfig()
    validate_neo4j_env_names(resolved)
    path = workspace.workspace_dir / "config.json"
    path.write_text(
        json.dumps(
            {
                "method_id": resolved.method_id,
                "dataset_id": workspace.dataset_id,
                "dataset_version": workspace.dataset_version,
                "artifact_dir": str(workspace.artifact_dir),
                "input_dir": str(workspace.input_dir),
                "raw_dir": str(workspace.raw_dir),
                "native_dir": str(workspace.native_dir),
                "neo4j_uri_env": resolved.neo4j_uri_env,
                "neo4j_user_env": resolved.neo4j_user_env,
                "neo4j_password_env": resolved.neo4j_password_env,
                "neo4j_database_env": resolved.neo4j_database_env,
                "dependency_mode": resolved.dependency_mode,
                "top_k": resolved.top_k,
                "max_documents": resolved.max_documents,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _assert_within(path, workspace.artifact_dir)
    return path


def validate_neo4j_env_names(config: GraphRAGNeo4jConfig) -> None:
    if config.method_id != MS_GRAPHRAG_NEO4J_METHOD_ID:
        raise ValueError("GraphRAG Neo4j config must use method_id='ms_graphrag_neo4j'")
    if config.neo4j_uri_env != GRAPHRAG_NEO4J_URI_ENV:
        if config.neo4j_uri_env == "NEO4J_URI":
            raise ValueError("ms_graphrag_neo4j must not use generic NEO4J_URI")
        if config.neo4j_uri_env == "LIGHTRAG_NEO4J_URI":
            raise ValueError("ms_graphrag_neo4j must not use LIGHTRAG_NEO4J_URI")
        raise ValueError("ms_graphrag_neo4j must use GRAPHRAG_NEO4J_URI")
    forbidden = {
        config.neo4j_user_env,
        config.neo4j_password_env,
        config.neo4j_database_env,
    } & FORBIDDEN_NEO4J_URI_ENVS
    if forbidden:
        raise ValueError(f"Forbidden Neo4j env names for ms_graphrag_neo4j: {sorted(forbidden)}")


def _assert_within(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes GraphRAG Neo4j artifact directory: {path}")
