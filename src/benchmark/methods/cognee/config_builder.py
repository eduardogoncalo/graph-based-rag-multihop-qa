from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.core.naming import slugify_dataset_version

COGNEE_METHOD_ID = "cognee"
COGNEE_GRAPH_PROVIDER = "neo4j"
COGNEE_GRAPH_URL = "bolt://localhost:18689"
COGNEE_GRAPH_USERNAME = "neo4j"
COGNEE_GRAPH_DATABASE = "neo4j"
COGNEE_NEO4J_PASSWORD_ENV = "COGNEE_NEO4J_PASSWORD"

COGNEE_RELATIONAL_PROVIDER = "postgres"
COGNEE_POSTGRES_HOST = "127.0.0.1"
COGNEE_POSTGRES_PORT = 15433
COGNEE_POSTGRES_USERNAME = "benchmark"
POSTGRES_BENCHMARK_DB = "benchmark"
# Nome da base do Cognee. Era `cognee_cuad_v1_50docs`, da linhagem CUAD que
# morreu em Junho de 2026 — um valor por omissão que nomeava um dataset que já
# não existe. Neutro de propósito: a separação entre datasets é feita pelo
# schema e pelo prefixo de tabela (ver `isolation_priority` no cognee.yaml),
# não pelo nome da base.
COGNEE_POSTGRES_DATABASE = "cognee_benchmark"
COGNEE_POSTGRES_SCHEMA = "cognee"
COGNEE_POSTGRES_PASSWORD_ENV = "COGNEE_POSTGRES_PASSWORD"
COGNEE_POSTGRES_HOST_ENV = "COGNEE_POSTGRES_HOST"
COGNEE_POSTGRES_PORT_ENV = "COGNEE_POSTGRES_PORT"
COGNEE_POSTGRES_USER_ENV = "COGNEE_POSTGRES_USER"
COGNEE_POSTGRES_DB_ENV = "COGNEE_POSTGRES_DB"
COGNEE_POSTGRES_SCHEMA_ENV = "COGNEE_POSTGRES_SCHEMA"

COGNEE_VECTOR_PROVIDER = "pgvector"
COGNEE_VECTOR_EXTENSION = "vector"
COGNEE_VECTOR_PROVIDER_ENV = "COGNEE_VECTOR_PROVIDER"
COGNEE_VECTOR_DB_HOST_ENV = "COGNEE_VECTOR_DB_HOST"
COGNEE_VECTOR_DB_PORT_ENV = "COGNEE_VECTOR_DB_PORT"
COGNEE_VECTOR_DB_USER_ENV = "COGNEE_VECTOR_DB_USER"
COGNEE_VECTOR_DB_PASSWORD_ENV = "COGNEE_VECTOR_DB_PASSWORD"
COGNEE_VECTOR_DB_NAME_ENV = "COGNEE_VECTOR_DB_NAME"
COGNEE_VECTOR_DB_SCHEMA_ENV = "COGNEE_VECTOR_DB_SCHEMA"

GRAPH_DATABASE_PROVIDER_ENV = "GRAPH_DATABASE_PROVIDER"
GRAPH_DATABASE_URL_ENV = "GRAPH_DATABASE_URL"
GRAPH_DATABASE_USERNAME_ENV = "GRAPH_DATABASE_USERNAME"
GRAPH_DATABASE_PASSWORD_ENV = "GRAPH_DATABASE_PASSWORD"
GRAPH_DATABASE_NAME_ENV = "GRAPH_DATABASE_NAME"

DB_PROVIDER_ENV = "DB_PROVIDER"
DB_HOST_ENV = "DB_HOST"
DB_PORT_ENV = "DB_PORT"
DB_USERNAME_ENV = "DB_USERNAME"
DB_PASSWORD_ENV = "DB_PASSWORD"
DB_NAME_ENV = "DB_NAME"
DB_SCHEMA_ENV = "DB_SCHEMA"

VECTOR_DB_PROVIDER_ENV = "VECTOR_DB_PROVIDER"
VECTOR_DB_HOST_ENV = "VECTOR_DB_HOST"
VECTOR_DB_PORT_ENV = "VECTOR_DB_PORT"
VECTOR_DB_USERNAME_ENV = "VECTOR_DB_USERNAME"
VECTOR_DB_PASSWORD_ENV = "VECTOR_DB_PASSWORD"
VECTOR_DB_NAME_ENV = "VECTOR_DB_NAME"
VECTOR_DB_SCHEMA_ENV = "VECTOR_DB_SCHEMA"

FORBIDDEN_COGNEE_ENV_PREFIXES = ("LIGHTRAG_NEO4J_", "GRAPHRAG_NEO4J_")


@dataclass(frozen=True)
class CogneeWorkspace:
    dataset_id: str
    dataset_version: str
    method_id: str
    artifact_dir: Path
    system_root: Path
    vector_root: Path
    db_root: Path
    neo4j_root: Path
    raw_dir: Path


@dataclass(frozen=True)
class CogneeGraphStoreConfig:
    provider: str = COGNEE_GRAPH_PROVIDER
    url: str = COGNEE_GRAPH_URL
    username: str = COGNEE_GRAPH_USERNAME
    password_env: str = COGNEE_NEO4J_PASSWORD_ENV
    database: str = COGNEE_GRAPH_DATABASE


@dataclass(frozen=True)
class CogneeRelationalStoreConfig:
    provider: str = COGNEE_RELATIONAL_PROVIDER
    host: str = COGNEE_POSTGRES_HOST
    port: int = COGNEE_POSTGRES_PORT
    username: str = COGNEE_POSTGRES_USERNAME
    password_env: str = COGNEE_POSTGRES_PASSWORD_ENV
    database: str = COGNEE_POSTGRES_DATABASE
    schema: str = COGNEE_POSTGRES_SCHEMA
    use_existing_container: bool = True


@dataclass(frozen=True)
class CogneeVectorStoreConfig:
    provider: str = COGNEE_VECTOR_PROVIDER
    host: str = COGNEE_POSTGRES_HOST
    port: int = COGNEE_POSTGRES_PORT
    username: str = COGNEE_POSTGRES_USERNAME
    password_env: str = COGNEE_POSTGRES_PASSWORD_ENV
    database: str = COGNEE_POSTGRES_DATABASE
    schema: str = COGNEE_POSTGRES_SCHEMA
    extension: str = COGNEE_VECTOR_EXTENSION
    use_existing_container: bool = True


@dataclass(frozen=True)
class CogneeLocalIndexProfile:
    """Connection/storage profile for an index built OUTSIDE the framework's
    default Postgres+pgvector deployment — i.e. cognee's own local stores
    (sqlite relational + lancedb vector under the workspace) plus an Option C
    Neo4j container. Mirrors byte-for-byte the env that
    ``scripts/cognee_index_musique_batched.py`` set at index time, so retrieval
    reads exactly what was written. When set on a ``CogneeConfig`` it overrides
    the Postgres/pgvector machinery in ``cognee_env``/``validate_config``."""

    graph_url: str
    graph_password: str  # literal value (matches indexer), not an env name
    graph_username: str = "neo4j"
    graph_database: str = "neo4j"
    vector_provider: str = "lancedb"
    relational_provider: str = "sqlite"
    # Pinned to the index-time models so the query embedding lands in the SAME
    # vector space as the indexed chunks (drift here silently breaks retrieval).
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"


@dataclass(frozen=True)
class CogneeConfig:
    method_id: str = COGNEE_METHOD_ID
    graph_store: CogneeGraphStoreConfig = field(default_factory=CogneeGraphStoreConfig)
    relational_store: CogneeRelationalStoreConfig = field(default_factory=CogneeRelationalStoreConfig)
    vector_store: CogneeVectorStoreConfig = field(default_factory=CogneeVectorStoreConfig)
    top_k: int = 5
    dataset_name: str = "benchmark_cognee"
    dataset_names: tuple[str, ...] = ()
    # When set, retrieval targets a self-contained local index (sqlite+lancedb+
    # Option C Neo4j) instead of the shared Postgres/pgvector deployment.
    local_index: CogneeLocalIndexProfile | None = None
    # Controlled-reader (Braço A) path keeps only_context=True: cognee returns the
    # resolved graph context and our fixed gpt-4o-mini reader answers. Setting
    # only_context=False runs cognee's OWN GRAPH_COMPLETION generation (native QA,
    # Braço B "as deployed") — the completion step then makes one gpt-4o-mini call
    # inside cognee and the answer surfaces as metadata["generated_answer"].
    only_context: bool = True


def build_workspace(
    *,
    artifacts_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    method_id: str = COGNEE_METHOD_ID,
) -> CogneeWorkspace:
    if method_id != COGNEE_METHOD_ID:
        raise ValueError("Cognee workspace only supports method_id='cognee'")
    # Slugify so the dir matches the Option C / indexer convention
    # (e.g. "ans_v1.0_eval1k" -> "musique_ans_v1_0_eval1k"). Para datasets cujo
    # "{id}_{version}" cru já não tenha '.' nem '-', o slug é idêntico.
    artifact_dir = Path(artifacts_dir) / slugify_dataset_version(dataset_id, dataset_version) / COGNEE_METHOD_ID
    workspace = CogneeWorkspace(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        method_id=method_id,
        artifact_dir=artifact_dir,
        system_root=artifact_dir / "system",
        vector_root=artifact_dir / "vector",
        db_root=artifact_dir / "db",
        neo4j_root=artifact_dir / "neo4j",
        raw_dir=artifact_dir / "raw",
    )
    for path in (
        workspace.artifact_dir,
        workspace.system_root,
        workspace.vector_root,
        workspace.db_root,
        workspace.neo4j_root,
        workspace.raw_dir,
    ):
        assert_within(path, workspace.artifact_dir)
    return workspace


def ensure_workspace(workspace: CogneeWorkspace) -> None:
    for path in (
        workspace.system_root,
        workspace.vector_root,
        workspace.db_root,
        workspace.neo4j_root,
        workspace.raw_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def cognee_env(
    *,
    workspace: CogneeWorkspace,
    config: CogneeConfig | None = None,
    include_neo4j_aliases: bool = True,
) -> dict[str, str]:
    resolved = config or CogneeConfig()
    validate_config(resolved)
    if resolved.local_index is not None:
        return _local_index_env(workspace=workspace, config=resolved)
    graph_password = _env_value(resolved.graph_store.password_env)
    relational = _resolved_relational_store(resolved.relational_store)
    relational_password = _env_value(resolved.relational_store.password_env)
    vector = _resolved_vector_store(resolved.vector_store)
    vector_password = _env_value(vector.password_env)
    model_env = _resolved_model_env()
    env = {
        "DATA_ROOT_DIRECTORY": str(workspace.raw_dir),
        "SYSTEM_ROOT_DIRECTORY": str(workspace.system_root),
        "CACHE_ROOT_DIRECTORY": str(workspace.db_root / "cache"),
        "COGNEE_LOGS_DIR": str(workspace.artifact_dir / "logs"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "CACHING": "false",
        GRAPH_DATABASE_PROVIDER_ENV: resolved.graph_store.provider,
        GRAPH_DATABASE_URL_ENV: resolved.graph_store.url,
        GRAPH_DATABASE_USERNAME_ENV: resolved.graph_store.username,
        GRAPH_DATABASE_PASSWORD_ENV: graph_password,
        GRAPH_DATABASE_NAME_ENV: resolved.graph_store.database,
        "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
        DB_PROVIDER_ENV: relational.provider,
        DB_HOST_ENV: relational.host,
        DB_PORT_ENV: str(relational.port),
        DB_USERNAME_ENV: relational.username,
        DB_PASSWORD_ENV: relational_password,
        DB_NAME_ENV: relational.database,
        DB_SCHEMA_ENV: relational.schema,
        "DB_PATH": str(workspace.db_root),
        VECTOR_DB_PROVIDER_ENV: vector.provider,
        VECTOR_DB_HOST_ENV: vector.host,
        VECTOR_DB_PORT_ENV: str(vector.port),
        VECTOR_DB_USERNAME_ENV: vector.username,
        VECTOR_DB_PASSWORD_ENV: vector_password,
        VECTOR_DB_NAME_ENV: vector.database,
        VECTOR_DB_SCHEMA_ENV: vector.schema,
        "VECTOR_DATASET_DATABASE_HANDLER": "pgvector",
        "VECTOR_DB_SUBPROCESS_ENABLED": "false",
        "COGNEE_SYSTEM_ROOT": str(workspace.system_root),
        "COGNEE_VECTOR_ROOT": str(workspace.vector_root),
        "COGNEE_DB_ROOT": str(workspace.db_root),
        **model_env,
    }
    if include_neo4j_aliases:
        env.update(
            {
                "NEO4J_URI": resolved.graph_store.url,
                "NEO4J_USER": resolved.graph_store.username,
                "NEO4J_PASSWORD": graph_password,
                "NEO4J_DATABASE": resolved.graph_store.database,
            }
        )
    return env


def validate_config(config: CogneeConfig) -> None:
    if config.method_id != COGNEE_METHOD_ID:
        raise ValueError("Cognee config must use method_id='cognee'")
    if config.local_index is not None:
        _validate_local_index(config)
        return
    if config.graph_store.provider != COGNEE_GRAPH_PROVIDER:
        raise ValueError("Cognee graph database provider must be 'neo4j'")
    if config.graph_store.url != COGNEE_GRAPH_URL:
        raise ValueError("Cognee must use bolt://localhost:18689")
    if config.graph_store.database != COGNEE_GRAPH_DATABASE:
        raise ValueError("Cognee must use Neo4j database 'neo4j'")
    if config.relational_store.provider != COGNEE_RELATIONAL_PROVIDER:
        raise ValueError("Cognee relational store provider must be 'postgres'")
    if config.relational_store.port <= 0:
        raise ValueError("Cognee Postgres port must be positive")
    if not config.relational_store.database:
        raise ValueError("Cognee Postgres database must be set")
    if not config.relational_store.schema:
        raise ValueError("Cognee Postgres schema must be set")
    if config.vector_store.provider != COGNEE_VECTOR_PROVIDER:
        raise ValueError("Cognee vector store provider must be 'pgvector'")
    if config.vector_store.database == POSTGRES_BENCHMARK_DB:
        raise ValueError("Cognee vector store must not use benchmark database")
    if config.vector_store.database != config.relational_store.database:
        raise ValueError("Cognee vector store must use the isolated Cognee Postgres database")
    if config.vector_store.schema != config.relational_store.schema:
        raise ValueError("Cognee vector store must use the isolated Cognee schema")
    if config.vector_store.extension != COGNEE_VECTOR_EXTENSION:
        raise ValueError("Cognee vector store extension must be 'vector'")
    if config.vector_store.port <= 0:
        raise ValueError("Cognee vector store port must be positive")
    if config.top_k <= 0:
        raise ValueError("top_k must be positive")


def _validate_local_index(config: CogneeConfig) -> None:
    profile = config.local_index
    if profile is None:  # pragma: no cover - guarded by caller
        raise ValueError("local index validation requires a CogneeLocalIndexProfile")
    if not profile.graph_url:
        raise ValueError("Cognee local index requires a graph_url")
    if not profile.graph_password:
        raise ValueError("Cognee local index requires a graph_password")
    if profile.relational_provider != "sqlite":
        raise ValueError("Cognee local index relational provider must be 'sqlite'")
    if profile.vector_provider != "lancedb":
        raise ValueError("Cognee local index vector provider must be 'lancedb'")
    if config.top_k <= 0:
        raise ValueError("top_k must be positive")


def _local_index_env(*, workspace: CogneeWorkspace, config: CogneeConfig) -> dict[str, str]:
    """Reproduce the indexer's env so retrieval reads the same local stores.

    Matches ``scripts/cognee_index_musique_batched.py`` exactly: roots under the
    workspace (lancedb + sqlite live in ``system/databases``), Option C Neo4j on
    the configured bolt URL, sqlite relational + lancedb vector. Model env comes
    from the shared resolver so the query embedding matches the index embedding.
    """
    profile = config.local_index
    assert profile is not None  # validated by validate_config
    model_env = _resolved_model_env()
    # cognee's BaseConfig rejects relative root paths; settings.artifacts_dir is
    # relative ("artifacts"), so resolve to absolute (against the project CWD,
    # exactly as the indexer's absolute --workspace did).
    return {
        "DATA_ROOT_DIRECTORY": str((workspace.artifact_dir / "data").resolve()),
        "SYSTEM_ROOT_DIRECTORY": str(workspace.system_root.resolve()),
        "CACHE_ROOT_DIRECTORY": str((workspace.artifact_dir / "cache").resolve()),
        "COGNEE_LOGS_DIR": str((workspace.artifact_dir / "logs").resolve()),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "CACHING": "false",
        "TELEMETRY_DISABLED": "true",
        GRAPH_DATABASE_PROVIDER_ENV: "neo4j",
        GRAPH_DATABASE_URL_ENV: profile.graph_url,
        GRAPH_DATABASE_USERNAME_ENV: profile.graph_username,
        GRAPH_DATABASE_PASSWORD_ENV: profile.graph_password,
        GRAPH_DATABASE_NAME_ENV: profile.graph_database,
        "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
        DB_PROVIDER_ENV: profile.relational_provider,
        VECTOR_DB_PROVIDER_ENV: profile.vector_provider,
        "NEO4J_URI": profile.graph_url,
        "NEO4J_USER": profile.graph_username,
        "NEO4J_PASSWORD": profile.graph_password,
        "NEO4J_DATABASE": profile.graph_database,
        **model_env,
        # Pin models AFTER model_env so index/query parity is guaranteed even if
        # the ambient EMBEDDING_MODEL/LLM_MODEL env drifts.
        "EMBEDDING_MODEL": profile.embedding_model,
        "EMBEDDING_DIMENSIONS": str(profile.embedding_dimensions),
        "LLM_MODEL": profile.llm_model,
    }


def validate_environment_isolation(env: dict[str, str] | None = None) -> None:
    import os

    resolved = dict(os.environ if env is None else env)
    forbidden = [
        name
        for name in resolved
        if any(name.startswith(prefix) for prefix in FORBIDDEN_COGNEE_ENV_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Cognee environment must not use method-specific Neo4j envs: {sorted(forbidden)}")


def validate_reset_target(target: str | Path, *, workspace: CogneeWorkspace) -> Path:
    resolved_target = Path(target).resolve()
    resolved_root = workspace.artifact_dir.resolve()
    if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
        raise ValueError(f"Cognee reset target escapes artifact directory: {target}")
    if resolved_target.is_symlink():
        raise ValueError(f"Cognee reset target must not be a symlink: {target}")
    for parent in [resolved_target, *resolved_target.parents]:
        if parent == resolved_root:
            break
        if parent.is_symlink():
            raise ValueError(f"Cognee reset target parent must not be a symlink: {parent}")
    return resolved_target


def reset_workspace_path(target: str | Path, *, workspace: CogneeWorkspace) -> None:
    resolved_target = validate_reset_target(target, workspace=workspace)
    if not resolved_target.exists():
        return
    if resolved_target.is_dir():
        shutil.rmtree(resolved_target)
    else:
        resolved_target.unlink()


def assert_within(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes Cognee artifact directory: {path}")


def _env_value(name: str) -> str:
    import os

    return os.environ.get(name, "")


def _resolved_model_env() -> dict[str, str]:
    import os

    try:
        from benchmark.core.settings import load_settings

        settings = load_settings()
    except Exception:
        settings = None

    openai_key = os.environ.get("OPENAI_API_KEY") or getattr(settings, "openai_api_key", None) or ""
    model_provider = (
        os.environ.get("LLM_PROVIDER")
        or os.environ.get("MODEL_PROVIDER")
        or getattr(settings, "model_provider", None)
        or ("openai" if openai_key else "")
    )
    chat_model = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_CHAT_MODEL")
        or getattr(settings, "openai_chat_model", None)
        or ""
    )
    embedding_provider = (
        os.environ.get("EMBEDDING_PROVIDER")
        or os.environ.get("MODEL_PROVIDER")
        or getattr(settings, "model_provider", None)
        or ("openai" if openai_key else "")
    )
    embedding_model = (
        os.environ.get("EMBEDDING_MODEL")
        or os.environ.get("OPENAI_EMBEDDING_MODEL")
        or getattr(settings, "openai_embedding_model", None)
        or ""
    )
    embedding_dimensions = (
        os.environ.get("EMBEDDING_DIMENSIONS")
        or os.environ.get("OPENAI_EMBEDDING_DIMENSIONS")
        or str(getattr(settings, "openai_embedding_dimensions", "") or "")
    )
    llm_key = os.environ.get("LLM_API_KEY") or openai_key
    embedding_key = os.environ.get("EMBEDDING_API_KEY") or openai_key
    litellm_key = os.environ.get("LITELLM_API_KEY") or openai_key

    return {
        "OPENAI_API_KEY": openai_key,
        "LLM_PROVIDER": model_provider,
        "LLM_MODEL": chat_model,
        "LLM_API_KEY": llm_key,
        "BAML_LLM_PROVIDER": model_provider,
        "BAML_LLM_MODEL": chat_model,
        "BAML_LLM_API_KEY": llm_key,
        "EMBEDDING_PROVIDER": embedding_provider,
        "EMBEDDING_MODEL": embedding_model,
        "EMBEDDING_API_KEY": embedding_key,
        "EMBEDDING_DIMENSIONS": embedding_dimensions,
        "LITELLM_API_KEY": litellm_key,
    }


def _resolved_relational_store(store: CogneeRelationalStoreConfig) -> CogneeRelationalStoreConfig:
    import os

    return CogneeRelationalStoreConfig(
        provider=store.provider,
        host=os.environ.get(COGNEE_POSTGRES_HOST_ENV, store.host),
        port=int(os.environ.get(COGNEE_POSTGRES_PORT_ENV, str(store.port))),
        username=os.environ.get(COGNEE_POSTGRES_USER_ENV, store.username),
        password_env=store.password_env,
        database=os.environ.get(COGNEE_POSTGRES_DB_ENV, store.database),
        schema=os.environ.get(COGNEE_POSTGRES_SCHEMA_ENV, store.schema),
        use_existing_container=store.use_existing_container,
    )


def _resolved_vector_store(store: CogneeVectorStoreConfig) -> CogneeVectorStoreConfig:
    import os

    return CogneeVectorStoreConfig(
        provider=os.environ.get(COGNEE_VECTOR_PROVIDER_ENV, store.provider),
        host=os.environ.get(COGNEE_VECTOR_DB_HOST_ENV, os.environ.get(COGNEE_POSTGRES_HOST_ENV, store.host)),
        port=int(os.environ.get(COGNEE_VECTOR_DB_PORT_ENV, os.environ.get(COGNEE_POSTGRES_PORT_ENV, str(store.port)))),
        username=os.environ.get(COGNEE_VECTOR_DB_USER_ENV, os.environ.get(COGNEE_POSTGRES_USER_ENV, store.username)),
        password_env=COGNEE_VECTOR_DB_PASSWORD_ENV
        if os.environ.get(COGNEE_VECTOR_DB_PASSWORD_ENV)
        else store.password_env,
        database=os.environ.get(COGNEE_VECTOR_DB_NAME_ENV, os.environ.get(COGNEE_POSTGRES_DB_ENV, store.database)),
        schema=os.environ.get(COGNEE_VECTOR_DB_SCHEMA_ENV, os.environ.get(COGNEE_POSTGRES_SCHEMA_ENV, store.schema)),
        extension=store.extension,
        use_existing_container=store.use_existing_container,
    )
