from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from benchmark.core.naming import slugify_dataset_version

LIGHTRAG_NEO4J_METHOD_ID = "lightrag_neo4j"
LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID = "lightrag_neo4j_doc_scoped_strict"
LIGHTRAG_NEO4J_BACKING_METHOD_IDS = {
    LIGHTRAG_NEO4J_METHOD_ID,
    LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
}
LIGHTRAG_NEO4J_URI_ENV = "LIGHTRAG_NEO4J_URI"
LIGHTRAG_NEO4J_USER_ENV = "LIGHTRAG_NEO4J_USER"
LIGHTRAG_NEO4J_PASSWORD_ENV = "LIGHTRAG_NEO4J_PASSWORD"
LIGHTRAG_NEO4J_DATABASE_ENV = "LIGHTRAG_NEO4J_DATABASE"
FORBIDDEN_NEO4J_URI_ENVS = {"NEO4J_URI", "GRAPHRAG_NEO4J_URI"}

# --- Reader context representation (what the constant reader receives) ---------
# Well-defined knob selecting HOW much of LightRAG's retrieval reaches the reader.
# LightRAG natively assembles entities + relationships + chunks. THREE modes:
#   "chunks"         -> only `data.chunks` (raw passages), matched to vector_rag's units.
#                       Deliberately drops the graph structure so the reader input is
#                       comparable to the dense baseline. DEFAULT (byte-identical to every
#                       run to date).
#   "structured"     -> OUR reconstruction of LightRAG's full context: entities +
#                       relationships + chunks, serialized by our output_parser from the
#                       query_data payload (no native generation). Isolates the saliency of
#                       graph structure. (alias: "structured_parsed")
#   "native_context" -> LightRAG's OWN assembled context STRING, via only_need_context=True
#                       (the `kg_query_context` template for graph modes; `naive_query_context`
#                       = chunks for mode="naive"). The truest analogue of cognee's
#                       only_context=True: the reader receives EXACTLY the string the lib
#                       builds, as a single block. Generation is skipped, so NO llm_response
#                       ever reaches the reader; doc_ids are recovered from raw_data.chunks so
#                       evidence_recall stays calculable. top_k is parameterizable via
#                       LIGHTRAG_NATIVE_TOPK (default = the caller's top_k, e.g. 40, for
#                       comparability; "default"/"none" = LightRAG's own default breadth).
# Resolved from LIGHTRAG_READER_CONTEXT (mirrors READER_GROUNDING / LIGHTRAG_DISABLE_LLM_CACHE).
LightRAGReaderContext = Literal["chunks", "structured", "native_context"]
READER_CONTEXT_CHUNKS: LightRAGReaderContext = "chunks"
READER_CONTEXT_STRUCTURED: LightRAGReaderContext = "structured"
READER_CONTEXT_NATIVE: LightRAGReaderContext = "native_context"
LIGHTRAG_NEO4J_READER_CONTEXT_ENV = "LIGHTRAG_READER_CONTEXT"
_STRUCTURED_ALIASES = {"structured", "structured_parsed", "graph", "full", "entities_relations"}
_NATIVE_ALIASES = {
    "native_context", "native", "only_context", "only_need_context", "context_string", "lib_context",
}


def resolve_lightrag_reader_context(explicit: str | None = None) -> LightRAGReaderContext:
    """Single source of truth for the reader-context mode. ``explicit`` (e.g. a
    config field or CLI flag) wins; otherwise read LIGHTRAG_READER_CONTEXT; default
    ``chunks`` so unset env == current behavior. Resolution order: native_context,
    then structured, else chunks."""
    raw = explicit if explicit is not None else os.getenv(
        LIGHTRAG_NEO4J_READER_CONTEXT_ENV, READER_CONTEXT_CHUNKS
    )
    value = str(raw).strip().lower()
    if value in _NATIVE_ALIASES:
        return READER_CONTEXT_NATIVE
    if value in _STRUCTURED_ALIASES:
        return READER_CONTEXT_STRUCTURED
    return READER_CONTEXT_CHUNKS


@dataclass(frozen=True)
class LightRAGNeo4jWorkspace:
    dataset_id: str
    dataset_version: str
    method_id: str
    artifact_dir: Path
    workspace_dir: Path
    input_dir: Path
    raw_dir: Path
    native_dir: Path


@dataclass(frozen=True)
class LightRAGNeo4jConfig:
    method_id: str = LIGHTRAG_NEO4J_METHOD_ID
    neo4j_uri_env: str = LIGHTRAG_NEO4J_URI_ENV
    neo4j_user_env: str = LIGHTRAG_NEO4J_USER_ENV
    neo4j_password_env: str = LIGHTRAG_NEO4J_PASSWORD_ENV
    neo4j_database_env: str = LIGHTRAG_NEO4J_DATABASE_ENV
    # Option C explicit connection overrides. When set (e.g. resolved per
    # dataset from the port registry), these take precedence over reading the
    # corresponding *_env variable, so the adapter connects to the
    # dataset-isolated container instead of any global LIGHTRAG_NEO4J_URI.
    # They are injected into the LightRAG process env at runtime only and are
    # never written to config.json (the password must not hit disk).
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None
    dependency_mode: str = "optional"
    query_mode: str = "mix"
    top_k: int = 5
    # None = todos os documentos. Era 5, um valor de desenvolvimento que
    # limitava o índice em silêncio; ver o comentário no YAML do método.
    max_documents: int | None = None
    disable_llm_cache_for_query: bool = field(
        default_factory=lambda: _env_bool("LIGHTRAG_DISABLE_LLM_CACHE", False)
    )
    # What the reader receives from LightRAG: "chunks" (default, raw passages,
    # matched to the dense baseline) or "structured" (entities + relationships +
    # chunks — the graph method's full retrieved context). See the module-level
    # docstring on READER_CONTEXT_* and resolve_lightrag_reader_context.
    reader_context: LightRAGReaderContext = field(
        default_factory=resolve_lightrag_reader_context
    )


def build_workspace(
    *,
    artifacts_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    method_id: str = LIGHTRAG_NEO4J_METHOD_ID,
) -> LightRAGNeo4jWorkspace:
    if method_id not in LIGHTRAG_NEO4J_BACKING_METHOD_IDS:
        raise ValueError(
            "LightRAG Neo4j adapter only supports method_id='lightrag_neo4j' "
            "or 'lightrag_neo4j_doc_scoped_strict'"
        )
    artifact_method_id = (
        LIGHTRAG_NEO4J_METHOD_ID
        if method_id == LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID
        else method_id
    )
    dataset_slug = slugify_dataset_version(dataset_id, dataset_version)
    artifact_dir = Path(artifacts_dir) / dataset_slug / artifact_method_id
    workspace = LightRAGNeo4jWorkspace(
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


def ensure_workspace(workspace: LightRAGNeo4jWorkspace) -> None:
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    workspace.raw_dir.mkdir(parents=True, exist_ok=True)
    workspace.native_dir.mkdir(parents=True, exist_ok=True)


def build_config_file(
    workspace: LightRAGNeo4jWorkspace,
    *,
    config: LightRAGNeo4jConfig | None = None,
) -> Path:
    ensure_workspace(workspace)
    resolved = config or LightRAGNeo4jConfig()
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
                "query_mode": resolved.query_mode,
                "top_k": resolved.top_k,
                "max_documents": resolved.max_documents,
                "disable_llm_cache_for_query": resolved.disable_llm_cache_for_query,
                "reader_context": resolved.reader_context,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _assert_within(path, workspace.artifact_dir)
    return path


def validate_neo4j_env_names(config: LightRAGNeo4jConfig) -> None:
    if config.method_id not in LIGHTRAG_NEO4J_BACKING_METHOD_IDS:
        raise ValueError(
            "LightRAG Neo4j config must use method_id='lightrag_neo4j' "
            "or 'lightrag_neo4j_doc_scoped_strict'"
        )
    if config.neo4j_uri_env != LIGHTRAG_NEO4J_URI_ENV:
        if config.neo4j_uri_env == "NEO4J_URI":
            raise ValueError("lightrag_neo4j must not use generic NEO4J_URI")
        if config.neo4j_uri_env == "GRAPHRAG_NEO4J_URI":
            raise ValueError("lightrag_neo4j must not use GRAPHRAG_NEO4J_URI")
        raise ValueError("lightrag_neo4j must use LIGHTRAG_NEO4J_URI")
    forbidden = {
        config.neo4j_user_env,
        config.neo4j_password_env,
        config.neo4j_database_env,
    } & FORBIDDEN_NEO4J_URI_ENVS
    if forbidden:
        raise ValueError(f"Forbidden Neo4j env names for lightrag_neo4j: {sorted(forbidden)}")


def _assert_within(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes LightRAG Neo4j artifact directory: {path}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
