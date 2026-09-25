from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from benchmark.core.naming import slugify_dataset_version

MS_GRAPHRAG_METHOD_ID = "ms_graphrag"

# Factory-default reader stack used across the benchmark (Bloco A):
# the only settings we impose on GraphRAG are the model bindings; every
# other knob keeps the `graphrag init` default (protocol: factory defaults).
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class GraphRAGWorkspace:
    dataset_id: str
    dataset_version: str
    method_id: str
    artifact_dir: Path
    workspace_dir: Path
    input_dir: Path
    raw_dir: Path


def build_workspace(
    *,
    artifacts_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    method_id: str = MS_GRAPHRAG_METHOD_ID,
) -> GraphRAGWorkspace:
    if method_id != MS_GRAPHRAG_METHOD_ID:
        raise ValueError("Microsoft GraphRAG adapter only supports method_id='ms_graphrag'")
    # slugified (dots -> underscores) to match the artifacts tree used by
    # cognee/lightrag (artifacts/musique_ans_v1_0_eval1k/...).
    artifact_dir = (
        Path(artifacts_dir).resolve() / slugify_dataset_version(dataset_id, dataset_version) / method_id
    )
    workspace_dir = artifact_dir / "workspace"
    input_dir = workspace_dir / "input"
    raw_dir = artifact_dir / "raw"
    workspace = GraphRAGWorkspace(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        method_id=method_id,
        artifact_dir=artifact_dir,
        workspace_dir=workspace_dir,
        input_dir=input_dir,
        raw_dir=raw_dir,
    )
    _assert_within(workspace.workspace_dir, workspace.artifact_dir)
    _assert_within(workspace.input_dir, workspace.artifact_dir)
    _assert_within(workspace.raw_dir, workspace.artifact_dir)
    return workspace


def ensure_workspace(workspace: GraphRAGWorkspace) -> None:
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    workspace.raw_dir.mkdir(parents=True, exist_ok=True)


def build_config_files(
    workspace: GraphRAGWorkspace,
    *,
    chat_model: str = DEFAULT_CHAT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Path]:
    """Bind the benchmark's models onto the ``graphrag init`` settings.

    Factory-defaults protocol: when ``settings.yaml`` already exists (written by
    ``graphrag init``) it is PATCHED in place — only the model bindings and the
    CSV input block change; every other knob keeps the shipped default. The
    minimal file is written from scratch only when init has not run (unit tests).
    """
    ensure_workspace(workspace)
    env_path = workspace.workspace_dir / ".env"
    settings_path = workspace.workspace_dir / "settings.yaml"
    env_path.write_text("GRAPHRAG_API_KEY=${GRAPHRAG_API_KEY}\n", encoding="utf-8")

    if settings_path.exists():
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    else:
        settings = {}

    settings.pop("models", None)  # alien section from earlier adapter versions
    completions = settings.setdefault("completion_models", {})
    chat = completions.setdefault("default_completion_model", {})
    chat.update({"model_provider": "openai", "model": chat_model, "api_key": "${GRAPHRAG_API_KEY}"})
    embeddings = settings.setdefault("embedding_models", {})
    embedding = embeddings.setdefault("default_embedding_model", {})
    embedding.update(
        {"model_provider": "openai", "model": embedding_model, "api_key": "${GRAPHRAG_API_KEY}"}
    )

    # CSV input with an `id` column preserves the benchmark's doc_* namespace
    # end-to-end (documents.id -> text_units.document_id): required for
    # document-level evidence recall.
    input_block = settings.setdefault("input", {})
    key = "file_type" if "file_type" in input_block or "storage" in input_block else "type"
    input_block[key] = "csv"
    # `$$` because graphrag's load_config runs string.Template.substitute over
    # the raw yaml text (a bare `$` raises "Invalid placeholder").
    input_block["file_pattern"] = r".*\.csv$$"
    input_block["id_column"] = "id"
    input_block["text_column"] = "text"
    input_block["title_column"] = "title"

    settings_path.write_text(
        "# Managed by the benchmark adapter: model bindings + CSV input only;\n"
        "# all other values are `graphrag init` factory defaults.\n"
        + yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _assert_within(env_path, workspace.artifact_dir)
    _assert_within(settings_path, workspace.artifact_dir)
    return {"env": env_path, "settings": settings_path}


def _assert_within(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes Microsoft GraphRAG artifact directory: {path}")
