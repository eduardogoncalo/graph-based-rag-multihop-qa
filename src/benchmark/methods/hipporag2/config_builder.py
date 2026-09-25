from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark.core.naming import slugify_dataset_version

HIPPORAG2_METHOD_ID = "hipporag2"

# Factory-default reader stack used across the benchmark (Bloco A): the same
# model bindings imposed on ms_graphrag/cognee. Everything else keeps the
# published HippoRAG 2 defaults (ICML 2025, Tab. 13: synonym 0.8, damping 0.5,
# temperature 0.0) — we do NOT re-tune.
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class Hipporag2Workspace:
    dataset_id: str
    dataset_version: str
    method_id: str
    artifact_dir: Path
    save_dir: Path        # HippoRAG(save_dir=...) — grafo igraph + embedding store
    raw_dir: Path         # stdout/stderr/commands preservados (auditoria)
    passage_map_path: Path  # sha1(text) -> document_id (traceabilidade doc_*)


def build_workspace(
    *,
    artifacts_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    method_id: str = HIPPORAG2_METHOD_ID,
) -> Hipporag2Workspace:
    if method_id != HIPPORAG2_METHOD_ID:
        raise ValueError("HippoRAG 2 adapter only supports method_id='hipporag2'")
    # slugified (dots -> underscores) to match the artifacts tree used by the
    # other substrates (artifacts/musique_ans_v1_0_eval1k/...).
    artifact_dir = (
        Path(artifacts_dir).resolve()
        / slugify_dataset_version(dataset_id, dataset_version)
        / method_id
    )
    workspace = Hipporag2Workspace(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        method_id=method_id,
        artifact_dir=artifact_dir,
        save_dir=artifact_dir / "workspace",
        raw_dir=artifact_dir / "raw",
        passage_map_path=artifact_dir / "workspace" / "passage_map.json",
    )
    _assert_within(workspace.save_dir, workspace.artifact_dir)
    _assert_within(workspace.raw_dir, workspace.artifact_dir)
    _assert_within(workspace.passage_map_path, workspace.artifact_dir)
    return workspace


def ensure_workspace(workspace: Hipporag2Workspace) -> None:
    workspace.save_dir.mkdir(parents=True, exist_ok=True)
    workspace.raw_dir.mkdir(parents=True, exist_ok=True)


def _assert_within(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes HippoRAG 2 artifact directory: {path}")
