from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LIGHTRAG_METHOD_ID = "lightrag"


@dataclass(frozen=True)
class LightRAGWorkspace:
    dataset_id: str
    dataset_version: str
    method_id: str
    artifact_dir: Path
    workspace_dir: Path
    input_dir: Path
    working_dir: Path
    raw_dir: Path


def build_workspace(
    *,
    artifacts_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    method_id: str = LIGHTRAG_METHOD_ID,
) -> LightRAGWorkspace:
    if method_id != LIGHTRAG_METHOD_ID:
        raise ValueError("LightRAG adapter only supports method_id='lightrag'")
    artifact_dir = Path(artifacts_dir) / f"{dataset_id}_{dataset_version}" / method_id
    workspace = LightRAGWorkspace(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        method_id=method_id,
        artifact_dir=artifact_dir,
        workspace_dir=artifact_dir / "workspace",
        input_dir=artifact_dir / "workspace" / "input",
        working_dir=artifact_dir / "workspace" / "lightrag_working_dir",
        raw_dir=artifact_dir / "raw",
    )
    for path in (
        workspace.workspace_dir,
        workspace.input_dir,
        workspace.working_dir,
        workspace.raw_dir,
    ):
        _assert_within(path, workspace.artifact_dir)
    return workspace


def ensure_workspace(workspace: LightRAGWorkspace) -> None:
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    workspace.working_dir.mkdir(parents=True, exist_ok=True)
    workspace.raw_dir.mkdir(parents=True, exist_ok=True)


def build_config_file(
    workspace: LightRAGWorkspace,
    *,
    query_mode: str = "mix",
    top_k: int = 5,
) -> Path:
    ensure_workspace(workspace)
    path = workspace.workspace_dir / "config.json"
    path.write_text(
        json.dumps(
            {
                "method_id": LIGHTRAG_METHOD_ID,
                "dataset_id": workspace.dataset_id,
                "dataset_version": workspace.dataset_version,
                "working_dir": str(workspace.working_dir),
                "query_mode": query_mode,
                "top_k": top_k,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _assert_within(path, workspace.artifact_dir)
    return path


def _assert_within(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes LightRAG artifact directory: {path}")
