from pathlib import Path

import pytest

from benchmark.methods.lightrag.config_builder import (
    LIGHTRAG_METHOD_ID,
    build_workspace,
    ensure_workspace,
)


def test_lightrag_workspace_path_is_isolated(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    assert workspace.method_id == LIGHTRAG_METHOD_ID
    assert workspace.artifact_dir == tmp_path / "musique_smoke_20_v1" / "lightrag"
    assert workspace.workspace_dir == workspace.artifact_dir / "workspace"
    assert workspace.input_dir == workspace.workspace_dir / "input"
    assert workspace.working_dir == workspace.workspace_dir / "lightrag_working_dir"
    assert workspace.raw_dir == workspace.artifact_dir / "raw"

    ensure_workspace(workspace)

    assert workspace.input_dir.exists()
    assert workspace.working_dir.exists()
    assert workspace.raw_dir.exists()


def test_lightrag_workspace_rejects_other_method_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lightrag"):
        build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique_smoke_20",
            dataset_version="v1",
            method_id="vector_rag",
        )
