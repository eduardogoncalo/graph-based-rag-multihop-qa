from pathlib import Path

import pytest

from benchmark.methods.hipporag2.config_builder import (
    HIPPORAG2_METHOD_ID,
    build_workspace,
    ensure_workspace,
)


def test_build_workspace_paths(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique",
        dataset_version="ans_v1.0_eval1k",
    )
    assert workspace.method_id == HIPPORAG2_METHOD_ID
    # slug: pontos viram underscore, árvore igual aos outros substratos
    assert workspace.artifact_dir == tmp_path.resolve() / "musique_ans_v1_0_eval1k" / "hipporag2"
    assert workspace.save_dir == workspace.artifact_dir / "workspace"
    assert workspace.raw_dir == workspace.artifact_dir / "raw"
    assert workspace.passage_map_path == workspace.save_dir / "passage_map.json"


def test_build_workspace_rejects_other_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_workspace(
            artifacts_dir=tmp_path,
            dataset_id="musique",
            dataset_version="ans_v1.0_eval1k",
            method_id="lightrag_neo4j",
        )


def test_ensure_workspace_creates_dirs(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique",
        dataset_version="ans_v1.0_eval1k",
    )
    ensure_workspace(workspace)
    assert workspace.save_dir.is_dir()
    assert workspace.raw_dir.is_dir()
