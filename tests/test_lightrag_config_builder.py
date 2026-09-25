import json
from pathlib import Path

from benchmark.methods.lightrag.config_builder import build_config_file, build_workspace


def test_lightrag_config_builder_writes_mix_mode_config(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    config_path = build_config_file(workspace, query_mode="mix", top_k=7)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["method_id"] == "lightrag"
    assert config["query_mode"] == "mix"
    assert config["top_k"] == 7
    assert config["working_dir"] == str(workspace.working_dir)
