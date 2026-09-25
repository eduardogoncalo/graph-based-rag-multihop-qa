from pathlib import Path

from benchmark.methods.ms_graphrag.config_builder import build_config_files, build_workspace


def test_ms_graphrag_config_builder_writes_placeholder_config(tmp_path: Path) -> None:
    workspace = build_workspace(
        artifacts_dir=tmp_path,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
    )

    files = build_config_files(workspace, chat_model="gpt-test", embedding_model="embed-test")

    assert files["env"] == workspace.workspace_dir / ".env"
    assert files["settings"] == workspace.workspace_dir / "settings.yaml"
    assert "GRAPHRAG_API_KEY=${GRAPHRAG_API_KEY}" in files["env"].read_text(encoding="utf-8")
    settings = files["settings"].read_text(encoding="utf-8")
    assert "gpt-test" in settings
    assert "embed-test" in settings
