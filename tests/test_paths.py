from pathlib import Path

from benchmark.core.paths import dataset_artifact_dir


def test_dataset_artifact_dir_is_method_scoped() -> None:
    path = dataset_artifact_dir("artifacts", "musique_smoke_20", "v1", "vector_rag")

    assert path == Path("artifacts/musique_smoke_20_v1/vector_rag")
