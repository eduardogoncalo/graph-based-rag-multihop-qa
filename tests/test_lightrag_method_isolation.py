import pytest

from benchmark.methods.lightrag.adapter import LightRAGAdapter
from benchmark.methods.lightrag.config_builder import build_workspace


def test_lightrag_adapter_rejects_non_lightrag_workspace(tmp_path) -> None:
    with pytest.raises(ValueError, match="lightrag"):
        LightRAGAdapter(
            workspace=build_workspace(
                artifacts_dir=tmp_path,
                dataset_id="musique_smoke_20",
                dataset_version="v1",
                method_id="vector_rag",
            )
        )
