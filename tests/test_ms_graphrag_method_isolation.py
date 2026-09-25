import pytest

from benchmark.methods.ms_graphrag.adapter import MicrosoftGraphRAGAdapter
from benchmark.methods.ms_graphrag.config_builder import build_workspace


def test_ms_graphrag_adapter_rejects_non_ms_workspace(tmp_path) -> None:
    with pytest.raises(ValueError, match="ms_graphrag"):
        MicrosoftGraphRAGAdapter(
            workspace=build_workspace(
                artifacts_dir=tmp_path,
                dataset_id="musique_smoke_20",
                dataset_version="v1",
                method_id="vector_rag",
            )
        )
