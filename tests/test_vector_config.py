from pathlib import Path

from benchmark.core.config_loader import load_method_config


def test_vector_rag_config_parses_phase_three_fields() -> None:
    config = load_method_config(Path("configs/methods/vector_rag.yaml"))

    assert config.method_id == "vector_rag"
    assert config.embedding_provider == "deterministic"
    assert config.embedding_dimension == 16
    assert config.storage["table"] == "vector_rag_chunk_embeddings"


