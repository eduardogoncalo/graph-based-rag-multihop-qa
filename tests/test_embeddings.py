from benchmark.embeddings import DeterministicEmbeddingProvider


def test_deterministic_embedding_provider_is_stable() -> None:
    provider = DeterministicEmbeddingProvider(dimension=8)

    first = provider.embed_texts(["Delaware governing law"])
    second = provider.embed_texts(["Delaware governing law"])

    assert first == second
    assert len(first) == 1
    assert len(first[0]) == 8
    assert provider.model_id == "deterministic-test-embedding"
