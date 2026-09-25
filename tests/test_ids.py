from benchmark.core.ids import deterministic_id


def test_deterministic_id_is_stable() -> None:
    first = deterministic_id("doc", ["musique_smoke_20", "v1", "passagem.txt"])
    second = deterministic_id("doc", ["musique_smoke_20", "v1", "passagem.txt"])

    assert first == second
    assert first.startswith("doc_")
