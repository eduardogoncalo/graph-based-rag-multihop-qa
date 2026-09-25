"""A exportação do conjunto canónico para JSONL.

O teste é sobre o `export_canonical_jsonl` — que ficheiros escreve e com que
forma. O carregador é só a fonte de um `CanonicalDataset` de verdade, e por
isso usa-se o do MuSiQue com uma amostra minúscula escrita aqui.

Usava o carregador do CUAD até 2026-08-09, pela mesma razão. O CUAD morreu e a
cobertura não tinha de morrer com ele.
"""

import json
from pathlib import Path

from benchmark.core.config_loader import DatasetConfig
from benchmark.ingestion.canonical_export import export_canonical_jsonl
from benchmark.ingestion.musique_loader import load_musique


def test_export_canonical_jsonl_writes_expected_paths(tmp_path: Path) -> None:
    config = _write_musique_fixture(tmp_path)
    dataset = load_musique(config, chunk_size=50, chunk_overlap=10)

    files = export_canonical_jsonl(dataset, config.canonical_path)

    assert files["documents"] == Path(config.canonical_path) / "documents.jsonl"
    assert files["chunks"] == Path(config.canonical_path) / "chunks.jsonl"
    assert files["questions"] == Path(config.canonical_path) / "questions.jsonl"
    assert files["gold_evidence"] == Path(config.canonical_path) / "gold_evidence.jsonl"
    assert all(path.exists() for path in files.values())

    first_document = json.loads(files["documents"].read_text(encoding="utf-8").splitlines()[0])
    first_question = json.loads(files["questions"].read_text(encoding="utf-8").splitlines()[0])

    assert first_document["dataset_id"] == "musique"
    assert first_question["gold_evidence"]


def test_uma_linha_por_registo(tmp_path: Path) -> None:
    """JSONL é uma linha por registo, e é isso que o resto da cadeia assume."""
    config = _write_musique_fixture(tmp_path)
    dataset = load_musique(config, chunk_size=50, chunk_overlap=10)
    files = export_canonical_jsonl(dataset, config.canonical_path)

    for chave, registos in (
        ("documents", dataset.documents),
        ("questions", dataset.questions),
        ("gold_evidence", dataset.gold_evidence),
        ("chunks", dataset.chunks),
    ):
        linhas = files[chave].read_text(encoding="utf-8").splitlines()
        assert len(linhas) == len(registos), chave
        assert all(json.loads(linha) for linha in linhas), chave


def _write_musique_fixture(tmp_path: Path) -> DatasetConfig:
    """Duas perguntas no formato bruto do MuSiQue, uma passagem partilhada.

    A passagem partilhada não é decorativa: prova que a exportação escreve um
    só Document quando duas perguntas citam a mesma passagem, que é a regra de
    deduplicação do carregador.
    """
    raw_dir = tmp_path / "raw" / "musique"
    raw_dir.mkdir(parents=True)
    partilhada = {
        "idx": 0,
        "title": "Delaware",
        "paragraph_text": "Delaware is a state in the Mid-Atlantic region.",
        "is_supporting": True,
    }
    perguntas = [
        {
            "id": "2hop__1",
            "question": "In which region is the state that governs the agreement?",
            "answer": "Mid-Atlantic",
            "answer_aliases": ["the Mid-Atlantic"],
            "paragraphs": [
                partilhada,
                {
                    "idx": 1,
                    "title": "Agreement",
                    "paragraph_text": "This agreement is governed by Delaware law.",
                    "is_supporting": True,
                },
            ],
        },
        {
            "id": "2hop__2",
            "question": "What is the capital of the state described above?",
            "answer": "Dover",
            "answer_aliases": [],
            "paragraphs": [
                partilhada,
                {
                    "idx": 1,
                    "title": "Dover",
                    "paragraph_text": "Dover is the capital of Delaware.",
                    "is_supporting": True,
                },
            ],
        },
    ]
    with (raw_dir / "musique_ans_v1.0_dev.jsonl").open("w", encoding="utf-8") as ficheiro:
        for pergunta in perguntas:
            ficheiro.write(json.dumps(pergunta) + "\n")

    return DatasetConfig(
        dataset_id="musique",
        dataset_version="v1",
        name="MuSiQue (amostra de teste)",
        raw_path=str(raw_dir),
        canonical_path=str(tmp_path / "canonical"),
        splits_path=str(tmp_path / "splits"),
    )
