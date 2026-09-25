from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmark.core.config_loader import DatasetConfig
from benchmark.ingestion.twowiki_loader import (
    TwowikiRawDataMissingError,
    find_twowiki_raw_file,
    load_twowiki,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "twowiki_sample.json"
ALIASES_FIXTURE = Path(__file__).parent / "fixtures" / "twowiki_id_aliases.jsonl"


def _make_config(tmp_path: Path) -> DatasetConfig:
    raw_dir = tmp_path / "raw" / "2wikimultihop"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ALIASES_FIXTURE, raw_dir / "id_aliases.json")
    return DatasetConfig(
        dataset_id="twowiki",
        dataset_version="fixture_v0",
        name="2Wiki fixture",
        raw_path=str(raw_dir),
        canonical_path=str(tmp_path / "canonical"),
        splits_path=str(tmp_path / "splits"),
    )


def test_load_twowiki_parses_and_deduplicates_shared_passages(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_twowiki(config, raw_file=FIXTURE_PATH)

    rows = json.loads(FIXTURE_PATH.read_text())
    assert len(dataset.questions) == len(rows)

    # "Film Alpha" aparece nas duas perguntas com o MESMO texto -> 1 Document só
    titles = [d.title for d in dataset.documents]
    assert titles.count("Film Alpha") == 1
    # 5 passagens distintas no fixture (Alpha, Beta, Distractor One, Prize, Distractor Two)
    assert len(dataset.documents) == 5
    # texto = title + parágrafo com sentenças unidas por espaço
    alpha = next(d for d in dataset.documents if d.title == "Film Alpha")
    assert alpha.text == "Film Alpha\n\nFilm Alpha is a 1990 drama. It was directed by Ana Silva."


def test_gold_evidence_maps_supporting_titles_to_documents(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_twowiki(config, raw_file=FIXTURE_PATH)

    by_question = {}
    for ev in dataset.gold_evidence:
        by_question.setdefault(ev.question_id, []).append(ev)

    assert all(len(evs) == 2 for evs in by_question.values())

    docs_by_id = {d.document_id: d for d in dataset.documents}
    for evs in by_question.values():
        for ev in evs:
            assert ev.document_id in docs_by_id
            assert ev.metadata["supporting_sent_ids"]


def test_question_metadata_carries_type_aliases_and_evidences(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_twowiki(config, raw_file=FIXTURE_PATH)

    q1 = next(q for q in dataset.questions if q.metadata["source_question_id"] == "q1-comparison")
    assert q1.gold_answer == "Film Alpha"
    assert q1.metadata["type"] == "comparison"
    # alias igual à resposta é filtrado; os demais ficam
    assert q1.metadata["answer_aliases"] == ["Alpha (film)", "The Alpha Movie"]
    assert len(q1.metadata["evidences"]) == 2

    q2 = next(q for q in dataset.questions if q.metadata["source_question_id"] == "q2-compositional")
    # Q200 não está no id_aliases -> sem aliases
    assert q2.metadata["answer_aliases"] == []


def test_sampling_is_deterministic_with_seed(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    first = load_twowiki(config, raw_file=FIXTURE_PATH, sample_size=1)
    second = load_twowiki(config, raw_file=FIXTURE_PATH, sample_size=1)
    assert [q.question_id for q in first.questions] == [
        q.question_id for q in second.questions
    ]


def test_find_twowiki_raw_file_error_when_missing(tmp_path: Path) -> None:
    with pytest.raises(TwowikiRawDataMissingError):
        find_twowiki_raw_file(tmp_path / "nowhere")


# ---------- variante HippoRAG (ficheiros de avaliação publicados) ----------

from benchmark.ingestion.twowiki_loader import load_twowiki_hipporag  # noqa: E402

HIPPORAG_QUESTIONS = Path(__file__).parent / "fixtures" / "twowiki_hipporag_questions.json"
HIPPORAG_CORPUS = Path(__file__).parent / "fixtures" / "twowiki_hipporag_corpus.json"
DEV_FIXTURE = FIXTURE_PATH  # twowiki_sample.json tem os mesmos _ids com supporting_facts/type


def test_hipporag_documents_come_from_corpus_file(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_twowiki_hipporag(
        config, raw_file=HIPPORAG_QUESTIONS, corpus_file=HIPPORAG_CORPUS, dev_file=DEV_FIXTURE
    )
    # documents = corpus deles (4), NÃO a união dos contexts (5: "Distractor Two" fica fora)
    assert len(dataset.documents) == 4
    titles = {d.title for d in dataset.documents}
    assert "Distractor Two" not in titles
    assert len(dataset.questions) == 2


def test_hipporag_gold_and_type_recovered_via_dev(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset = load_twowiki_hipporag(
        config, raw_file=HIPPORAG_QUESTIONS, corpus_file=HIPPORAG_CORPUS, dev_file=DEV_FIXTURE
    )
    by_q = {}
    for ev in dataset.gold_evidence:
        by_q.setdefault(ev.question_id, []).append(ev)
    assert all(len(evs) == 2 for evs in by_q.values())
    doc_ids = {d.document_id for d in dataset.documents}
    assert all(ev.document_id in doc_ids for ev in dataset.gold_evidence)
    q1 = next(q for q in dataset.questions if q.metadata["source_question_id"] == "q1-comparison")
    assert q1.metadata["type"] == "comparison"
    assert q1.metadata["provenance"] == "hipporag_reproduce_dataset"
    assert q1.metadata["answer_aliases"] == ["Alpha (film)", "The Alpha Movie"]


# ---------- amostragem pelo config, e a restrição do corpus que ela obriga ----------


def _config_com_amostra(tmp_path: Path, num_questions: int, seed: int = 42) -> DatasetConfig:
    """O mesmo config, mais o bloco `sample:` que o YAML do smoke traz.

    `DatasetConfig` tem `extra="allow"`, portanto o bloco chega ao carregador
    como atributo — que é como o `twowiki_smoke_20.yaml` o entrega.
    """
    config = _make_config(tmp_path)
    config.sample = {"num_questions": num_questions, "seed": seed}
    return config


def test_sem_bloco_sample_o_corpus_vem_inteiro(tmp_path: Path) -> None:
    """O caminho do eval1k, que produziu os resultados da dissertação.

    É a regressão que mais importa desta alteração: sem `sample:` e sem
    `sample_size`, nada da amostragem corre e o corpus é o ficheiro publicado
    na íntegra.
    """
    dataset = load_twowiki_hipporag(
        _make_config(tmp_path),
        raw_file=HIPPORAG_QUESTIONS,
        corpus_file=HIPPORAG_CORPUS,
        dev_file=DEV_FIXTURE,
    )
    assert len(dataset.questions) == 2
    assert len(dataset.documents) == 4  # o corpus todo, incluindo "Distractor One"


def test_bloco_sample_amostra_perguntas_e_encolhe_o_corpus(tmp_path: Path) -> None:
    """O 2Wiki não encolhe sozinho, ao contrário do MuSiQue.

    Os documentos vêm de um ficheiro de corpus à parte e não das perguntas, por
    isso amostrar sem restringir daria 20 perguntas contra 6119 documentos —
    treze horas de indexação para uma «amostra de smoke». O corpus passa a ser o
    das passagens que aparecem no contexto das perguntas escolhidas.
    """
    dataset = load_twowiki_hipporag(
        _config_com_amostra(tmp_path, num_questions=1),
        raw_file=HIPPORAG_QUESTIONS,
        corpus_file=HIPPORAG_CORPUS,
        dev_file=DEV_FIXTURE,
    )
    assert len(dataset.questions) == 1
    # Só as passagens do contexto da pergunta escolhida, e nunca mais do que o
    # corpus inteiro tinha.
    titulos = {d.title for d in dataset.documents}
    assert titulos < {"Film Alpha", "Film Beta", "Fictional Prize", "Distractor One"}
    assert len(dataset.documents) < 4


def test_a_amostra_do_config_e_determinista(tmp_path: Path) -> None:
    """Mesma semente, mesma amostra — senão a impressão digital não serve."""
    primeira, segunda = (
        load_twowiki_hipporag(
            _config_com_amostra(tmp_path, num_questions=1),
            raw_file=HIPPORAG_QUESTIONS,
            corpus_file=HIPPORAG_CORPUS,
            dev_file=DEV_FIXTURE,
        )
        for _ in range(2)
    )
    assert [q.question_id for q in primeira.questions] == [
        q.question_id for q in segunda.questions
    ]
    assert {d.document_id for d in primeira.documents} == {
        d.document_id for d in segunda.documents
    }


def test_toda_a_evidencia_aponta_a_documentos_que_ficaram(tmp_path: Path) -> None:
    """A restrição do corpus não pode deixar evidência órfã.

    Se uma passagem de suporte fosse removida com o corpus, a pergunta ficava
    sem recuperação possível e o `recall@5` era zero por construção — o género
    de defeito que só se vê nos números.
    """
    dataset = load_twowiki_hipporag(
        _config_com_amostra(tmp_path, num_questions=1),
        raw_file=HIPPORAG_QUESTIONS,
        corpus_file=HIPPORAG_CORPUS,
        dev_file=DEV_FIXTURE,
    )
    doc_ids = {d.document_id for d in dataset.documents}
    assert dataset.gold_evidence
    assert all(ev.document_id in doc_ids for ev in dataset.gold_evidence)


def test_sample_size_explicito_ganha_ao_config(tmp_path: Path) -> None:
    """Quem escreve --sample-size está a pedir outra coisa de propósito."""
    dataset = load_twowiki_hipporag(
        _config_com_amostra(tmp_path, num_questions=2),
        raw_file=HIPPORAG_QUESTIONS,
        corpus_file=HIPPORAG_CORPUS,
        dev_file=DEV_FIXTURE,
        sample_size=1,
    )
    assert len(dataset.questions) == 1


def test_hipporag_missing_corpus_raises(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with pytest.raises(TwowikiRawDataMissingError):
        load_twowiki_hipporag(
            config,
            raw_file=HIPPORAG_QUESTIONS,
            corpus_file=tmp_path / "nowhere.json",
            dev_file=DEV_FIXTURE,
        )
