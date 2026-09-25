"""A verificação de impressão digital corre sozinha, dentro do `ingest`.

Antes disto existia o módulo que compara, mas ninguém o chamava: a amostra
divergente só se notaria nos números, quando já tivesse custado a indexação, os
leitores e o juiz. O que estes testes fixam é que a verificação corre **entre
carregar e escrever**, e que uma amostra errada não chega a tocar no disco.

O idioma é o dos testes vizinhos: duplos escritos à mão, `tmp_path`, sem
`unittest.mock`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import typer

from benchmark.cli.app import _check_sample_fingerprint
from benchmark.ingestion.fingerprints import (
    AmostraDivergenteError,
    sha256_de_ids,
    verificar_amostra_registada,
)


@dataclass
class ConfigFalso:
    dataset_id: str
    dataset_version: str


@dataclass
class ItemFalso:
    question_id: str = ""
    document_id: str = ""


@dataclass
class CanonicoFalso:
    questions: list[ItemFalso]
    documents: list[ItemFalso]


def _canonico(question_ids: list[str], document_ids: list[str]) -> CanonicoFalso:
    return CanonicoFalso(
        questions=[ItemFalso(question_id=qid) for qid in question_ids],
        documents=[ItemFalso(document_id=did) for did in document_ids],
    )


def _escrever_impressao(
    diretorio: Path,
    *,
    dataset_id: str = "amostrinha",
    dataset_version: str = "v1",
    question_ids: list[str],
    document_ids: list[str],
) -> Path:
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / f"{dataset_id}_{dataset_version}.json"
    caminho.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "num_questions": len(question_ids),
                "num_documents": len(document_ids),
                "sha256_question_ids": sha256_de_ids(question_ids),
                "sha256_document_ids": sha256_de_ids(document_ids),
                "question_ids": question_ids,
            }
        )
    )
    return caminho


# --------------------------------------------------------------------------- #
# verificar_amostra_registada — o ponto de entrada que a ingestão usa
# --------------------------------------------------------------------------- #


def test_dataset_sem_impressao_devolve_none(tmp_path: Path) -> None:
    """Um dataset sem impressão não é erro: é um sobre o qual não há nada a dizer.

    É o caso da amostra de smoke, que por construção não bate com a da
    dissertação e não devia bater.
    """
    resultado = verificar_amostra_registada(
        "smoke",
        "v0",
        question_ids=["q_1"],
        document_ids=["doc_1"],
        diretorio=tmp_path,
    )
    assert resultado is None


def test_amostra_certa_devolve_a_impressao(tmp_path: Path) -> None:
    _escrever_impressao(
        tmp_path, question_ids=["q_1", "q_2"], document_ids=["doc_1", "doc_2", "doc_3"]
    )
    impressao = verificar_amostra_registada(
        "amostrinha",
        "v1",
        question_ids=["q_2", "q_1"],  # a ordem não conta
        document_ids=["doc_3", "doc_1", "doc_2"],
        diretorio=tmp_path,
    )
    assert impressao is not None
    assert impressao.num_questions == 2
    assert impressao.num_documents == 3


def test_amostra_divergente_levanta(tmp_path: Path) -> None:
    _escrever_impressao(tmp_path, question_ids=["q_1", "q_2"], document_ids=["doc_1"])
    with pytest.raises(AmostraDivergenteError):
        verificar_amostra_registada(
            "amostrinha",
            "v1",
            question_ids=["q_1", "q_9"],
            document_ids=["doc_1"],
            diretorio=tmp_path,
        )


# --------------------------------------------------------------------------- #
# _check_sample_fingerprint — o que o `ingest` faz de facto
# --------------------------------------------------------------------------- #


def test_amostra_divergente_aborta_antes_de_escrever(monkeypatch, capsys) -> None:
    """O ponto todo: sai com código não-zero, e sai ANTES do export.

    O teste que interessa não é a mensagem, é o `typer.Exit` — porque é ele que
    impede o `export_canonical_jsonl` de correr.
    """
    def _recusa(*_args, **_kwargs):
        raise AmostraDivergenteError("the musique_x sample is not the one the thesis used")

    monkeypatch.setattr("benchmark.cli.app.verificar_amostra_registada", _recusa)

    with pytest.raises(typer.Exit) as saida:
        _check_sample_fingerprint(
            config=ConfigFalso("musique", "ans_v1.0_eval1k"),
            canonical=_canonico(["q_1"], ["doc_1"]),
            sample_size=None,
            allow_divergent=False,
        )
    assert saida.value.exit_code == 1
    erro = capsys.readouterr().err
    assert "is not the one the thesis used" in erro
    assert "Nothing was written" in erro
    assert "--allow-divergent-sample" in erro


def test_saida_explicita_permite_ingerir_a_divergente(monkeypatch, capsys) -> None:
    """A saída existe para não empurrar ninguém a apagar o ficheiro de impressão.

    Se a única forma de ingerir outra amostra fosse apagar a impressão, alguém
    apagava-a — e aí a verificação desaparecia para sempre, para toda a gente.
    """
    def _recusa(*_args, **_kwargs):
        raise AmostraDivergenteError("diverge")

    monkeypatch.setattr("benchmark.cli.app.verificar_amostra_registada", _recusa)

    _check_sample_fingerprint(
        config=ConfigFalso("musique", "ans_v1.0_eval1k"),
        canonical=_canonico(["q_1"], ["doc_1"]),
        sample_size=None,
        allow_divergent=True,
    )
    saida = capsys.readouterr().out
    assert "--allow-divergent-sample" in saida


def test_sample_size_desliga_a_verificacao_mas_diz_ao_que_vem(monkeypatch, capsys) -> None:
    """Uma amostra pedida à mão diverge por construção — e isso diz-se.

    Calar aqui seria a mesma armadilha ao contrário: alguém corre com
    --sample-size 20, vê os números, e compara-os com a dissertação.
    """
    def _nao_devia_ser_chamado(*_args, **_kwargs):
        raise AssertionError("com --sample-size não se verifica coisa nenhuma")

    monkeypatch.setattr(
        "benchmark.cli.app.verificar_amostra_registada", _nao_devia_ser_chamado
    )

    _check_sample_fingerprint(
        config=ConfigFalso("musique", "ans_v1.0_eval1k"),
        canonical=_canonico(["q_1"], ["doc_1"]),
        sample_size=20,
        allow_divergent=False,
    )
    saida = capsys.readouterr().out
    assert "--sample-size 20" in saida
    assert "NOT comparable" in saida


def test_dataset_sem_impressao_passa_e_diz_que_nao_verificou(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "benchmark.cli.app.verificar_amostra_registada", lambda *a, **k: None
    )
    _check_sample_fingerprint(
        config=ConfigFalso("musique_smoke_20", "v1"),
        canonical=_canonico(["q_1"], ["doc_1"]),
        sample_size=None,
        allow_divergent=False,
    )
    saida = capsys.readouterr().out
    assert "no fingerprint recorded" in saida
    assert "was not verified" in saida


def test_amostra_certa_confirma_em_voz_alta(monkeypatch, capsys) -> None:
    """Uma verificação silenciosa quando passa é uma verificação que ninguém sabe
    se correu. A linha de confirmação é o que distingue "verificado" de
    "esquecido"."""
    from benchmark.ingestion.fingerprints import Impressao

    impressao = Impressao(
        dataset_id="musique",
        dataset_version="ans_v1.0_eval1k",
        num_questions=1000,
        num_documents=11515,
        sha256_question_ids="x",
        sha256_document_ids="y",
        question_ids=(),
    )
    monkeypatch.setattr(
        "benchmark.cli.app.verificar_amostra_registada", lambda *a, **k: impressao
    )
    _check_sample_fingerprint(
        config=ConfigFalso("musique", "ans_v1.0_eval1k"),
        canonical=_canonico(["q_1"], ["doc_1"]),
        sample_size=None,
        allow_divergent=False,
    )
    saida = capsys.readouterr().out
    assert "1000 questions" in saida
    assert "11515 documents" in saida
    assert "musique_ans_v1.0_eval1k" in saida


def test_a_confirmacao_nao_invoca_a_dissertacao_para_a_amostra_de_smoke(
    monkeypatch, capsys
) -> None:
    """A amostra de smoke tem impressão própria, e não é a da dissertação.

    Uma frase fixa a dizer "iguais aos da dissertação" seria falsa aqui — e
    falsa exactamente no caminho que alguém de fora corre.
    """
    from benchmark.ingestion.fingerprints import Impressao

    impressao = Impressao(
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        num_questions=20,
        num_documents=399,
        sha256_question_ids="x",
        sha256_document_ids="y",
        question_ids=(),
    )
    monkeypatch.setattr(
        "benchmark.cli.app.verificar_amostra_registada", lambda *a, **k: impressao
    )
    _check_sample_fingerprint(
        config=ConfigFalso("musique_smoke_20", "v1"),
        canonical=_canonico(["q_1"], ["doc_1"]),
        sample_size=None,
        allow_divergent=False,
    )
    saida = capsys.readouterr().out
    assert "musique_smoke_20_v1" in saida
    assert "dissertação" not in saida
