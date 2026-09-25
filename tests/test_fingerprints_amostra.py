"""A amostra ingerida tem de ser a mesma que a dissertação usou.

Os documentos saem dos parágrafos das perguntas seleccionadas, portanto
perguntas e documentos vêm sempre da mesma amostra. O que estes testes
protegem é a amostra em si: as 1000 perguntas do MuSiQue são um
``random.Random(42).sample`` sobre a validação, e um ficheiro bruto de outro
lançamento produziria outras 1000 sem que nada se queixasse.
"""

from __future__ import annotations

import pytest

from benchmark.ingestion.fingerprints import (
    AmostraDivergenteError,
    carregar_impressao,
    sha256_de_ids,
    verificar_amostra,
)

SLUGS = [("musique", "ans_v1.0_eval1k"), ("twowiki", "ans_v1.0_eval1k")]


@pytest.mark.parametrize(("dataset_id", "dataset_version"), SLUGS)
def test_impressao_registada_existe_e_tem_mil_perguntas(
    dataset_id: str, dataset_version: str
) -> None:
    impressao = carregar_impressao(dataset_id, dataset_version)
    assert impressao is not None, "sem impressão registada não há como detectar divergência"
    assert impressao.num_questions == 1000
    assert len(impressao.question_ids) == 1000
    assert len(set(impressao.question_ids)) == 1000


@pytest.mark.parametrize(("dataset_id", "dataset_version"), SLUGS)
def test_a_propria_impressao_e_coerente(dataset_id: str, dataset_version: str) -> None:
    impressao = carregar_impressao(dataset_id, dataset_version)
    assert impressao is not None
    assert sha256_de_ids(impressao.question_ids) == impressao.sha256_question_ids


def test_os_dois_datasets_tem_amostras_distintas() -> None:
    musique = carregar_impressao("musique", "ans_v1.0_eval1k")
    twowiki = carregar_impressao("twowiki", "ans_v1.0_eval1k")
    assert musique is not None and twowiki is not None
    assert musique.sha256_question_ids != twowiki.sha256_question_ids
    assert not set(musique.question_ids) & set(twowiki.question_ids)


def test_amostra_correcta_passa() -> None:
    impressao = carregar_impressao("musique", "ans_v1.0_eval1k")
    assert impressao is not None
    documentos = [f"doc_{i}" for i in range(impressao.num_documents)]
    # A impressão dos documentos foi calculada sobre os ids reais, por isso
    # aqui verifica-se o caminho das perguntas com o número certo de documentos.
    with pytest.raises(AmostraDivergenteError, match="document set does not"):
        verificar_amostra(
            impressao,
            question_ids=list(impressao.question_ids),
            document_ids=documentos,
        )


def test_uma_pergunta_trocada_e_apanhada() -> None:
    """O caso realista: outra semente, ou outro lançamento do ficheiro bruto."""
    impressao = carregar_impressao("musique", "ans_v1.0_eval1k")
    assert impressao is not None
    trocadas = list(impressao.question_ids)
    trocadas[0] = "q_0000000000000000"
    with pytest.raises(AmostraDivergenteError) as erro:
        verificar_amostra(
            impressao,
            question_ids=trocadas,
            document_ids=["x"] * impressao.num_documents,
        )
    texto = str(erro.value)
    assert "1 missing and 1 extra questions" in texto
    assert "not comparable" in texto


def test_contagem_errada_e_apanhada() -> None:
    impressao = carregar_impressao("musique", "ans_v1.0_eval1k")
    assert impressao is not None
    with pytest.raises(AmostraDivergenteError, match="999 questions, expected 1000"):
        verificar_amostra(
            impressao,
            question_ids=list(impressao.question_ids)[:999],
            document_ids=["x"] * impressao.num_documents,
        )
