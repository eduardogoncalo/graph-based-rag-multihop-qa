"""Verifica que a amostra ingerida é a mesma que a dissertação usou.

PORQUÊ. O MuSiQue não vem pronto: as 1000 perguntas de avaliação saem de
``random.Random(42).sample(...)`` sobre a divisão de validação. Isso é
determinista dado o mesmo ficheiro bruto e a mesma ordem de leitura — e deixa
de o ser em silêncio se o ficheiro descarregado for de outro lançamento, ou se
a ordem mudar. O 2Wiki não é amostrado, vem dos ficheiros publicados do
HippoRAG, mas o mesmo raciocínio se aplica: pode descarregar-se o ficheiro
errado.

Os documentos não correm esse risco por vias próprias — são recolhidos dos
parágrafos das perguntas seleccionadas, portanto saem da mesma amostra por
construção. Ainda assim conferem-se, porque um número diferente de documentos
denuncia um corpus diferente.

Sem esta verificação, uma amostra diferente produz números diferentes sem que
nada se queixe, e a comparação com a dissertação passa a ser inválida sem
ninguém dar por isso.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DIRETORIO_PADRAO = Path("configs/datasets/fingerprints")


class AmostraDivergenteError(RuntimeError):
    """A amostra ingerida não é a que a dissertação usou."""


@dataclass(frozen=True)
class Impressao:
    dataset_id: str
    dataset_version: str
    num_questions: int
    num_documents: int
    sha256_question_ids: str
    sha256_document_ids: str
    question_ids: tuple[str, ...]

    @property
    def slug(self) -> str:
        return f"{self.dataset_id}_{self.dataset_version}"


def sha256_de_ids(ids: list[str] | tuple[str, ...]) -> str:
    """Impressão de um conjunto de identificadores, independente da ordem."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def carregar_impressao(
    dataset_id: str, dataset_version: str, *, diretorio: Path | None = None
) -> Impressao | None:
    """Devolve a impressão registada, ou None se o dataset não tiver uma."""
    base = diretorio if diretorio is not None else DIRETORIO_PADRAO
    caminho = base / f"{dataset_id}_{dataset_version}.json"
    if not caminho.is_file():
        return None
    dados = json.loads(caminho.read_text())
    return Impressao(
        dataset_id=dados["dataset_id"],
        dataset_version=dados["dataset_version"],
        num_questions=dados["num_questions"],
        num_documents=dados["num_documents"],
        sha256_question_ids=dados["sha256_question_ids"],
        sha256_document_ids=dados["sha256_document_ids"],
        question_ids=tuple(dados["question_ids"]),
    )


def verificar_amostra(
    impressao: Impressao,
    *,
    question_ids: list[str],
    document_ids: list[str],
) -> None:
    """Levanta ``AmostraDivergenteError`` se a amostra não bater certo.

    A mensagem diz quantas perguntas faltam e quantas sobram, porque a
    diferença entre "descarreguei outro lançamento" e "a semente mudou" está
    justamente aí.
    """
    problemas: list[str] = []

    if len(question_ids) != impressao.num_questions:
        problemas.append(
            f"{len(question_ids)} questions, expected {impressao.num_questions}"
        )
    if len(document_ids) != impressao.num_documents:
        problemas.append(
            f"{len(document_ids)} documents, expected {impressao.num_documents}"
        )

    if sha256_de_ids(question_ids) != impressao.sha256_question_ids:
        obtidas = set(question_ids)
        esperadas = set(impressao.question_ids)
        problemas.append(
            f"{len(esperadas - obtidas)} missing and "
            f"{len(obtidas - esperadas)} extra questions against the thesis sample"
        )
    elif sha256_de_ids(document_ids) != impressao.sha256_document_ids:
        problemas.append("the questions match but the document set does not")

    if not problemas:
        return

    raise AmostraDivergenteError(
        f"The {impressao.slug} sample is not the one the thesis used: "
        + "; ".join(problemas)
        + ". Likely cause: the raw file downloaded comes from a different release. "
        "Numbers produced from here are not comparable with the document."
    )


def verificar_amostra_registada(
    dataset_id: str,
    dataset_version: str,
    *,
    question_ids: list[str],
    document_ids: list[str],
    diretorio: Path | None = None,
) -> Impressao | None:
    """Verifica contra a impressão registada, se existir uma.

    Devolve a impressão usada, ou ``None`` quando o dataset não tem nenhuma —
    o caso de uma amostra de smoke, ou de um dataset novo. Um dataset sem
    impressão não é um erro: é um dataset sobre o qual não há nada a afirmar.

    É este o ponto de entrada que a ingestão usa. A verificação tinha de deixar
    de depender de alguém se lembrar de a invocar, que era o estado anterior.
    """
    impressao = carregar_impressao(dataset_id, dataset_version, diretorio=diretorio)
    if impressao is None:
        return None
    verificar_amostra(impressao, question_ids=question_ids, document_ids=document_ids)
    return impressao
