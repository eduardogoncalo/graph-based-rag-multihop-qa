"""Um dataset só é utilizável se as CINCO peças dele existirem.

Este ficheiro existe por causa de uma lição que custou uma noite: até
2026-08-10 os datasets completos tinham config, impressão digital, carregador e
experimento — e **não tinham porta utilizável**. A peça em falta não dava erro
na revisão de código nem na suíte; dava um `PortConflictError` no momento de
subir o container, depois de a ingestão já ter corrido.

As peças, e onde vivem:

1. `configs/datasets/<id>.yaml` ................ o config
2. `LOADERS` em `benchmark.cli.app` ............ o carregador
3. `configs/datasets/fingerprints/<slug>.json` . a impressão digital
4. `configs/infra/neo4j_ports.yaml` ............ as portas, por método
5. `configs/experiments/*.yaml` ................ o experimento

Um teste por peça, sobre a matriz toda. Quem acrescentar um dataset novo vê
aqui exactamente o que lhe falta, em vez de o descobrir a correr.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.cli.app import LOADERS
from benchmark.core.config_loader import load_dataset_config, load_experiment_config
from benchmark.core.naming import slugify_dataset_version
from benchmark.infra.neo4j_containers import Neo4jPortRegistry
from benchmark.ingestion.fingerprints import carregar_impressao

RAIZ = Path(__file__).resolve().parents[1]

# Os quatro datasets do pacote. Os dois `_smoke_20` são o que se corre para
# exercitar a cadeia; os dois `eval1k` são o protocolo da dissertação.
DATASETS = [
    ("musique_smoke_20", "v1", "musique_smoke_20.yaml"),
    ("twowiki_smoke_20", "v1", "twowiki_smoke_20.yaml"),
    ("musique", "ans_v1.0_eval1k", "musique.yaml"),
    ("twowiki", "ans_v1.0_eval1k", "twowiki.yaml"),
]

METODOS_COM_CONTAINER = ("graphrag", "lightrag", "cognee")


@pytest.mark.parametrize("dataset_id,versao,ficheiro", DATASETS)
def test_1_o_config_existe_e_declara_o_que_diz(
    dataset_id: str, versao: str, ficheiro: str
) -> None:
    config = load_dataset_config(RAIZ / "configs" / "datasets" / ficheiro)
    assert config.dataset_id == dataset_id
    assert config.dataset_version == versao


@pytest.mark.parametrize("dataset_id,versao,ficheiro", DATASETS)
def test_2_o_carregador_esta_registado_na_cli(
    dataset_id: str, versao: str, ficheiro: str
) -> None:
    """Sem entrada em LOADERS, `benchmark ingest --dataset X` não sabe o que fazer."""
    assert dataset_id in LOADERS


@pytest.mark.parametrize("dataset_id,versao,ficheiro", DATASETS)
def test_3_tem_impressao_digital(dataset_id: str, versao: str, ficheiro: str) -> None:
    """Sem impressão, um ficheiro bruto de outro lançamento passa despercebido.

    O `benchmark ingest` compara e recusa escrever uma amostra divergente — mas
    só se houver com que comparar.
    """
    impressao = carregar_impressao(dataset_id, versao)
    assert impressao is not None, f"{dataset_id} {versao} não tem impressão registada"
    assert impressao.num_questions > 0
    assert impressao.num_documents > 0
    assert len(impressao.question_ids) == impressao.num_questions


@pytest.mark.parametrize("metodo_curto", METODOS_COM_CONTAINER)
@pytest.mark.parametrize("dataset_id,versao,ficheiro", DATASETS)
def test_4_tem_portas_utilizaveis_para_cada_metodo(
    dataset_id: str, versao: str, ficheiro: str, metodo_curto: str
) -> None:
    """A peça que faltava, e que este ficheiro existe para nunca mais faltar."""
    registo = Neo4jPortRegistry.load()
    slug = slugify_dataset_version(dataset_id, versao)
    http, bolt = registo.ports_for(metodo_curto, slug)
    assert http not in registo.reserved, f"{slug}/{metodo_curto}: http {http} reservada"
    assert bolt not in registo.reserved, f"{slug}/{metodo_curto}: bolt {bolt} reservada"


@pytest.mark.parametrize("dataset_id,versao,ficheiro", DATASETS)
def test_5_tem_experimento_declarado(dataset_id: str, versao: str, ficheiro: str) -> None:
    """O `reproduce.sh` deriva `{dataset}_{versão}`; tem de existir config a bater."""
    esperado = f"{dataset_id}_{versao}"
    declarados = {
        load_experiment_config(caminho).experiment_id
        for caminho in (RAIZ / "configs" / "experiments").glob("*.yaml")
        if caminho.name != "musique_eval1k_runs_registry.yaml"
    }
    assert esperado in declarados, (
        f"nenhum config de experimento declara {esperado!r}; o reproduce.sh vai "
        f"escrever resultados debaixo desse nome e o report não os encontra"
    )


def test_o_reproduce_conhece_exactamente_estes_datasets() -> None:
    """O script tem a lista dele, em `case`. Se divergir desta, alguém a esqueceu.

    O sintoma seria «dataset desconhecido» para um dataset que o pacote traz —
    ou, pior, um dataset que o script aceita e para o qual não sabe que ficheiro
    bruto descarregar.
    """
    texto = (RAIZ / "scripts" / "reproduce.sh").read_text(encoding="utf-8")
    for dataset_id, _versao, _ficheiro in DATASETS:
        assert dataset_id in texto, f"o reproduce.sh não conhece {dataset_id}"


def test_as_duas_amostras_de_smoke_cabem_num_smoke() -> None:
    """Um «smoke» de milhares de documentos não é um smoke.

    O 2Wiki é o caso perigoso: o corpus dele é um ficheiro publicado à parte,
    com 6119 passagens que não dependem das perguntas. Sem a restrição feita
    pelo carregador, 20 perguntas traziam 6119 documentos — cerca de treze horas
    só de indexação do LightRAG, contra os ~50 minutos das 399 do MuSiQue.
    """
    for dataset_id in ("musique_smoke_20", "twowiki_smoke_20"):
        impressao = carregar_impressao(dataset_id, "v1")
        assert impressao is not None
        assert impressao.num_questions == 20
        assert impressao.num_documents < 500, (
            f"{dataset_id}: {impressao.num_documents} documentos é indexação de horas"
        )
