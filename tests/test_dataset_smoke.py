"""A amostra de smoke tem de estar isolada da dissertação, por construção.

O modo `smoke` é o que o avaliador corre e o que o vídeo mostra. Corre na mesma
máquina onde pode estar o ambiente original, e usa os mesmos métodos — portanto
tudo o que o distingue é o **identificador do dataset**. É ele que decide o
nome do container Neo4j, o do volume, o directório canónico e as portas.

Estes testes fixam esse isolamento nos quatro sítios onde ele se materializa.
Um `musique_smoke_20` que resolvesse para os nomes do `musique` ia direito aos
containers que guardam os índices da dissertação.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.cli.app import LOADERS
from benchmark.core.config_loader import load_dataset_config
from benchmark.core.naming import resolve_neo4j_names, slugify_dataset_version
from benchmark.infra.neo4j_containers import Neo4jPortRegistry
from benchmark.ingestion.fingerprints import carregar_impressao

CONFIG = Path("configs/datasets/musique_smoke_20.yaml")
SLUG_SMOKE = "musique_smoke_20_v1"
SLUG_TESE = "musique_ans_v1_0_eval1k"


@pytest.fixture
def config():
    return load_dataset_config(CONFIG)


def test_o_config_existe_e_tem_identificador_proprio(config) -> None:
    assert config.dataset_id == "musique_smoke_20"
    assert config.dataset_version == "v1"
    assert slugify_dataset_version(config.dataset_id, config.dataset_version) == SLUG_SMOKE


def test_o_carregador_esta_registado_na_cli() -> None:
    """Sem isto, `benchmark ingest --dataset musique_smoke_20` não corre."""
    assert "musique_smoke_20" in LOADERS
    carregador_smoke, _ = LOADERS["musique_smoke_20"]
    carregador_musique, _ = LOADERS["musique"]
    assert carregador_smoke is carregador_musique, "é o mesmo carregador, outro config"


def test_o_canonico_nao_colide_com_o_das_mil_perguntas(config) -> None:
    """A razão nº 1 para isto ser um dataset e não um --sample-size.

    Verificado a 2026-08-09: ingerir 20 perguntas contra o config do `musique`
    escreve por cima do canónico das 1000, porque o canonical_path vem do
    config e o --sample-size não lhe toca.
    """
    musique = load_dataset_config(Path("configs/datasets/musique.yaml"))
    assert config.canonical_path != musique.canonical_path
    assert config.splits_path != musique.splits_path
    assert SLUG_SMOKE in config.canonical_path


def test_partilha_o_ficheiro_bruto_com_o_musique(config) -> None:
    """O fetch_datasets.py só descarrega um ficheiro; os dois configs lêem-no."""
    musique = load_dataset_config(Path("configs/datasets/musique.yaml"))
    assert config.raw_path == musique.raw_path


def test_os_containers_nao_colidem_com_os_da_dissertacao() -> None:
    """A razão nº 2. Os containers são globais ao podman, não à pasta."""
    for metodo in ("lightrag_neo4j", "ms_graphrag_neo4j", "cognee"):
        container_smoke, volume_smoke = resolve_neo4j_names(metodo, "musique_smoke_20", "v1")
        container_tese, volume_tese = resolve_neo4j_names(metodo, "musique", "ans_v1.0_eval1k")
        assert container_smoke != container_tese
        assert volume_smoke != volume_tese
        assert SLUG_SMOKE in container_smoke
        assert SLUG_TESE not in container_smoke


def test_as_portas_do_smoke_estao_na_banda_18xxx_e_nao_sao_reservadas() -> None:
    """A banda 17xxx está inteira reservada: é a do ambiente original.

    Antes disto o registo não tinha uma única atribuição utilizável — todas as
    `assignments` apontavam para portas que o próprio registo recusa.
    """
    registo = Neo4jPortRegistry.load()
    for metodo_curto in ("graphrag", "lightrag", "cognee"):
        http, bolt = registo.ports_for(metodo_curto, SLUG_SMOKE)
        assert 18000 <= http < 19000, f"{metodo_curto}: http {http} fora da banda 18xxx"
        assert 18000 <= bolt < 19000, f"{metodo_curto}: bolt {bolt} fora da banda 18xxx"
        assert http not in registo.reserved
        assert bolt not in registo.reserved


def test_as_portas_do_smoke_nao_colidem_com_o_compose() -> None:
    """18474/18687 e 18475/18688 são do docker-compose.yml, e sobem sempre."""
    do_compose = {18474, 18687, 18475, 18688}
    registo = Neo4jPortRegistry.load()
    for metodo_curto in ("graphrag", "lightrag", "cognee"):
        assert not set(registo.ports_for(metodo_curto, SLUG_SMOKE)) & do_compose


def test_todas_as_portas_atribuidas_ao_smoke_sao_distintas() -> None:
    registo = Neo4jPortRegistry.load()
    portas = [
        porta
        for metodo_curto in ("graphrag", "lightrag", "cognee")
        for porta in registo.ports_for(metodo_curto, SLUG_SMOKE)
    ]
    assert len(portas) == len(set(portas))


def test_os_datasets_da_dissertacao_tem_portas_utilizaveis() -> None:
    """Este teste dizia o CONTRÁRIO até 2026-08-10, e a inversão é deliberada.

    A versão anterior exigia que as portas dos datasets da dissertação
    estivessem todas em `reserved` — «o mapa do que o preflight tem de
    recusar». Só que uma atribuição que o preflight recusa não é um mapa, é uma
    atribuição inutilizável: **nenhum dataset real era indexável**, nem aqui
    nem na máquina de quem recebe o pacote, onde não há nada a proteger.

    A protecção mudou de sítio, não desapareceu — ver
    `guard.exigir_dataset_permitido`, que recusa estes slugs pelo NOME do
    container e só enquanto o ambiente original estiver presente. O teste que
    prova isso vive em `test_guard_ambiente_original.py`.
    """
    registo = Neo4jPortRegistry.load()
    for metodo_curto in ("graphrag", "lightrag", "cognee"):
        for porta in registo.ports_for(metodo_curto, SLUG_TESE):
            assert porta not in registo.reserved, (
                f"{porta} está reservada: o dataset da dissertação voltou a ser "
                f"inindexável para quem recebe o pacote"
            )
            assert 18000 <= porta < 19000, f"{porta} fora da banda deste repositório"


def test_a_amostra_de_smoke_tambem_tem_impressao_digital() -> None:
    """O caminho que o avaliador corre é o que mais precisa de ser verificado.

    O plano não pedia impressão para o smoke — mas é o modo que alguém de fora
    executa, e é onde um ficheiro bruto errado passaria despercebido por não
    haver com que comparar.
    """
    impressao = carregar_impressao("musique_smoke_20", "v1")
    assert impressao is not None
    assert impressao.num_questions == 20
    assert impressao.num_documents == 399
    assert len(impressao.question_ids) == 20


def test_a_impressao_do_smoke_e_disjunta_da_do_eval1k() -> None:
    """Ids diferentes por construção: o dataset_id entra no hash do id."""
    smoke = carregar_impressao("musique_smoke_20", "v1")
    eval1k = carregar_impressao("musique", "ans_v1.0_eval1k")
    assert smoke is not None and eval1k is not None
    assert not set(smoke.question_ids) & set(eval1k.question_ids)


# --------------------------------------------------------------------------
# O 2Wiki — metade do protocolo, e a que nunca tinha sido exercitada aqui
# --------------------------------------------------------------------------

CONFIG_2WIKI = Path("configs/datasets/twowiki_smoke_20.yaml")
SLUG_SMOKE_2WIKI = "twowiki_smoke_20_v1"
SLUG_TESE_2WIKI = "twowiki_ans_v1_0_eval1k"


@pytest.fixture
def config_2wiki():
    return load_dataset_config(CONFIG_2WIKI)


def test_o_config_do_2wiki_tem_identificador_proprio(config_2wiki) -> None:
    assert config_2wiki.dataset_id == "twowiki_smoke_20"
    assert config_2wiki.dataset_version == "v1"
    assert (
        slugify_dataset_version(config_2wiki.dataset_id, config_2wiki.dataset_version)
        == SLUG_SMOKE_2WIKI
    )


def test_o_smoke_do_2wiki_esta_registado_na_cli() -> None:
    """Sem isto, `benchmark ingest --dataset twowiki_smoke_20` não encontra carregador."""
    assert "twowiki_smoke_20" in LOADERS


def test_o_smoke_do_2wiki_nao_resolve_para_os_nomes_da_dissertacao(config_2wiki) -> None:
    for metodo in ("lightrag_neo4j", "cognee"):
        container, volume = resolve_neo4j_names(
            metodo, config_2wiki.dataset_id, config_2wiki.dataset_version
        )
        assert SLUG_TESE_2WIKI not in container
        assert SLUG_TESE_2WIKI not in volume
        assert SLUG_SMOKE_2WIKI in container
        assert SLUG_SMOKE_2WIKI in volume


def test_o_smoke_do_2wiki_tem_portas_proprias_na_banda_18xxx() -> None:
    registo = Neo4jPortRegistry.load()
    for metodo_curto in ("graphrag", "lightrag", "cognee"):
        http, bolt = registo.ports_for(metodo_curto, SLUG_SMOKE_2WIKI)
        assert 18000 <= http < 19000
        assert 18000 <= bolt < 19000
        assert http not in registo.reserved
        assert bolt not in registo.reserved


def test_o_smoke_do_2wiki_nao_partilha_portas_com_o_do_musique() -> None:
    """Os dois smokes podem estar de pé ao mesmo tempo, e vão estar."""
    registo = Neo4jPortRegistry.load()
    portas = [
        porta
        for slug in (SLUG_SMOKE, SLUG_SMOKE_2WIKI)
        for metodo_curto in ("graphrag", "lightrag", "cognee")
        for porta in registo.ports_for(metodo_curto, slug)
    ]
    assert len(portas) == len(set(portas))


def test_a_amostra_de_smoke_do_2wiki_tem_impressao_digital() -> None:
    """191 documentos, e o número não é um acaso.

    O corpus do 2Wiki é um ficheiro publicado à parte, com 6119 passagens, e
    não depende das perguntas. Sem a restrição do corpus feita pelo carregador
    esta impressão registaria 6119 — e o «smoke» custava treze horas de
    indexação do LightRAG.
    """
    impressao = carregar_impressao("twowiki_smoke_20", "v1")
    assert impressao is not None
    assert impressao.num_questions == 20
    assert impressao.num_documents == 191
    assert len(impressao.question_ids) == 20


def test_a_impressao_do_smoke_do_2wiki_e_disjunta_da_do_eval1k() -> None:
    smoke = carregar_impressao("twowiki_smoke_20", "v1")
    eval1k = carregar_impressao("twowiki", "ans_v1.0_eval1k")
    assert smoke is not None and eval1k is not None
    assert not set(smoke.question_ids) & set(eval1k.question_ids)


def test_os_dois_smokes_nao_partilham_identificadores() -> None:
    musique = carregar_impressao("musique_smoke_20", "v1")
    twowiki = carregar_impressao("twowiki_smoke_20", "v1")
    assert musique is not None and twowiki is not None
    assert not set(musique.question_ids) & set(twowiki.question_ids)
