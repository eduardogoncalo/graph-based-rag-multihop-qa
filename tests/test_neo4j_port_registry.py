from __future__ import annotations

import pytest

from benchmark.infra.neo4j_containers import (
    Neo4jPortRegistry,
    PortConflictError,
    preflight_ports,
    resolve_ports,
)

# Os três métodos que recebem container próprio por dataset, e os datasets que
# o registo tem de saber servir. É a matriz que o `benchmark` consegue indexar.
METODOS = ("graphrag", "lightrag", "cognee")
SLUGS = (
    "musique_smoke_20_v1",
    "musique_ans_v1_0_eval1k",
    "twowiki_smoke_20_v1",
    "twowiki_ans_v1_0_eval1k",
)

# A banda do ambiente original. Reservada, nunca atribuída.
BANDA_ORIGINAL = frozenset(range(17474, 17479)) | frozenset(range(17687, 17692))


def test_registry_loads_musique_assignments() -> None:
    registry = Neo4jPortRegistry.load()
    assert registry.ports_for("graphrag", "musique_ans_v1_0_eval1k") == (18479, 18693)
    assert registry.ports_for("lightrag", "musique_ans_v1_0_eval1k") == (18480, 18694)
    assert registry.ports_for("cognee", "musique_ans_v1_0_eval1k") == (18481, 18695)


def test_registry_loads_twowiki_assignments() -> None:
    """O 2Wiki é metade do protocolo, e tem de ser indexável como o MuSiQue."""
    registry = Neo4jPortRegistry.load()
    assert registry.ports_for("graphrag", "twowiki_ans_v1_0_eval1k") == (18482, 18696)
    assert registry.ports_for("lightrag", "twowiki_ans_v1_0_eval1k") == (18483, 18697)
    assert registry.ports_for("cognee", "twowiki_ans_v1_0_eval1k") == (18484, 18698)


@pytest.mark.parametrize("slug", SLUGS)
@pytest.mark.parametrize("metodo", METODOS)
def test_todos_os_pares_metodo_dataset_tem_atribuicao(metodo: str, slug: str) -> None:
    """A matriz inteira, e não só as entradas que alguém se lembrou de usar.

    Uma atribuição em falta não dá erro na indexação — dá um KeyError no
    momento de subir o container, depois de o dinheiro da preparação já ter
    sido gasto.
    """
    http, bolt = Neo4jPortRegistry.load().ports_for(metodo, slug)
    assert 18476 <= http <= 18999
    assert 18689 <= bolt <= 18999


def test_nenhuma_atribuicao_esta_reservada() -> None:
    """O invariante que estava partido até 2026-08-10, e o motivo deste teste.

    As atribuições dos datasets `*_eval1k` apontavam para a banda 17xxx, que
    está inteira em `reserved` — logo o preflight recusava-as todas e nenhum
    dataset real era indexável. Uma atribuição que o preflight recusa não é uma
    atribuição: é uma armadilha que só se descobre a correr.
    """
    registry = Neo4jPortRegistry.load()
    reservadas_atribuidas = {
        (chave, porta)
        for chave, portas in registry.assignments.items()
        for porta in (portas["http"], portas["bolt"])
        if porta in registry.reserved
    }
    assert reservadas_atribuidas == set()


def test_a_banda_do_ambiente_original_continua_reservada() -> None:
    """Mover as atribuições não podia enfraquecer a protecção — não enfraqueceu."""
    registry = Neo4jPortRegistry.load()
    assert BANDA_ORIGINAL <= set(registry.reserved)


def test_a_banda_do_ambiente_original_nao_e_atribuida_a_ninguem() -> None:
    registry = Neo4jPortRegistry.load()
    atribuidas = {
        porta
        for portas in registry.assignments.values()
        for porta in (portas["http"], portas["bolt"])
    }
    assert atribuidas & BANDA_ORIGINAL == set()


def test_nenhuma_porta_e_atribuida_duas_vezes() -> None:
    """Duas entradas na mesma porta é um container a roubar o outro.

    O sintoma seria um índice a aparecer no dataset errado, que é o género de
    coisa que só se vê nos números.
    """
    registry = Neo4jPortRegistry.load()
    todas = [
        porta
        for portas in registry.assignments.values()
        for porta in (portas["http"], portas["bolt"])
    ]
    assert len(todas) == len(set(todas))


def test_atribuicoes_nao_colidem_com_o_docker_compose() -> None:
    """18474/18687 e 18475/18688 são dos serviços fixos do compose."""
    registry = Neo4jPortRegistry.load()
    do_compose = {18474, 18475, 18687, 18688}
    atribuidas = {
        porta
        for portas in registry.assignments.values()
        for porta in (portas["http"], portas["bolt"])
    }
    assert atribuidas & do_compose == set()


@pytest.mark.parametrize("slug", SLUGS)
@pytest.mark.parametrize("metodo", METODOS)
def test_resolve_ports_passa_o_preflight_das_reservadas(metodo: str, slug: str) -> None:
    """O preflight completo sonda a porta no host, e aqui não se quer isso.

    Verifica-se só o ramo que interessa a este registo — que nenhuma atribuição
    é recusada por estar reservada. Uma porta ocupada é uma questão da máquina,
    não do ficheiro.
    """
    registry = Neo4jPortRegistry.load()
    http, bolt = resolve_ports(metodo, slug, registry=registry, preflight=False)
    for porta in (http, bolt):
        assert porta not in registry.reserved


def test_registry_reserves_host_neo4j_and_benchmark_postgres() -> None:
    registry = Neo4jPortRegistry.load()
    assert {5432, 7474, 7687} <= set(registry.reserved)


def test_ports_for_unknown_key_raises() -> None:
    registry = Neo4jPortRegistry.load()
    with pytest.raises(KeyError):
        registry.ports_for("graphrag", "no_such_dataset")


def test_preflight_rejects_reserved_port() -> None:
    with pytest.raises(PortConflictError):
        preflight_ports(7687, reserved=frozenset({7687}))
