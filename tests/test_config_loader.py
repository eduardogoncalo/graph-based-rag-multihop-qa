from pathlib import Path

from benchmark.core.config_loader import (
    load_dataset_config,
    load_experiment_config,
    load_method_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_dataset_config() -> None:
    config = load_dataset_config(ROOT / "configs/datasets/musique_smoke_20.yaml")

    assert config.dataset_id == "musique_smoke_20"
    assert config.dataset_version == "v1"


def test_load_method_config() -> None:
    config = load_method_config(ROOT / "configs/methods/vector_rag.yaml")

    assert config.method_id == "vector_rag"
    assert config.enabled is True


def test_load_experiment_config() -> None:
    config = load_experiment_config(ROOT / "configs/experiments/smoke.yaml")

    # `musique_smoke_20_v1` desde 2026-08-10, era `smoke_v1`. O identificador
    # passou a derivar do dataset porque o `reproduce.sh` deriva o mesmo valor,
    # e duas fontes de verdade que discordam sobre o nome do experimento
    # espalham resultados por dois sítios no Postgres.
    assert config.experiment_id == "musique_smoke_20_v1"
    assert "single_agent" in config.agent_modes


def test_todos_os_experimentos_declaram_um_id_derivado_do_dataset() -> None:
    """O `reproduce.sh` calcula `{dataset}_{versão}` — os configs têm de bater.

    Se um config declarar outro identificador, o `benchmark report` procura os
    resultados debaixo de um nome e o `reproduce.sh` escreveu-os debaixo de
    outro. Ninguém dá por isso até a tabela sair vazia.
    """
    for caminho in sorted((ROOT / "configs/experiments").glob("*.yaml")):
        if caminho.name == "musique_eval1k_runs_registry.yaml":
            continue  # registo de corridas da fase experimental, não é um experimento
        config = load_experiment_config(caminho)
        esperado = f"{config.dataset_id}_{config.dataset_version}"
        assert config.experiment_id == esperado, (
            f"{caminho.name}: experiment_id {config.experiment_id!r} não deriva "
            f"do dataset ({esperado!r})"
        )


def test_os_experimentos_nao_nomeiam_o_ms_graphrag_neo4j() -> None:
    """A variante Neo4j não tem cliente vivo: o adaptador recusa-se a correr.

    `ms_graphrag_neo4j live execution is not configured` — um experimento que a
    declare manda o professor a um método que rebenta com uma mensagem que
    parece um problema de instalação dele.
    """
    for caminho in sorted((ROOT / "configs/experiments").glob("*.yaml")):
        if caminho.name == "musique_eval1k_runs_registry.yaml":
            continue
        config = load_experiment_config(caminho)
        assert "ms_graphrag_neo4j" not in config.methods, caminho.name
