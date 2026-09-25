"""O travão que impede este repositório de tocar no ambiente da dissertação.

Estes testes existem porque a protecção é invisível quando funciona. Se
alguém repuser um valor por omissão antigo, é aqui que se dá por isso.

Há dois níveis, e os testes separam-nos de propósito. As portas da banda 17xxx
são recusadas em qualquer máquina. As portas padrão do Postgres — 5432 e 5433 —
só o são enquanto o ambiente original estiver presente, porque no pacote final
pertencem legitimamente a quem o recebe. Os testes que dependem dessa distinção
declaram sempre o estado do ambiente, para a suíte dar o mesmo resultado aqui e
na máquina de quem clonar.
"""

from __future__ import annotations

import sys

import pytest

from benchmark.core.settings import Settings
from benchmark.infra.guard import (
    VARIAVEL_DO_AMBIENTE_ORIGINAL,
    LigacaoAoAmbienteOriginalError,
    caminho_do_ambiente_original,
    exigir_alvo_permitido,
    exigir_caminho_permitido,
    porta_e_do_ambiente_original,
)
from scripts import (
    cognee_index_musique,
    cognee_index_musique_batched,
    cognee_start_neo4j,
    hipporag2_index_runner,
)


@pytest.fixture
def sem_ambiente_original(monkeypatch) -> None:
    """A máquina de quem recebe o pacote: não há original nenhum ao lado."""
    monkeypatch.setenv(VARIAVEL_DO_AMBIENTE_ORIGINAL, "")


ASSINATURA = ("artifacts", "artifacts_variant", "status", "writing")


@pytest.fixture
def com_ambiente_original(monkeypatch, tmp_path):
    """A máquina do autor: o repositório original existe e tem de ser poupado."""
    original = tmp_path / "thesis"
    for marca in ASSINATURA:
        (original / marca).mkdir(parents=True)
    monkeypatch.setenv(VARIAVEL_DO_AMBIENTE_ORIGINAL, str(original))
    return original


# --------------------------------------------------------------------------
# Nível 1 — recusado sempre, em qualquer máquina
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "porta",
    [
        7689,  # Neo4j do Cognee da era CUAD
        17474,  # Neo4j Option C, primeiro HTTP
        17478,  # Neo4j Option C, último HTTP
        17687,  # Neo4j Option C, primeiro Bolt
        17689,  # o que os scripts do Cognee traziam por omissão
        17691,  # Neo4j Option C, último Bolt
    ],
)
def test_portas_reservadas_sao_recusadas_mesmo_sem_o_original(
    porta: int, sem_ambiente_original
) -> None:
    """Estas portas foram escolhidas aqui e não significam nada noutro sítio.

    Recusá-las numa máquina alheia não custa nada a ninguém, por isso a recusa
    não depende de o ambiente original existir.
    """
    assert porta_e_do_ambiente_original(porta, "localhost")
    with pytest.raises(LigacaoAoAmbienteOriginalError):
        exigir_alvo_permitido(f"bolt://localhost:{porta}", origem="TESTE")


# --------------------------------------------------------------------------
# Nível 2 — protecção local, não proibição universal
# --------------------------------------------------------------------------


@pytest.mark.parametrize("porta", [5432, 5433])
def test_portas_padrao_do_postgres_sao_recusadas_com_o_original_presente(
    porta: int, com_ambiente_original
) -> None:
    assert porta_e_do_ambiente_original(porta, "localhost")
    with pytest.raises(LigacaoAoAmbienteOriginalError):
        exigir_alvo_permitido(f"postgresql://u:p@localhost:{porta}/x", origem="TESTE")


@pytest.mark.parametrize("porta", [5432, 5433])
def test_portas_padrao_do_postgres_passam_no_pacote_final(
    porta: int, sem_ambiente_original
) -> None:
    """A 5432 é a porta por omissão do Postgres de toda a gente.

    Quem recebe este pacote tem todo o direito de a usar. Transformá-la numa
    proibição universal partia a instalação de quem não tem nada a ver com esta
    dissertação — a lista antiga é protecção local, não regra do pacote.
    """
    uri = f"postgresql://u:p@localhost:{porta}/x"
    assert not porta_e_do_ambiente_original(porta, "localhost")
    assert exigir_alvo_permitido(uri, origem="TESTE") == uri


@pytest.mark.parametrize("porta", [7474, 7687])
def test_neo4j_nativo_do_host_e_recusado_com_o_original_presente(
    porta: int, com_ambiente_original
) -> None:
    """Entrou a 2026-08-10, e foi um caso real, não hipotético.

    O `.env` deste repositório trazia `GRAPHRAG_NEO4J_URI=bolt://localhost:7687`
    — o Neo4j NATIVO do host desta máquina, que estava aberto e teria aceite
    escrita. Veio na bagagem do ambiente original, como o `DATABASE_URL` na 5432
    que se corrigiu a 2026-08-09: nessa altura corrigiu-se só a linha do
    Postgres, e as do Neo4j ficaram.

    Este repositório nunca usa a 7687 — o compose publica 18687 e 18688.
    """
    assert porta_e_do_ambiente_original(porta, "localhost")
    with pytest.raises(LigacaoAoAmbienteOriginalError):
        exigir_alvo_permitido(f"bolt://localhost:{porta}", origem="TESTE")


@pytest.mark.parametrize("porta", [7474, 7687])
def test_neo4j_nativo_do_host_passa_no_pacote_final(
    porta: int, sem_ambiente_original
) -> None:
    """7474/7687 são as portas por omissão do Neo4j de toda a gente.

    Mesma lógica da 5432: protecção local, não regra do pacote. Quem recebe isto
    numa máquina limpa pode ter lá um Neo4j, e não é da conta deste travão.
    """
    uri = f"bolt://localhost:{porta}"
    assert not porta_e_do_ambiente_original(porta, "localhost")
    assert exigir_alvo_permitido(uri, origem="TESTE") == uri


def test_ambiente_original_e_detectado_pela_assinatura(com_ambiente_original) -> None:
    assert caminho_do_ambiente_original() == com_ambiente_original


def test_deteccao_por_vizinhanca_exige_a_assinatura_completa(monkeypatch, tmp_path) -> None:
    """Sem a variável, o original reconhece-se por ser irmão e ter a assinatura.

    Uma pasta parecida não chega. `artifacts/` e `status/` sozinhas são nomes
    comuns — nesta máquina batem em cinco e seis projectos sem relação nenhuma
    com a dissertação, e foi assim que a detecção escolheu a árvore errada. A
    assinatura são as quatro pastas que o plano marca como «fica».
    """
    from benchmark.infra import guard

    monkeypatch.delenv(VARIAVEL_DO_AMBIENTE_ORIGINAL, raising=False)
    pacote = tmp_path / "graph_based_rag"
    pacote.mkdir()
    monkeypatch.setattr(guard, "_raiz_deste_repositorio", lambda: pacote)

    vizinho = tmp_path / "outro_projecto_qualquer"
    (vizinho / "artifacts").mkdir(parents=True)
    (vizinho / "status").mkdir()
    assert caminho_do_ambiente_original() is None, "dois nomes comuns não são o original"

    original = tmp_path / "thesis"
    for marca in ASSINATURA:
        (original / marca).mkdir(parents=True)
    assert caminho_do_ambiente_original() == original


def test_deteccao_devolve_todos_os_candidatos(monkeypatch, tmp_path) -> None:
    """Parar no primeiro por ordem alfabética já protegeu a pasta errada."""
    from benchmark.infra import guard

    monkeypatch.delenv(VARIAVEL_DO_AMBIENTE_ORIGINAL, raising=False)
    pacote = tmp_path / "graph_based_rag"
    pacote.mkdir()
    monkeypatch.setattr(guard, "_raiz_deste_repositorio", lambda: pacote)

    for nome in ("aaa_copia", "thesis"):
        for marca in ASSINATURA:
            (tmp_path / nome / marca).mkdir(parents=True)

    assert guard.caminhos_do_ambiente_original() == (tmp_path / "aaa_copia", tmp_path / "thesis")
    # E o travão de caminho protege o segundo, não só o primeiro.
    with pytest.raises(LigacaoAoAmbienteOriginalError):
        exigir_caminho_permitido(
            tmp_path / "thesis" / "artifacts_variant" / "x", origem="--save-dir"
        )


# --------------------------------------------------------------------------
# As portas deste repositório, que têm de continuar a passar
# --------------------------------------------------------------------------


@pytest.mark.parametrize("porta", [15432, 15433, 18474, 18687, 18688, 18689])
def test_portas_deste_repositorio_passam(porta: int, com_ambiente_original) -> None:
    """Mesmo com o original presente — senão o travão travava-nos a nós."""
    assert not porta_e_do_ambiente_original(porta, "localhost")
    uri = f"bolt://localhost:{porta}"
    assert exigir_alvo_permitido(uri, origem="TESTE") == uri


def test_anfitriao_remoto_nao_e_travado(com_ambiente_original) -> None:
    """Um 5432 noutra máquina é outra base de dados, não a da dissertação."""
    uri = "postgresql://u:p@db.exemplo.org:5432/x"
    assert exigir_alvo_permitido(uri, origem="TESTE") == uri


def test_mensagem_diz_onde_se_corrige(com_ambiente_original) -> None:
    with pytest.raises(LigacaoAoAmbienteOriginalError) as erro:
        exigir_alvo_permitido("postgresql://u:p@localhost:5432/x", origem="DATABASE_URL")
    texto = str(erro.value)
    assert "DATABASE_URL" in texto
    assert "15432" in texto


def test_definicoes_por_omissao_nao_apontam_ao_ambiente_original() -> None:
    definicoes = Settings(_env_file=None)
    assert ":15432/" in definicoes.database_url
    assert definicoes.graphrag_neo4j_uri.endswith(":18687")
    assert definicoes.lightrag_neo4j_uri.endswith(":18688")


def test_env_mal_preenchido_falha_ao_carregar() -> None:
    """A protecção que mais importa: um .env com as URI antigas não arranca."""
    with pytest.raises(LigacaoAoAmbienteOriginalError):
        Settings(_env_file=None, lightrag_neo4j_uri="bolt://localhost:17688")


@pytest.mark.parametrize("dataset_id", ["musique", "twowiki"])
def test_cognee_option_c_nao_alcanca_o_grafo_da_dissertacao(
    dataset_id: str, com_ambiente_original
) -> None:
    """O caminho do Cognee liga-se a containers já em execução.

    Corre com preflight desligado de propósito, portanto a recusa por porta
    reservada — que trava os outros métodos — não se aplica aqui.

    **Este teste mudou a 2026-08-10, e o que ele protege não.** Até essa data a
    recusa vinha da porta: o registo atribuía a banda 17xxx a estes datasets, e
    o `exigir_alvo_permitido` recusava a URI. Só que essa atribuição também
    tornava os datasets completos inindexáveis por QUALQUER pessoa — o preflight
    recusava-os na máquina de quem recebe o pacote, onde não há nada a
    proteger.

    Movidas as portas para a banda 18xxx, a recusa passou a vir do nome do
    dataset, que é onde o perigo realmente mora: o container Neo4j chama-se
    `neo4j_cognee_<slug>` e o espaço de nomes do podman é global à máquina.
    """
    from benchmark.methods.cognee.option_c import resolve_cognee_option_c

    with pytest.raises(LigacaoAoAmbienteOriginalError):
        resolve_cognee_option_c(
            dataset_id=dataset_id, dataset_version="ans_v1.0_eval1k", password="x"
        )


@pytest.mark.parametrize("dataset_id", ["musique", "twowiki"])
def test_cognee_option_c_indexa_os_datasets_completos_na_maquina_limpa(
    dataset_id: str, sem_ambiente_original
) -> None:
    """O outro lado, e é o que o pacote existe para permitir.

    Quem recebe isto tem de conseguir indexar exactamente os datasets que a
    dissertação usou. Sem ambiente original não há container nenhum com esse
    nome, e o travão fica inerte — como a 5432.
    """
    from benchmark.methods.cognee.option_c import resolve_cognee_option_c

    config = resolve_cognee_option_c(
        dataset_id=dataset_id, dataset_version="ans_v1.0_eval1k", password="x"
    )
    assert config.local_index is not None
    # Banda 18xxx, deste repositório — nunca a 17xxx do original.
    porta = int(config.local_index.graph_url.rsplit(":", 1)[1])
    assert 18689 <= porta <= 18999


@pytest.mark.parametrize("dataset_id", ["musique", "twowiki"])
def test_lightrag_option_c_nao_alcanca_o_container_da_dissertacao(
    dataset_id: str, com_ambiente_original
) -> None:
    """O mesmo buraco existia no LightRAG, e fechou-se ao mesmo tempo."""
    from benchmark.methods.lightrag_neo4j.option_c import resolve_lightrag_option_c

    with pytest.raises(LigacaoAoAmbienteOriginalError):
        resolve_lightrag_option_c(
            dataset_id=dataset_id,
            dataset_version="ans_v1.0_eval1k",
            preflight=False,
        )


@pytest.mark.parametrize("dataset_id", ["musique", "twowiki"])
def test_lightrag_option_c_indexa_os_datasets_completos_na_maquina_limpa(
    dataset_id: str, sem_ambiente_original
) -> None:
    from benchmark.methods.lightrag_neo4j.option_c import resolve_lightrag_option_c

    binding = resolve_lightrag_option_c(
        dataset_id=dataset_id,
        dataset_version="ans_v1.0_eval1k",
        preflight=False,
    )
    assert 18689 <= binding.bolt_port <= 18999


@pytest.mark.parametrize(
    "dataset_id,version",
    [
        ("musique_smoke_20", "v1"),
        ("twowiki_smoke_20", "v1"),
    ],
)
def test_as_amostras_de_smoke_correm_mesmo_com_o_original_presente(
    dataset_id: str, version: str, com_ambiente_original
) -> None:
    """É esta a saída que a mensagem de recusa oferece, e tem de funcionar.

    As amostras têm identificador próprio, logo container e volume próprios, e
    não colidem com nada da dissertação. Recusá-las seria deixar quem trabalha
    nesta máquina sem nenhuma forma de exercitar a cadeia.
    """
    from benchmark.methods.lightrag_neo4j.option_c import resolve_lightrag_option_c

    binding = resolve_lightrag_option_c(
        dataset_id=dataset_id, dataset_version=version, preflight=False
    )
    assert binding.slug == f"{dataset_id}_{version}"
    assert 18689 <= binding.bolt_port <= 18999


# --------------------------------------------------------------------------
# Os subprocessos oficiais — os caminhos que corriam fora do alcance do travão
# --------------------------------------------------------------------------

INDEXADORES_DO_COGNEE = [cognee_index_musique, cognee_index_musique_batched]


@pytest.mark.parametrize("modulo", INDEXADORES_DO_COGNEE, ids=lambda m: m.__name__)
def test_indexador_do_cognee_recusa_bolt_do_ambiente_original(modulo) -> None:
    """`--bolt` era obrigatório, mas obrigatório não é validado.

    Tirar-lhe o valor por omissão impediu o acidente distraído. Não impedia
    escrever a URI antiga à mão, e o script corre num venv isolado onde nada
    mais do travão chega.
    """
    with pytest.raises(LigacaoAoAmbienteOriginalError):
        modulo.exigir_bolt_permitido("bolt://localhost:17689")


@pytest.mark.parametrize("modulo", INDEXADORES_DO_COGNEE, ids=lambda m: m.__name__)
def test_indexador_do_cognee_aceita_bolt_deste_repositorio(modulo) -> None:
    uri = "bolt://localhost:18689"
    assert modulo.exigir_bolt_permitido(uri) == uri


def test_cognee_start_neo4j_recusa_porta_do_ambiente_original(monkeypatch) -> None:
    """Este arranca containers: um `start` sobre o container do original

    punha de pé o Neo4j da dissertação. A recusa tem de vir antes de tocar no
    podman.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cognee_start_neo4j.py",
            "--name", "neo4j_cognee_musique_ans_v1_0_eval1k",
            "--http-port", "17476",
            "--bolt-port", "17689",
            "--password", "x",
        ],
    )

    def _nunca_chega_aqui(nome: str) -> bool:  # pragma: no cover - o travão dispara antes
        raise AssertionError("o travão deixou passar e o podman foi contactado")

    monkeypatch.setattr(cognee_start_neo4j, "container_exists", _nunca_chega_aqui)

    with pytest.raises(LigacaoAoAmbienteOriginalError):
        cognee_start_neo4j.main()


def test_cognee_start_neo4j_aceita_portas_deste_repositorio(monkeypatch) -> None:
    class ChegouAoPodman(Exception):
        """Sentinela: prova que o travão deixou passar."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cognee_start_neo4j.py",
            "--name", "neo4j_cognee_musique_smoke",
            "--http-port", "18476",
            "--bolt-port", "18689",
            "--password", "x",
        ],
    )

    def _marca(nome: str) -> bool:
        raise ChegouAoPodman

    monkeypatch.setattr(cognee_start_neo4j, "container_exists", _marca)

    with pytest.raises(ChegouAoPodman):
        cognee_start_neo4j.main()


def test_hipporag2_recusa_save_dir_dentro_do_ambiente_original(com_ambiente_original) -> None:
    """O HippoRAG 2 é o único método sem URI.

    O que há para fechar não é uma ligação, é o destino de escrita: o
    `artifacts_variant/` do original guarda os índices oficiais, que custaram
    uma passagem completa sobre o corpus e não voltam a ser gerados.
    """
    alvo = com_ambiente_original / "artifacts_variant" / "hipporag2_musique"
    with pytest.raises(LigacaoAoAmbienteOriginalError) as erro:
        hipporag2_index_runner.exigir_save_dir_permitido(str(alvo))
    assert "--save-dir" in str(erro.value)


def test_hipporag2_aceita_save_dir_deste_repositorio(
    com_ambiente_original, tmp_path
) -> None:
    destino = tmp_path / "artifacts" / "hipporag2_musique_smoke"
    assert hipporag2_index_runner.exigir_save_dir_permitido(str(destino)) == destino.resolve()


def test_caminho_e_livre_sem_ambiente_original(sem_ambiente_original, tmp_path) -> None:
    """Sem original presente não há nada a proteger, e nenhum destino é proibido."""
    destino = tmp_path / "qualquer" / "sitio"
    assert exigir_caminho_permitido(destino, origem="--save-dir") == destino.resolve()
