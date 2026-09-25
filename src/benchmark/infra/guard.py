"""Recusa qualquer ligação apontada ao ambiente experimental original.

Este repositório nasceu de um ambiente onde a fase experimental correu durante
dois meses e meio. Esse ambiente ainda existe na mesma máquina, nos mesmos
espaços de nomes do podman e nas mesmas portas de `localhost`, e guarda os
índices e os julgamentos que sustentam a dissertação.

Nada aqui pode tocar-lhe. O lançador de containers já protege contra
reutilizar um container pelo nome, mas não protege contra uma URI apontada à
mão — num `.env` mal preenchido, num argumento de linha de comando, ou num
valor por omissão esquecido. Do ponto de vista do programa essa ligação é
legítima, e por isso escreveria sem se queixar.

Este módulo é esse travão. Falha alto, cedo, e com o nome de quem tentou.

Há dois níveis de recusa, e a distinção importa para quem recebe o pacote:

- **Sempre recusadas** — a banda 17xxx e a 7689. Foram escolhidas aqui, não
  significam nada em mais máquina nenhuma, e recusá-las não custa nada a
  ninguém.
- **Só enquanto o original existir** — a 5432 e a 5433. São as portas por
  omissão do Postgres. Quem receber este pacote tem todo o direito de correr o
  Postgres dele na 5432, e transformá-la numa proibição universal partia a
  instalação de quem não tem nada a ver com esta dissertação.

O segundo nível liga-se sozinho quando o ambiente original está mesmo presente,
e fica inerte quando não está.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

# Portas do ambiente original. Derivadas do registo em
# configs/infra/neo4j_ports.yaml, repetidas aqui de propósito: o travão não
# pode depender de um ficheiro de configuração que alguém possa editar sem
# perceber o que está a desactivar.
PORTAS_SEMPRE_RECUSADAS: frozenset[int] = frozenset(
    {
        7689,  # Neo4j do Cognee da era CUAD (thesis_neo4j_cognee_1)
        *range(17474, 17479),  # Neo4j Option C, HTTP
        *range(17687, 17692),  # Neo4j Option C, Bolt
    }
)

# Portas padrão dos serviços: perigosas aqui, legítimas em qualquer outro sítio.
#
# O Neo4j entrou nesta lista a 2026-08-10, e foi um caso real. O `.env` deste
# repositório tinha `GRAPHRAG_NEO4J_URI=bolt://localhost:7687` — o Neo4j NATIVO
# do host desta máquina, que estava aberto e teria aceite escrita. Veio na
# bagagem do ambiente original, como o `DATABASE_URL` na 5432 que se corrigiu a
# 2026-08-09: nessa altura corrigiu-se só a linha do Postgres, e estas ficaram.
#
# São do mesmo nível que a 5432 e não do primeiro: 7474/7687 são as portas por
# omissão do Neo4j, e quem recebe o pacote tem todo o direito de correr um Neo4j
# nelas. Este repositório não as usa — o compose publica 18474/18687 e
# 18475/18688 — mas proibi-las universalmente partia a instalação de quem não
# tem nada a ver com esta dissertação.
PORTAS_RECUSADAS_SO_COM_O_ORIGINAL_PRESENTE: frozenset[int] = frozenset(
    {
        5432,  # Postgres do benchmark: respostas, julgamentos, McNemar
        5433,  # Postgres do Cognee
        7474,  # Neo4j nativo do host, HTTP
        7687,  # Neo4j nativo do host, Bolt
    }
)

PORTAS_DO_AMBIENTE_ORIGINAL: frozenset[int] = (
    PORTAS_SEMPRE_RECUSADAS | PORTAS_RECUSADAS_SO_COM_O_ORIGINAL_PRESENTE
)

# Os datasets sobre os quais a fase experimental correu. Os containers Neo4j
# têm nome derivado de *(método, dataset)* e são globais ao podman, não à
# pasta: `neo4j_cognee_musique_ans_v1_0_eval1k` nesta máquina É o container que
# guarda o índice da dissertação, venha o pedido de onde vier.
#
# Isto é do MESMO nível que a 5432 — recusado só enquanto o original estiver
# presente. Quem recebe o pacote tem todo o direito de indexar os datasets
# completos, e é exactamente isso que se quer que consiga fazer; o que não pode
# é acontecer nesta máquina, onde o nome colide.
#
# Até 2026-08-10 esta protecção era um efeito colateral: as atribuições de
# porta destes datasets apontavam para a banda 17xxx, reservada, e o preflight
# recusava-as. Mas isso também tornava os datasets reais inindexáveis por
# QUALQUER pessoa, incluindo quem recebe o pacote numa máquina limpa. Movidas
# as portas para a banda 18xxx, a protecção passou a ter de ser dita em voz
# alta — que é o que este par de nomes faz.
SLUGS_DO_AMBIENTE_ORIGINAL: frozenset[str] = frozenset(
    {
        "musique_ans_v1_0_eval1k",
        "twowiki_ans_v1_0_eval1k",
    }
)

# Aponta o travão ao repositório original. Serve para o localizar quando ele
# não está ao lado, e — posta a vazio — para o desligar explicitamente em quem
# recebe o pacote e nunca teve ambiente original nenhum.
VARIAVEL_DO_AMBIENTE_ORIGINAL = "BENCHMARK_AMBIENTE_ORIGINAL"

# Assinatura do repositório original: quatro pastas que a Fase 1 do plano marca
# como «fica», ou seja, que por construção nunca viajam neste pacote. É assim
# que o original se reconhece sem escrever no código o caminho pessoal de
# ninguém, e é por serem quatro que a assinatura não apanha um projecto
# qualquer — `artifacts/` e `status/` sozinhas são nomes demasiado comuns, e
# apanhavam vizinhos que nada têm a ver com a dissertação.
_ASSINATURA_DO_ORIGINAL = ("artifacts", "artifacts_variant", "status", "writing")

_ANFITRIOES_LOCAIS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


class LigacaoAoAmbienteOriginalError(RuntimeError):
    """Tentativa de ligar a um serviço do ambiente experimental original."""


def _raiz_deste_repositorio() -> Path:
    # src/benchmark/infra/guard.py -> raiz
    return Path(__file__).resolve().parents[3]


def caminhos_do_ambiente_original() -> tuple[Path, ...]:
    """Os repositórios originais presentes nesta máquina.

    Procura por duas vias, e nenhuma escreve no código o caminho pessoal de
    quem quer que seja:

    1. a variável `BENCHMARK_AMBIENTE_ORIGINAL`, que o localiza onde estiver —
       ou, posta a vazio, declara que não há nenhum;
    2. os irmãos deste repositório com a assinatura do original. O plano diz
       que o pacote nasce «ao lado do repositório original e fora dele», e é
       essa vizinhança que se procura.

    Devolve **todos** os candidatos, e não o primeiro: parar no primeiro por
    ordem alfabética já escolheu a árvore errada uma vez, e o travão de
    caminho ficou a proteger uma pasta que não era a da dissertação.

    Vazio quando não há ambiente original — o caso de quem recebe o pacote, e
    em que as portas padrão do Postgres deixam de ser proibidas.
    """
    declarado = os.environ.get(VARIAVEL_DO_AMBIENTE_ORIGINAL)
    if declarado is not None:
        if not declarado:
            return ()
        candidato = Path(declarado).expanduser()
        return (candidato,) if candidato.is_dir() else ()

    raiz = _raiz_deste_repositorio()
    try:
        irmaos = sorted(raiz.parent.iterdir())
    except OSError:
        return ()
    return tuple(
        irmao
        for irmao in irmaos
        if irmao != raiz
        and irmao.is_dir()
        and all((irmao / marca).is_dir() for marca in _ASSINATURA_DO_ORIGINAL)
    )


def caminho_do_ambiente_original() -> Path | None:
    """O primeiro ambiente original detectado, ou `None` se não houver nenhum."""
    caminhos = caminhos_do_ambiente_original()
    return caminhos[0] if caminhos else None


def porta_e_do_ambiente_original(porta: int | None, anfitriao: str | None) -> bool:
    """True se (anfitrião, porta) pertencer ao ambiente original.

    Só se aplica a anfitriões locais: uma porta 5432 numa máquina remota é
    outra base de dados e não diz respeito a este travão.
    """
    if porta is None:
        return False
    if anfitriao is not None and anfitriao.lower() not in _ANFITRIOES_LOCAIS:
        return False
    if porta in PORTAS_SEMPRE_RECUSADAS:
        return True
    if porta not in PORTAS_RECUSADAS_SO_COM_O_ORIGINAL_PRESENTE:
        return False
    return bool(caminhos_do_ambiente_original())


def exigir_alvo_permitido(uri: str, *, origem: str) -> str:
    """Devolve a URI, ou levanta se ela apontar ao ambiente original.

    `origem` identifica quem pediu a ligação — nome da variável de ambiente, do
    argumento, ou do ficheiro de configuração — para a mensagem dizer onde se
    corrige.
    """
    try:
        partes = urlparse(uri)
    except ValueError:
        return uri

    if not porta_e_do_ambiente_original(partes.port, partes.hostname):
        return uri

    raise LigacaoAoAmbienteOriginalError(
        f"{origem} points at {partes.hostname}:{partes.port}, which belongs to the "
        f"original experimental environment and holds the thesis data. "
        f"This repository must not touch it. "
        f"This repository's ports: Postgres 15432, Cognee Postgres 15433, "
        f"Neo4j in the 18xxx band. Fix {origem} and run again."
    )


def slug_e_do_ambiente_original(slug: str) -> bool:
    """True se o slug for de um dataset da fase experimental E o original existir.

    Os dois lados da conjunção importam. Numa máquina limpa não há container
    nenhum com esse nome, e recusar o slug proibiria quem recebe o pacote de
    indexar exactamente os datasets que a dissertação usou — que é o contrário
    do que se quer.
    """
    if slug not in SLUGS_DO_AMBIENTE_ORIGINAL:
        return False
    return bool(caminhos_do_ambiente_original())


def exigir_dataset_permitido(slug: str, *, origem: str) -> str:
    """Devolve o slug, ou levanta se ele nomear um container da dissertação.

    O terceiro irmão do `exigir_alvo_permitido` e do `exigir_caminho_permitido`,
    e o que fecha o caminho que os outros dois não veem: o nome do container.
    Uma URI pode estar certa e um caminho pode estar certo, e ainda assim
    `podman` entregar o container errado — porque o nome deriva de *(método,
    dataset)* e o espaço de nomes do podman é global à máquina.

    A mensagem diz explicitamente para NÃO remover o container. O
    `start_neo4j_container` já recusa um container existente, mas a mensagem
    dele — «remove it first» — é um convite a destruir um índice que não volta
    a ser gerado.
    """
    if not slug_e_do_ambiente_original(slug):
        return slug

    raise LigacaoAoAmbienteOriginalError(
        f"{origem} names dataset {slug!r}, one of the datasets from the original "
        f"experimental phase. Neo4j containers are named after (method, dataset) "
        f"and podman's namespace is global to the machine: here, that name "
        f"belongs to the container holding the thesis index. "
        f"DO NOT remove that container — its indexes cannot be regenerated. "
        f"To exercise the chain, use the smoke samples "
        f"(musique_smoke_20, twowiki_smoke_20), which have identifiers, "
        f"containers, volumes and ports of their own. "
        f"On a machine without the original environment this brake sits inert "
        f"and the complete datasets index normally."
    )


def exigir_caminho_permitido(caminho: str | Path, *, origem: str) -> Path:
    """Devolve o caminho resolvido, ou levanta se ele cair dentro do original.

    O irmão do `exigir_alvo_permitido` para os métodos que não têm URI. O
    HippoRAG 2 é o caso: guarda o índice em disco, não numa base de dados, e a
    forma de lhe estragar o trabalho é apontar-lhe o destino de escrita ao
    `artifacts_variant/` do original, onde vivem os índices oficiais.

    Sem ambiente original presente não há nada a proteger, e devolve sempre.
    """
    alvo = Path(caminho).expanduser().resolve()
    for candidato in caminhos_do_ambiente_original():
        original = candidato.resolve()
        if alvo == original or original in alvo.parents:
            break
    else:
        return alvo

    raise LigacaoAoAmbienteOriginalError(
        f"{origem} points at {alvo}, which sits inside the original experimental "
        f"environment ({original}) and holds the thesis artefacts. "
        f"This repository must not touch it, and writing there would destroy "
        f"indexes that cannot be regenerated. Point {origem} at a destination "
        f"inside this repository and run again."
    )
