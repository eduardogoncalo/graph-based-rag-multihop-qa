#!/usr/bin/env bash
#
# The official path: from ingestion to report, in one command.
#
#   ./scripts/reproduce.sh                 # smoke mode, what the video shows
#   ./scripts/reproduce.sh --mode full     # all five methods, isolated envs
#   ./scripts/reproduce.sh --dry-run       # checks everything, spends nothing
#
#   ./scripts/reproduce.sh --dataset twowiki_smoke_20   # the other half of the protocol
#   ./scripts/reproduce.sh --dataset musique --version ans_v1.0_eval1k --mode full
#                                          # the full protocol: 1000 questions.
#                                          # Days, and tens of dollars. See the README.
#
# `--mode` picks WHICH METHODS run; `--dataset` picks HOW MANY QUESTIONS.
# Different things, and confusing them costs money: `--mode full` over the
# smoke sample costs cents, over the complete dataset it costs tens of
# dollars and ~38 h in Cognee indexing alone.
#
# WHAT THIS SCRIPT IS. A chain with brakes. It checks every precondition
# BEFORE the first paid call, because failing halfway through indexing means
# losing the money already spent. Every known trap is here as a check, a
# pinned value or a warning — the list is in scripts/README.md, and it does
# not rediscover itself.
#
# WHAT IT IS NOT. It does not reproduce the thesis's numbers. Indexing,
# reading and judging are done by language models, and none of that is
# deterministic. It reproduces the PROTOCOL.

set -euo pipefail

cd "$(dirname "$0")/.."
RAIZ="$PWD"

MODO="smoke"
DRY_RUN=0
LIMITE=""
SALTAR_INDEXACAO=0
DATASET="musique_smoke_20"
VERSAO="v1"
EXPERIMENTO=""

while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODO="${2:-}"; shift ;;
    --dataset) DATASET="${2:-}"; shift ;;
    --version) VERSAO="${2:-}"; shift ;;
    --experiment-id) EXPERIMENTO="${2:-}"; shift ;;
    --limit) LIMITE="${2:-}"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --skip-index) SALTAR_INDEXACAO=1 ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$MODO" in
  smoke|full) ;;
  *) echo "mode has to be 'smoke' or 'full', not '$MODO'" >&2; exit 2 ;;
esac

# O identificador do experimento deriva do dataset, e a razão não é estética.
# Dois datasets debaixo do mesmo `experiment_id` misturam-se no Postgres — a
# selecção de perguntas filtra por dataset, mas o relatório e o juiz agregam por
# experimento. Já aconteceu com o `--out-tag`, a 2026-08-09, e os ficheiros de
# sumário colapsaram num só.
EXPERIMENTO="${EXPERIMENTO:-${DATASET}_${VERSAO}}"

# Que ficheiros brutos é que este dataset precisa, e como se descarregam. As
# amostras de smoke partilham os ficheiros do dataset completo — o
# `fetch_datasets.py` só descarrega um conjunto por família.
case "$DATASET" in
  musique|musique_smoke_20)
    FAMILIA="musique"
    FICHEIRO_BRUTO="data/raw/musique/musique_ans_v1.0_dev.jsonl"
    ;;
  twowiki|twowiki_smoke_20)
    FAMILIA="twowiki"
    FICHEIRO_BRUTO="data/raw/2wikimultihop/2wikimultihopqa.json"
    ;;
  *)
    echo "unknown dataset: '$DATASET'" >&2
    echo "known: musique, musique_smoke_20, twowiki, twowiki_smoke_20" >&2
    exit 2
    ;;
esac

PYTHON="$RAIZ/.venv/bin/python"

if [ "$MODO" = "smoke" ]; then
  METODOS=(vector_rag lightrag_neo4j)
else
  # O Microsoft GraphRAG do modo full é o `ms_graphrag` — a implementação
  # oficial em ficheiros, indexada pela CLI do .venvs/graphrag. NÃO é o
  # `ms_graphrag_neo4j`: essa variante não tem cliente vivo, e o adaptador
  # levanta "live execution is not configured". Ver o PLANO_RELEASE.md.
  METODOS=(vector_rag lightrag_neo4j ms_graphrag)
  # cognee e hipporag2 não estão na CLI: correm por scripts próprios e
  # despacham para os ambientes isolados.
fi

# --------------------------------------------------------------------------- #

titulo() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
log()    { printf '   %s\n' "$*"; }
aviso()  { printf '   \033[33mAVISO:\033[0m %s\n' "$*"; }
erro()   { printf '\033[31mERRO:\033[0m %s\n' "$*" >&2; }
morrer() { erro "$*"; exit 1; }

# --------------------------------------------------------------------------- #
titulo "1. Preconditions"
# Tudo o que se verifica aqui é uma coisa que já falhou a meio de uma execução
# paga. A ordem é por custo de descoberta: primeiro o que é barato de detectar.
# --------------------------------------------------------------------------- #

[ -x "$PYTHON" ] || morrer "$PYTHON does not exist. Run ./scripts/bootstrap_envs.sh first."
log "main environment: $("$PYTHON" --version)"

[ -f .env ] || morrer "no .env found. Copy .env.example and fill in OPENAI_API_KEY."

# O .env TEM de ir para o ambiente do processo, e não basta o pydantic lê-lo.
# Descoberto a 2026-08-09, a doer: metade dos adaptadores lê os.environ
# directamente — MODEL_PROVIDER, READER_GROUNDING, LIGHTRAG_REUSE_RAG e outras
# treze — e essas variáveis nunca chegam lá pelo Settings. Sem este `set -a`, o
# LightRAG rebenta com "embedding_func is required for vector storage", que não
# faz nenhuma alusão à causa.
set -a
# shellcheck disable=SC1091
. ./.env
set +a
log ".env exported into the environment (not just loaded by Settings)"

[ -n "${OPENAI_API_KEY:-}" ] || morrer "OPENAI_API_KEY is empty in .env. It is required: the judge fails immediately without it."
log "OPENAI_API_KEY: set"

if [ "${MODEL_PROVIDER:-}" != "openai" ]; then
  morrer "MODEL_PROVIDER=${MODEL_PROVIDER:-<empty>}, it has to be 'openai'. The code default is 'fake', which returns synthetic answers and exists for the test suite."
fi
log "MODEL_PROVIDER: openai"

# READER_GROUNDING is the instruction given to the fixed reader of the
# controlled arm. v1 (free) is the thesis configuration; v2 (grounded) is an
# ablation whose runs were discarded. Set wrong, it silently produces a
# different experiment, so the script always says out loud which one runs, and
# refuses values the code would not recognise.
export READER_GROUNDING="${READER_GROUNDING:-v1}"
case "$(printf '%s' "$READER_GROUNDING" | tr '[:upper:]' '[:lower:]')" in
  v1|off|free|0|no|none|false)
    LEITOR=v1
    log "READER_GROUNDING=$READER_GROUNDING — FREE reader (v1), the thesis configuration"
    ;;
  v2|on|grounded|1|yes|true)
    LEITOR=v2
    aviso "READER_GROUNDING=$READER_GROUNDING — GROUNDED reader (v2), an ablation, NOT the thesis configuration"
    ;;
  *)
    morrer "READER_GROUNDING=$READER_GROUNDING is not recognised. Use v1 (the thesis configuration) or v2 (an ablation)."
    ;;
esac

# Sem isto o LightRAG instancia um objecto RAG por pergunta e estoura a
# memória. Custou execuções perdidas.
export LIGHTRAG_REUSE_RAG="${LIGHTRAG_REUSE_RAG:-1}"
log "LIGHTRAG_REUSE_RAG=$LIGHTRAG_REUSE_RAG"

# O travão do dataset, verificado AQUI e não na hora de subir o container. Numa
# máquina com o ambiente experimental original ao lado, os containers Neo4j dos
# datasets completos são literalmente os que guardam os índices da dissertação —
# o nome deriva de (método, dataset) e o espaço de nomes do podman é global.
#
# Sem esta verificação, a recusa vinha só na indexação, depois de a ingestão e a
# persistência já terem corrido. Numa máquina limpa isto não diz nada e não
# custa nada: o travão fica inerte.
if ! "$PYTHON" - "$DATASET" "$VERSAO" <<'FIM'
import sys
from benchmark.core.naming import slugify_dataset_version
from benchmark.infra.guard import LigacaoAoAmbienteOriginalError, exigir_dataset_permitido

slug = slugify_dataset_version(sys.argv[1], sys.argv[2])
try:
    exigir_dataset_permitido(slug, origem="--dataset of reproduce.sh")
except LigacaoAoAmbienteOriginalError as erro:
    print(erro, file=sys.stderr)
    sys.exit(1)
FIM
then
  morrer "the requested dataset belongs to the original experimental environment"
fi
log "dataset allowed on this machine"

if ! command -v podman >/dev/null 2>&1 && ! command -v docker >/dev/null 2>&1; then
  morrer "neither podman nor docker on PATH. The infrastructure is a compose file."
fi
COMPOSE="podman compose"
CONTENTOR="podman"
if ! command -v podman >/dev/null 2>&1; then
  COMPOSE="docker compose"
  CONTENTOR="docker"
fi
log "compose: $COMPOSE"

# O compose prefixa os volumes com o nome do DIRECTÓRIO. Subir isto de uma
# pasta com outro nome cria um banco vazio e choca na porta.
log "compose project: $(basename "$RAIZ") (volumes inherit this name)"

if [ "$MODO" = "full" ]; then
  for amb in cognee graphrag hipporag2; do
    [ -x ".venvs/$amb/bin/python" ] || morrer "full mode needs .venvs/$amb. Run ./scripts/bootstrap_envs.sh --full"
  done
  log "isolated environments: all three present"

  # As palavras-passe do Cognee. Verificadas AQUI e não na hora de as usar: a
  # indexação do Cognee leva mais de uma hora, e descobrir uma palavra-passe em
  # falta no fim disso é perder a hora.
  export COGNEE_POSTGRES_PASSWORD="${COGNEE_POSTGRES_PASSWORD:-benchmark}"
  export COGNEE_NEO4J_PASSWORD="${COGNEE_NEO4J_PASSWORD:-benchmark_cognee}"
  log "Cognee passwords: set"
fi

# --------------------------------------------------------------------------- #
titulo "2. Data"
# --------------------------------------------------------------------------- #

log "dataset: ${DATASET} ${VERSAO}   experiment: ${EXPERIMENTO}"

if [ ! -f "$FICHEIRO_BRUTO" ]; then
  log "raw file missing; downloading ($FAMILIA)"
  [ "$DRY_RUN" = "1" ] || "$PYTHON" scripts/fetch_datasets.py --dataset "$FAMILIA"
else
  log "$FAMILIA raw file present"
  [ "$DRY_RUN" = "1" ] || "$PYTHON" scripts/fetch_datasets.py --dataset "$FAMILIA" --verify-only >/dev/null
  log "sha256 matches"
fi

# Aviso de custo, antes da primeira chamada paga e não depois. As amostras de
# smoke custam cêntimos; os datasets completos são outra ordem de grandeza —
# 1000 perguntas, 4 indexações, 2000 chamadas de juiz.
case "$DATASET" in
  *_smoke_20) ;;
  *)
    aviso "COMPLETE dataset ($DATASET $VERSAO), not the smoke sample."
    aviso "That is ~1000 questions. In full mode, Cognee indexing alone"
    aviso "cost \$3.88 and ~38 h in the experimental phase. Ctrl-C now if this is not what you meant."
    [ "$DRY_RUN" = "1" ] || sleep 5
    ;;
esac

# --------------------------------------------------------------------------- #
titulo "3. Infrastructure"
# --------------------------------------------------------------------------- #

esperar_porta() {
  local porta="$1" nome="$2" limite="${3:-120}"
  for _ in $(seq 1 "$limite"); do
    if "$PYTHON" - "$porta" <<'FIM' 2>/dev/null
import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
FIM
    then log "$nome ready on port $porta"; return 0; fi
    sleep 2
  done
  # A saída do compose só interessa quando uma porta não responde — e é
  # precisamente aí que ela costumava estar perdida, mandada para /dev/null.
  if [ -n "${SAIDA_COMPOSE:-}" ]; then
    erro "what compose said:"
    printf '%s\n' "$SAIDA_COMPOSE" >&2
  fi
  morrer "$nome did not answer on port $porta after $((limite * 2))s"
}

# A PORTA ABERTA NÃO É O SERVIÇO PRONTO, e a diferença custou uma execução.
#
# O `esperar_porta` acima só confirma que alguém aceita um socket TCP — e no
# podman quem aceita é o encaminhamento de porta, que sobe **antes** do processo
# lá dentro. Num container quente isto nunca se nota; num container acabado de
# criar, o `db migrate` a seguir rebenta com «server closed the connection
# unexpectedly», que parece avaria do Postgres e é o Postgres ainda a arrancar.
#
# Apanhado a 2026-08-10, no teste de aceite sobre o zip, numa pasta limpa —
# exactamente o caminho que quem recebe o pacote percorre na primeira execução,
# e onde os containers são sempre novos.
#
# Estas duas funções falam mesmo com o serviço: uma ligação e uma consulta.
esperar_postgres() {
  local url="$1" nome="$2" limite="${3:-60}"
  for _ in $(seq 1 "$limite"); do
    if "$PYTHON" - "$url" <<'FIM' 2>/dev/null
import sys
import psycopg

with psycopg.connect(sys.argv[1], connect_timeout=3) as ligacao:
    with ligacao.cursor() as cursor:
        cursor.execute("SELECT 1")
FIM
    then log "$nome accepting queries"; return 0; fi
    sleep 2
  done
  morrer "$nome opened the port but never accepted a query ($((limite * 2))s)"
}

esperar_neo4j() {
  local uri="$1" utilizador="$2" palavra="$3" nome="$4" limite="${5:-60}"
  for _ in $(seq 1 "$limite"); do
    if "$PYTHON" - "$uri" "$utilizador" "$palavra" <<'FIM' 2>/dev/null
import sys
from neo4j import GraphDatabase

condutor = GraphDatabase.driver(sys.argv[1], auth=(sys.argv[2], sys.argv[3]))
try:
    condutor.verify_connectivity()
finally:
    condutor.close()
FIM
    then log "$nome accepting bolt connections"; return 0; fi
    sleep 2
  done
  morrer "$nome opened the port but never completed a bolt handshake ($((limite * 2))s)"
}

SERVICOS=(postgres neo4j_lightrag)
[ "$MODO" = "full" ] && SERVICOS=(postgres postgres_cognee neo4j_lightrag neo4j_graphrag)

if [ "$DRY_RUN" = "1" ]; then
  log "(dry-run) would start: ${SERVICOS[*]}"
else
  # O `podman compose up -d` NÃO é idempotente: com os containers já criados
  # devolve «container name is already in use» e sai a 125, mesmo que eles
  # estejam a correr e perfeitamente saudáveis. A consequência é que a SEGUNDA
  # execução deste script falhava sempre — e falhava com «o compose não subiu»,
  # porque a saída ia para /dev/null e a causa perdia-se.
  #
  # Apanhado a 2026-08-10 ao correr o smoke do 2Wiki numa máquina onde o smoke
  # do MuSiQue já tinha deixado os containers de pé. É o caminho normal de quem
  # corre os dois datasets, um a seguir ao outro.
  #
  # O que interessa não é o comando ter corrido: é a PORTA RESPONDER. Tenta-se
  # subir, guarda-se o que ele disse, arrancam-se os que já existiam, e o
  # veredicto fica para o `esperar_porta` — que mostra a saída do compose se
  # alguma porta faltar.
  if ! SAIDA_COMPOSE="$($COMPOSE up -d "${SERVICOS[@]}" 2>&1)"; then
    log "compose did not create the services; starting the existing ones"
    for servico in "${SERVICOS[@]}"; do
      for nome in "$(basename "$RAIZ")_${servico}_1" "$(basename "$RAIZ")-${servico}-1"; do
        $CONTENTOR start "$nome" >/dev/null 2>&1 && log "  $nome started" && break
      done
    done
  fi
  # Primeiro a porta, que é barato e apanha um serviço que nem sequer subiu.
  esperar_porta 15432 "benchmark Postgres"
  esperar_porta 18688 "LightRAG Neo4j"
  if [ "$MODO" = "full" ]; then
    esperar_porta 15433 "Cognee Postgres"
    esperar_porta 18687 "GraphRAG Neo4j"
  fi
  # Em podman rootless, o encaminhamento de porta do Neo4j só volta com um
  # stop seguido de start. Se a porta não respondeu acima, é isto.

  # E só depois o serviço, que é o que o `db migrate` a seguir precisa mesmo.
  esperar_postgres "$DATABASE_URL" "benchmark Postgres"
  if [ "$MODO" = "full" ]; then
    esperar_postgres \
      "postgresql://${COGNEE_POSTGRES_USER:-benchmark}:${COGNEE_POSTGRES_PASSWORD}@${COGNEE_POSTGRES_HOST:-127.0.0.1}:${COGNEE_POSTGRES_PORT:-15433}/postgres" \
      "Cognee Postgres"
  fi

  "$PYTHON" -m benchmark.cli.app db migrate
fi

# --------------------------------------------------------------------------- #
# As URI que o Neo4j vai MESMO usar, e não as que se supõe.
#
# O `set -a; . ./.env` acima exportou o que estiver no `.env`, e a expansão
# `${VAR:-omissão}` só se aplica quando a variável não existe. Logo, um `.env`
# com a porta errada GANHA silenciosamente ao valor do compose.
#
# Aconteceu a 2026-08-10, e custou uma execução: o `.env` desta máquina trazia
# `LIGHTRAG_NEO4J_URI=bolt://localhost:7688` na bagagem do ambiente original.
# A indexação do LightRAG arrancou, ligou-se ao vazio, e a única pista foi um
# traceback de quarenta linhas do driver do Neo4j a acabar em
# «ServiceUnavailable: Couldn't connect to localhost:7688». Nada nele dizia
# «o teu .env aponta para a porta errada».
#
# Por isso a resolução passou para aqui, ao lado das outras pré-condições, e
# diz-se em voz alta que porta vai ser usada.
# --------------------------------------------------------------------------- #

export LIGHTRAG_NEO4J_URI="${LIGHTRAG_NEO4J_URI:-bolt://localhost:18688}"
export LIGHTRAG_NEO4J_USER="${LIGHTRAG_NEO4J_USER:-neo4j}"
export LIGHTRAG_NEO4J_PASSWORD="${LIGHTRAG_NEO4J_PASSWORD:-benchmark_lightrag}"
export LIGHTRAG_NEO4J_DATABASE="${LIGHTRAG_NEO4J_DATABASE:-neo4j}"
export GRAPHRAG_NEO4J_URI="${GRAPHRAG_NEO4J_URI:-bolt://localhost:18687}"
export GRAPHRAG_NEO4J_USER="${GRAPHRAG_NEO4J_USER:-neo4j}"
export GRAPHRAG_NEO4J_PASSWORD="${GRAPHRAG_NEO4J_PASSWORD:-benchmark_graphrag}"
export GRAPHRAG_NEO4J_DATABASE="${GRAPHRAG_NEO4J_DATABASE:-neo4j}"

verificar_bolt() {
  local variavel="$1" uri="$2" esperada="$3"
  local porta="${uri##*:}"

  if [ "$porta" != "$esperada" ]; then
    aviso "$variavel=$uri — this repository's compose publishes $esperada."
    aviso "  It comes from your .env. If that was not deliberate, align it with .env.example."
  fi

  if [ "$DRY_RUN" = "1" ]; then
    log "$variavel=$uri (dry-run: not probed)"
    return 0
  fi

  if "$PYTHON" - "$porta" <<'FIM' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
FIM
  then
    log "$variavel=$uri (answers)"
  else
    erro "$variavel points at $uri, and nothing is listening on that port."
    erro "Fix it in .env — this repository's value is bolt://localhost:$esperada."
    morrer "without this, indexing runs, spends money, and dies in a Neo4j driver traceback"
  fi
}

titulo "3b. Neo4j — the port that will actually be used"
verificar_bolt LIGHTRAG_NEO4J_URI "$LIGHTRAG_NEO4J_URI" 18688
[ "$MODO" = "full" ] && verificar_bolt GRAPHRAG_NEO4J_URI "$GRAPHRAG_NEO4J_URI" 18687

# O aperto de mão bolt, e não só a porta. Um Neo4j acabado de criar leva
# dezenas de segundos a aceitar autenticação depois de a porta abrir, e a
# indexação do LightRAG cai lá dentro — depois de já ter gasto nos embeddings.
if [ "$DRY_RUN" != "1" ]; then
  esperar_neo4j "$LIGHTRAG_NEO4J_URI" "$LIGHTRAG_NEO4J_USER" "$LIGHTRAG_NEO4J_PASSWORD" "LightRAG Neo4j"
  if [ "$MODO" = "full" ]; then
    esperar_neo4j "$GRAPHRAG_NEO4J_URI" "$GRAPHRAG_NEO4J_USER" "$GRAPHRAG_NEO4J_PASSWORD" "GraphRAG Neo4j"
  fi
fi

# --------------------------------------------------------------------------- #
titulo "4. Ingestion"
# --------------------------------------------------------------------------- #

if [ "$DRY_RUN" = "1" ]; then
  log "(dry-run) would ingest and persist $DATASET $VERSAO"
else
  "$PYTHON" -m benchmark.cli.app ingest --dataset "$DATASET" --version "$VERSAO"
  "$PYTHON" -m benchmark.cli.app persist --dataset "$DATASET" --version "$VERSAO"
fi

# --------------------------------------------------------------------------- #
titulo "5. Indexing"
# A partir daqui gasta-se dinheiro a sério.
# --------------------------------------------------------------------------- #

# As URI do Neo4j foram resolvidas e sondadas na secção 3b, antes de aqui se
# chegar. NUNCA a banda 17xxx: essa é a do ambiente experimental original, e o
# travão em benchmark.infra.guard recusa-a.

# O slug é o que os construtores de workspace calculam a partir de
# (dataset_id, dataset_version). Os scripts de indexação recebem o destino por
# ARGUMENTO, mas quem lê o índice CALCULA-O — e se os dois não coincidirem ao
# carácter, a indexação corre bem, gasta dinheiro, e o leitor não encontra
# nada. Os sintomas não fazem alusão à causa: "shapes (0,) and (1536,) not
# aligned" no HippoRAG 2, "DatasetNotFoundError" no Cognee. Daqui em diante os
# caminhos vêm todos desta variável.
SLUG="$("$PYTHON" -c "
from benchmark.core.naming import slugify_dataset_version
print(slugify_dataset_version('$DATASET', '$VERSAO'))")"
CANONICO="data/canonical/${SLUG}"

if [ "$SALTAR_INDEXACAO" = "1" ]; then
  log "--skip-index: skipping"
elif [ "$DRY_RUN" = "1" ]; then
  log "(dry-run) would index: ${METODOS[*]}"
  [ "$MODO" = "full" ] && log "(dry-run) and, through their own scripts: cognee hipporag2"
else
  for metodo in "${METODOS[@]}"; do
    log "indexing $metodo"
    "$PYTHON" -m benchmark.cli.app index --dataset "$DATASET" --version "$VERSAO" --method "$metodo"
  done

  if [ "$MODO" = "full" ]; then
    # O HippoRAG 2 indexa no venv isolado. O --save-dir TEM de acabar em
    # /workspace: é o que hipporag2/config_builder.py calcula.
    log "indexing hipporag2 (.venvs/hipporag2)"
    if [ -e "artifacts/${SLUG}/hipporag2/workspace/passage_map.json" ]; then
      log "  index already exists; skipping (the runner would refuse, and rightly so)"
    else
      .venvs/hipporag2/bin/python scripts/hipporag2_index_runner.py \
        --save-dir "artifacts/${SLUG}/hipporag2/workspace" \
        --canonical-dir "$CANONICO"
    fi

    # O Cognee indexa no venv isolado e escreve no grafo do Option C. Duas
    # coisas que têm de bater certo: --workspace é o directório do método SEM
    # /workspace, e --dataset-name é o SLUG, que é o que cognee/option_c.py
    # procura na recuperação.
    log "indexing cognee (.venvs/cognee)"
    COGNEE_BOLT="bolt://localhost:$("$PYTHON" -c "
from benchmark.infra.neo4j_containers import Neo4jPortRegistry
print(Neo4jPortRegistry.load().ports_for('cognee', '$SLUG')[1])")"
    "$PYTHON" scripts/cognee_start_neo4j.py \
      --name "neo4j_cognee_${SLUG}" \
      --http-port "$("$PYTHON" -c "
from benchmark.infra.neo4j_containers import Neo4jPortRegistry
print(Neo4jPortRegistry.load().ports_for('cognee', '$SLUG')[0])")" \
      --bolt-port "${COGNEE_BOLT##*:}" \
      --password "${COGNEE_NEO4J_PASSWORD:-benchmark_cognee}"
    "$PYTHON" scripts/bootstrap_cognee_postgres.py \
      --create-database --create-schema --create-vector-extension --execute
    .venvs/cognee/bin/python scripts/cognee_index_musique.py \
      --documents "${CANONICO}/documents.jsonl" \
      --workspace "artifacts/${SLUG}/cognee" \
      --out-dir "artifacts/${SLUG}/cognee/out" \
      --dataset-name "$SLUG" \
      --bolt "$COGNEE_BOLT" \
      --password "${COGNEE_NEO4J_PASSWORD:-benchmark_cognee}"
  fi
fi

# --------------------------------------------------------------------------- #
titulo "6. Readers"
# --------------------------------------------------------------------------- #

ARGS_LIMITE=()
[ -n "$LIMITE" ] && ARGS_LIMITE=(--limit "$LIMITE")

LEITORES=("${METODOS[@]}")
# O cognee e o hipporag2 não estão na CLI de indexação, mas ESTÃO no despacho
# de recuperação do run_single — logo, correm pelo mesmo batch runner.
[ "$MODO" = "full" ] && LEITORES=("${METODOS[@]}" cognee hipporag2)

if [ "$DRY_RUN" = "1" ]; then
  log "(dry-run) would run the readers for: ${LEITORES[*]}"
else
  for metodo in "${LEITORES[@]}"; do
    log "reader: $metodo"
    "$PYTHON" scripts/run_experiment_batch.py \
      --method "$metodo" \
      --experiment-id "$EXPERIMENTO" \
      --dataset-id "$DATASET" \
      --dataset-version "$VERSAO" \
      "${ARGS_LIMITE[@]}" \
      || aviso "the $metodo reader failed; the others carry on"
  done
fi

# --------------------------------------------------------------------------- #
titulo "7. Judge"
# --------------------------------------------------------------------------- #

if [ "$DRY_RUN" = "1" ]; then
  log "(dry-run) would judge experiment $EXPERIMENTO"
else
  # Duas opções que não são cosméticas:
  #
  # --out-tag tem de ser distinto por dataset. Igual nos dois, os ficheiros de
  # sumário colapsam num só. Já aconteceu, a 2026-08-09.
  #
  # --out-dir porque o valor por omissão do juiz é `artifacts/musique/reports`,
  # fixo, e escreveria os resultados de qualquer dataset debaixo do nome
  # `musique`. Passa-se aqui em vez de se corrigir lá: o `run_llm_judge.py` é
  # dos ficheiros que o plano fecha à alteração, e mudar o chamador consegue o
  # mesmo sem tocar no que produziu os resultados da dissertação.
  "$PYTHON" scripts/run_llm_judge.py \
    --experiment-id "$EXPERIMENTO" \
    --full \
    --out-tag "${DATASET}_${VERSAO}" \
    --out-dir "artifacts/${DATASET}_${VERSAO}/reports" \
    || aviso "the judge failed; the report comes out without strict accuracy"
fi

# --------------------------------------------------------------------------- #
titulo "8. Report"
# --------------------------------------------------------------------------- #

if [ "$DRY_RUN" = "1" ]; then
  log "(dry-run) everything checked. Nothing was spent."
  exit 0
fi

"$PYTHON" -m benchmark.cli.app compare --experiment-id "$EXPERIMENTO" || true
"$PYTHON" -m benchmark.cli.app report --experiment-id "$EXPERIMENTO" || true

titulo "Done"
cat <<FIM
   Mode: $MODO   Dataset: ${DATASET}_${VERSAO}   Experiment: $EXPERIMENTO
   Reader: READER_GROUNDING=$READER_GROUNDING

   Low F1 and EM are NOT a fault. Values around F1 ~ 0.09 and EM ~ 0 are an
   artefact of verbosity and of measurement, diagnosed on 2026-06-23. The
   metric the thesis uses is the judge's strict accuracy.

   This reproduces the protocol, not the numbers.
FIM
