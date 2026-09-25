#!/usr/bin/env bash
#
# Monta os ambientes Python do benchmark.
#
# O ambiente principal (.venv) chega para o modo `smoke`. Os três isolados
# (.venvs/*) só são precisos no modo `full`.
#
# A separação é por INDEXAÇÃO, não por método, e a distinção custou uma noite a
# perceber:
#
#   * a INDEXAÇÃO do Cognee corre no `.venvs/cognee`, que fixa `neo4j==5.28.4`
#     — o principal tem o 6.2.0;
#   * a INDEXAÇÃO e a consulta do HippoRAG 2 correm no `.venvs/hipporag2`, que
#     puxa torch e transformers e ocupa 7 GB;
#   * a CLI oficial do Microsoft GraphRAG vive no `.venvs/graphrag`;
#   * mas a RECUPERAÇÃO do Cognee corre no ambiente PRINCIPAL — o
#     `methods/cognee/adapter.py` faz `import cognee` directamente, e por isso
#     o `cognee` está declarado no `pyproject.toml`.
#
# Uso:
#   ./scripts/bootstrap_envs.sh            # o principal, e mais nada
#   ./scripts/bootstrap_envs.sh --full     # o principal e os três isolados
#   ./scripts/bootstrap_envs.sh --only hipporag2
#
# Precisa do `uv`. Não é preferência: os quatro ambientes foram criados com ele,
# cada um sobre um CPython que o próprio `uv` instala — o que dispensa quem
# recebe o pacote de ter um Python de sistema na versão certa, que é metade dos
# problemas de "clonei e não corre".

set -euo pipefail

cd "$(dirname "$0")/.."
RAIZ="$PWD"

FULL=0
APENAS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1 ;;
    --only) APENAS="${2:-}"; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "opção desconhecida: $1" >&2; exit 2 ;;
  esac
  shift
done

# --------------------------------------------------------------------------- #

log() { printf '%s\n' "$*"; }
erro() { printf 'ERRO: %s\n' "$*" >&2; }

if ! command -v uv >/dev/null 2>&1; then
  erro "o \`uv\` não está no PATH."
  cat >&2 <<'FIM'

    O uv é um binário único e instala-se sem privilégios:
        curl -LsSf https://astral.sh/uv/install.sh | sh

    Alternativa sem uv: os requirements/*.txt são ficheiros pip normais, e
    dá para montar cada ambiente com o python certo e `pip install -r`. As
    versões de Python estão em cada cabeçalho.
FIM
  exit 1
fi
log "uv: $(uv --version)"

# --------------------------------------------------------------------------- #
# O ambiente principal. Vem do uv.lock, que é resolução completa — e não de um
# requirements gerado, que não tem essa garantia.
# --------------------------------------------------------------------------- #

montar_principal() {
  log ""
  log "=== ambiente principal (.venv) ==="
  if [ -f uv.lock ]; then
    uv sync --frozen
    log "montado a partir do uv.lock (resolução completa, fixada)"
  else
    erro "uv.lock em falta; a montar a partir do pyproject.toml, sem garantia de versões"
    uv sync
  fi
}

# --------------------------------------------------------------------------- #
# Os três isolados. Versões completamente fixadas em requirements/, tiradas dos
# ambientes em que a fase experimental correu.
# --------------------------------------------------------------------------- #

python_de() {
  case "$1" in
    hipporag2|cognee) echo "3.11" ;;
    graphrag) echo "3.13" ;;
    *) erro "ambiente desconhecido: $1"; exit 2 ;;
  esac
}

montar_isolado() {
  local nome="$1"
  local py; py="$(python_de "$nome")"
  local req="requirements/${nome}.txt"
  local destino=".venvs/${nome}"

  log ""
  log "=== ${nome} (python ${py}) ==="
  if [ ! -f "$req" ]; then
    erro "$req em falta"
    return 1
  fi

  uv venv --python "$py" "$destino"
  uv pip install --python "${destino}/bin/python" -r "$req"

  # Verificação: importar o pacote que justifica o ambiente. Um venv que se
  # monta mas não importa não serve de nada, e é melhor sabê-lo agora do que a
  # meio de uma indexação que já gastou dinheiro.
  local modulo
  case "$nome" in
    hipporag2) modulo="hipporag" ;;
    cognee) modulo="cognee" ;;
    graphrag) modulo="graphrag" ;;
  esac
  if "${destino}/bin/python" -c "import ${modulo}" 2>/dev/null; then
    log "ok: ${modulo} importa em ${destino}"
  else
    erro "${destino} montou mas \`import ${modulo}\` falha"
    return 1
  fi
}

# --------------------------------------------------------------------------- #

if [ -n "$APENAS" ]; then
  montar_isolado "$APENAS"
else
  montar_principal
  if [ "$FULL" = "1" ]; then
    for nome in cognee graphrag hipporag2; do
      montar_isolado "$nome"
    done
  else
    log ""
    log "Só o ambiente principal, que é o que o modo \`smoke\` precisa."
    log "Para o modo \`full\`: ./scripts/bootstrap_envs.sh --full"
  fi
fi

log ""
log "Feito. A seguir:"
log "  cp .env.example .env    e preencher a OPENAI_API_KEY"
log "  ./scripts/reproduce.sh"
