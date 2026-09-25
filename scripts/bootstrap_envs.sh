#!/usr/bin/env bash
#
# Builds the benchmark's Python environments.
#
# The main environment (.venv) is enough for `smoke` mode. The three isolated
# ones (.venvs/*) are needed for `full` mode and for --protocol thesis.
#
# The split follows INDEXING, not method:
#
#   * Cognee INDEXING runs in `.venvs/cognee`, which pins `neo4j==5.28.4`
#     (the main environment has 6.2.0);
#   * HippoRAG 2 INDEXING and querying run in `.venvs/hipporag2`, which pulls
#     torch and transformers and takes 7 GB;
#   * Microsoft GraphRAG's official CLI lives in `.venvs/graphrag`;
#   * but Cognee RETRIEVAL runs in the MAIN environment: methods/cognee/adapter.py
#     does `import cognee` directly, which is why `cognee` is in pyproject.toml.
#
# Usage:
#   ./scripts/bootstrap_envs.sh            # the main one, nothing else
#   ./scripts/bootstrap_envs.sh --full     # the main one and the three isolated
#   ./scripts/bootstrap_envs.sh --only hipporag2
#
# Needs `uv`: every environment is built on a CPython that uv installs itself,
# so no system Python at a particular version is required.

set -euo pipefail

cd "$(dirname "$0")/.."
RAIZ="$PWD"

FULL=0
APENAS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1 ;;
    --only) APENAS="${2:-}"; shift ;;
    -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# --------------------------------------------------------------------------- #

log() { printf '%s\n' "$*"; }
erro() { printf 'ERROR: %s\n' "$*" >&2; }

if ! command -v uv >/dev/null 2>&1; then
  erro "\`uv\` is not on PATH."
  cat >&2 <<'FIM'

    uv is a single binary and installs without privileges:
        curl -LsSf https://astral.sh/uv/install.sh | sh

    Without uv: requirements/*.txt are plain pip files; build each
    environment with the right Python and `pip install -r`. The Python
    version is in each file's header.
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
  log "=== main environment (.venv) ==="
  if [ -f uv.lock ]; then
    uv sync --frozen
    log "built from uv.lock (full, pinned resolution)"
  else
    erro "uv.lock missing; building from pyproject.toml, with no version guarantee"
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
    *) erro "unknown environment: $1"; exit 2 ;;
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
    erro "$req is missing"
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
    log "ok: ${modulo} imports in ${destino}"
  else
    erro "${destino} was built but \`import ${modulo}\` fails"
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
    log "Only the main environment, which is what \`smoke\` mode needs."
    log "For \`full\` mode and --protocol thesis: ./scripts/bootstrap_envs.sh --full"
  fi
fi

log ""
log "Done. Next:"
log "  cp .env.example .env    and fill in OPENAI_API_KEY"
log "  ./scripts/reproduce.sh"
