#!/usr/bin/env bash
#
# Constrói o zip de entrega, e recusa-se a construir um que vaze.
#
#   ./scripts/empacotar_entrega.sh                    # do HEAD
#   ./scripts/empacotar_entrega.sh --ref entrega-v1   # de um tag
#
# PORQUE ISTO EXISTE, E NÃO UM `zip -r`.
#
# O `.env` com as chaves reais de API está na raiz desta pasta, ao lado do
# `_prep/`, do `artifacts/` e de um cache de 247 MB. Um `zip -r` distraído
# embala tudo isso, e o erro só se vê depois de o ficheiro ter saído da máquina.
#
# O que este script faz, por essa ordem:
#
#   1. `git archive` — só o que está versionado, e respeitando o
#      `.gitattributes` (que tira os documentos de bastidores e o `docs/`);
#   2. acrescenta o `data/`, que é ignorado pelo git de propósito mas que o
#      pacote leva — MENOS o `data/raw/.cache/`, que é o zip de 247 MB do
#      Dropbox de onde o 2Wiki é extraído e que o `fetch_datasets.py` volta a
#      descarregar se precisar;
#   3. acrescenta o `results/`, se existir. São os resultados da experiência da
#      dissertação: dados, e nenhum código os lê;
#   4. **verifica o zip construído** e apaga-o se encontrar o que não devia lá
#      estar. É esta a parte que interessa: a lista de exclusões acima é uma
#      intenção, e a verificação é a prova.

set -euo pipefail

cd "$(dirname "$0")/.."
RAIZ="$PWD"

REF="HEAD"
SAIDA="graph_based_rag_entrega.zip"
PREFIXO="graph_based_rag"

while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF="${2:-}"; shift ;;
    --out) SAIDA="${2:-}"; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "opção desconhecida: $1" >&2; exit 2 ;;
  esac
  shift
done

log()    { printf '   %s\n' "$*"; }
titulo() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
morrer() { printf '\033[31mERRO:\033[0m %s\n' "$*" >&2; exit 1; }

git rev-parse --verify "$REF" >/dev/null 2>&1 || morrer "ref desconhecida: $REF"

# Um zip de um estado com alterações por commitar é um zip que não corresponde a
# nenhum commit — e portanto não se pode voltar a gerar igual.
if [ "$REF" = "HEAD" ] && ! git diff-index --quiet HEAD -- 2>/dev/null; then
  morrer "a árvore tem alterações por commitar. Commite, ou empacote de um tag com --ref."
fi

TRABALHO="$(mktemp -d)"
trap 'rm -rf "$TRABALHO"' EXIT

titulo "1. O código versionado"
git archive --format=tar --prefix="${PREFIXO}/" "$REF" | tar -x -C "$TRABALHO"
log "$(find "$TRABALHO" -type f | wc -l) ficheiros de $REF"

titulo "2. Os datasets"
if [ -d data ]; then
  # O `data/` é ignorado pelo git, portanto o `git archive` não o trouxe.
  # Copia-se com tar, que respeita `--exclude` em qualquer sítio onde este
  # script corra.
  mkdir -p "$TRABALHO/${PREFIXO}/data"
  tar -c --exclude='./raw/.cache' -C data . | tar -x -C "$TRABALHO/${PREFIXO}/data"
  log "data/ copiado, sem raw/.cache ($(du -sh "$TRABALHO/${PREFIXO}/data" | cut -f1))"
else
  log "data/ não existe; o pacote vai sem datasets"
fi

titulo "3. Os resultados"
if [ -d results ]; then
  cp -r results "$TRABALHO/${PREFIXO}/results"
  log "results/ copiado ($(du -sh "$TRABALHO/${PREFIXO}/results" | cut -f1))"
else
  log "results/ não existe ainda — o pacote vai sem ele"
fi

titulo "4. A construir"
rm -f "$RAIZ/$SAIDA"
( cd "$TRABALHO" && zip -q -r "$RAIZ/$SAIDA" "$PREFIXO" )
log "$SAIDA ($(du -h "$RAIZ/$SAIDA" | cut -f1))"

titulo "5. A verificar — é esta a parte que interessa"

CONTEUDO="$(unzip -Z1 "$RAIZ/$SAIDA")"
FALHOU=0

# NOTA SOBRE OS TUBOS, e custou um zip apagado por engano.
#
# `printf '%s\n' "$CONTEUDO" | grep -q ...` parece inofensivo e não é: o
# `grep -q` sai assim que encontra a primeira ocorrência e fecha o tubo, o
# `printf` morre com SIGPIPE (141), e o `set -o pipefail` no topo transforma
# isso na falha da pipeline INTEIRA. O `if` lê uma correspondência encontrada
# como ficheiro em falta.
#
# É latente: só dispara quando o conteúdo passa do tamanho do buffer do tubo,
# o que aconteceu quando o `results/` entrou no pacote e a listagem cresceu.
#
# A correcção é não haver tubo. `<<<` alimenta o grep por redirecção.
#
# A direcção do defeito era segura — recusava um pacote bom em vez de aceitar
# um mau — porque o `recusar` abaixo usa `grep` sem `-q` e lê tudo. Mas um
# verificador que falha ao acaso não é um verificador.
recusar() {
  local padrao="$1" motivo="$2"
  local encontrados
  encontrados="$(grep -E "$padrao" <<< "$CONTEUDO" || true)"
  if [ -n "$encontrados" ]; then
    printf '\033[31m   FUGA (%s):\033[0m\n' "$motivo"
    printf '%s\n' "$encontrados" | sed 's/^/     /'
    FALHOU=1
  else
    log "sem $motivo"
  fi
}

exigir() {
  local padrao="$1" o_que="$2"
  if grep -qE "$padrao" <<< "$CONTEUDO"; then
    log "traz $o_que"
  else
    printf '\033[31m   EM FALTA:\033[0m %s\n' "$o_que"
    FALHOU=1
  fi
}

# O que NUNCA pode lá estar.
recusar '/\.env$|/\.env\.[^e]|/[^/]*\.env$' "ficheiros .env (chaves reais de API)"
recusar '/_prep/'                            "_prep/ (retratos de ambiente)"
recusar '/artifacts/'                        "artifacts/ (resultados de execução)"
recusar '/\.venv'                            ".venv/ (ambientes virtuais)"
recusar '/data/raw/\.cache/'                 "o cache de 247 MB do Dropbox"
recusar '/docs/'                             "docs/ (specs internas)"
recusar '/\.git/'                            ".git/"


# A REGRA DOS .md, verificada e não assumida: só READMEs.
#
# O `.gitattributes` exclui `*.md` e reabre excepção aos READMEs, mas uma regra
# de exclusão é uma intenção — um `.gitattributes` mal editado, ou um ficheiro
# acrescentado por outro caminho, passa. Aqui olha-se para o zip construído.
INTRUSOS_MD="$(grep -E '\.md$' <<< "$CONTEUDO" | grep -vE '/README\.md$' || true)"
if [ -n "$INTRUSOS_MD" ]; then
  printf '\033[31m   FUGA (.md que não é README):\033[0m\n'
  printf '%s\n' "$INTRUSOS_MD" | sed 's/^/     /'
  FALHOU=1
else
  log "sem .md além dos READMEs"
fi

# O que TEM de lá estar. Um pacote sem isto não é entregável, e a ausência é
# tão silenciosa como uma fuga.
exigir '/\.env\.example$'         ".env.example (o ficheiro que se preenche)"
exigir '/README\.md$'             "o README"
exigir '/scripts/README\.md$'     "o README da arquitectura"
exigir '/scripts/reproduce\.sh$'  "o reproduce.sh"
exigir '/src/benchmark/'          "o código"
exigir '/tests/'                  "os testes"
exigir '/configs/datasets/fingerprints/' "as impressões digitais"

if [ "$FALHOU" = "1" ]; then
  rm -f "$RAIZ/$SAIDA"
  morrer "o zip foi APAGADO. Corrija o que está acima e volte a correr."
fi

titulo "Feito"
cat <<FIM
   $SAIDA   ($(du -h "$RAIZ/$SAIDA" | cut -f1))
   Estado: $REF ($(git rev-parse --short "$REF"))

   O zip e o commit têm de ser o mesmo estado. Se alterar alguma coisa,
   volte a correr isto — não edite o zip.
FIM
