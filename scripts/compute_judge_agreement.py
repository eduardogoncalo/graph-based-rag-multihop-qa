#!/usr/bin/env python
"""Concordância humano-juiz e κ de Cohen, para a validação S10.

Só biblioteca padrão — incluindo a leitura dos `.xlsx`, que é um zip com XML
lá dentro. Isto é avaliação, e tem de correr sem depender de o ambiente estar
montado nem de o `openpyxl` estar instalado.

OS DOIS LADOS, E ONDE VIVEM
---------------------------
- **O juiz** está em `keys/judge_key_*.csv`: `question_id`, `cell`,
  `judge_label`. Imutáveis, como diz o README da pasta.
- **O humano** está em `<dataset>/s10_*.xlsx`, coluna `human_evaluation`.

Cruzam-se por `(question_id, cell)`. As chaves principais escrevem a célula com
espaços à volta da barra e as do suplemento sem — normaliza-se, senão o
cruzamento devolvia zero pares.

DOIS MODOS
----------
- **por omissão** — o cálculo a sério, sobre os rótulos dos dois lados, e
  compara o resultado com o `s10_results.json`.
- **`--verify-published`** — verifica só a aritmética dos números publicados,
  sem tocar nos rótulos humanos. É possível porque o `s10_results.json` regista
  as marginais dos dois lados e a lista de divergências, e isso **determina a
  matriz de confusão por inteiro**. Serve para quando os xlsx não estiverem
  disponíveis.

O segundo não substitui o primeiro: prova que os números publicados são
aritmeticamente consistentes com o que ficou registado, não que a anotação
humana foi feita como se diz.

O QUE A TESE REPORTA, E O QUE NÃO
---------------------------------
- **Protocolo principal** (120 itens, verificação com o rótulo do juiz à
  vista): concordância e κ.
- **Suplemento** (30 itens, cego ao rótulo do juiz, amostrado de propósito
  por estratos): concordância global e por estrato, **sem κ** (Tabela B.4). A
  amostra é desbalanceada por desenho, e o próprio `s10_results.json` diz
  «reportar acordo por estrato, NAO kappa». O κ por dataset continua a ser
  calculado, porque o `s10_results.json` o regista e a verificação confere-o,
  mas não é impresso: não é um número da tese.

CONVENÇÃO NUM CASO DEGENERADO
-----------------------------
Quando os dois anotadores concordam em tudo **e** usam uma só classe, o κ é
0/0. É o que acontece ao κ binário do suplemento do MuSiQue: nenhum dos lados
atribuiu `correct`, logo `pe = 1`. Aqui devolve-se **1.0**, que é a convenção
usada nos números publicados. Concordância perfeita não pode dar «indefinido»
só porque a amostra não tinha a classe positiva.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

CLASSE_POSITIVA = "correct"
CAMINHO_VALIDACAO = Path("validation/s10_judge")
CAMINHO_RESULTADOS = CAMINHO_VALIDACAO / "s10_results.json"
CAMINHO_CHAVES = CAMINHO_VALIDACAO / "keys"

# Convenção de emparelhamento: `judge_key_<dataset>[_supplement].csv` cruza com
# `<dataset>/s10_<dataset>_human_validation.xlsx` ou `<dataset>/s10_<dataset>_supplement.xlsx`.
_SUFIXO_PRINCIPAL = "human_validation"
_SUFIXO_SUPLEMENTO = "supplement"

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
COLUNAS_HUMANAS = ("human_evaluation", "human_label")


class RotulosHumanosEmFaltaError(RuntimeError):
    """O lado humano não está no pacote, e sem ele não há nada a calcular."""


class ValoresPublicadosDivergemError(RuntimeError):
    """O recálculo não bate com o que está publicado."""


# --------------------------------------------------------------------------- #
# A aritmética. Funções puras sobre a matriz de confusão, para se poderem
# verificar contra valores calculados à mão.
# --------------------------------------------------------------------------- #


def acordo_observado(confusao: dict[tuple[str, str], int]) -> float:
    """Fracção de itens em que os dois lados deram o mesmo rótulo."""
    total = sum(confusao.values())
    if not total:
        return 0.0
    return sum(n for (a, b), n in confusao.items() if a == b) / total


def kappa_de_cohen(confusao: dict[tuple[str, str], int]) -> float:
    """κ de Cohen sobre a matriz de confusão.

    ``κ = (po - pe) / (1 - pe)``, com ``pe`` a concordância esperada pelas
    marginais. Ver a nota do cabeçalho sobre o caso ``pe == 1``.
    """
    total = sum(confusao.values())
    if not total:
        return 0.0

    po = acordo_observado(confusao)
    if po == 1.0:
        # Concordância perfeita. Vale mesmo quando pe == 1 e a fórmula daria 0/0.
        return 1.0

    categorias = {c for par in confusao for c in par}
    por_linha = Counter()
    por_coluna = Counter()
    for (a, b), n in confusao.items():
        por_linha[a] += n
        por_coluna[b] += n

    pe = sum(por_linha[c] * por_coluna[c] for c in categorias) / (total * total)
    if pe == 1.0:
        # po < 1 e pe == 1 é impossível com marginais consistentes, mas se
        # acontecer é melhor dizer 0.0 do que rebentar com ZeroDivisionError.
        return 0.0
    return (po - pe) / (1 - pe)


def binarizar(
    confusao: dict[tuple[str, str], int], *, positiva: str = CLASSE_POSITIVA
) -> dict[tuple[str, str], int]:
    """Colapsa em ``correct`` contra tudo o resto.

    É a leitura estrita: para a dissertação, `partial` não conta como acerto.
    """
    binaria: Counter[tuple[str, str]] = Counter()
    for (a, b), n in confusao.items():
        binaria[(a == positiva, b == positiva)] += n
    return {(str(a), str(b)): n for (a, b), n in binaria.items()}


def matriz_de_confusao(pares: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """``[(rótulo_do_juiz, rótulo_humano), ...]`` -> matriz."""
    return dict(Counter(pares))


# --------------------------------------------------------------------------- #
# Leitura dos ficheiros
# --------------------------------------------------------------------------- #


def normalizar_celula(celula: str) -> str:
    """``"cognee | controlado"`` e ``"cognee|controlado"`` são a mesma célula.

    As chaves principais usam espaços à volta da barra e as do suplemento não.
    Cruzar por string crua perderia todos os pares.
    """
    return "|".join(parte.strip() for parte in str(celula).split("|"))


def _ler_csv(caminho: Path) -> list[dict[str, str]]:
    with caminho.open(newline="", encoding="utf-8") as ficheiro:
        return list(csv.DictReader(ficheiro))


def ler_chave_do_juiz(caminho: Path) -> dict[tuple[str, str], str]:
    """``(question_id, cell) -> judge_label``."""
    rotulos: dict[tuple[str, str], str] = {}
    for linha in _ler_csv(caminho):
        chave = (linha["question_id"], normalizar_celula(linha["cell"]))
        rotulos[chave] = linha["judge_label"].strip()
    return rotulos


def ler_estratos(caminho: Path) -> dict[tuple[str, str], str]:
    """``(question_id, cell) -> estrato``, quando a chave o traz.

    Só as chaves do suplemento têm coluna `estrato`. Devolve vazio para as
    outras — a amostra principal é de prevalência natural e não é estratificada.
    """
    linhas = _ler_csv(caminho)
    if not linhas or "estrato" not in linhas[0]:
        return {}
    return {
        (linha["question_id"], normalizar_celula(linha["cell"])): linha["estrato"].strip()
        for linha in linhas
    }


def _indice_da_coluna(referencia: str) -> int:
    """``"A1"`` -> 0, ``"AB7"`` -> 27. Necessário porque células vazias não
    aparecem no XML, e contar por ordem de aparecimento desalinhava tudo."""
    letras = re.match(r"([A-Z]+)", referencia).group(1)
    indice = 0
    for letra in letras:
        indice = indice * 26 + (ord(letra) - 64)
    return indice - 1


def ler_xlsx(caminho: Path) -> list[dict[str, str]]:
    """Lê a primeira folha de um `.xlsx` sem dependências externas.

    Um `.xlsx` é um zip: as strings estão numa tabela partilhada
    (`sharedStrings.xml`) e as células referenciam-na por índice. Ler isto à
    mão custa trinta linhas e poupa uma dependência a um script cuja razão de
    ser é correr sem ambiente montado.
    """
    with zipfile.ZipFile(caminho) as arquivo:
        nomes = arquivo.namelist()
        partilhadas: list[str] = []
        if "xl/sharedStrings.xml" in nomes:
            raiz = ElementTree.fromstring(arquivo.read("xl/sharedStrings.xml"))
            partilhadas = [
                "".join(texto.text or "" for texto in item.iter(f"{_NS}t"))
                for item in raiz.findall(f"{_NS}si")
            ]
        folhas = sorted(n for n in nomes if n.startswith("xl/worksheets/sheet"))
        if not folhas:
            raise RotulosHumanosEmFaltaError(f"{caminho} não tem folhas")
        raiz = ElementTree.fromstring(arquivo.read(folhas[0]))

    linhas: list[dict[int, str]] = []
    for linha_xml in raiz.iter(f"{_NS}row"):
        celulas: dict[int, str] = {}
        for celula in linha_xml.findall(f"{_NS}c"):
            valor_xml = celula.find(f"{_NS}v")
            tipo = celula.get("t")
            if tipo == "s" and valor_xml is not None:
                valor = partilhadas[int(valor_xml.text)]
            elif tipo == "inlineStr":
                valor = "".join(t.text or "" for t in celula.iter(f"{_NS}t"))
            else:
                valor = valor_xml.text if valor_xml is not None else None
            if valor is not None:
                celulas[_indice_da_coluna(celula.get("r"))] = valor
        linhas.append(celulas)

    if not linhas:
        return []
    cabecalho = linhas[0]
    return [
        {nome: linha.get(indice, "") for indice, nome in cabecalho.items()}
        for linha in linhas[1:]
    ]


def _rotulos_de(linhas: list[dict[str, str]], *, origem: Path) -> dict[tuple[str, str], str]:
    if not linhas:
        raise RotulosHumanosEmFaltaError(f"{origem} está vazio")
    colunas = set(linhas[0])
    for nome in COLUNAS_HUMANAS:
        if nome in colunas:
            coluna = nome
            break
    else:
        raise RotulosHumanosEmFaltaError(
            f"{origem} não tem coluna de rótulo humano. Esperava uma de "
            f"{list(COLUNAS_HUMANAS)}; tem {sorted(colunas)}"
        )

    rotulos: dict[tuple[str, str], str] = {}
    for linha in linhas:
        valor = (linha.get(coluna) or "").strip()
        if not valor:
            continue
        rotulos[(linha["question_id"], normalizar_celula(linha["cell"]))] = valor
    return rotulos


def ler_rotulos_humanos(caminho: Path) -> dict[tuple[str, str], str]:
    """``(question_id, cell) -> rótulo humano``, de um `.xlsx` ou de um `.csv`.

    O `.xlsx` é o formato original da anotação; o `.csv` aceita-se para quem
    preferir exportar. Em qualquer dos dois, a coluna do rótulo é
    `human_evaluation` (o nome nos ficheiros originais) ou `human_label`.
    """
    if not caminho.is_file():
        raise RotulosHumanosEmFaltaError(
            f"{caminho} não existe.\n"
            "Os rótulos humanos vivem nos ficheiros s10_*_human_validation.xlsx\n"
            "e s10_*_supplement.xlsx, dentro de validation/s10_judge/<dataset>/.\n"
            "Se não os tiver, --verify-published verifica os números publicados\n"
            "sem eles."
        )
    linhas = ler_xlsx(caminho) if caminho.suffix == ".xlsx" else _ler_csv(caminho)
    return _rotulos_de(linhas, origem=caminho)


def emparelhar(chave: Path, raiz_validacao: Path) -> Path:
    """Do ficheiro de chave do juiz para o ficheiro de rótulos humanos.

    ``judge_key_musique.csv``            -> ``musique/s10_musique_human_validation.xlsx``
    ``judge_key_musique_supplement.csv`` -> ``musique/s10_musique_supplement.xlsx``

    Emparelhar por convenção e não juntar tudo num saco só é deliberado: os
    dois protocolos são diferentes — o principal foi verificação-e-correcção
    com o rótulo do juiz pré-preenchido, o suplemento foi rotulagem cega — e
    misturá-los apagaria essa distinção sem ninguém dar por ela.
    """
    nome = chave.stem.replace("judge_key_", "")
    if nome.endswith("_" + _SUFIXO_SUPLEMENTO):
        dataset = nome[: -len("_" + _SUFIXO_SUPLEMENTO)]
        ficheiro = f"s10_{dataset}_{_SUFIXO_SUPLEMENTO}.xlsx"
    else:
        dataset = nome
        ficheiro = f"s10_{dataset}_{_SUFIXO_PRINCIPAL}.xlsx"

    candidato = raiz_validacao / dataset / ficheiro
    if candidato.is_file():
        return candidato
    alternativa = candidato.with_suffix(".csv")
    if alternativa.is_file():
        return alternativa
    raise RotulosHumanosEmFaltaError(
        f"não encontrei os rótulos humanos de {chave.name}.\n"
        f"Esperava {candidato} (ou o mesmo em .csv)."
    )


# --------------------------------------------------------------------------- #
# O cálculo
# --------------------------------------------------------------------------- #


def acordo_por_estrato(
    juiz: dict[tuple[str, str], str],
    humano: dict[tuple[str, str], str],
    estratos: dict[tuple[str, str], str],
) -> dict[str, dict[str, int]]:
    """``estrato -> {"acordo": quantos concordaram, "n": quantos há}``.

    O suplemento foi amostrado por estratos de propósito — sobreamostrando
    recusas e discordâncias entre passes — e por isso o próprio
    `s10_results.json` diz, para o combinado: «reportar acordo por estrato, NÃO
    kappa». Um κ sobre uma amostra desbalanceada de propósito não descreve nada.
    """
    resumo: dict[str, dict[str, int]] = {}
    for chave in sorted(set(juiz) & set(humano) & set(estratos)):
        entrada = resumo.setdefault(estratos[chave], {"acordo": 0, "n": 0})
        entrada["n"] += 1
        if juiz[chave] == humano[chave]:
            entrada["acordo"] += 1
    return resumo


def avaliar(
    juiz: dict[tuple[str, str], str],
    humano: dict[tuple[str, str], str],
    estratos: dict[tuple[str, str], str] | None = None,
) -> dict:
    """Concordância, κ e divergências entre os dois conjuntos de rótulos.

    Cruza por ``(question_id, cell)``, e só sobre a intersecção. Os itens que
    só existem de um lado são relatados em vez de descartados em silêncio: um
    cruzamento que perde metade da amostra tem de se notar.
    """
    comuns = sorted(set(juiz) & set(humano))
    pares = [(juiz[chave], humano[chave]) for chave in comuns]
    confusao = matriz_de_confusao(pares)
    binaria = binarizar(confusao)

    divergencias = [
        {
            "question_id": question_id,
            "cell": cell,
            "judge": juiz[(question_id, cell)],
            "humano": humano[(question_id, cell)],
        }
        for question_id, cell in comuns
        if juiz[(question_id, cell)] != humano[(question_id, cell)]
    ]

    bloco = {
        "n": len(comuns),
        "acordo": round(acordo_observado(confusao), 4),
        "kappa_5": round(kappa_de_cohen(confusao), 4),
        "acordo_binario": round(acordo_observado(binaria), 4),
        "kappa_binario": round(kappa_de_cohen(binaria), 4),
        "dist_judge": dict(Counter(juiz[chave] for chave in comuns)),
        "dist_humano": dict(Counter(humano[chave] for chave in comuns)),
        "divergencias": divergencias,
        "so_no_juiz": len(set(juiz) - set(humano)),
        "so_no_humano": len(set(humano) - set(juiz)),
    }
    if estratos:
        bloco["por_estrato"] = acordo_por_estrato(juiz, humano, estratos)
    return bloco


# --------------------------------------------------------------------------- #
# Verificação dos números publicados, sem os rótulos humanos
# --------------------------------------------------------------------------- #


def confusao_a_partir_do_registo(bloco: dict) -> dict[tuple[str, str], int]:
    """Reconstrói a matriz de confusão de um bloco do ``s10_results.json``.

    As marginais dos dois lados mais a lista de divergências determinam-na por
    inteiro: cada divergência é uma célula fora da diagonal, e o que sobra das
    marginais é a diagonal.

    Levanta ``KeyError`` se o bloco não tiver as marginais — é o caso do
    ``suplemento_combinado``, que o próprio ficheiro marca como «reportar
    acordo por estrato, NÃO kappa».
    """
    dist_judge = Counter(bloco["dist_judge"])
    dist_humano = Counter(bloco["dist_humano"])

    confusao: Counter[tuple[str, str]] = Counter()
    for divergencia in bloco.get("divergencias") or []:
        rotulo_juiz = divergencia["judge"]
        rotulo_humano = divergencia["humano"]
        confusao[(rotulo_juiz, rotulo_humano)] += 1
        dist_judge[rotulo_juiz] -= 1
        dist_humano[rotulo_humano] -= 1

    if dist_judge != dist_humano:
        raise ValoresPublicadosDivergemError(
            "as marginais não fecham depois de tirar as divergências: "
            f"juiz {dict(+dist_judge)} contra humano {dict(+dist_humano)}"
        )

    for rotulo, quantos in dist_judge.items():
        if quantos:
            confusao[(rotulo, rotulo)] += quantos
    return dict(confusao)


def verificar_publicados(caminho: Path) -> list[dict]:
    """Recalcula o κ de cada bloco e compara com o publicado."""
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    relatorio: list[dict] = []

    for nome, bloco in dados["resultados"].items():
        if "dist_judge" not in bloco:
            relatorio.append({"bloco": nome, "estado": "sem marginais, não verificável"})
            continue

        confusao = confusao_a_partir_do_registo(bloco)
        recalculado = {
            "n": sum(confusao.values()),
            "acordo": round(acordo_observado(confusao), 4),
            "kappa_5": round(kappa_de_cohen(confusao), 4),
            "acordo_binario": round(acordo_observado(binarizar(confusao)), 4),
            "kappa_binario": round(kappa_de_cohen(binarizar(confusao)), 4),
        }
        diferencas = {
            campo: {"publicado": bloco[campo], "recalculado": valor}
            for campo, valor in recalculado.items()
            if campo in bloco and abs(bloco[campo] - valor) > 5e-4
        }
        relatorio.append(
            {
                "bloco": nome,
                "estado": "confere" if not diferencas else "DIVERGE",
                "recalculado": recalculado,
                "diferencas": diferencas,
            }
        )
    return relatorio


# --------------------------------------------------------------------------- #


def resumo_da_tese(resultados: dict[str, dict]) -> list[str]:
    """The lines the thesis reports (Table B.4), from the recomputed blocks."""
    principal = [b for nome, b in resultados.items() if not nome.endswith("_" + _SUFIXO_SUPLEMENTO)]
    suplemento = [b for nome, b in resultados.items() if nome.endswith("_" + _SUFIXO_SUPLEMENTO)]
    linhas = ["", "As the thesis reports it (Table B.4):"]
    if principal:
        n = sum(b["n"] for b in principal)
        acordo = sum(round(b["acordo"] * b["n"]) for b in principal)
        kappas = sorted({b["kappa_5"] for b in principal})
        linhas.append(
            f"  primary protocol:      {acordo}/{n} agreement ({acordo / n:.1%}), "
            f"Cohen's κ {', '.join(f'{k:.3f}' for k in kappas)} "
            "(judge label visible: anchoring may inflate it)"
        )
    if suplemento:
        n = sum(b["n"] for b in suplemento)
        acordo = sum(round(b["acordo"] * b["n"]) for b in suplemento)
        recusas = [
            valores
            for b in suplemento
            for estrato, valores in (b.get("por_estrato") or {}).items()
            if estrato.startswith("refusal")
        ]
        texto_recusas = ""
        if recusas:
            texto_recusas = (
                f", {sum(v['acordo'] for v in recusas)}/{sum(v['n'] for v in recusas)} "
                "on native-arm refusals"
            )
        linhas.append(
            f"  supplement (blind):    {acordo}/{n} agreement ({acordo / n:.1%}){texto_recusas}; "
            "κ not computed (unbalanced by design)"
        )
    return linhas


def _travao():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from benchmark.infra import guard

    return guard


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concordância humano-juiz e κ de Cohen (validação S10).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--validation-dir",
        default=str(CAMINHO_VALIDACAO),
        help="Raiz da validação S10 (tem keys/ e as pastas por dataset).",
    )
    ap.add_argument("--out", default=None, help="Onde escrever o JSON do resultado.")
    ap.add_argument(
        "--verify-published",
        action="store_true",
        help=(
            "Recalcula o κ a partir do que o s10_results.json regista e compara "
            "com o publicado. Não precisa dos rótulos humanos."
        ),
    )
    args = ap.parse_args()

    raiz = Path(args.validation_dir)
    caminho_resultados = raiz / "s10_results.json"

    if args.verify_published:
        relatorio = verificar_publicados(caminho_resultados)
        for entrada in relatorio:
            print(f"{entrada['bloco']:26} {entrada['estado']}")
            for campo, valores in (entrada.get("diferencas") or {}).items():
                print(f"    {campo}: {valores}")
        divergem = [e for e in relatorio if e["estado"] == "DIVERGE"]
        if divergem:
            print(f"\n{len(divergem)} blocks do not match the published values.")
            return 1
        print("\nEvery verifiable block matches the published values.")
        return 0

    publicado = json.loads(caminho_resultados.read_text(encoding="utf-8"))["resultados"]
    resultados: dict[str, dict] = {}
    divergem: list[str] = []

    for chave in sorted((raiz / "keys").glob("judge_key_*.csv")):
        nome = chave.stem.replace("judge_key_", "")
        ficheiro_humano = emparelhar(chave, raiz)
        bloco = avaliar(
            ler_chave_do_juiz(chave),
            ler_rotulos_humanos(ficheiro_humano),
            ler_estratos(chave),
        )
        resultados[nome] = bloco

        if nome.endswith("_" + _SUFIXO_SUPLEMENTO):
            # The thesis reports no κ for the supplement (Table B.4).
            print(
                f"{nome:22} n={bloco['n']:4}  agreement={bloco['acordo']:.4f}  "
                f"(κ not reported: unbalanced by design)   <- {ficheiro_humano.name}"
            )
        else:
            print(
                f"{nome:22} n={bloco['n']:4}  agreement={bloco['acordo']:.4f}  "
                f"kappa_5={bloco['kappa_5']:.4f}  kappa_bin={bloco['kappa_binario']:.4f}"
                f"   <- {ficheiro_humano.name}"
            )
        if bloco["so_no_juiz"] or bloco["so_no_humano"]:
            print(
                f"{'':22} WARNING: {bloco['so_no_juiz']} only in the judge key, "
                f"{bloco['so_no_humano']} only in the human labels"
            )

        # Comparação com o publicado. O nome do bloco no s10_results.json é
        # `principal_<dataset>` ou `suplemento_<dataset>`.
        if nome.endswith("_" + _SUFIXO_SUPLEMENTO):
            chave_publicada = f"suplemento_{nome[: -len('_' + _SUFIXO_SUPLEMENTO)]}"
        else:
            chave_publicada = f"principal_{nome}"
        esperado = publicado.get(chave_publicada)
        if esperado is None:
            print(f"{'':22} (no published block named {chave_publicada})")
            continue
        diferencas = {
            campo: (esperado[campo], bloco[campo])
            for campo in ("n", "acordo", "kappa_5", "acordo_binario", "kappa_binario")
            if campo in esperado and abs(esperado[campo] - bloco[campo]) > 5e-4
        }
        if "por_estrato" in bloco and "por_estrato" in esperado:
            if bloco["por_estrato"] != esperado["por_estrato"]:
                diferencas["por_estrato"] = (esperado["por_estrato"], bloco["por_estrato"])
        if diferencas:
            divergem.append(chave_publicada)
            print(f"{'':22} DIFFERS from the published values ({chave_publicada}): {diferencas}")
        else:
            print(f"{'':22} matches the published values ({chave_publicada})")
            if "por_estrato" in bloco:
                for estrato, valores in sorted(bloco["por_estrato"].items()):
                    print(f"{'':22}   {estrato:24} {valores['acordo']}/{valores['n']}")

    if args.out:
        destino = Path(_travao().exigir_caminho_permitido(args.out, origem="--out"))
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps({"resultados": resultados}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nwritten: {destino}")

    for linha in resumo_da_tese(resultados):
        print(linha)

    if divergem:
        print(f"\n{len(divergem)} blocks do not reproduce the published values: {', '.join(divergem)}")
        return 1
    print("\nEvery block reproduces the published values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
