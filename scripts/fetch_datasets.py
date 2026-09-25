#!/usr/bin/env python
"""Descarrega os ficheiros brutos do MuSiQue e do 2WikiMultiHopQA.

Só a biblioteca padrão, de propósito: isto corre **antes** de o ambiente estar
montado. Quem clona o repositório precisa dos dados para correr fosse o que
fosse, e não faz sentido exigir que instale o pacote primeiro.

O QUE ISTO GARANTE, E O QUE NÃO GARANTE
---------------------------------------
Cada ficheiro tem um sha256 fixado aqui, verificado depois de descarregar. Um
espelho que mude, um download truncado ou uma página de erro em vez do ficheiro
são apanhados — não passam por bons.

O sha256 prova que o ficheiro é o mesmo. **Não prova que é o certo.** Essa prova
é outra, e está noutro sítio: `configs/datasets/fingerprints/` regista os
identificadores das 1000 perguntas de cada dataset, e `benchmark ingest`
recusa-se a escrever uma amostra que não bata certo. Os sha256 abaixo foram
escolhidos porque os ficheiros que descrevem **reproduzem essas impressões
exactamente** — verificado a 2026-08-09, 1000/11515 no MuSiQue e 1000/6119 no
2Wiki.

PORQUE É QUE O MUSIQUE VEM DE ESPELHOS
--------------------------------------
A distribuição oficial do MuSiQue é um zip no Google Drive
(https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view), e o
Drive recusa downloads automáticos quando a quota do ficheiro está esgotada —
que é o que estava a 2026-08-09. Não é uma origem em que se possa confiar num
script.

Os espelhos usados são três repositórios independentes do HuggingFace que têm o
ficheiro original, byte a byte: os três dão o mesmo sha256, e esse sha256
reproduz a amostra da dissertação. Se todos falharem, a mensagem de erro diz
como fazer à mão a partir do Drive.

PORQUE É QUE O 2WIKI VEM DE DOIS SÍTIOS
---------------------------------------
Perguntas e corpus vêm dos ficheiros publicados pelo HippoRAG
(OSU-NLP-Group/HippoRAG, reproduce/dataset/) e **não** da distribuição original
— decisão do autor a 2026-07-09, para comparabilidade item a item com o
HippoRAG 2. Da distribuição original só se usam `dev.json`, para recuperar
`supporting_facts` e `type` por `_id`, e `id_aliases.json`, para os aliases de
resposta.

Uso:
  python scripts/fetch_datasets.py                    # os dois datasets
  python scripts/fetch_datasets.py --dataset musique
  python scripts/fetch_datasets.py --verify-only      # não descarrega nada
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

TAMANHO_DO_BLOCO = 1 << 20
UA = "graph_based_rag-fetch_datasets/1.0"

# --------------------------------------------------------------------------- #
# Registo dos ficheiros. Cada entrada é um ficheiro que a ingestão precisa.
#
# `sha256` e `bytes` foram medidos a 2026-08-09 sobre os ficheiros que
# reproduzem as impressões digitais da dissertação. NÃO os actualizar por um
# download ter falhado a verificação: se o ficheiro mudou, mudou a amostra, e é
# isso que se quer saber.
# --------------------------------------------------------------------------- #

_HF = "https://huggingface.co/datasets/{repo}/resolve/main/{ficheiro}"
_HIPPORAG = (
    "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset/{ficheiro}"
)

# O zip do 2Wiki com evidences_id/answer_id e a segmentação de frases corrigida.
# 259 MB para dois ficheiros, e não há forma de extrair só parte de um zip
# remoto — por isso fica em cache e não se volta a descarregar.
_ZIP_2WIKI = {
    "nome": "data_ids_april7.zip",
    "url": "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",
    "sha256": "95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef",
    "bytes": 258968175,
}

FICHEIROS: dict[str, list[dict]] = {
    "musique": [
        {
            "destino": "musique/musique_ans_v1.0_dev.jsonl",
            "sha256": "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
            "bytes": 30439728,
            "porque": "as 1000 perguntas de avaliação saem daqui, por random.Random(42).sample",
            "urls": [
                _HF.format(repo="voidful/MuSiQue", ficheiro="musique_ans_v1.0_dev.jsonl"),
                _HF.format(repo="dgslibisey/MuSiQue", ficheiro="musique_ans_v1.0_dev.jsonl"),
                _HF.format(repo="bdsaglam/musique", ficheiro="musique_ans_v1.0_dev.jsonl"),
            ],
            "manual": (
                "Descarregue musique_v1.0.zip de "
                "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view "
                "e coloque o musique_ans_v1.0_dev.jsonl de dentro dele em {destino}"
            ),
        },
    ],
    "twowiki": [
        {
            "destino": "2wikimultihop/2wikimultihopqa.json",
            "sha256": "895cba294064df0c3302c76847b1fc08d99b5619f7663dfaa3b65cd780f1cac4",
            "bytes": 6505789,
            "porque": "as EXACTAS 1000 perguntas que o HippoRAG 2 avaliou",
            "urls": [_HIPPORAG.format(ficheiro="2wikimultihopqa.json")],
            "manual": (
                "Copie reproduce/dataset/2wikimultihopqa.json do repositório "
                "OSU-NLP-Group/HippoRAG para {destino}"
            ),
        },
        {
            "destino": "2wikimultihop/2wikimultihopqa_corpus.json",
            "sha256": "9d6e352952aafb18dab22bf8195039461321a44a949df902ae83bce134ad238a",
            "bytes": 3083943,
            "porque": "o corpus de 6.119 passagens; os Documents vêm daqui",
            "urls": [_HIPPORAG.format(ficheiro="2wikimultihopqa_corpus.json")],
            "manual": (
                "Copie reproduce/dataset/2wikimultihopqa_corpus.json do repositório "
                "OSU-NLP-Group/HippoRAG para {destino}"
            ),
        },
        {
            "destino": "2wikimultihop/dev.json",
            "sha256": "79f77ae104088ea8e25b1a65dbece768d45771194663bc5660ec9a98070dadf5",
            "bytes": 57614142,
            "porque": "recupera supporting_facts e type por _id, ausentes no ficheiro do HippoRAG",
            "zip": _ZIP_2WIKI,
            "membro": "dev.json",
            "manual": (
                "Descarregue {zip_url} e coloque o dev.json de dentro dele em {destino}"
            ),
        },
        {
            "destino": "2wikimultihop/id_aliases.json",
            "sha256": "f08ffcb6c2cefca9bdbe86b4248d6ad7a7743762d3f7264c14ff0bae85726fb6",
            "bytes": 17501406,
            "porque": "aliases de resposta por answer_id (Wikidata)",
            "zip": _ZIP_2WIKI,
            "membro": "id_aliases.json",
            "manual": (
                "Descarregue {zip_url} e coloque o id_aliases.json de dentro dele em {destino}"
            ),
        },
    ],
}


class DescarregamentoFalhouError(RuntimeError):
    """Nenhuma origem serviu, ou o que veio não é o que devia ser."""


def _travao():
    """Importa o travão do ambiente original.

    Import tardio e com `sys.path` próprio: este script tem de correr antes de
    o pacote `benchmark` estar instalado. O travão só depende da biblioteca
    padrão, portanto importa na mesma.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from benchmark.infra import guard

    return guard


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_do_ficheiro(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as ficheiro:
        while bloco := ficheiro.read(TAMANHO_DO_BLOCO):
            digest.update(bloco)
    return digest.hexdigest()


def _descarregar_para(url: str, destino: Path, *, esperado_bytes: int | None = None) -> None:
    """Descarrega para um temporário ao lado e só depois renomeia.

    Um ficheiro parcial com o nome final é pior do que nenhum: a execução
    seguinte encontra-o, e só a verificação de sha256 o distingue de um bom.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    pedido = urllib.request.Request(url, headers={"User-Agent": UA})
    temporario = destino.with_suffix(destino.suffix + ".parcial")
    try:
        with (
            urllib.request.urlopen(pedido, timeout=120) as resposta,
            temporario.open("wb") as saida,
        ):
            lidos = 0
            marco = 0
            while bloco := resposta.read(TAMANHO_DO_BLOCO):
                saida.write(bloco)
                lidos += len(bloco)
                if esperado_bytes and lidos - marco >= 20 * TAMANHO_DO_BLOCO:
                    marco = lidos
                    log(f"      {lidos / esperado_bytes:.0%} ({lidos // (1 << 20)} MB)")
        temporario.replace(destino)
    except BaseException:
        temporario.unlink(missing_ok=True)
        raise


def _verificar(caminho: Path, *, sha256: str, esperado_bytes: int) -> None:
    tamanho = caminho.stat().st_size
    if tamanho != esperado_bytes:
        raise DescarregamentoFalhouError(
            f"{caminho.name} tem {tamanho} bytes, esperados {esperado_bytes}"
        )
    obtido = sha256_do_ficheiro(caminho)
    if obtido != sha256:
        raise DescarregamentoFalhouError(
            f"{caminho.name} tem sha256 {obtido}, esperado {sha256}.\n"
            "    O ficheiro não é o que a dissertação usou. Se a origem mudou de "
            "lançamento, a amostra muda com ela e os números deixam de ser "
            "comparáveis — NÃO actualize o sha256 neste script para calar isto."
        )


def _obter_zip(zip_info: dict, cache: Path) -> Path:
    """Descarrega (uma vez) o zip do 2Wiki e devolve o caminho em cache."""
    arquivo = cache / zip_info["nome"]
    if arquivo.is_file():
        try:
            _verificar(arquivo, sha256=zip_info["sha256"], esperado_bytes=zip_info["bytes"])
        except DescarregamentoFalhouError:
            log(f"    arquivo em cache não confere, a descarregar de novo: {arquivo}")
            arquivo.unlink()
        else:
            log(f"    arquivo já em cache: {arquivo}")
            return arquivo

    log(f"    a descarregar {zip_info['nome']} ({zip_info['bytes'] // (1 << 20)} MB)")
    _descarregar_para(zip_info["url"], arquivo, esperado_bytes=zip_info["bytes"])
    _verificar(arquivo, sha256=zip_info["sha256"], esperado_bytes=zip_info["bytes"])
    return arquivo


def _extrair_do_zip(arquivo: Path, membro: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arquivo) as zf:
        nomes = zf.namelist()
        if membro not in nomes:
            candidatos = [n for n in nomes if n.endswith("/" + membro)]
            if not candidatos:
                raise DescarregamentoFalhouError(
                    f"{arquivo.name} não contém {membro}. Tem: {', '.join(nomes[:10])}"
                )
            membro = candidatos[0]
        with zf.open(membro) as origem, tempfile.NamedTemporaryFile(
            dir=destino.parent, delete=False
        ) as temporario:
            shutil.copyfileobj(origem, temporario)
            caminho_temporario = Path(temporario.name)
    caminho_temporario.replace(destino)


def _tratar_ficheiro(
    entrada: dict, *, raiz: Path, cache: Path, verify_only: bool, force: bool
) -> str:
    """Devolve 'ok', 'descarregado' ou levanta. `raiz` é o data/raw resolvido."""
    destino = raiz / entrada["destino"]
    sha256 = entrada["sha256"]
    esperado_bytes = entrada["bytes"]

    log(f"  {entrada['destino']}")
    log(f"    porquê: {entrada['porque']}")

    if destino.is_file() and not force:
        try:
            _verificar(destino, sha256=sha256, esperado_bytes=esperado_bytes)
        except DescarregamentoFalhouError as erro:
            raise DescarregamentoFalhouError(
                f"{destino} já existe e não confere.\n    {erro}\n"
                "    Apague-o, ou corra com --force para o substituir."
            ) from erro
        log("    já lá está e confere")
        return "ok"

    if verify_only:
        raise DescarregamentoFalhouError(
            f"{destino} não existe. Corra sem --verify-only para o descarregar."
        )

    erros: list[str] = []

    if "zip" in entrada:
        arquivo = _obter_zip(entrada["zip"], cache)
        log(f"    a extrair {entrada['membro']}")
        _extrair_do_zip(arquivo, entrada["membro"], destino)
    else:
        for url in entrada["urls"]:
            log(f"    a descarregar de {url.split('/')[2]}")
            try:
                _descarregar_para(url, destino, esperado_bytes=esperado_bytes)
                break
            except (urllib.error.URLError, OSError, TimeoutError) as erro:
                erros.append(f"{url}: {erro}")
                log(f"      falhou: {erro}")
        else:
            manual = entrada["manual"].format(destino=destino, zip_url="")
            raise DescarregamentoFalhouError(
                "Nenhuma origem serviu para "
                f"{entrada['destino']}:\n      " + "\n      ".join(erros) + "\n"
                f"    À mão: {manual}"
            )

    _verificar(destino, sha256=sha256, esperado_bytes=esperado_bytes)
    log("    descarregado e confere")
    return "descarregado"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Descarrega os ficheiros brutos do MuSiQue e do 2WikiMultiHopQA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Depois disto: `benchmark ingest --dataset musique --version "
            "ans_v1.0_eval1k`, que verifica a impressão digital da amostra."
        ),
    )
    ap.add_argument(
        "--dataset",
        choices=[*sorted(FICHEIROS), "all"],
        default="all",
        help="Qual descarregar. Por omissão, os dois.",
    )
    ap.add_argument(
        "--data-dir",
        default="data/raw",
        help="Onde escrever. Por omissão data/raw, que é o que os configs esperam.",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Só verifica o que já lá está; não descarrega nada.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Volta a descarregar mesmo que o ficheiro já esteja bom.",
    )
    args = ap.parse_args()

    # O destino é validado antes de se escrever o que quer que seja: apontar
    # isto ao ambiente original despejava 350 MB dentro dele.
    raiz = Path(_travao().exigir_caminho_permitido(args.data_dir, origem="--data-dir"))
    cache = raiz / ".cache"

    escolhidos = sorted(FICHEIROS) if args.dataset == "all" else [args.dataset]
    contagem = {"ok": 0, "descarregado": 0}
    falhas: list[str] = []

    for dataset in escolhidos:
        log(f"\n{dataset}")
        for entrada in FICHEIROS[dataset]:
            try:
                contagem[_tratar_ficheiro(
                    entrada,
                    raiz=raiz,
                    cache=cache,
                    verify_only=args.verify_only,
                    force=args.force,
                )] += 1
            except DescarregamentoFalhouError as erro:
                log(f"    ERRO: {erro}")
                falhas.append(entrada["destino"])

    total = contagem["ok"] + contagem["descarregado"]
    log(
        f"\n{total} ficheiros prontos "
        f"({contagem['descarregado']} descarregados, {contagem['ok']} já existiam)"
    )

    if falhas:
        log(f"{len(falhas)} por resolver: {', '.join(falhas)}")
        return 1

    arquivo_em_cache = cache / _ZIP_2WIKI["nome"]
    if arquivo_em_cache.is_file():
        log(f"\nPode apagar o arquivo em cache, já não é preciso: {arquivo_em_cache}")

    log(
        "\nA seguir, e é o que confirma que os ficheiros são os certos:\n"
        "  benchmark ingest --dataset musique --version ans_v1.0_eval1k\n"
        "  benchmark ingest --dataset twowiki --version ans_v1.0_eval1k"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
