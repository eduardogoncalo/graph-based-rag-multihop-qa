"""O script que descarrega os ficheiros brutos.

Não há rede nestes testes, e não é preciso: o que interessa fixar é a parte que
decide — que um ficheiro que não confere é recusado em vez de aceite, que o
sha256 esperado não se ajusta ao que veio, e que o destino passa pelo travão do
ambiente original antes de se escrever fosse o que fosse.

O sha256 de cada ficheiro está no registo do script. Estes testes verificam o
registo, e não os valores: os valores foram verificados contra as impressões
digitais da dissertação a 2026-08-09, e é isso que os torna certos.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import fetch_datasets


def _escrever(caminho: Path, conteudo: bytes) -> tuple[str, int]:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(conteudo)
    return hashlib.sha256(conteudo).hexdigest(), len(conteudo)


# --------------------------------------------------------------------------- #
# O registo
# --------------------------------------------------------------------------- #


def test_registo_cobre_os_dois_datasets() -> None:
    assert set(fetch_datasets.FICHEIROS) == {"musique", "twowiki"}


def test_todo_o_ficheiro_tem_sha256_tamanho_e_razao() -> None:
    """Sem sha256 não há verificação; sem razão escrita ninguém sabe para que serve."""
    for dataset, entradas in fetch_datasets.FICHEIROS.items():
        for entrada in entradas:
            assert len(entrada["sha256"]) == 64, f"{dataset}: {entrada['destino']}"
            assert entrada["bytes"] > 0
            assert entrada["porque"].strip()
            assert entrada["manual"].strip()
            assert "urls" in entrada or "zip" in entrada


def test_musique_vem_de_mais_do_que_uma_origem() -> None:
    """O Drive oficial esgota quota. Uma origem só era um ponto único de falha."""
    (entrada,) = fetch_datasets.FICHEIROS["musique"]
    assert len(entrada["urls"]) >= 2
    anfitrioes = {url.split("/")[2] for url in entrada["urls"]}
    assert anfitrioes, "as origens têm de ser URLs completos"


def test_o_2wiki_vem_do_hipporag_e_nao_da_distribuicao_original() -> None:
    """Decisão de 2026-07-09: perguntas e corpus são os ficheiros do HippoRAG.

    Se alguém trocar isto pela distribuição original do 2Wiki, a amostra deixa
    de ser item-a-item comparável com o HippoRAG 2 — e a impressão digital
    apanha-o, mas depois de o download já ter corrido.
    """
    por_destino = {e["destino"]: e for e in fetch_datasets.FICHEIROS["twowiki"]}
    perguntas = por_destino["2wikimultihop/2wikimultihopqa.json"]
    corpus = por_destino["2wikimultihop/2wikimultihopqa_corpus.json"]
    for entrada in (perguntas, corpus):
        assert all("HippoRAG" in url for url in entrada["urls"])
    # dev.json e id_aliases.json vêm do lançamento original, dentro de um zip
    assert "zip" in por_destino["2wikimultihop/dev.json"]
    assert "zip" in por_destino["2wikimultihop/id_aliases.json"]


def test_os_destinos_batem_com_os_raw_path_dos_configs() -> None:
    """Descarregar para o sítio errado é o mesmo que não descarregar."""
    destinos = {
        e["destino"] for entradas in fetch_datasets.FICHEIROS.values() for e in entradas
    }
    assert "musique/musique_ans_v1.0_dev.jsonl" in destinos  # configs/datasets/musique.yaml
    assert any(d.startswith("2wikimultihop/") for d in destinos)  # twowiki.yaml


# --------------------------------------------------------------------------- #
# A verificação
# --------------------------------------------------------------------------- #


def test_ficheiro_certo_passa(tmp_path: Path) -> None:
    caminho = tmp_path / "bom.jsonl"
    sha256, tamanho = _escrever(caminho, b"conteudo exacto")
    fetch_datasets._verificar(caminho, sha256=sha256, esperado_bytes=tamanho)


def test_tamanho_errado_e_recusado_antes_do_hash(tmp_path: Path) -> None:
    """Um download truncado apanha-se pelo tamanho, que é mais barato."""
    caminho = tmp_path / "curto.jsonl"
    sha256, tamanho = _escrever(caminho, b"curto")
    with pytest.raises(fetch_datasets.DescarregamentoFalhouError, match="bytes, esperados"):
        fetch_datasets._verificar(caminho, sha256=sha256, esperado_bytes=tamanho + 1)


def test_sha256_errado_diz_para_nao_actualizar_o_sha256(tmp_path: Path) -> None:
    """A mensagem é o travão: a reacção errada a este erro é ajustar o esperado.

    Se a origem mudou de lançamento, mudou a amostra. Actualizar o sha256 para
    o download passar é apagar exactamente o aviso que se queria.
    """
    caminho = tmp_path / "outro.jsonl"
    _, tamanho = _escrever(caminho, b"outro lancamento")
    with pytest.raises(fetch_datasets.DescarregamentoFalhouError) as erro:
        fetch_datasets._verificar(caminho, sha256="0" * 64, esperado_bytes=tamanho)
    texto = str(erro.value)
    assert "NÃO actualize o sha256" in texto
    assert "deixam de ser" in texto


def test_ficheiro_existente_que_nao_confere_nao_e_substituido_em_silencio(
    tmp_path: Path,
) -> None:
    """Sem --force, um ficheiro estragado é um erro e não uma oportunidade."""
    destino = tmp_path / "musique" / "musique_ans_v1.0_dev.jsonl"
    _escrever(destino, b"lixo")
    entrada = {
        "destino": "musique/musique_ans_v1.0_dev.jsonl",
        "sha256": "1" * 64,
        "bytes": 4,
        "porque": "teste",
        "urls": ["https://exemplo.invalido/x"],
        "manual": "à mão para {destino}",
    }
    with pytest.raises(fetch_datasets.DescarregamentoFalhouError) as erro:
        fetch_datasets._tratar_ficheiro(
            entrada, raiz=tmp_path, cache=tmp_path / ".cache", verify_only=True, force=False
        )
    assert "--force" in str(erro.value)
    assert destino.read_bytes() == b"lixo", "não podia ter mexido no ficheiro"


def test_verify_only_nao_descarrega_nada(tmp_path: Path) -> None:
    entrada = {
        "destino": "musique/em_falta.jsonl",
        "sha256": "2" * 64,
        "bytes": 10,
        "porque": "teste",
        "urls": ["https://exemplo.invalido/x"],
        "manual": "à mão para {destino}",
    }
    with pytest.raises(fetch_datasets.DescarregamentoFalhouError, match="verify-only"):
        fetch_datasets._tratar_ficheiro(
            entrada, raiz=tmp_path, cache=tmp_path / ".cache", verify_only=True, force=False
        )
    assert not (tmp_path / "musique" / "em_falta.jsonl").exists()


def test_ficheiro_ja_bom_nao_e_descarregado_de_novo(tmp_path: Path) -> None:
    destino = tmp_path / "musique" / "bom.jsonl"
    sha256, tamanho = _escrever(destino, b"exactamente isto")
    entrada = {
        "destino": "musique/bom.jsonl",
        "sha256": sha256,
        "bytes": tamanho,
        "porque": "teste",
        "urls": ["https://exemplo.invalido/x"],
        "manual": "à mão para {destino}",
    }
    resultado = fetch_datasets._tratar_ficheiro(
        entrada, raiz=tmp_path, cache=tmp_path / ".cache", verify_only=False, force=False
    )
    assert resultado == "ok"


# --------------------------------------------------------------------------- #
# O travão
# --------------------------------------------------------------------------- #


def test_o_destino_passa_pelo_travao(monkeypatch, tmp_path: Path) -> None:
    """--data-dir apontado ao original despejava 350 MB lá dentro."""
    from benchmark.infra.guard import LigacaoAoAmbienteOriginalError

    original = tmp_path / "thesis"
    for marca in ("artifacts", "artifacts_variant", "status", "writing"):
        (original / marca).mkdir(parents=True)
    monkeypatch.setenv("BENCHMARK_AMBIENTE_ORIGINAL", str(original))
    monkeypatch.setattr("sys.argv", ["fetch_datasets.py", "--data-dir", str(original / "data")])

    with pytest.raises(LigacaoAoAmbienteOriginalError, match="--data-dir"):
        fetch_datasets.main()
