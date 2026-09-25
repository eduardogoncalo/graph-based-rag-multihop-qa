"""A indexação do HippoRAG 2 tem de ser uma passagem única, e prová-lo.

Indexar por partes corrompe o grafo: cada chamada a `index()` volta a correr o
OpenIE e volta a acrescentar as arestas que dele saem. A deduplicação por hash
de conteúdo é do *embedding store*, e não cobre esse passo. Medido a
2026-08-09: 19.392.908 arestas para 319.817 pares distintos, contra ~1,58M
arestas de um grafo são.

Estes testes correm sobre uma **amostra pequena** e com um duplo do `hipporag`
no lugar da biblioteca — não indexam nada a sério, não precisam do
`.venvs/hipporag2`, não gastam API e não tocam nos 11.515 documentos do corpus
completo.

> A amostra é gerada aqui, com vinte documentos, porque o dataset de smoke
> ainda não existe — é o trabalho 4 da lista do `PLANO_RELEASE.md`. Quando ele
> existir, `_amostra_pequena` passa a lê-lo em vez de o simular.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from scripts import hipporag2_index_runner as runner

AMOSTRA = 20


class GrafoFalso:
    """A forma mínima que o runner lê de um grafo — o igraph expõe estas três."""

    def __init__(self, arestas: list[tuple[int, int]]) -> None:
        self._arestas = list(arestas)
        self._vertices = {vertice for par in self._arestas for vertice in par}

    def get_edgelist(self) -> list[tuple[int, int]]:
        return list(self._arestas)

    def vcount(self) -> int:
        return len(self._vertices)

    def ecount(self) -> int:
        return len(self._arestas)


class HippoRAGFalso:
    """Regista as chamadas a `index()`. Uma passagem única dá exactamente uma."""

    ultima: HippoRAGFalso | None = None
    arestas: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 3)]

    def __init__(self, *, save_dir: str, llm_model_name: str, embedding_model_name: str) -> None:
        self.save_dir = save_dir
        self.chamadas: list[list[str]] = []
        self.graph = GrafoFalso(type(self).arestas)
        HippoRAGFalso.ultima = self

    def index(self, docs: list[str]) -> None:
        self.chamadas.append(list(docs))


@pytest.fixture
def hipporag_falso(monkeypatch):
    modulo = types.ModuleType("hipporag")
    modulo.HippoRAG = HippoRAGFalso
    monkeypatch.setitem(sys.modules, "hipporag", modulo)
    HippoRAGFalso.ultima = None
    HippoRAGFalso.arestas = [(0, 1), (1, 2), (2, 3)]
    return HippoRAGFalso


def _amostra_pequena(tmp_path, n: int = AMOSTRA):
    """Os documentos associados às perguntas de uma amostra de smoke."""
    canonical = tmp_path / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    linhas = [
        json.dumps({"document_id": f"doc_{i:04d}", "text": f"Documento {i} da amostra."})
        for i in range(n)
    ]
    (canonical / "documents.jsonl").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return canonical


def _correr(monkeypatch, capsys, save_dir, canonical, *extra: str) -> dict:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hipporag2_index_runner.py",
            "--save-dir", str(save_dir),
            "--canonical-dir", str(canonical),
            *extra,
        ],
    )
    assert runner.main() == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# 1, 2 e 3 — workspace novo e vazio, corpus todo junto, uma só chamada
# --------------------------------------------------------------------------


def test_indexa_a_amostra_toda_numa_so_chamada(tmp_path, monkeypatch, capsys, hipporag_falso):
    canonical = _amostra_pequena(tmp_path)
    save_dir = tmp_path / "workspace_novo"

    sumario = _correr(monkeypatch, capsys, save_dir, canonical)

    rag = hipporag_falso.ultima
    assert len(rag.chamadas) == 1, "mais do que uma chamada a index() duplica arestas"
    assert rag.chamadas[0] == [f"Documento {i} da amostra." for i in range(AMOSTRA)]
    assert sumario["indexed_total"] == AMOSTRA
    assert sumario["chamadas_index"] == 1


def test_workspace_tem_de_estar_vazio_a_partida(tmp_path, monkeypatch, capsys, hipporag_falso):
    canonical = _amostra_pequena(tmp_path)
    save_dir = tmp_path / "workspace_novo"
    save_dir.mkdir()
    (save_dir / "sobras_de_um_indice_anterior.bin").write_text("x", encoding="utf-8")

    with pytest.raises(runner.WorkspaceJaUsadoError):
        _correr(monkeypatch, capsys, save_dir, canonical)


def test_janela_parcial_nao_pode_ser_continuada_no_mesmo_destino(
    tmp_path, monkeypatch, capsys, hipporag_falso
):
    """A segunda porta para a mesma corrupção: `--skip`/`--limit` em duas voltas.

    Indexar [0:10] e depois [10:20] no mesmo destino são duas chamadas a
    `index()`, exactamente como dois lotes eram.
    """
    canonical = _amostra_pequena(tmp_path)
    save_dir = tmp_path / "workspace_novo"

    _correr(monkeypatch, capsys, save_dir, canonical, "--limit", "10")

    with pytest.raises(runner.WorkspaceJaUsadoError):
        _correr(monkeypatch, capsys, save_dir, canonical, "--skip", "10", "--limit", "10")


# --------------------------------------------------------------------------
# 4 — recusa de workspace já preenchido, e a única saída
# --------------------------------------------------------------------------


def test_segunda_indexacao_no_mesmo_destino_e_recusada(
    tmp_path, monkeypatch, capsys, hipporag_falso
):
    canonical = _amostra_pequena(tmp_path)
    save_dir = tmp_path / "workspace_novo"

    _correr(monkeypatch, capsys, save_dir, canonical)
    assert (save_dir / "index_receipt.json").exists()

    with pytest.raises(runner.WorkspaceJaUsadoError) as erro:
        _correr(monkeypatch, capsys, save_dir, canonical)
    assert "--reset" in str(erro.value), "a mensagem tem de dizer qual é a saída"


def test_reset_recomeca_do_zero(tmp_path, monkeypatch, capsys, hipporag_falso):
    canonical = _amostra_pequena(tmp_path)
    save_dir = tmp_path / "workspace_novo"

    _correr(monkeypatch, capsys, save_dir, canonical)
    (save_dir / "restos.bin").write_text("x", encoding="utf-8")

    sumario = _correr(monkeypatch, capsys, save_dir, canonical, "--reset")

    assert sumario["indexed_total"] == AMOSTRA
    assert len(hipporag_falso.ultima.chamadas) == 1
    assert not (save_dir / "restos.bin").exists(), "--reset apaga o destino"


def test_batch_size_deixou_de_existir(tmp_path, monkeypatch, capsys, hipporag_falso):
    """A opção que produzia os lotes não volta por distracção."""
    canonical = _amostra_pequena(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hipporag2_index_runner.py",
            "--save-dir", str(tmp_path / "w"),
            "--canonical-dir", str(canonical),
            "--batch-size", "100",
        ],
    )
    with pytest.raises(SystemExit):
        runner.main()


# --------------------------------------------------------------------------
# 5 — o relatório do grafo
# --------------------------------------------------------------------------


def test_relatorio_do_grafo_no_sumario(tmp_path, monkeypatch, capsys, hipporag_falso):
    canonical = _amostra_pequena(tmp_path)
    save_dir = tmp_path / "workspace_novo"

    sumario = _correr(monkeypatch, capsys, save_dir, canonical)

    grafo = sumario["grafo"]
    assert grafo["vertices"] == 4
    assert grafo["arestas"] == 3
    assert grafo["pares_distintos"] == 3
    assert grafo["multiplicidade_maxima"] == 1
    # e fica no recibo, para quem for ver depois sem repetir a execução
    recibo = json.loads((save_dir / "index_receipt.json").read_text(encoding="utf-8"))
    assert recibo["grafo"] == grafo


def test_relatorio_torna_a_duplicacao_visivel() -> None:
    """A assinatura da corrupção de 2026-08-09, em ponto pequeno.

    Seis arestas para dois pares distintos: multiplicidade 3. É este o número
    que denuncia um grafo indexado por partes.
    """
    grafo = GrafoFalso([(0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (1, 2)])

    estatisticas = runner.estatisticas_do_grafo(grafo)

    assert estatisticas["arestas"] == 6
    assert estatisticas["pares_distintos"] == 2
    assert estatisticas["multiplicidade_maxima"] == 3


def test_par_nao_e_ordenado() -> None:
    """{u,v} e {v,u} são o mesmo par — senão a duplicação passava despercebida."""
    estatisticas = runner.estatisticas_do_grafo(GrafoFalso([(0, 1), (1, 0)]))

    assert estatisticas["pares_distintos"] == 1
    assert estatisticas["multiplicidade_maxima"] == 2


def test_relatorio_nao_inventa_numeros_quando_nao_ha_grafo() -> None:
    """Se a biblioteca não expuser o grafo, diz-se isso — não se estima."""
    estatisticas = runner.estatisticas_do_grafo(None)

    assert estatisticas["vertices"] is None
    assert estatisticas["pares_distintos"] is None
    assert "não expôs" in estatisticas["nota"]


def test_grafo_e_ignorado_se_nao_tiver_a_forma_esperada() -> None:
    class SemGrafo:
        graph = object()

    assert runner.grafo_do_rag(SemGrafo()) is None


# --------------------------------------------------------------------------
# O contrato com o src/, que lê a última linha do stdout
# --------------------------------------------------------------------------


def test_ultima_linha_do_stdout_continua_a_ser_o_sumario(
    tmp_path, monkeypatch, capsys, hipporag_falso
):
    """`benchmark.methods.hipporag2.indexer` faz splitlines()[-1] e json.loads."""
    canonical = _amostra_pequena(tmp_path)
    sumario = _correr(monkeypatch, capsys, tmp_path / "w", canonical)

    assert sumario["indexed_total"] == AMOSTRA
    assert "corpus_total" in sumario
