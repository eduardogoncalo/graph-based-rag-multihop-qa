"""Concordância humano-juiz e κ de Cohen.

O plano listava este cálculo como a lacuna da validação: o `s10_results.json`
existe, mas foi produzido por cálculo pontual, e não havia código nenhum que o
reproduzisse.

Estes testes fazem três coisas distintas, e a distinção importa:

1. **Verificam a aritmética** contra valores calculáveis à mão. Um κ errado que
   devolva um número plausível é pior do que um que rebente.
2. **Reproduzem os valores registados no `s10_results.json` a partir dos
   rótulos brutos** — 1,0/1,0 nos blocos principais, 0,877 e 0,8855 nos
   suplementos — cruzando de novo as chaves do juiz com a anotação humana.
   Atenção: a tese só reporta κ no protocolo principal. No suplemento reporta
   concordância (28/30, 16/16 nas recusas) e diz que o κ não é calculado; os
   κ do suplemento são conferidos aqui só como consistência do registo.
3. **Verificam o segundo caminho**, o recálculo a partir das marginais que o
   `s10_results.json` regista, e que os dois caminhos concordam. A redundância
   existe para que um erro num deles não passe.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from scripts import compute_judge_agreement as agreement

VALIDACAO = Path("validation/s10_judge")
RESULTADOS = VALIDACAO / "s10_results.json"
CHAVES = VALIDACAO / "keys"


# --------------------------------------------------------------------------- #
# A aritmética, contra valores calculados à mão
# --------------------------------------------------------------------------- #


def test_kappa_de_um_caso_calculado_a_mao() -> None:
    """Duas classes, 50 itens: po = 0,70, pe = 0,50, κ = 0,40.

    Marginais: juiz 25/25, humano 30/20.
    pe = (25×30 + 25×20) / 50² = 1250/2500 = 0,5.
    """
    confusao = {
        ("sim", "sim"): 20,
        ("sim", "nao"): 5,
        ("nao", "sim"): 10,
        ("nao", "nao"): 15,
    }
    assert agreement.acordo_observado(confusao) == pytest.approx(0.70)
    assert agreement.kappa_de_cohen(confusao) == pytest.approx(0.40)


def test_kappa_e_zero_quando_o_acordo_e_o_do_acaso() -> None:
    """po = pe = 0,5, logo κ = 0. Metade certa não é meio bom — é acaso."""
    confusao = {
        ("sim", "sim"): 25,
        ("sim", "nao"): 25,
        ("nao", "sim"): 25,
        ("nao", "nao"): 25,
    }
    assert agreement.acordo_observado(confusao) == pytest.approx(0.5)
    assert agreement.kappa_de_cohen(confusao) == pytest.approx(0.0)


def test_acordo_perfeito_da_kappa_um() -> None:
    assert agreement.kappa_de_cohen({("a", "a"): 3, ("b", "b"): 7}) == pytest.approx(1.0)


def test_acordo_perfeito_numa_so_classe_da_um_e_nao_indefinido() -> None:
    """O caso degenerado, e é real: é o κ binário do suplemento do MuSiQue.

    Nenhum dos lados atribuiu `correct`, portanto na leitura binária os dois
    deram sempre "resto": pe = 1 e a fórmula dá 0/0. Concordância perfeita não
    pode virar «indefinido» só porque a amostra não tinha a classe positiva.
    """
    assert agreement.kappa_de_cohen({("resto", "resto"): 15}) == pytest.approx(1.0)


def test_kappa_de_matriz_vazia_nao_rebenta() -> None:
    assert agreement.kappa_de_cohen({}) == 0.0
    assert agreement.acordo_observado({}) == 0.0


def test_binarizacao_e_estrita_partial_nao_conta_como_acerto() -> None:
    """A dissertação mede acurácia estrita. `partial` fica do lado do resto."""
    confusao = {
        ("correct", "correct"): 4,
        ("partial", "partial"): 3,
        ("incorrect", "partial"): 2,
        ("correct", "partial"): 1,
    }
    binaria = agreement.binarizar(confusao)
    assert binaria[("True", "True")] == 4
    assert binaria[("False", "False")] == 5  # partial+partial e incorrect+partial
    assert binaria[("True", "False")] == 1
    # o acordo binário sobe face ao de 5 classes, porque incorrect/partial
    # deixam de ser divergência
    assert agreement.acordo_observado(binaria) > agreement.acordo_observado(confusao)


# --------------------------------------------------------------------------- #
# O cruzamento
# --------------------------------------------------------------------------- #


def test_a_celula_normaliza_espacos_a_volta_da_barra() -> None:
    """As chaves principais escrevem "cognee | controlado" e as do suplemento
    "cognee|controlado". Cruzar por string crua perdia todos os pares."""
    assert agreement.normalizar_celula("cognee | controlado") == "cognee|controlado"
    assert agreement.normalizar_celula("cognee|controlado") == "cognee|controlado"


def test_as_chaves_reais_usam_mesmo_as_duas_formas() -> None:
    """Não é hipotético — é assim que os ficheiros estão."""
    formas = set()
    for ficheiro in CHAVES.glob("judge_key_*.csv"):
        with ficheiro.open(newline="", encoding="utf-8") as f:
            for linha in csv.DictReader(f):
                formas.add(" | " in linha["cell"])
    assert formas == {True, False}, "esperava as duas formas nos ficheiros reais"


def test_avaliar_conta_divergencias_e_itens_sem_par() -> None:
    juiz = {("q1", "c"): "correct", ("q2", "c"): "incorrect", ("q3", "c"): "refusal"}
    humano = {("q1", "c"): "correct", ("q2", "c"): "partial", ("q4", "c"): "correct"}
    resultado = agreement.avaliar(juiz, humano)
    assert resultado["n"] == 2
    assert resultado["so_no_juiz"] == 1
    assert resultado["so_no_humano"] == 1
    assert resultado["divergencias"] == [
        {"question_id": "q2", "cell": "c", "judge": "incorrect", "humano": "partial"}
    ]


def test_itens_sem_par_sao_relatados_e_nao_descartados_em_silencio() -> None:
    """Um cruzamento que perde metade da amostra tem de se notar."""
    juiz = {("q1", "c"): "correct", ("q2", "c"): "correct"}
    humano = {("q1", "c"): "correct"}
    assert agreement.avaliar(juiz, humano)["so_no_juiz"] == 1


# --------------------------------------------------------------------------- #
# Os números publicados
# --------------------------------------------------------------------------- #


def test_reproduz_os_valores_publicados() -> None:
    """O requisito do plano: reproduzir o que está na dissertação.

    Possível sem os rótulos humanos porque o `s10_results.json` regista as
    marginais dos dois lados e a lista de divergências — e isso determina a
    matriz de confusão por inteiro.
    """
    relatorio = agreement.verificar_publicados(RESULTADOS)
    conferidos = [e for e in relatorio if e["estado"] == "confere"]
    assert {e["bloco"] for e in conferidos} == {
        "principal_musique",
        "principal_twowiki",
        "suplemento_musique",
        "suplemento_twowiki",
    }
    assert not [e for e in relatorio if e["estado"] == "DIVERGE"]


def test_os_kappa_dos_suplementos_batem_ao_milesimo() -> None:
    """Consistência do registo: 0,877 e 0,8855 estão no `s10_results.json`. Não
    são números da tese, que não calcula κ no suplemento (Tabela B.4)."""
    publicado = json.loads(RESULTADOS.read_text())["resultados"]
    for bloco, esperado in (("suplemento_musique", 0.877), ("suplemento_twowiki", 0.8855)):
        confusao = agreement.confusao_a_partir_do_registo(publicado[bloco])
        assert agreement.kappa_de_cohen(confusao) == pytest.approx(esperado, abs=5e-4)


def test_os_blocos_combinados_nao_sao_verificaveis_e_diz_se() -> None:
    """O `suplemento_combinado` não traz marginais — e o próprio ficheiro diz
    que ali se reporta acordo por estrato, NÃO κ. Não é falha: é uma decisão
    metodológica, e o script tem de a respeitar em vez de inventar um número."""
    relatorio = {e["bloco"]: e for e in agreement.verificar_publicados(RESULTADOS)}
    assert relatorio["suplemento_combinado"]["estado"] == "sem marginais, não verificável"
    assert relatorio["principal_combinado"]["estado"] == "sem marginais, não verificável"


def test_uma_divergencia_a_mais_no_registo_e_apanhada() -> None:
    """Se as marginais não fecharem, o recálculo recusa em vez de arredondar."""
    bloco = {
        "dist_judge": {"correct": 5, "incorrect": 5},
        "dist_humano": {"correct": 5, "incorrect": 5},
        "divergencias": [{"judge": "correct", "humano": "incorrect"}],
    }
    with pytest.raises(agreement.ValoresPublicadosDivergemError):
        agreement.confusao_a_partir_do_registo(bloco)


# --------------------------------------------------------------------------- #
# Os ficheiros de anotação humana
# --------------------------------------------------------------------------- #


def test_os_quatro_ficheiros_de_anotacao_estao_no_pacote() -> None:
    """Sem estes, metade dos dados falta e não há concordância a calcular."""
    for dataset in ("musique", "twowiki"):
        for sufixo in ("human_validation", "supplement"):
            caminho = VALIDACAO / dataset / f"s10_{dataset}_{sufixo}.xlsx"
            assert caminho.is_file(), f"falta {caminho}"


def test_le_xlsx_sem_dependencias_externas() -> None:
    """O `openpyxl` não está no ambiente principal, e este script tem de correr
    antes de o ambiente estar montado. Daí o leitor de xlsx em stdlib."""
    linhas = agreement.ler_xlsx(VALIDACAO / "musique" / "s10_musique_human_validation.xlsx")
    assert len(linhas) == 60
    assert {"question_id", "cell", "human_evaluation"} <= set(linhas[0])
    assert all(linha["human_evaluation"].strip() for linha in linhas), (
        "há linhas por rotular — a amostra estaria incompleta"
    )


def test_as_colunas_vazias_nao_desalinham_as_restantes() -> None:
    """No XML de um xlsx as células vazias não aparecem. Contar por ordem de
    aparecimento em vez de pela referência A1/B1 desalinhava tudo à direita da
    primeira célula vazia — e `notes`, que é quase toda vazia, vem antes do
    fim."""
    linhas = agreement.ler_xlsx(VALIDACAO / "musique" / "s10_musique_human_validation.xlsx")
    assert all(linha["question_id"].startswith("q_") for linha in linhas)
    assert all("|" in linha["cell"] for linha in linhas)


def test_emparelhamento_de_chave_para_rotulos_humanos() -> None:
    principal = agreement.emparelhar(CHAVES / "judge_key_musique.csv", VALIDACAO)
    suplemento = agreement.emparelhar(CHAVES / "judge_key_twowiki_supplement.csv", VALIDACAO)
    assert principal.name == "s10_musique_human_validation.xlsx"
    assert suplemento.name == "s10_twowiki_supplement.xlsx"


def test_emparelhamento_em_falta_diz_o_que_procurou(tmp_path: Path) -> None:
    (tmp_path / "keys").mkdir()
    chave = tmp_path / "keys" / "judge_key_inexistente.csv"
    chave.write_text("question_id,cell,judge_label\n", encoding="utf-8")
    with pytest.raises(agreement.RotulosHumanosEmFaltaError, match="Esperava"):
        agreement.emparelhar(chave, tmp_path)


def test_aceita_as_duas_grafias_da_coluna_humana(tmp_path: Path) -> None:
    """`human_evaluation` é o nome nos xlsx originais; `human_label` é o óbvio."""
    for coluna in ("human_label", "human_evaluation"):
        caminho = tmp_path / f"{coluna}.csv"
        caminho.write_text(
            f"question_id,cell,{coluna}\nq1,a | b,correct\n", encoding="utf-8"
        )
        assert agreement.ler_rotulos_humanos(caminho) == {("q1", "a|b"): "correct"}


def test_ficheiro_sem_coluna_humana_e_recusado(tmp_path: Path) -> None:
    caminho = tmp_path / "sem.csv"
    caminho.write_text("question_id,cell,notas\nq1,a|b,seja o que for\n", encoding="utf-8")
    with pytest.raises(agreement.RotulosHumanosEmFaltaError, match="human_evaluation"):
        agreement.ler_rotulos_humanos(caminho)


def test_ficheiro_em_falta_diz_onde_ele_devia_estar(tmp_path: Path) -> None:
    with pytest.raises(agreement.RotulosHumanosEmFaltaError) as erro:
        agreement.ler_rotulos_humanos(tmp_path / "nao_existe.xlsx")
    texto = str(erro.value)
    assert "s10_" in texto
    assert "--verify-published" in texto


# --------------------------------------------------------------------------- #
# A reprodução a sério: dos rótulos brutos aos números publicados
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("chave", "bloco_publicado"),
    [
        ("judge_key_musique.csv", "principal_musique"),
        ("judge_key_twowiki.csv", "principal_twowiki"),
        ("judge_key_musique_supplement.csv", "suplemento_musique"),
        ("judge_key_twowiki_supplement.csv", "suplemento_twowiki"),
    ],
)
def test_reproduz_o_publicado_a_partir_dos_rotulos_brutos(chave, bloco_publicado) -> None:
    """O requisito do plano, cumprido pelo caminho difícil.

    Não a partir das marginais registadas no s10_results.json — a partir dos
    rótulos do juiz e da anotação humana, cruzados de novo.
    """
    caminho = CHAVES / chave
    resultado = agreement.avaliar(
        agreement.ler_chave_do_juiz(caminho),
        agreement.ler_rotulos_humanos(agreement.emparelhar(caminho, VALIDACAO)),
        agreement.ler_estratos(caminho),
    )
    publicado = json.loads(RESULTADOS.read_text())["resultados"][bloco_publicado]

    assert resultado["n"] == publicado["n"]
    for campo in ("acordo", "kappa_5", "acordo_binario", "kappa_binario"):
        assert resultado[campo] == pytest.approx(publicado[campo], abs=5e-4), campo
    assert resultado["dist_judge"] == publicado["dist_judge"]
    assert resultado["dist_humano"] == publicado["dist_humano"]


def test_nenhum_item_fica_por_cruzar() -> None:
    """Um cruzamento que perdesse itens dava um κ sobre outra amostra."""
    for chave in sorted(CHAVES.glob("judge_key_*.csv")):
        resultado = agreement.avaliar(
            agreement.ler_chave_do_juiz(chave),
            agreement.ler_rotulos_humanos(agreement.emparelhar(chave, VALIDACAO)),
        )
        assert resultado["so_no_juiz"] == 0, chave.name
        assert resultado["so_no_humano"] == 0, chave.name


def test_o_acordo_por_estrato_reproduz_o_publicado() -> None:
    """O suplemento é desbalanceado por desenho, e o próprio s10_results.json
    diz que ali se reporta acordo por estrato, NÃO κ."""
    publicado = json.loads(RESULTADOS.read_text())["resultados"]
    for chave, bloco in (
        ("judge_key_musique_supplement.csv", "suplemento_musique"),
        ("judge_key_twowiki_supplement.csv", "suplemento_twowiki"),
    ):
        caminho = CHAVES / chave
        resultado = agreement.avaliar(
            agreement.ler_chave_do_juiz(caminho),
            agreement.ler_rotulos_humanos(agreement.emparelhar(caminho, VALIDACAO)),
            agreement.ler_estratos(caminho),
        )
        assert resultado["por_estrato"] == publicado[bloco]["por_estrato"]


def test_a_amostra_principal_nao_tem_estratos() -> None:
    """Prevalência natural, sem balanceamento — balancear distorceria o κ."""
    assert agreement.ler_estratos(CHAVES / "judge_key_musique.csv") == {}
    assert agreement.ler_estratos(CHAVES / "judge_key_musique_supplement.csv") != {}


def test_as_divergencias_encontradas_sao_as_registadas() -> None:
    """Uma a uma, e não só a contagem: a célula e os dois rótulos."""
    publicado = json.loads(RESULTADOS.read_text())["resultados"]
    for chave, bloco in (
        ("judge_key_musique_supplement.csv", "suplemento_musique"),
        ("judge_key_twowiki_supplement.csv", "suplemento_twowiki"),
    ):
        caminho = CHAVES / chave
        resultado = agreement.avaliar(
            agreement.ler_chave_do_juiz(caminho),
            agreement.ler_rotulos_humanos(agreement.emparelhar(caminho, VALIDACAO)),
        )
        registadas = publicado[bloco]["divergencias"]
        assert len(resultado["divergencias"]) == len(registadas)
        for encontrada, registada in zip(resultado["divergencias"], registadas, strict=True):
            assert encontrada["cell"] == registada["cell"]
            assert encontrada["judge"] == registada["judge"]
            assert encontrada["humano"] == registada["humano"]


def test_os_dois_caminhos_de_calculo_concordam() -> None:
    """O cálculo a partir dos rótulos brutos e o recálculo a partir das
    marginais publicadas têm de dar o mesmo. Se divergissem, um dos dois estava
    errado — e a redundância existe para isso."""
    publicado = json.loads(RESULTADOS.read_text())["resultados"]
    for chave, bloco in (
        ("judge_key_musique_supplement.csv", "suplemento_musique"),
        ("judge_key_twowiki_supplement.csv", "suplemento_twowiki"),
    ):
        caminho = CHAVES / chave
        dos_rotulos = agreement.avaliar(
            agreement.ler_chave_do_juiz(caminho),
            agreement.ler_rotulos_humanos(agreement.emparelhar(caminho, VALIDACAO)),
        )
        confusao = agreement.confusao_a_partir_do_registo(publicado[bloco])
        assert dos_rotulos["kappa_5"] == pytest.approx(
            agreement.kappa_de_cohen(confusao), abs=5e-4
        )


# --------------------------------------------------------------------------- #
# Privacidade
# --------------------------------------------------------------------------- #


def test_os_xlsx_nao_identificam_o_anotador() -> None:
    """A Fase 1 do plano diz que identificação de anotador não viaja.

    O `.xlsx` guarda autoria em docProps/core.xml, e é fácil ela ir sem
    ninguém reparar.
    """
    import zipfile

    for caminho in sorted(VALIDACAO.glob("*/*.xlsx")):
        with zipfile.ZipFile(caminho) as arquivo:
            if "docProps/core.xml" not in arquivo.namelist():
                continue
            core = arquivo.read("docProps/core.xml").decode("utf-8", "replace")
        for etiqueta in ("dc:creator", "cp:lastModifiedBy"):
            conteudo = re.search(rf"<{etiqueta}>(.*?)</{etiqueta}>", core)
            valor = (conteudo.group(1) if conteudo else "").strip()
            assert valor in ("", "openpyxl"), (
                f"{caminho.name} tem {etiqueta}={valor!r} — identificação de "
                "anotador não pode viajar no pacote"
            )


def test_o_resumo_diz_o_que_a_tese_reporta() -> None:
    """Tabela B.4: principal 120/120 e κ 1,000; suplemento 28/30 e 16/16 nas
    recusas nativas, sem κ."""
    publicado = json.loads(RESULTADOS.read_text())["resultados"]
    blocos = {
        nome: agreement.avaliar(
            *(
                {("q%d" % i, "c"): rotulo for i, rotulo in enumerate(lado)}
                for lado in _lados(publicado[bloco])
            ),
        )
        | ({"por_estrato": publicado[bloco]["por_estrato"]} if "por_estrato" in publicado[bloco] else {})
        for nome, bloco in (
            ("musique", "principal_musique"),
            ("twowiki", "principal_twowiki"),
            ("musique_supplement", "suplemento_musique"),
            ("twowiki_supplement", "suplemento_twowiki"),
        )
    }
    linhas = "\n".join(agreement.resumo_da_tese(blocos))
    assert "120/120 agreement (100.0%), Cohen's κ 1.000" in linhas
    assert "28/30 agreement (93.3%), 16/16 on native-arm refusals" in linhas
    assert "κ not computed" in linhas
    suplemento = next(l for l in linhas.splitlines() if "supplement" in l)
    assert "0.877" not in suplemento and "0.8855" not in suplemento


def _lados(bloco: dict) -> tuple[list[str], list[str]]:
    """Os rótulos do juiz e do humano, item a item, a partir da matriz registada."""
    confusao = agreement.confusao_a_partir_do_registo(bloco)
    juiz, humano = [], []
    for (rotulo_juiz, rotulo_humano), n in sorted(confusao.items()):
        juiz += [rotulo_juiz] * n
        humano += [rotulo_humano] * n
    return juiz, humano
