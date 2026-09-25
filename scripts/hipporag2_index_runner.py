"""HippoRAG 2 index runner — roda DENTRO do venv isolado (.venvs/hipporag2).

Lê documents.jsonl do canônico e chama `HippoRAG.index(docs=...)` **uma única
vez**, com a janela inteira de uma assentada, e mantém passage_map.json
(sha1(text) -> document_id) para a traceabilidade doc_* do benchmark.

**Porque é uma só chamada, e porque não há retoma.** Indexar por partes corrompe
o grafo: cada chamada volta a correr o OpenIE e volta a acrescentar as arestas
que dele saem. A deduplicação por hash de conteúdo é do *embedding store* — das
passagens e das frases — e não cobre o passo que deriva as arestas. Medido a
2026-08-09: 19.392.908 arestas para 319.817 pares distintos, contra ~1,58M
arestas de um grafo são. Por isso não há `--batch-size`, e por isso um workspace
já usado é recusado em vez de retomado.

`--skip` e `--limit` continuam a definir a janela — servem a amostra pequena —
mas cada janela exige um workspace próprio. Indexar `[0:500]` e depois
`[500:1000]` no mesmo destino são duas chamadas a `index()`, exactamente como
dois lotes eram.

stdout: a ÚLTIMA linha é o sumário JSON {"indexed_total": N, ...}, que inclui as
estatísticas do grafo. Ruído do hipporag vai para stderr.

Uso (dentro do venv):
  .venvs/hipporag2/bin/python scripts/hipporag2_index_runner.py \
      --save-dir artifacts/<slug>/hipporag2/workspace \
      --canonical-dir data/canonical/<slug> \
      [--limit N] [--skip N] [--reset]

**O `--save-dir` tem de acabar em `/workspace`.** É o que o
`hipporag2/config_builder.py` calcula e o que o adaptador vai procurar na
consulta. Indexar um nível acima produz um índice válido que o leitor não
encontra: o runner de consulta arranca com `passages_mapped=0` e a primeira
pergunta rebenta com `shapes (0,) and (1536,) not aligned`, que não faz
alusão nenhuma à causa. Aconteceu a 2026-08-09.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path


class WorkspaceJaUsadoError(RuntimeError):
    """O destino já tem um índice, e indexar por cima duplicaria arestas."""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _travao():
    """Importa o travão do ambiente original.

    Import tardio e com `sys.path` próprio de propósito: este script corre
    dentro de `.venvs/hipporag2`, que não tem o pacote `benchmark` instalado. O
    travão só depende da biblioteca padrão, portanto importa em qualquer venv.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from benchmark.infra import guard

    return guard


def exigir_save_dir_permitido(save_dir: str) -> Path:
    """Recusa um --save-dir apontado ao ambiente da dissertação.

    O HippoRAG 2 é o único método sem URI: guarda o índice em disco, não numa
    base de dados. A forma de lhe estragar o trabalho não é uma ligação, é o
    destino de escrita — o `artifacts_variant/` do original guarda os índices
    oficiais, que custaram uma passagem completa sobre o corpus.
    """
    return _travao().exigir_caminho_permitido(save_dir, origem="--save-dir")


def exigir_workspace_vazio(save_dir: Path, *, reset: bool = False) -> None:
    """Garante que se indexa para um destino novo, e não por cima de um índice.

    Verifica o conteúdo e não só o recibo: uma execução que morra a meio deixa
    um índice parcial sem recibo nenhum, e indexar por cima desse é a mesma
    duplicação por outra porta.

    `reset` apaga o destino, e é sempre explícito — nunca implícito.
    """
    if not save_dir.exists():
        return
    if reset:
        shutil.rmtree(save_dir)
        return
    conteudo = sorted(caminho.name for caminho in save_dir.iterdir())
    if not conteudo:
        return
    raise WorkspaceJaUsadoError(
        f"--save-dir {save_dir} already has content ({', '.join(conteudo[:5])}"
        f"{'...' if len(conteudo) > 5 else ''}) and this runner does not resume: "
        f"calling index() on an existing index derives the edges again and "
        f"duplicates them (19,392,908 edges for 319,817 pairs, measured on "
        f"2026-08-09). Use --reset to start from scratch, or point --save-dir "
        f"at a new destination."
    )


def grafo_do_rag(rag: object) -> object | None:
    """O grafo do HippoRAG, se ele o expuser de forma reconhecível."""
    for nome in ("graph", "g", "igraph"):
        candidato = getattr(rag, nome, None)
        if candidato is not None and callable(getattr(candidato, "get_edgelist", None)):
            return candidato
    return None


def estatisticas_do_grafo(grafo: object | None) -> dict[str, object]:
    """Vértices, arestas, pares distintos e multiplicidade máxima.

    É este o instrumento que torna a duplicação visível a olho: o grafo
    corrompido tinha 19.392.908 arestas para 319.817 pares distintos — uma
    multiplicidade média de ~60.

    **Relata, não trava.** O HippoRAG liga o mesmo par por mais do que um
    motivo (relation, synonym, context), portanto uma multiplicidade de 2 ou 3
    pode ser legítima; 60 não é. Quem lê o número é que decide.

    Um par é não-ordenado: {u,v} e {v,u} são o mesmo par.
    """
    if grafo is None:
        return {
            "vertices": None,
            "arestas": None,
            "pares_distintos": None,
            "multiplicidade_maxima": None,
            "nota": "o objecto HippoRAG não expôs um grafo com get_edgelist()",
        }

    arestas = [tuple(sorted(par)) for par in grafo.get_edgelist()]
    contagem = Counter(arestas)
    vcount = getattr(grafo, "vcount", None)
    ecount = getattr(grafo, "ecount", None)
    return {
        "vertices": vcount() if callable(vcount) else None,
        "arestas": ecount() if callable(ecount) else len(arestas),
        "pares_distintos": len(contagem),
        "multiplicidade_maxima": max(contagem.values()) if contagem else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--passage-map", default=None)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete --save-dir before indexing. Explicit and destructive.",
    )
    args = parser.parse_args()

    save_dir = exigir_save_dir_permitido(args.save_dir)
    exigir_workspace_vazio(save_dir, reset=args.reset)
    save_dir.mkdir(parents=True, exist_ok=True)
    map_path = (
        Path(_travao().exigir_caminho_permitido(args.passage_map, origem="--passage-map"))
        if args.passage_map
        else save_dir / "passage_map.json"
    )

    documents_path = Path(args.canonical_dir) / "documents.jsonl"
    docs: list[tuple[str, str]] = []  # (document_id, text)
    with documents_path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            docs.append((record["document_id"], record["text"]))
    window = docs[args.skip : (args.skip + args.limit) if args.limit else None]
    log(f"corpus={len(docs)} window=[{args.skip}:{args.skip + len(window)}]")

    passage_map: dict[str, str] = {}
    if map_path.exists():
        passage_map = json.loads(map_path.read_text(encoding="utf-8"))
    collisions = 0
    for document_id, text in window:
        key = sha1(text)
        if key in passage_map and passage_map[key] != document_id:
            collisions += 1  # textos idênticos com ids distintos: mantém o primeiro
        passage_map.setdefault(key, document_id)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(passage_map), encoding="utf-8")

    with contextlib.redirect_stdout(sys.stderr):
        from hipporag import HippoRAG

        rag = HippoRAG(
            save_dir=str(save_dir),
            llm_model_name=args.llm_model,
            embedding_model_name=args.embedding_model,
        )

    started = time.time()
    # UMA chamada, com a janela inteira. Ver o cabeçalho para o porquê.
    with contextlib.redirect_stdout(sys.stderr):
        rag.index(docs=[text for _, text in window])
    elapsed = round(time.time() - started, 1)

    estatisticas = estatisticas_do_grafo(grafo_do_rag(rag))
    janela_sha256 = hashlib.sha256(
        "\n".join(document_id for document_id, _ in window).encode("utf-8")
    ).hexdigest()

    recibo = {
        "janela_sha256": janela_sha256,
        "indexed_total": len(window),
        "corpus_total": len(docs),
        "skip": args.skip,
        "limit": args.limit,
        "grafo": estatisticas,
    }
    (save_dir / "index_receipt.json").write_text(
        json.dumps(recibo, sort_keys=True), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "indexed_total": len(window),
                "corpus_total": len(docs),
                "passage_map_entries": len(passage_map),
                "text_collisions": collisions,
                "chamadas_index": 1,
                "janela_sha256": janela_sha256,
                "grafo": estatisticas,
                "elapsed_s": elapsed,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
