"""HippoRAG 2 query runner — roda DENTRO do venv isolado (.venvs/hipporag2).

Contrato: stdout carrega EXCLUSIVAMENTE JSON (1 documento por linha); todo o
ruído do hipporag (prints/logging/tqdm) vai para stderr via redirect_stdout.

Modos:
  --query "..."   one-shot: imprime um JSON e sai (smoke/testes). --mode qa
                  usa a geração nativa (rag_qa) em vez de só retrieval.
  --serve         servidor JSONL: lê {"query","top_k","mode"} por linha do
                  stdin, responde por linha; linha {"ready":true} após
                  carregar o índice. EOF encerra. mode default = "retrieve"
                  (retrocompatível com o adapter do braço A); mode="qa" roda
                  o rag_qa nativo e devolve {"answer","raw_response",...}.

Payload de item: {"rank","text","score","document_id"} — document_id vem do
passage_map.json (sha1(text) -> doc_*) escrito pelo index runner. No modo qa
os items são os top qa_top_k docs (os que o reader nativo efetivamente leu).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_passage_map(save_dir: Path) -> dict[str, str]:
    path = save_dir / "passage_map.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    log(f"WARN: passage_map.json ausente em {save_dir} — document_id ficará null")
    return {}


def build_rag(save_dir: str, llm_model: str, embedding_model: str):
    # hipporag imprime no stdout durante o load; desviar tudo para stderr.
    with contextlib.redirect_stdout(sys.stderr):
        from hipporag import HippoRAG

        rag = HippoRAG(
            save_dir=save_dir,
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
        )
    return rag


def run_query(rag, passage_map: dict[str, str], query: str, top_k: int) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        solutions = rag.retrieve(queries=[query], num_to_retrieve=top_k)
    sol = solutions[0]
    items = []
    for rank, (doc, score) in enumerate(zip(sol.docs, sol.doc_scores), start=1):
        items.append(
            {
                "rank": rank,
                "text": doc,
                "score": float(score),
                "document_id": passage_map.get(sha1(doc)),
            }
        )
    return {
        "query": query,
        "items": items,
        "stats": {"n_docs_returned": len(items)},
    }


def run_qa(rag, passage_map: dict[str, str], query: str) -> dict:
    """QA nativo do HippoRAG 2: retrieve interno (retrieval_top_k) + reader
    do pacote sobre os top qa_top_k docs. Devolve a short answer parseada."""
    with contextlib.redirect_stdout(sys.stderr):
        solutions, responses, metadata = rag.rag_qa(queries=[query])
    sol = solutions[0]
    qa_top_k = int(getattr(rag.global_config, "qa_top_k", 5))
    items = []
    docs = sol.docs or []
    scores = sol.doc_scores if sol.doc_scores is not None else []
    for rank, (doc, score) in enumerate(zip(docs[:qa_top_k], scores[:qa_top_k]), start=1):
        items.append(
            {
                "rank": rank,
                "text": doc,
                "score": float(score),
                "document_id": passage_map.get(sha1(doc)),
            }
        )
    stats = {"n_docs_returned": len(items), "qa_top_k": qa_top_k}
    meta = metadata[0] if metadata else {}
    if isinstance(meta, dict):
        for key in ("prompt_tokens", "completion_tokens", "num_input_tokens", "num_output_tokens"):
            if key in meta:
                stats[key] = meta[key]
    return {
        "query": query,
        "answer": sol.answer,
        "raw_response": responses[0] if responses else "",
        "items": items,
        "stats": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", default="retrieve", choices=["retrieve", "qa"])
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    rag = build_rag(args.save_dir, args.llm_model, args.embedding_model)
    passage_map = load_passage_map(save_dir)
    log(f"runner ready: save_dir={save_dir} passages_mapped={len(passage_map)}")

    if args.serve:
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                if request.get("mode", "retrieve") == "qa":
                    payload = run_qa(rag, passage_map, request["query"])
                else:
                    payload = run_query(
                        rag, passage_map, request["query"], int(request.get("top_k", 5))
                    )
            except Exception as exc:  # noqa: BLE001 — o servidor não pode morrer
                payload = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(payload), flush=True)
        return 0

    if args.query is None:
        parser.error("--query é obrigatório sem --serve")
    if args.mode == "qa":
        print(json.dumps(run_qa(rag, passage_map, args.query)), flush=True)
    else:
        print(json.dumps(run_query(rag, passage_map, args.query, args.top_k)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
