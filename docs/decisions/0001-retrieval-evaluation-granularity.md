# ADR 0001 — Granularidade da avaliação de retrieval (document-level) e correção do bug de namespace

- **Status:** Aceito e implementado
- **Data:** 2026-06-22
- **Contexto do experimento:** `musique_eval1k_first_results` (MuSiQue eval1k, 1000 perguntas)
- **Artefatos relacionados:** `artifacts/musique/reports/retrieval_metric_fix_strategy.md` (estratégia + fundamentação bibliográfica completa, com páginas); `artifacts/musique/reports/first_results_crash_report.md` (quebra por OOM)

## Contexto

A métrica `evidence_recall@5` registrou **0.0 em todos os 1000 itens** do `vector_rag`. Investigação (leitura de código + dry-run em dados reais) mostrou que **não é falha de recuperação, e sim um bug de namespace de IDs na avaliação**:

- **Gold:** `evidence_id = deterministic_id("ev", [question_id, document_id])` → `ev_<hash>` (`musique_loader.py:246`). É o conjunto que `load_evaluation_input` passava como `gold_evidence_ids`.
- **Recuperado:** `source_document_id = doc_<hash>` canônico, `source_chunk_id`, `item_id` (`pgvector_store.py:127-132`).

`evidence_recall_at_k` cruza os dois conjuntos (`retrieval_metrics.py`). Como `ev_*` nunca coincide com `doc_*`/`chunk_*`, a interseção é **vazia por construção** → recall ≡ 0. O mesmo afeta `precision@5`; e `citation_accuracy` está quebrado por uma terceira razão (citações são chunk-level, gold é ev-level).

## Decisão

Adotada a **Option B** (de A/B/C/D avaliadas na estratégia):

1. Introduzir um conjunto gold **document-level** explícito, `gold_document_ids` (ids canônicos `doc_*`), usado por `evidence_recall@k` e `precision@k`.
2. **Preservar** `gold_evidence_ids` (`ev_*`, evidence-level) reservado para citação/atribuição.
3. **Manter a chave** `evidence_recall_at_5` (sem renomear), pois (a) `compare.py`/`report.py`/`app.py` a fixam e (b) na literatura multi-hop "evidência de suporte" é medida em nível de passagem — e, no nosso schema, **uma passagem de suporte do MuSiQue é um documento canônico**.

Definição final adotada:

> `evidence_recall_at_5` = fração das **passagens/documentos de suporte** (gold `doc_*`) recuperados no top-5. Não é recall de `ev_*`.

Descartadas: **A** (sobrecarrega um único campo, confla níveis evidence/document), **C** (aceitar múltiplos namespaces mascara bugs), **D** (reconciliação por método — adiada; necessária para grafos, ver abaixo).

## Fundamentação (peer-reviewed)

- **Medir retrieval recall/precision separado da resposta (EM/F1):** Lewis et al., *RAG for Knowledge-Intensive NLP Tasks*, **NeurIPS 2020** (ablações de retriever, §4.5 p.7); Kim & Lee, *Where Does Legal AI Fail?*, **CIKM '25** (ACM, "we separately evaluate retriever and reranker", p.1397, Recall@k como métrica primária p.1398); Houimli et al., **IEEE Big Data 2025** ("we evaluate retrieval separately from generation", p.3).
- **Recall@k = fração de evidência gold no top-k:** Houimli et al., IEEE Big Data 2025 (definição, p.2); GraphRAG-SF, **IEEE Big Data 2025** (fórmulas Precision@K/Recall@K, Eqs.8–11 p.2566).
- **Match por ID exato contra gold (pressupõe namespace único):** Lewis et al., NeurIPS 2020 (overlap de títulos de artigo vs. gold evidence do FEVER, p.7); Kim & Lee, CIKM '25 (match por ID de statute; LLM-judge só na geração, p.1398). **É exatamente o pré-requisito violado pelo bug e restaurado pela Option B.**
- **Cobertura de retrieval ≠ citação/atribuição (justifica B sobre A):** Imtiyaz et al., *Assessing RAG*, **ICCIKE 2025** (IEEE; CoFE-RAG lista "Citation Accuracy" como métrica distinta, Fig.1 p.2); ARES (juízes independentes por componente — publicado em **NAACL 2024**; o PDF em `references/` é o preprint, citar a versão NAACL).
- **Granularidade deve ser explicitada:** Imtiyaz et al., ICCIKE 2025 ("the granularity of metrics is related to the computational cost and proximity to domain-specific needs", p.1).
- **Contra-exemplo (o que evitar):** *Medical RAG: Hybrid Retrieval & Re-Ranking*, **IEEE ResGenXAI 2025** — reporta recall/precision **sem definir o gold-match nem o k** (p.4); exatamente a métrica não-verificável que esta decisão evita.

## Implementação (feita em 2026-06-22)

- `src/benchmark/evaluation/evaluator.py`: campo obrigatório `gold_document_ids` em `EvaluationInput` (sem default — evita o caso "gold vazio → recall 1.0"); `evidence_recall@k`/`precision@k` passam a usá-lo; `citation_accuracy` mantém `gold_evidence_ids`.
- `src/benchmark/storage/experiment_store.py`: `load_evaluation_input` consulta `SELECT DISTINCT document_id FROM gold_evidence WHERE question_id = …` e popula `gold_document_ids`.
- `tests/test_evaluator.py`: construtores atualizados + teste de regressão de namespace (`test_recall_uses_document_namespace_not_evidence_namespace`) + caso MuSiQue-like de recall parcial. **Suíte: 275 passed, 2 skipped.**
- `scripts/recompute_eval.py`: recompute determinístico **sem retrieval/LLM**, com **DELETE escopado antes** (necessário porque `evaluation_result_id` inclui o `metric_value` na chave — `experiment_store.py:305` — então valor alterado criaria linha nova em vez de atualizar).

## Resultado (backfill do vector_rag, 1000/1000)

| métrica | antes | depois |
|---|---|---|
| `evidence_recall_at_5` (mean) | 0.0000 | **0.0954** (211/1000 > 0; max 1.0) |
| `precision_at_5` (mean) | 0.0000 | **0.0472** |
| `answer_f1` / `exact_match` | 0.054 / 0.001 | inalterados (independentes do bug) |
| `citation_accuracy` | 0.0000 | 0.0000 (ainda quebrado — ver consequências) |

0 duplicatas de `(run_id, metric_name)` após o backfill.

**Leitura:** o recall baixo-mas-variável é o resultado *esperado* — recuperação multi-hop com top-5 denso sem reranker é genuinamente difícil no MuSiQue (motivação direta para os métodos de grafo da tese). O fix prova que a **métrica** funciona, não que o vector_rag seja forte.

## Consequências

- **`citation_accuracy` permanece inválido** (citações chunk-level vs gold ev-level): **não reportar como métrica válida** até consertar (mapear `chunk_id → evidence_id`).
- **Métodos de grafo (LightRAG/Cognee/MS GraphRAG) exigem reconciliação própria (Option D, adiada):** seus itens recuperados não carregam `doc_*` canônico (LightRAG retorna "Original Chunks ID"; Cognee liga fragmentos à fonte; MS GraphRAG retorna *community summaries* sem proveniência por documento). **Comparações de recall entre vector_rag e métodos de grafo não são válidas** até a reconciliação. (Fontes desta seção são preprints — ⚠️ usar só descritivamente; detalhes na estratégia, §10/D6.)
- `precision@5` mistura itens chunk-level (recuperado) com gold document-level; limpo enquanto 1 passagem MuSiQue ≈ 1 chunk.

## Referências citadas

1. P. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." **NeurIPS 2020**.
2. Y. Kim, W. Lee. "Where Does Legal AI Fail? Evaluating RAG Pipelines." **CIKM '25** (ACM). DOI 10.1145/3746252.3761151.
3. M. Afane et al. "Benchmarking Legal RAG: The Promise and Limits of AI Statutory Surveys." **CSLAW '26** (ACM). DOI 10.1145/3788646.3789533.
4. A. Houimli, Z. Gabsi, S. Skhiri. "Evaluation of GraphRAG Strategies for Efficient Information Retrieval." **IEEE Big Data 2025** (verificar dígitos do DOI 10.1109/BIGDATA…11400992).
5. I. Imtiyaz et al. "Assessing RAG: A Comprehensive Review of Evaluation Frameworks." **ICCIKE 2025** (IEEE). DOI 10.1109/ICCIKE67667.2025.11318213.
6. "Improving LLM Retrieval with GraphRAG-SF: Semantic Filtering for Graph-Based RAG Systems." **IEEE Big Data 2025**.
7. U. K. S. et al. "Medical RAG: Hybrid Retrieval & Re-Ranking." **IEEE ResGenXAI 2025**. DOI 10.1109/ResGenXAI64788.2025.11344015. *(contra-exemplo)*
8. J. Saad-Falcon et al. "ARES: An Automated Evaluation Framework for RAG Systems." **NAACL 2024** *(o PDF local é o preprint arXiv:2311.09476)*.
