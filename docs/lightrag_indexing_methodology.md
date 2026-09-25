# Metodologia de indexação — LightRAG sobre MuSiQue (Phase 3E)

> Documento de metodologia da tese (RAG com grafos / graph-based RAG). Descreve
> em detalhe **como** o dataset MuSiQue é indexado pelo método `lightrag_neo4j`,
> **por quê** cada decisão foi tomada e **em quais trabalhos revisados por pares**
> ela se apoia. Gerado 2026-06-20, em paralelo à execução do índice 1k.
>
> Convenção de fontes: apenas trabalhos **publicados/revisados por pares** (sem
> preprints arXiv), conforme a diretriz da tese. As entradas bibliográficas
> completas estão consolidadas em `references/graphrag_datasets_report.md`.

---

## 1. Visão geral do pipeline

A indexação não é um passo único; é o último estágio de um pipeline canônico que
separa **representação do dado** (independente de método) da **indexação
específica de cada método**:

```
raw MuSiQue (JSONL)
   │  loader (musique_loader.py)
   ▼
dataset canônico (documents / chunks / questions / gold_evidence em JSONL)
   │
   ├──► vector_rag  : embeddings de chunks (pgvector)         [baseline denso]
   └──► lightrag_neo4j : grafo de conhecimento (Neo4j) + vector store  ◄── este documento
```

A separação é deliberada: todos os métodos consomem **o mesmo dataset canônico**,
de modo que diferenças de desempenho refletem o **método de recuperação**, não
diferenças de pré-processamento. Esse princípio de comparação controlada segue a
prática estabelecida desde o trabalho seminal de RAG (Lewis et al., *Retrieval-
Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020), que
formaliza a separação entre o **índice de conhecimento** e o **gerador**.

---

## 2. Por que MuSiQue, por que graph-based RAG

- **MuSiQue** (Trivedi et al., *MuSiQue: Multihop Questions via Single-hop
  Question Composition*, TACL 2022) é um benchmark de QA **multi-hop** construído
  por composição de perguntas single-hop, com **parágrafos distratores**
  desenhados para penalizar atalhos de raciocínio. Cada pergunta exige combinar
  evidências de 2 a 4 parágrafos distintos — exatamente o cenário em que a
  recuperação puramente densa/lexical tende a falhar e em que estruturas de
  **grafo** podem ajudar a "encadear" evidências.
- **Graph-based RAG**: a hipótese da tese é que representar o corpus como um
  **grafo de entidades e relações** (em vez de apenas chunks isolados) melhora a
  recuperação multi-hop, pois o caminho entre evidências passa a ser explícito.
  LightRAG (Guo et al., *LightRAG: Simple and Fast Retrieval-Augmented
  Generation*, Findings of EMNLP 2025) é um dos métodos revisados por pares que
  materializa essa ideia com custo de indexação menor que abordagens hierárquicas,
  e por isso é o primeiro método de grafo avaliado.
- **Amostra de 1000 perguntas** (`ans_v1.0_eval1k`): segue o protocolo de
  avaliação adotado por IRCoT (Trivedi et al., *Interleaving Retrieval with
  Chain-of-Thought Reasoning…*, ACL 2023) e HippoRAG (Gutiérrez et al.,
  *HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs*, NeurIPS 2024),
  que avaliam sobre 1000 perguntas do split público de validação do MuSiQue
  (`seed=42`). O split de teste do MuSiQue é privado.

---

## 3. Representação canônica (a unidade indexada)

No loader (`src/benchmark/ingestion/musique_loader.py`), **cada parágrafo do
MuSiQue vira um `Document`**:

- `Document.text = f"{title}\n\n{paragraph_text}"` — título e corpo concatenados,
  para que o título (frequentemente a entidade-âncora, ex.: nome de uma pessoa ou
  obra) participe da extração de entidades.
- `document_id = deterministic_id("doc", [dataset_id, dataset_version, title, paragraph_text])`
  — um **hash SHA-256** sobre o conteúdo. Isso dá **deduplicação natural**: o
  mesmo parágrafo referenciado por várias perguntas colapsa em **um único
  documento**. Por isso as 1000 perguntas (≈20 parágrafos cada) reduzem-se a
  **11.515 documentos únicos** (e não ~20.000).
- `Question.document_id = None` e `question_type = "multi_hop_reasoning"`: a
  pergunta é multi-documento por construção; as evidências corretas ficam em
  `gold_evidence` (ligando pergunta ↔ parágrafos *supporting*).

> Implicação para a indexação: o LightRAG indexa **passages (parágrafos)**, não
> perguntas. O grafo final representa o conhecimento contido nas evidências das
> 1000 perguntas — supporting **e** distratores —, que é o corpus sobre o qual a
> recuperação será avaliada.

---

## 4. Chunking — dois regimes distintos

Há **dois** mecanismos de chunking no projeto; é importante não confundi-los.

### 4.1 Chunks canônicos (para vector_rag) — baseados em caracteres
`src/benchmark/ingestion/chunking.py`:

| Parâmetro | Valor | Observação |
|---|---|---|
| `chunk_size` | **1200 caracteres** | janela deslizante |
| `chunk_overlap` | **200 caracteres** | ~16,7% de sobreposição |
| ID | `deterministic_id("chunk", [doc_id, start, end, text])` | reprodutível |
| metadados | `start_char`, `end_char`, `chunk_index` | rastreabilidade |

A sobreposição de 200 caracteres existe para **evitar que uma entidade/sentença
seja cortada na fronteira** entre chunks, preservando contexto local — prática
padrão em pipelines RAG desde Lewis et al. (NeurIPS 2020). Como os parágrafos do
MuSiQue são curtos, na prática **cada parágrafo cabe em um único chunk** (≈11.888
chunks para 11.515 documentos — pouquíssimos parágrafos ultrapassam 1200 chars).
**Estes chunks NÃO são usados pelo LightRAG** — apenas pelos baselines denso e
lexical.

### 4.2 Chunking interno do LightRAG — baseado em tokens
O adaptador (`lightrag_neo4j/adapter.py`) entrega ao LightRAG o **texto completo
do documento** (`rag.insert(texts=[document.text], ids=[document_id])`); o
**LightRAG faz seu próprio chunking**, em **tokens** (confirmado no log:
`Chunking F(legacy): size=1200, split_only=False, overlap=100`):

| Parâmetro (LightRAG) | Valor | Fonte |
|---|---|---|
| `chunk_token_size` | **1200 tokens** | default LightRAG |
| `chunk_overlap_token_size` | **100 tokens** | default LightRAG |
| tokenizer (`tiktoken_model_name`) | **`gpt-4o-mini`** | default LightRAG |

Como os parágrafos do MuSiQue têm tipicamente ~100–350 tokens, **cada documento
gera 1 chunk** no LightRAG (o log mostra `Chunk 1 of 1` por documento). Ou seja,
para este corpus, a unidade efetiva de extração de grafo é **o parágrafo
inteiro** — o que é desejável em multi-hop, pois mantém a evidência de cada hop
coesa, sem fragmentá-la.

> **Por que deixar o LightRAG chunkar em vez de passar os chunks canônicos?**
> Para preservar a fidelidade ao método publicado: o LightRAG (Guo et al.,
> Findings EMNLP 2025) define a extração de entidades/relações **sobre seus
> próprios chunks tokenizados**. Reusar o método "as published" evita introduzir
> um viés de pré-processamento não previsto pelos autores e mantém a comparação
> honesta. A sobreposição em tokens (100) cumpre o mesmo papel anti-corte da
> seção 4.1, mas no espaço de tokens do modelo.

---

## 5. Indexação de grafo do LightRAG (o coração do método)

Para cada chunk, o LightRAG executa o pipeline abaixo (todos os parâmetros são os
**defaults publicados**, salvo a concorrência ajustada na seção 6):

1. **Extração de entidades e relações via LLM.** Um prompt instrui o
   `gpt-4o-mini` a extrair entidades (com `entity_type` e descrição) e relações
   (com descrição) do chunk. Tipos observados no corpus MuSiQue incluem `person`,
   `organization`, `geo`, `event`, `content` etc. (o tipo é inferido pelo LLM).
   - **Gleaning** = `entity_extract_max_gleaning = 1`: após a primeira extração, o
     LLM faz **uma** passada adicional ("você esqueceu alguma entidade?") para
     aumentar o recall de entidades. Por isso observam-se ~2 chamadas LLM de
     extração por documento.
2. **Merge incremental no grafo.** Entidades/relações novas são **mescladas** com
   as existentes (mesmo nome → mesmo nó). Quando um nó/aresta acumula muitas
   descrições, o LightRAG resume via LLM (`force_llm_summary_on_merge = 8`;
   `summary_max_tokens = 1200`) para manter a descrição compacta. Esse merge é o
   que conecta evidências de **documentos diferentes** num mesmo subgrafo —
   essencial para multi-hop.
3. **Embeddings.** Entidades, relações e chunks são vetorizados com
   `text-embedding-3-small` (**dimensão 1536**) e gravados no vector store do
   LightRAG (arquivos `vdb_entities.json`, `vdb_relationships.json`,
   `vdb_chunks.json` no workspace). Recuperação densa default: `top_k = 40`,
   `cosine_threshold = 0.2`.
4. **Persistência no grafo Neo4j.** Os nós e arestas são gravados no Neo4j via
   `graph_storage="Neo4JStorage"`. Cada nó recebe um **label de workspace**
   (backtick-quoted) que namespaceia o dataset dentro do banco
   (`lightrag_neo4j_musique_ans_v1.0_eval1k`). As relações são do tipo `DIRECTED`.
   Estado on-disk do KV (`kv_store_doc_status.json` etc.) permite **retomar** a
   indexação se interrompida.

### Modelos usados (escolhas de implementação)
| Papel | Modelo | Observação |
|---|---|---|
| LLM de extração/summary | `gpt-4o-mini` | barato; suficiente para extração estruturada |
| Embeddings | `text-embedding-3-small` (1536-d) | mesmo provedor; bom custo/qualidade |
| Provedor | OpenAI (`MODEL_PROVIDER=openai`) | resolvido por env, não pelo YAML do método (ver §8) |

---

## 6. Concorrência (ajuste operacional, não metodológico)

Os defaults do LightRAG (`MAX_PARALLEL_INSERT=3`, `MAX_ASYNC=4`) são muito
conservadores frente à folga da conta OpenAI (5000 RPM / 4.000.000 TPM). Para o
índice 1k, foram elevados **por variável de ambiente** (sem alterar código):

```
MAX_PARALLEL_INSERT=16   # documentos em paralelo
MAX_ASYNC=32             # chamadas LLM simultâneas
EMBEDDING_FUNC_MAX_ASYNC=16
```

Isto é um ajuste de **throughput**, não afeta o resultado do grafo (mesmos
prompts, mesmos modelos, mesmos defaults de extração). Reduziu o ETA de ~2,3 dias
para a ordem de horas. Justificativa e medições em
`artifacts/musique/reports/lightrag_throughput_investigation.md`.

---

## 7. Isolamento por dataset (Option C)

Cada par `(método, dataset, versão)` tem **container + volume Neo4j próprios**
(Option C). Para LightRAG·MuSiQue:

- container `neo4j_lightrag_musique_ans_v1_0_eval1k`, volume `…_data`;
- portas HTTP 17475 / Bolt 17688 (banda 17xxx; evita o Neo4j nativo do host em
  7474/7687);
- workspace on-disk `artifacts/musique_ans_v1_0_eval1k/lightrag_neo4j/`.

**Por quê:** o Neo4j local é **Community Edition**, que não suporta
`CREATE DATABASE` (multi-database é Enterprise). Isolar por *labels/propriedades*
num único banco (Option B) é **metodologicamente arriscado** para graph-RAG,
porque algoritmos de grafo do LightRAG podem operar **sobre o grafo inteiro
antes** de qualquer filtro de dataset, contaminando resultados. Isolar por
**processo + container + volume** (Option C) garante que o grafo de um dataset
nunca "vaze" para outro. Diagnóstico e decisão em
`artifacts/musique/reports/neo4j_isolation_diagnostic.md` e
`…/option_c_infrastructure_plan.md`.

---

## 8. Reprodutibilidade e armadilhas

- **Amostragem determinística:** `seed=42`, `split=validation`,
  `num_questions=1000` (em `configs/datasets/musique.yaml`).
- **IDs determinísticos:** documentos e chunks usam SHA-256 do conteúdo → o mesmo
  raw produz sempre os mesmos IDs (e a mesma deduplicação).
- **Resume:** o índice retoma do `kv_store_doc_status.json`; reiniciar não
  reprocessa documentos já concluídos.
- **Armadilha de provedor:** o provedor de embeddings/LLM é resolvido por
  `MODEL_PROVIDER` no `.env` (lido via `os.environ`), **não** pelo campo
  `embedding_provider` do YAML do método. Para indexar offline seria preciso
  sobrescrever `MODEL_PROVIDER`; editar o YAML não tem efeito. (Ver
  `project-musique-phase1-done` na memória.)
- **Custo real:** a indexação faz chamadas reais ao `gpt-4o-mini` (extração +
  gleaning + summaries) e ao `text-embedding-3-small`. ~2 chamadas LLM de
  extração por documento × 11.515 documentos.

---

## 9. Parâmetros — tabela consolidada

| Categoria | Parâmetro | Valor |
|---|---|---|
| Dataset | split / nº perguntas / seed | validation / 1000 / 42 |
| Dataset | documentos únicos (pós-dedup) | 11.515 |
| Documento | formato do texto | `"{title}\n\n{paragraph_text}"` |
| Chunk canônico (vector_rag) | tamanho / overlap | 1200 chars / 200 chars |
| Chunk LightRAG | tamanho / overlap / tokenizer | 1200 tok / 100 tok / tiktoken(gpt-4o-mini) |
| Extração | LLM / gleaning | gpt-4o-mini / 1 passada extra |
| Merge | summary on merge / max tokens | limiar 8 / 1200 tokens |
| Embeddings | modelo / dimensão | text-embedding-3-small / 1536 |
| Recuperação (default) | top_k / cosine threshold | 40 / 0.2 |
| Grafo | storage / label / tipo de relação | Neo4JStorage / workspace-label / DIRECTED |
| Concorrência | parallel insert / max async / embed async | 16 / 32 / 16 |
| Isolamento | container / bolt | neo4j_lightrag_musique_ans_v1_0_eval1k / 17688 |

---

## 10. Referências (revisadas por pares)

1. Lewis, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP
   Tasks.** *NeurIPS*, 2020.
2. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A. **MuSiQue: Multihop
   Questions via Single-hop Question Composition.** *TACL*, 2022.
3. Trivedi, H. et al. **Interleaving Retrieval with Chain-of-Thought Reasoning
   for Knowledge-Intensive Multi-Step Questions (IRCoT).** *ACL*, 2023.
4. Gutiérrez, B. J. et al. **HippoRAG: Neurobiologically Inspired Long-Term
   Memory for Large Language Models.** *NeurIPS*, 2024.
5. Guo, Z. et al. **LightRAG: Simple and Fast Retrieval-Augmented Generation.**
   *Findings of EMNLP*, 2025.

> Entradas completas (autores, páginas, DOI) e a triagem "peer-reviewed" estão em
> `references/graphrag_datasets_report.md`. Verifique a entrada do LightRAG ao
> citar na tese (venue confirmado como Findings of EMNLP 2025 no levantamento do
> projeto).
