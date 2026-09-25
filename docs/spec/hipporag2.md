# Plano de Implementação — HippoRAG 2 como 4º substrato (MuSiQue eval1k)

**Data:** 2026-07-06 · **Status:** PLANO (aguardando aprovação para P0)
**Fontes:** paper `references/hipporag2_icml2025_published.pdf` (ICML 2025, PMLR 267) ·
repo oficial `OSU-NLP-Group/HippoRAG` · auditoria do contrato de integração (agente Explore, 2026-07-06).

> **Anotado a 2026-08-09, não reescrito.** Este documento é o plano de
> implementação tal como foi aprovado, e a maior parte dele continua a valer —
> as decisões D1 a D6 são as que o código faz hoje. **Três pontos foram
> superados pela execução** e estão marcados no sítio onde aparecem: a decisão
> **D7** (indexação em lotes), a descrição do `hipporag2_index_runner.py` em
> **§4/P1.2** (idempotente e resumível) e o **risco #6** (índice não-incremental,
> classificado como probabilidade *Baixa*).
>
> O que se passou: o risco #6 materializou-se, e a mitigação escolhida — a D7 —
> era a causa. Indexar em lotes produziu **19.392.908 arestas para 319.817 pares
> distintos**, multiplicidade ~60 contra as ~1,58M arestas do grafo saudável
> previsto na §1. A deduplicação por hash de conteúdo é do *embedding store* e
> cobre as passagens; o passo de OpenIE que **deriva as arestas** volta a correr
> a cada chamada a `index()` e volta a acrescentá-las. A premissa de
> incrementalidade era falsa para as arestas.
>
> A indexação é hoje **passagem única**, imposta em código e não por convenção:
> o `--batch-size` não existe e um `save_dir` com conteúdo é recusado. Ver
> `PLANO_RELEASE.md` (ESTADO, item 3) e `refino_plano.md` para o registo
> completo. **Quem ler esta spec não deve montar um supervisor em lotes.**

---

## 0. Objetivo e enquadramento

Adicionar o **HippoRAG 2** como 4º substrato graph-RAG do benchmark (ao lado de
`lightrag_neo4j`, `cognee`, `ms_graphrag`), avaliado nas **1000 queries MuSiQue eval1k**,
encaixando no desenho de 2 braços (nativo × controlado):

- **Braço controlado:** retrieval HippoRAG 2 → reader do harness (`READER_GROUNDING=v2`).
- **Braço nativo:** resposta própria do `.rag_qa()` (short-answer CoT — o MESMO prompt já
  portado na Rota B), materializada via ETL como experimento próprio.

**Por que vale a pena (ângulo científico):**
1. Substrato graph-RAG SOTA com números MuSiQue **publicados e comparáveis**
   (gpt-4o-mini: **EM 35.0 / F1 49.3** QA; **recall@2 53.5 / recall@5 74.2** — Tab. 8/9).
2. É o único dos 4 que **produz ranking de passagens diretamente** → `evidence_recall@5`
   funciona sem gambiarra (lightrag/cognee/graphrag não expõem passagens rankeadas;
   o próprio paper exclui GraphRAG/LightRAG da Tab. 3 por isso).
3. Resposta nativa já é span curta (prompt CoT) → F1 nativo comparável sem rephrase.
4. Caveat honesto: F1 49.3 (deles, gpt-4o-mini) ≈ LightRAG rephrased 48.1 (nosso).
   Pode não dominar neste setup — o que é um achado em si.

**Dificuldade estimada: MÉDIA — ~2–4 dias focados.** O algoritmo vem pronto (repo oficial);
o esforço é canalização: venv isolado + build de índice supervisionada + ~6 ficheiros de plumbing.

---

## 1. O método (resumo operacional do paper)

Duas fases (Fig. 2 do paper):

**Indexação offline**
1. **OpenIE por LLM** por passagem → triplas schema-less; sujeito/objeto = *phrase nodes*,
   predicado = *relation edge*.
2. **Synonym detection** por embedding: pares de phrases com similaridade > threshold
   ganham *synonym edge*.
3. **Dense-sparse integration:** cada passagem vira *passage node* ligado por *context edge*
   ("contains") a todas as suas phrases. KG final = phrases + passagens.

**Retrieval online**
1. **Query→triple:** embedder rankeia triplas do grafo contra a query (top-k).
2. **Recognition memory (filtro):** LLM filtra as triplas recuperadas (prompt Fig. 4;
   tunado com DSPy MIPROv2 — **usar o prompt final publicado, sem re-tunar**).
   Se o filtro devolve vazio → fallback para dense retrieval puro.
3. **Seed nodes + reset probabilities:** phrase nodes das triplas filtradas (até 5, score =
   média dos scores das triplas onde aparecem) + **todos** os passage nodes (score = similaridade
   de embedding × **weight factor 0.05**).
4. **PPR** (Personalized PageRank) sobre o KG via **python-igraph** — sem Neo4j.
5. Ranking final de passagens = PageRank dos passage nodes → top-5 vai ao reader QA.

**Hiperparâmetros publicados (Tab. 13 + §6.2):**

| Parâmetro | Valor |
|---|---|
| Synonym threshold | 0.8 |
| PPR damping factor | 0.5 |
| Temperature | 0.0 |
| Passage-node weight factor | 0.05 |
| Top-k triplas (query→triple) | 5 |
| Passagens para QA | top-5 |

**Escala esperada do KG MuSiQue (Tab. 10, gpt-4o-mini, 11.656 passagens):**
101.641 phrase nodes · 11.656 passage nodes · 125.903 relation edges ·
1.304.605 synonym edges · 146.293 context edges → **~1,58M arestas**.

**Custo/tempo (Tab. 12, MuSiQue completo):** 9,2M tokens in + 3,0M out na indexação ·
~100 min de indexação (com Llama-70B em 4×H100; conosco será API-bound, estimar 2–5 h
em lotes) · ~1,2 s/query. Ordem de custo ≈ índice cognee (US$3,88). **Estimativa: US$3–6
de índice + ~US$2–4 de retrieval/QA nas 1000 queries.**

---

## 2. Decisões de arquitetura (fechadas antes de codar)

| # | Decisão | Justificativa |
|---|---|---|
| D1 | **Template estrutural = `ms_graphrag`** (NÃO cognee/lightrag_neo4j) | HippoRAG 2 = igraph + ficheiros locais, sem Neo4j → mesmo perfil do ms_graphrag: workspace local em parquet/ficheiros, adapter por subprocess, runner JSON via stdout. |
| D2 | **SEM Option C** (sem contentor Neo4j, sem portas 17xxx, sem `option_c.py`, sem entrada em `neo4j_ports.yaml` nem `METHOD_SHORT_TAGS`) | Não há Neo4j. Isolamento = diretório de artefatos por dataset (`build_workspace` + `_assert_within`), como no ms_graphrag. |
| D3 | **Isolamento de dependências = Padrão B (subprocess → venv dedicado `.venvs/hipporag2`)** | HippoRAG puxa torch/transformers/dspy/igraph — manter FORA do `.venv` principal e do `pyproject.toml`. Mesmo padrão do `.venvs/graphrag` + `scripts/graphrag_query_runner.py`. |
| D4 | **LLM = `gpt-4o-mini`** (OpenIE + triple filter + QA nativo) | Config oficialmente reportada no paper (Tab. 8/9) → números de referência diretos. Mesma família do resto do pipeline. |
| D5 | **Embedder = OpenAI `text-embedding-3-*`** (o mesmo do resto do benchmark), NÃO NV-Embed-v2 | Sem GPU na máquina (verificado 2026-07-06). O repo oficial suporta embeddings OpenAI. NÃO é réplica exata do paper → declarar na tese como comparação controlada; o próprio paper defende robustez a retrievers diferentes (§1, "robust to different retrievers"). |
| D6 | **Hiperparâmetros = defaults publicados** (Tab. 13), prompt do filtro = o publicado (Fig. 4 / repo), **sem re-otimização DSPy** | Reprodutibilidade citável; re-tunar custaria tempo e quebraria comparabilidade. |
| D7 | ~~**Índice em lotes com supervisor self-healing**~~ **SUPERADA a 2026-08-09 — ver abaixo** | ~~Histórico: OOM/quedas no cognee (693/1000) e reboots. Grafo de 1,58M arestas em memória exige monitorização de RSS.~~ |

> **D7 foi revertida.** A indexação é **passagem única sobre a janela inteira**,
> uma só chamada a `index()` por `save_dir`. Lotes corrompem o grafo: cada
> chamada volta a derivar as arestas via OpenIE e volta a acrescentá-las —
> 19.392.908 arestas para 319.817 pares distintos, contra as ~1,58M da §1.
>
> **O risco que D7 mitigava não desapareceu.** O OOM do risco #2 continua real,
> e a passagem única troca uma falha de correcção por um risco de memória. A
> troca é deliberada: um grafo corrompido em silêncio é pior do que um OOM
> barulhento. Quem correr o corpus inteiro é aí que arrisca, e a saída é
> **reduzir a janela com `--limit` para um destino próprio**, não retomar com
> `--skip` sobre o mesmo `save_dir` — que é a mesma duplicação por outro
> caminho.

---

## 3. Contrato de integração (mapeado no código, 2026-07-06)

> Não há registry/factory de métodos. Dispatch = cadeias `if method_id == ...` hard-coded.
> "Registar" = editar os sítios listados em §3.2.

### 3.1 O que um método expõe (`src/benchmark/methods/<name>/`)

| Ficheiro | Conteúdo | Template |
|---|---|---|
| `config_builder.py` | `HIPPORAG2_METHOD_ID = "hipporag2"`; dataclass congelada `Hipporag2Config` (defaults + `validate_*`); `build_workspace(*, artifacts_dir, dataset_id, dataset_version, method_id=HIPPORAG2_METHOD_ID) -> Hipporag2Workspace` com raiz `artifacts/{slug}/hipporag2/...` e guardas `_assert_within` | `ms_graphrag/config_builder.py:10,30,124` |
| `adapter.py` | `Hipporag2Adapter(*, workspace, runner=None)`; expõe `.workspace`; `query_context(query=, ...)` faz subprocess para `.venvs/hipporag2/bin/python scripts/hipporag2_query_runner.py`; método de index idem. Env overrides: `HIPPORAG2_VENV_PYTHON` / `HIPPORAG2_QUERY_RUNNER`. **Usar `.absolute()`, NUNCA `.resolve()`** no python do venv (resolver o symlink perde `pyvenv.cfg` → perde site-packages; aviso em `ms_graphrag/adapter.py:132-134`) | `ms_graphrag/adapter.py:61,125,135-155` |
| `output_parser.py` | `parse_query_output(*, query, stdout, top_k, ...) -> RetrievalResult`; mapeia JSON do runner → `RetrievedItem`; **OBRIGATÓRIO popular `source_document_id`** (e `metadata["document_ids"]`) — senão `evidence_recall@5` = 0.0 (bug histórico do vector_rag, namespace ev_/doc_) | `ms_graphrag/output_parser.py:14,70` |
| `retriever.py` | `retrieve(*, adapter, query, top_k=5, experiment_store=None, run_id=None, ...) -> RetrievalResult`; chama `experiment_store.persist_retrieval_result(...)` quando recebe store | `ms_graphrag/retriever.py:11,31-38` |
| `indexer.py` | `load_canonical_documents(canonical_path)` (lê `documents.jsonl` → `Document`) + `index(*, adapter, documents) -> Hipporag2IndexSummary` | `lightrag_neo4j/indexer.py:20,40` |
| `__init__.py` | re-exporta method id, `build_workspace`, adapter, `index`, `retrieve` | `ms_graphrag/__init__.py` |
| ~~`option_c.py`~~ / ~~`neo4j_schema.py`~~ | **NÃO criar** (D2) | — |

**Tipo de retorno — `RetrievalResult`** (`src/benchmark/core/schemas.py:76-82`, pydantic
`extra="forbid"` → campos extra são ERRO, específicos vão em `metadata`):

```python
class RetrievalResult(BenchmarkModel):
    method_id: str
    query: str
    items: list[RetrievedItem] = []   # RetrievedItem: schemas.py:67-73
    raw_response: dict | list | str | None = None
    latency_ms: float
    metadata: dict = {}

class RetrievedItem(BenchmarkModel):
    item_id: str
    text: str
    score: float | None
    source_document_id: str | None    # ← evidence_recall lê daqui
    source_chunk_id: str | None       # ← reader usa "{source_chunk_id or item_id}: {text}"
    metadata: dict
```

- Reader monta o contexto em `agents/nodes.py:251-260` a partir de `items`.
- PPR score do passage node → `score`; texto da passagem → `text`; id canónico do doc
  MuSiQue → `source_document_id`. Resposta nativa do `.rag_qa()` (quando pedida) →
  `raw_response` (padrão lightrag para o ETL nativo).

### 3.2 Sítios de registo (hard-coded — editar 3, opcionalmente 4)

| # | Ficheiro | O quê |
|---|---|---|
| R1 | `src/benchmark/experiments/run_single.py::_build_retriever` (linhas 111–198) | Adicionar branch `if method_id == HIPPORAG2_METHOD_ID:` (modelo: branch cognee `:162-180` ou ms_graphrag `:181-195`) + imports no topo (`:22-46`). O `raise ValueError` final (`:196`) lista os métodos suportados — atualizar. |
| R2 | `src/benchmark/agents/nodes.py:200-218` | Adicionar `"hipporag2"` a `_SUPPORTED_METHODS` (senão o nó retriever rejeita antes do dispatch). |
| R3 | `src/benchmark/cli/app.py` | Guardas + branches dos comandos `index` (`:179-190`, `:200-354`) e `retrieve` (`:385-396`, `:415-461`) + helper de factory do adapter (padrão `_ms_graphrag_adapter`). **Só necessário para os caminhos CLI**; o batch (P4) precisa apenas de R1+R2. |
| R4 | `core/naming.py::METHOD_SHORT_TAGS` (`:13-17`) | **PULAR** — só para Neo4j/Option C; `method_short_tag` auto-slugifica ids desconhecidos. |

`scripts/run_experiment_batch.py` **não muda** — `--method` passa direto (`:40`, `:121`).

### 3.3 Config YAML

- Criar `configs/methods/hipporag2.yaml`. **Único campo obrigatório:** `method_id: hipporag2`
  (validado em `cli/app.py:197-198`). Convencionais: `enabled`, `description`, `index:`
  (`max_documents`), `query:` (`top_k`).
- **Atenção:** o caminho batch (`run_experiment_batch` → `run_single`) NÃO lê o YAML —
  a config runtime real é a dataclass congelada do `config_builder.py`. YAML serve os
  comandos CLI `benchmark index/retrieve`.
- **NÃO** mexer em `configs/infra/neo4j_ports.yaml`.

### 3.4 Persistência e eval (já existente, sem mudanças)

- `ExperimentStore.persist_retrieval_result(...)` (`storage/experiment_store.py:83`) →
  tabelas `retrieval_results` (`:100`) + `retrieval_items` (`:133`), id determinístico,
  `ON CONFLICT DO UPDATE` (resumível).
- `persist_answer` (`:199`) → `answers`; `evaluate_run(run_id=, store=, k=)`
  (`evaluation/evaluator.py`) → EM / answer_f1 / evidence_recall_at_k em
  `evaluation_results`; chamado no loop batch (`run_experiment_batch.py:127`).
- Reader/grounding: `READER_GROUNDING` (v2 grounded default / v1 livre),
  `nodes.py:244-248`, instruções `:224`/`:237`.

---

## 4. Plano faseado (P0–P5)

> Convenção do projeto: ao fim de CADA fase, **Step Status Report** estruturado + aguardar
> aprovação antes de avançar.

### P0 — Venv isolado + shakeout da lib (½–1 dia) ⚠️ maior risco do plano

1. `uv venv .venvs/hipporag2 && .venvs/hipporag2/bin/pip install hipporag` (ou clone do repo
   se o pacote PyPI estiver desatualizado face ao paper — verificar tag/commit vs ICML).
2. Confirmar que importa e roda **sem GPU e sem vLLM**: LLM via OpenAI API (`gpt-4o-mini`)
   e embedder via OpenAI API. Se o path OpenAI-embedder tiver arestas upstream, resolver
   aqui (é o item mais provável de comer meio-dia).
3. Smoke mínimo: indexar 5 passagens sintéticas, `retrieve()` 1 query, inspecionar objeto
   devolvido (formato dos docs/scores) — isto fixa o contrato do runner.
4. Registrar versões pinadas (`pip freeze > .venvs/hipporag2.lock.txt` ou equivalente).

**Gate P0:** import OK, index+retrieve smoke OK em CPU, API-only; formato de saída documentado.

### P1 — Runner + pacote do método (1 dia)

1. `scripts/hipporag2_query_runner.py` — roda DENTRO do venv: carrega índice do workspace,
   executa retrieve (e opcionalmente `rag_qa` com flag `--generate`, padrão
   `ms_graphrag/adapter.py:157`), imprime **um único JSON** no stdout:
   `{items: [{passage_id, document_id, text, ppr_score}], native_answer?, stats}`.
2. `scripts/hipporag2_index_runner.py` — idem para indexação. ~~em lotes (`--limit N`,
   idempotente/resumível: o HippoRAG persiste o índice em disco e suporta inserção incremental —
   validar em P0)~~ **SUPERADO a 2026-08-09:** a validação em P0 nunca chegou a
   ser feita e a premissa era falsa. **Não é idempotente nem resumível.** Uma
   chamada a `index()` por `save_dir`; `--skip`/`--limit` definem a janela e cada
   janela exige destino próprio; um `save_dir` com conteúdo é recusado, com
   `--reset` destrutivo como única saída. Ver a anotação da D7.
3. Pacote `src/benchmark/methods/hipporag2/` completo (6 ficheiros, §3.1).
4. Registo R1+R2 (+R3 se quisermos os comandos CLI).
5. `configs/methods/hipporag2.yaml`.
6. Testes: unit do `output_parser` (JSON fixture → `RetrievalResult` com
   `source_document_id` correto) + teste de dispatch (method id aceito). Suíte verde.

**Gate P1:** suíte de testes verde; `retrieve` end-to-end contra o índice smoke de P0
devolve `RetrievalResult` válido com ids de documentos canónicos.

### P2 — Shakeout de indexação (15 docs) (½ dia)

1. Indexar ~15 docs MuSiQue reais (mesmo protocolo do shakeout cognee de 2026-06-27).
2. Verificar: contagens de nodes/edges plausíveis, não-duplicação em re-run (idempotência),
   retrieve devolve passagens certas para queries dos docs indexados.
3. Medir: tokens/doc, tempo/doc, RSS → extrapolar custo/tempo do full e calibrar tamanho
   de lote do supervisor.

**Gate P2:** shakeout PASS + projeção de custo/tempo/memória do full index.

### P3 — Full index (11.656 passagens) + eval1k retrieval (1 dia, majoritariamente wall-clock)

1. `scripts/hipporag2_supervisor.sh` — lotes (calibrados em P2), self-healing
   (queda/hibernação/OOM), heartbeat em `artifacts/musique/reports/hipporag2_index_status.txt`,
   log próprio. Padrão: `cognee_supervisor_batched.sh` + `rephrase_supervisor.sh`.
2. Rodar full index. Reconciliação final (contagens vs Tab. 10 como sanity de ordem de
   grandeza: ~100k phrase nodes, ~1,5M edges).
3. Eval batch controlado: `python scripts/run_experiment_batch.py --method hipporag2
   --experiment-id musique_eval1k_hipporag2 --top-k 5` (com `READER_GROUNDING=v2` default).

**Gate P3:** índice completo reconciliado; 1000/1000 queries com retrieval + answer + eval
persistidos; `evidence_recall@5 > 0` (sanity do namespace de ids).

### P4 — Braço nativo + ETL (½ dia)

1. Runner com `--generate` → `.rag_qa()` (prompt CoT oficial = o mesmo da Rota B) →
   resposta nativa em `raw_response`.
2. ETL `scripts/etl_hipporag2_native.py` (padrão `etl_lightrag_native.py`) → experimento
   `musique_eval1k_hipporag2_native` com EM/F1/containment.
   - Nota: a resposta nativa JÁ é span curta → coluna comparável à "rephrased" dos outros
     3 sem passo extra de rephrase.

**Gate P4:** experimento nativo com 1000/1000, EM/F1 persistidos.

### P5 — Análise + relatório (½ dia)

1. Tabela consolidada 4 substratos × {evidence_recall@5, EM/F1 nativo-curto/rephrased,
   judge* (se/quando rodar)} × comparação com paper (Tab. 8/9 gpt-4o-mini:
   EM 35.0 / F1 49.3 / recall@5 74.2).
2. Step report em `artifacts/musique/reports/step_report_hipporag2_<data>.md` +
   atualização de `status/`, `metodologia.md`, memória.
3. Decidir se entra o judge (custo ~US$6,6/método — mesma pendência da Rota B).

**Gate P5:** relatório aprovado; números na faixa esperada ou desvio explicado.

---

## 5. Riscos e mitigações (por ordem)

| # | Risco | Prob. | Mitigação |
|---|---|---|---|
| 1 | **Instalação/execução sem GPU** — HippoRAG puxa torch/dspy/vLLM; path OpenAI-embedder menos testado upstream | Alta | P0 dedicado só a isto; venv isolado; se preciso, pin de versões do repo na tag do paper; issues do repo como referência |
| 2 | **OOM na build/carga do grafo** (~1,58M arestas igraph em RAM; histórico cognee 693/1000) | Média | Supervisor em lotes self-healing (D7); monitorizar RSS; heartbeat |
| 3 | **`evidence_recall@5 = 0` por namespace de ids** (bug histórico ev_/doc_) | Média | `output_parser` testado com fixture contra ids canónicos; gate P3 exige recall > 0 |
| 4 | **Divergência do paper** (NV-Embed-v2 → text-embedding-3) | Certa (aceite) | Declarar na tese como comparação controlada (D5); ancorar na config gpt-4o-mini reportada |
| 5 | Custo derrapar (retry storms na API) | Baixa | Lotes + resumível via `ON CONFLICT`; orçamento US$10–15 total com margem |
| 6 | Índice não-incremental/não-idempotente na lib | ~~Baixa~~ **MATERIALIZOU-SE** | ~~Validar em P0.3; se não, checkpoint por snapshot do workspace entre lotes~~ → passagem única imposta em código, 2026-08-09 |

> **O risco #6 era o que estava mal avaliado.** Classificado como *Baixa*, foi o
> único que se concretizou, e a mitigação prevista para ele — *checkpoint* por
> lotes — era a própria causa. A validação em P0.3 não foi feita; a premissa da
> incrementalidade seguiu para o código como facto e para os docstrings como
> ensinamento, e só apareceu ao contar as arestas do grafo já construído.
>
> A lição que sobra, e que vale para além do HippoRAG 2: **um risco cuja
> detecção depende de olhar para o resultado não se classifica como *Baixa*.**
> É por isso que o sumário da indexação passou a trazer vértices, arestas, pares
> distintos e multiplicidade máxima — o número que torna esta classe de defeito
> visível a olho, em vez de silenciosa.

---

## 6. Orçamento e cronograma

| Fase | Esforço ativo | Wall-clock extra | Custo API |
|---|---|---|---|
| P0 venv+shakeout lib | ½–1 dia | — | ~US$0,10 |
| P1 runner+pacote+registo | 1 dia | — | — |
| P2 shakeout 15 docs | ½ dia | — | ~US$0,10 |
| P3 full index + eval1k | ½ dia | 2–5 h índice + ~2 h eval | ~US$4–8 |
| P4 braço nativo + ETL | ½ dia | ~1 h | ~US$1–2 |
| P5 análise + relatório | ½ dia | — | — |
| **Total** | **~3–4 dias** | | **~US$6–10** (+US$6,6 se judge) |

---

## 7. Ficheiros load-bearing (ler durante a implementação)

- `src/benchmark/core/schemas.py:67-82` — tipo de retorno
- `src/benchmark/experiments/run_single.py:111-198` — dispatch principal
- `src/benchmark/agents/nodes.py:200-270` — métodos suportados + reader/grounding
- `src/benchmark/methods/ms_graphrag/{adapter,retriever,output_parser,config_builder}.py` — template 1:1
- `src/benchmark/storage/experiment_store.py:83-161` — persistência
- `scripts/graphrag_query_runner.py` — padrão runner venv-isolado
- `scripts/run_experiment_batch.py` — loop batch/eval (não muda)
- `scripts/etl_lightrag_native.py` — padrão ETL do braço nativo
- `scripts/cognee_supervisor_batched.sh`, `scripts/rephrase_supervisor.sh` — padrão supervisor
- Paper: §3 (método), §G.1 (PPR/seeds/reset), Fig. 4 (prompt filtro), Tab. 8/9/10/12/13
  (números de referência, escala do KG, custo, hiperparâmetros)

---

## 8. Critérios de aceitação globais

1. Suíte de testes verde em todas as fases (nenhuma regressão nos 4 métodos existentes).
2. Índice full reconciliado (ordem de grandeza da Tab. 10).
3. 1000/1000 queries nos dois braços, persistidas no Postgres, resumíveis.
4. `evidence_recall@5` > 0 e plausível face a 0.742 do paper (desvio explicável pelo embedder).
5. EM/F1 nativo-curto comparável à Tab. 8 (EM 35.0 / F1 49.3) — desvio ±10 pts investigado.
6. Nenhuma dependência pesada vazando para o `.venv` principal / `pyproject.toml`.
7. Step Status Report aprovado ao fim de cada fase antes de avançar.
