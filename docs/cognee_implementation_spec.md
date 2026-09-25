# Especificação de implementação — método `cognee`

> Documento de planejamento. Escrito em 2026-06-24 a partir da investigação de como o
> `lightrag_neo4j` foi especificado. **Objetivo dos próximos dias:** implementar o `cognee` e
> rodar o **full 1000** (MuSiQue `musique_ans_v1_0_eval1k`), com as mesmas métricas de nível-resposta
> dos outros métodos.

> **Anotado a 2026-08-09, não reescrito.** A §2 abaixo tem uma lista intitulada
> **«Quebrado / CUAD-stale (footgun ativo)»**, e a §3 a tabela de deltas que dela
> decorre. **Nada nessa lista continua activo** — está tudo feito, e ler aquilo
> hoje é ler um aviso sobre um perigo que já não existe:
>
> | o que a spec dá como quebrado | estado a 2026-08-09 |
> |---|---|
> | `COGNEE_GRAPH_URL = bolt://localhost:7689` | é `18689`, banda deste repositório |
> | `raise ValueError("Cognee must use bolt://localhost:7689")` | exige `18689` |
> | `COGNEE_POSTGRES_DATABASE = cognee_cuad_v1_50docs` | é `cognee_benchmark` |
> | `dataset_name = cuad_v1_50docs_cognee` | é `benchmark_cognee` |
> | `audit.py` fala em «canonical CUAD chunk_id» | generalizado |
> | `docker-compose.cognee.yml` da era CUAD | não viaja; o `postgres_cognee` está no compose principal |
>
> Os **números de linha da §3 já não resolvem** — o ficheiro mudou muito desde
> Junho. Ficam como registo do que se encontrou, não como coordenadas.
>
> O que continua a valer, e é a maior parte do documento: o padrão de
> isolamento (seguir o `lightrag`, não o `ms-graphrag`), o contrato de
> integração, e as decisões de configuração do Cognee.

---

## 1. Padrão a seguir = **lightrag**, NÃO ms-graphrag

Há dois padrões de isolamento no projeto, e eles são opostos:

| | **lightrag_neo4j** (← seguir este) | **ms_graphrag** (← NÃO seguir) |
|---|---|---|
| Deps Python | **compartilha o `.venv` principal**; `lightrag-hku>=1.5.0` declarado no `pyproject.toml` | venv isolado `.venvs/ms-graphrag-neo4j` + **subprocess** |
| Princípio | *"isolation is graph/backend isolation, **not** Python dependency isolation"* (plano de dependência do lightrag, apagado em 2026-07-31) | isolamento de deps Python |
| Invocação da lib | **direto, import lazy** dentro do adapter | via subprocess no `ms_graphrag/adapter.py` |
| Isolamento real | **Neo4j próprio** + env vars dedicadas (`LIGHTRAG_NEO4J_*`) + Option C por dataset | venv + subprocess |

**Fato que reverteu a decisão anterior:** o `.venvs/lightrag-neo4j/` existe no disco (Python 3.11) mas é
**vestigial** — `grep` em `src/` não acha nenhum dispatch para ele; o lightrag roda da `.venv` principal.
A escolha registrada antes ("Opção 1 = venv próprio `.venvs/cognee`") foi **superada** por esta diretriz:
**cognee segue o lightrag → compartilha a `.venv` principal + isolamento só de backend.**

### Implicação para o cognee
- cognee **1.1.2 já está pip-instalado na `.venv` principal** (Python 3.13; `Requires-Python >=3.10,<3.15`),
  mas **não declarado** em `pyproject.toml`/`uv.lock`.
- Seguir lightrag = **declarar/pinar `cognee==1.1.2` no `pyproject`** e manter na `.venv` compartilhada.
- Isolar só o backend: **Neo4j próprio** (bolt **17689** / http **17476**, já reservado em
  `configs/infra/neo4j_ports.yaml: cognee_musique_ans_v1_0_eval1k`) + env vars `COGNEE_*` espelhando
  `LIGHTRAG_NEO4J_*` + Option C por dataset + Postgres de tracking compartilhado.
- ⚠️ **Risco real (lightrag não tinha):** cognee arrasta MUITAS deps transitivas — mais pesado que
  `lightrag-hku`. Declarar no `pyproject` e re-resolver o `uv.lock` pode perturbar deps do core
  (pydantic/httpx/openai/numpy). Por isso o **gate da Fase 1 é "suíte continua verde após re-resolver"**.

---

## 2. Estado atual do scaffolding (o que existe)

`src/benchmark/methods/cognee/`: `adapter.py`, `audit.py`, `config_builder.py`, `indexer.py`,
`parser.py`, `retriever.py`, `trace_writer.py`, `__init__.py`.

**Bom (já method-agnostic / pronto):**
- `adapter.py:131` `retrieve_async(...)` devolve **só um `RetrievalResult`** → a geração é do nó
  compartilhado; **cognee não precisa de código de geração novo**.
- `adapter.py:198` retrieval usa **`only_context=True`** (não usa o LLM do cognee na recuperação).
- `RealCogneeClient` (`adapter.py:49`) faz **import lazy** (`importlib.import_module("cognee")`) — sem
  import no topo, então não quebra coleta de testes.
- `adapter.py:234` `_load_chunk_mappings` já tenta reconciliar provenance via Postgres (enriquecimento doc_*).

**Quebrado / CUAD-stale (footgun ativo):**
- `config_builder.py:9` `COGNEE_GRAPH_URL = "bolt://localhost:7689"` (CUAD) — conflita com Option C (17689).
- `config_builder.py:19` `COGNEE_POSTGRES_DATABASE = "cognee_cuad_v1_50docs"`.
- `config_builder.py:117` `dataset_name: str = "cuad_v1_50docs_cognee"`.
- `config_builder.py:232` `raise ValueError("Cognee must use bolt://localhost:7689")` — **força** o bolt do CUAD.
- `audit.py:482,542,543` linguagem de provenance específica de **"canonical CUAD chunk_id"**.
- `docker-compose.cognee.yml` é da era CUAD.

**Faltando (do status 2026-06-24 §5 + investigação):**
1. cognee não declarado/pinado no `pyproject`.
2. config CUAD → MuSiQue + Option C.
3. **Wiring no CLI**: factory + dispatch (index/retrieve) + allowlist — cognee não registrado. *(local exato do factory ainda a confirmar — investigação foi interrompida.)*
4. Validação da integração real do `RealCogneeClient` contra a API do cognee 1.1.2 (ver §4 Fase 5).
5. Suíte de testes espelhando o lightrag.
6. Scripts de validação (real-import, env, smoke) + projeção de custo do cognify.

---

## 3. Deltas concretos CUAD → MuSiQue/Option C

| Arquivo:linha | De (CUAD) | Para (MuSiQue + Option C) |
|---|---|---|
| `config_builder.py:9` | `bolt://localhost:7689` | `bolt://localhost:17689` (ler de `neo4j_ports.yaml`, não hardcode) |
| `config_builder.py:19` | `cognee_cuad_v1_50docs` | `cognee_musique_ans_v1_0_eval1k` |
| `config_builder.py:117` | `cuad_v1_50docs_cognee` | dataset canônico MuSiQue (alinhar com a chave do `neo4j_ports.yaml`) |
| `config_builder.py:232` | `raise ... must use 7689` | aceitar 17689 (idealmente derivar do Option C, sem raise hardcoded) |
| `audit.py:482,542,543` | "canonical CUAD chunk_id" | provenance MuSiQue (doc_*) |
| `docker-compose.cognee.yml` | serviço/volume CUAD | container `cognee_musique` http 17476 / bolt 17689 / volume dedicado |

Env vars a definir (espelhando `LIGHTRAG_NEO4J_*`): `COGNEE_NEO4J_URI/USER/PASSWORD/DATABASE` +
`COGNEE_POSTGRES_*`; o `scoped_cognee_env` (`adapter.py:253`) mapeia para o que o cognee espera.

---

## 4. Plano por fases (com gates e o que muta)

> Disciplina mantida: **só métricas aditivas**; nunca sobrescrever baseline; **sem validação humana**;
> **confirmar custo de LLM antes de gastar**; sem ops destrutivas de docker; parar para Step Status
> Report antes de qualquer gasto. Não tocar no Neo4j host (7474/7687).

**Fase 1 — Declarar dependência (espelha a dep-strategy do lightrag).** *Muta: `pyproject.toml`, `uv.lock`, doc novo. Gasto: $0.*
- Declarar `cognee==1.1.2` (pin exato) no `pyproject.toml`; re-resolver `uv.lock`.
- Escrever `docs/cognee_dependency_plan.md` (espelhando o do lightrag).
- **GATE:** suíte completa + smoke-import dos outros métodos **verdes após re-resolver**.
- **Contingência (não é o default):** se a re-resolução perturbar o core e quebrar a suíte → cair para o
  padrão ms-graphrag (venv `.venvs/cognee` isolado + subprocess). Registrar como fallback.

**Fase 2 — De-CUAD o config (mata o footgun).** *Muta: `config_builder.py`, `audit.py`, `configs/methods/cognee.yaml`. Gasto: $0.*
- Aplicar os deltas da §3.
- **GATE:** testes unitários do `config_builder` do cognee verdes; `validate_environment_isolation`
  rejeita `NEO4J_*` genérico e valores CUAD.

**Fase 3 — Infra de backend (Option C, espelha lightrag).** *Muta: `docker-compose.cognee.yml`, wiring Option C. Gasto: $0.*
- Migrar o compose para o container `cognee_musique` (17476/17689, volume dedicado).
- Wire do Option C por dataset (espelhar `lightrag_neo4j/option_c.py`).
- **GATE:** container sobe; cognee conecta (cypher count read-only).

**Fase 4 — Wiring no CLI (factory + dispatch + allowlist).** *Muta: factory/registry/cli + testes. Gasto: $0.*
- Registrar `cognee` no factory/registry + allowlist; ligar dispatch de `index` e `retrieve`
  (espelhar a rota do `lightrag_neo4j`).
- Espelhar `tests/test_lightrag_neo4j_cli_routing.py` → `tests/test_cognee_cli_routing.py`.
- **GATE:** CLI reconhece `cognee`; testes de routing verdes.

**Fase 5 — Validação da API real (gasto ~$0).** *Muta: scripts/docs + dados de 1 doc. Gasto: mínimo.*
- `scripts/validate_cognee_real_import.py` (espelha `validate_lightrag_neo4j_real_import.py`): confirmar
  que o cognee 1.1.2 expõe **exatamente** o que o `RealCogneeClient` assume —
  `recall(query, datasets=, top_k=, only_context=, auto_route=False, scope="graph")`, `search` +
  `SearchType.CHUNKS`, `add`, `cognify(datasets=)`, e como apontar o graph store p/ um Neo4j específico e
  o relational store p/ um Postgres específico. **Se a assinatura não bater, corrigir o client.**
- Check de isolamento de env (espelha `validate_lightrag_neo4j_env`).
- **GATE:** real-import + env-check passam.

**Fase 6 — Smoke + PROJEÇÃO DE CUSTO DO COGNIFY (o gate de custo).** *Muta: dados de poucos docs. Gasto: pequeno (confirmar antes).*
- Indexar **1–5 docs** MuSiQue via `add+cognify`; medir tokens e **$/doc**; extrapolar para ~1000 docs.
  Esse é o **entregável central** do smoke (mesma disciplina dos $11 do judge).
- Recuperar 3 perguntas (`only_context`) → confirmar que o `RetrievalResult` parseia e que a reconciliação
  de provenance (doc_*) funciona — **check de viabilidade barato, desacoplado** do número de nível-resposta.
- **GATE go/no-go p/ Fase 7:** custo projetado do cognify no corpus **aprovado pelo usuário** →
  **STEP STATUS REPORT + aprovação explícita.**

**Fase 7 — FULL 1000 (o objetivo).** *Muta: dados cognee + linhas aditivas no Postgres. Gasto: cognify full + judge full (ambos pré-aprovados).*
- Indexar o `musique_ans_v1_0_eval1k` completo via cognify (após aprovação de custo).
- Recuperar as 1000 perguntas (`only_context`) → nó de geração compartilhado responde → persistir
  **aditivamente** (`method='cognee'`).
- Rodar as **mesmas métricas dos outros métodos**: `exact_match`, `answer_f1` e o **LLM-judge**
  (strict/lenient), aditivo, sem sobrescrever baseline, com guarda de fingerprint.
- Comparar: cognee vs **lightrag (strict 0,4315)** vs **vector (strict 0,2218)**.

---

## 5. Decisões em aberto (mudam o trabalho)
- **String canônica do dataset** do cognee MuSiQue (alinhar com a chave de `neo4j_ports.yaml`:
  `cognee_musique_ans_v1_0_eval1k`).
- **Judge do cognee:** rodar no mesmo passe ou separado (custo ~metade do passe de 2000, pois é 1000).
- **Python da contingência** (só se o gate da Fase 1 falhar): 3.11 como os outros venvs isolados.

## 6. Riscos de custo
- **cognify do corpus inteiro = maior incógnita** (extração LLM por chunk). Projetar no smoke antes de rodar.
- Judge sobre as 1000 respostas do cognee (~metade dos $11 do passe de 2000).

## 7. Próximo passo imediato
Fase 1 (declarar `cognee==1.1.2` + re-resolver + gate verde) — **sem gasto**. Aguardando "go".
