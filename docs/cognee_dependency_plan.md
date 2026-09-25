# cognee — Plano de dependência (venv isolado)

> Escrito 2026-06-27 ao executar a **Fase 1 revisada**. Espelha em espírito
> os planos de dependência do ms-graphrag e do lightrag (apagados na limpeza de
> 2026-07-31; o essencial ficou em `writing/thesis/text/methodology.tex` §3.3),
> mas **inverte** a decisão de 2026-06-24 ("cognee segue o lightrag / `.venv` compartilhada").

## 1. Decisão (revisada pelo usuário em 2026-06-27)

cognee **NÃO** entra no `.venv` principal e **NÃO** é declarado em `pyproject.toml`/`uv.lock`.
Em vez disso, segue o padrão de **isolamento de dependência do ms-graphrag**: um venv próprio.

**Motivo:** declarar `cognee==1.1.2` no core e re-resolver o `uv.lock` arrastaria ~100 deps
transitivas e forçaria downgrades de `jiter`/`rich`/`packaging` no ambiente compartilhado que roda
`vector_rag`, `lightrag_neo4j`, Postgres e o driver Neo4j 6.2.0 do benchmark. O risco a esse core
(que já produziu o baseline) não é aceitável. Isolar elimina o risco por construção.

> O isolamento de **backend** (Option C / Neo4j próprio por dataset) decidido antes **continua valendo**.
> O que mudou é só *onde a dependência Python vive*.

## 2. O que foi feito na Fase 1 (sem gasto, sem indexar, sem LLM)

```bash
uv venv --python 3.11 .venvs/cognee            # Python 3.11.14 (convenção dos outros isolados)
uv pip install --python .venvs/cognee/bin/python "cognee==1.1.2"   # bare, SEM extras
```

- **Pin exato:** `cognee==1.1.2`. Freeze completo (130 pacotes) em `docs/cognee_venv_freeze.txt`.
- **Extras evitados de propósito:** `cognee[neo4j]` (fixa `neo4j<6`, conflita com o driver 6.2.0 do core)
  e `cognee[dev]` (fixa `pytest<8`, conflita com pytest 9.0.3 do core). O bare não puxa nenhum dos dois.
- **Verificações:**
  - `import cognee` OK no venv isolado → `cognee.__version__ == "1.1.2"`.
  - `uv pip check` no venv isolado: **130 pacotes, todos compatíveis** (0 conflitos — mais limpo que o
    `.venv` principal, que tem 3 conflitos pré-existentes de jiter/rich/packaging por um pip-install
    antigo de cognee, ver §5).
  - Suíte do core (`.venv` principal): **283 passed, 2 skipped** — inalterada → core não foi tocado.

## 3. Como o CLI/wiring chamará o cognee (a desenhar na Fase 4 — aqui só o contrato)

**Problema:** o adapter atual (`src/benchmark/methods/cognee/adapter.py`, `RealCogneeClient`) faz
**import direto in-process** (`importlib.import_module("cognee")`). Isso **não funciona** com cognee num
venv isolado — o processo principal roda no `.venv` (3.13) e não enxerga o `cognee` do `.venvs/cognee`.

**Precedente no repo:** `ms_graphrag` (`adapter.py:31` `GraphRAGSubprocessRunner`) chama o framework
pesado via `subprocess.run(args, capture_output=True, text=True)`. Não há, em `src/`, nenhum caminho
hardcoded para um venv isolado — o contrato subprocess do cognee é **novo** e fica desenhado assim:

- **Worker no venv isolado:** um entrypoint (ex.: `scripts/cognee_worker.py`) executado por
  `.venvs/cognee/bin/python`. É o ÚNICO código que faz `import cognee`. Recebe um comando
  (`index` | `retrieve`) + payload **JSON via stdin**, devolve **JSON via stdout** (sem prints soltos;
  o cognee loga em stderr, que é separado).
- **Lado do core (`.venv`):** um `CogneeSubprocessRunner` que monta `args = [COGNEE_VENV_PYTHON, worker, cmd]`,
  injeta o env escopado (Option C: `COGNEE_NEO4J_*`/`COGNEE_POSTGRES_*` apontando para bolt 17689 /
  Postgres dedicado) e faz `subprocess.run`. Parseia o JSON de volta para `RetrievalResult` (retrieve)
  ou um summary de indexação (index). A geração de resposta continua no **nó compartilhado**
  (`gpt-4o-mini`), fora do worker.
- **Caminho do python isolado:** resolver de forma não-hardcoded — env var `COGNEE_VENV_PYTHON`
  com default `<repo>/.venvs/cognee/bin/python` (espelhar como o lightrag resolve `LIGHTRAG_*`).
- **Retrofit do adapter:** `RealCogneeClient` deixa de importar cognee no processo principal; vira
  cliente do subprocess. A lógica de parsing (`parser.py`) e o `CogneeAdapter` permanecem no core.

## 4. Pendências para as próximas fases (NÃO feitas agora)

- **Driver Neo4j para o cognee:** o bare **não instala** o driver `neo4j`. Para o Option C funcionar, o
  worker vai precisar do driver que o cognee espera — e como o venv é **isolado**, dá para instalar
  `neo4j<6` ali **sem afetar** o 6.2.0 do core (esse é justamente o payoff do isolamento). Confirmar a
  versão exata na Fase 3 (infra). Não tocado na Fase 1.
- **De-CUAD do config** (Fase 2): defaults CUAD hardcoded + `raise` da porta 7689 + `cognee.yaml`.
- **Worker subprocess + retrofit do adapter** (Fase 4).
- **Smoke + projeção de custo do cognify** sobre o corpus canônico **~11.515 docs** (Fase 5).

## 5. Riscos restantes / transparência

- **cognee pré-existe no `.venv` principal** (pip-install não-declarado de sessão anterior; é a fonte dos
  3 conflitos que o `uv pip check` do core acusa). A Fase 1 **não** o removeu — remover pacotes do
  ambiente que produziu o baseline é, por si só, uma perturbação que merece aprovação própria. Fica como
  recomendação de limpeza para uma etapa gated, não silenciosa.
- **Custo de manutenção:** dois ambientes (core + `.venvs/cognee`) e um contrato subprocess a manter.
- **Não verificado nesta fase:** que o worker subprocess realmente conecte ao Neo4j Option C
  (depende do driver, Fase 3) e que o cognee 1.1.2 rode idêntico em 3.11.14 vs o 3.13 auditado
  (o import smoke passou; o comportamento de runtime se confirma no smoke da Fase 5).
