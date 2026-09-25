# Minimal Retrieval Trace Specification

## Objective

Design a simple, framework-neutral way to record what evidence was retrieved and sent to the answering step for each benchmark question.

The immediate goal is not to build a complex graph analytics schema.

The immediate goal is to answer:

```text
For this question, which nodes, relationships, chunks, and context were retrieved or sent to the agent/model?
```

This should work first for LightRAG Neo4j, but the design should remain simple enough to support future Cognee and GraphRAG implementations.

---

## Current Problem

The current LightRAG execution stores an aggregated retrieval item with metadata similar to:

```json
{
  "kind": "answer",
  "method_id": "lightrag_neo4j",
  "query_mode": "mix"
}
```

This is not enough to understand what LightRAG actually retrieved.

We need to know, for each question:

* which graph nodes/entities were retrieved;
* which relationships were retrieved;
* which chunks or context blocks were sent to the answer generation step;
* what final context was used;
* whether the trace is complete, partial, or unavailable.

---

## Important Design Principle

Do not over-engineer this.

At this stage, prefer a minimal design with one trace record per run/question/method.

Avoid creating many relational tables unless there is a strong reason.

A simple JSONB-based trace is acceptable and probably preferable.

---

## Proposed Minimal Table

Create one generic table, for example:

```text
retrieval_traces
```

One row per `run_id`.

Suggested fields:

```text
trace_id
run_id
method_id
framework_id
dataset_id
dataset_version
question_id
query_text
query_mode
retrieval_strategy
retrieved_nodes jsonb
retrieved_relationships jsonb
retrieved_chunks jsonb
final_context jsonb
raw_trace jsonb
trace_status
metadata jsonb
created_at
```

Suggested `trace_status` values:

```text
complete
partial
missing
not_supported
error
```

---

## JSON Structure

### retrieved_nodes

A list of nodes/entities recovered during retrieval.

Example:

```json
[
  {
    "id": "entity_123",
    "label": "Termination",
    "type": "clause_topic",
    "score": 0.82,
    "rank": 1,
    "source": "local",
    "metadata": {
      "framework": "lightrag"
    }
  }
]
```

### retrieved_relationships

A list of relationships/edges recovered during retrieval.

Example:

```json
[
  {
    "id": "rel_456",
    "source": "entity_123",
    "target": "entity_789",
    "type": "RELATED_TO",
    "label": "termination provision relates to agreement obligations",
    "score": 0.67,
    "rank": 1,
    "metadata": {
      "framework": "lightrag"
    }
  }
]
```

### retrieved_chunks

A list of text chunks or source snippets recovered or used as context.

Example:

```json
[
  {
    "chunk_id": "chunk_001",
    "document_id": "doc_01",
    "rank": 1,
    "score": 0.91,
    "text": "...",
    "source": "final_context",
    "metadata": {
      "framework": "lightrag"
    }
  }
]
```

### final_context

The final assembled context sent to the answer generation step, when available.

Example:

```json
{
  "text": "...",
  "token_count": 1320,
  "source": "assembled_context",
  "metadata": {
    "query_mode": "mix"
  }
}
```

### raw_trace

Raw framework-specific trace information.

This should preserve original data from LightRAG, Cognee or GraphRAG when available, without forcing it into the minimal schema.

---

## Mapping for LightRAG Neo4j

LightRAG should attempt to populate:

```text
retrieved_nodes          -> LightRAG entities/nodes
retrieved_relationships  -> LightRAG relationships/edges
retrieved_chunks         -> source chunks or text units used in context
final_context            -> final context sent to generation, if accessible
raw_trace                -> any raw or partially parsed LightRAG trace
```

If LightRAG does not expose these directly, the implementation plan should explain whether capture requires:

* parsing LightRAG output;
* instrumenting the adapter;
* instrumenting the LLM/context function;
* using callbacks/hooks if available;
* controlled monkey patching;
* accepting partial trace coverage.

---

## Mapping for Future GraphRAG

GraphRAG should be able to use the same table.

Possible mapping:

```text
retrieved_nodes          -> entities
retrieved_relationships  -> relationships
retrieved_chunks         -> source chunks
final_context            -> context passed to answer generation
raw_trace                -> communities, reports, claims, source mappings
```

If communities are available, they can be stored either in `retrieved_nodes` with `type = "community"` or inside `raw_trace`.

Do not create a separate communities table yet unless truly necessary.

---

## Mapping for Future Cognee

Cognee should be able to use the same table.

Possible mapping:

```text
retrieved_nodes          -> Cognee nodes or memory nodes
retrieved_relationships  -> Cognee edges
retrieved_chunks         -> source text or memory/context snippets
final_context            -> context sent to the model/agent
raw_trace                -> Cognee-specific retrieval output
```

The design should not require Cognee implementation now.

---

## Relationship With Existing retrieval_items

Do not remove or break existing `retrieval_items`.

Current behavior should remain:

```text
Vector RAG -> retrieval_items with chunks
LightRAG   -> existing aggregated item may remain, but must not be treated as top-k chunk retrieval
```

The new `retrieval_traces` table is an additional audit layer.

For Vector RAG, a future simple trace could store:

```text
retrieved_chunks = top-k chunks
retrieved_nodes = []
retrieved_relationships = []
```

But this is optional for now.

---

## Required Feedback Before Implementation

Before implementing, provide a critical feedback report.

The feedback should answer:

1. Is this JSONB one-row-per-run approach enough to answer the main question?

```text
Which nodes and relationships were retrieved for this question?
```

2. Can this be simplified even further?

3. Is a new table necessary, or can this be stored safely inside existing `retrieval_results` or `retrieval_items.metadata`?

4. What are the trade-offs between:

   * one JSONB row per run;
   * many rows, one per retrieved item;
   * using existing `retrieval_items`;
   * creating normalized trace tables later?

5. What is the minimal implementation that delivers the objective?

6. What can be done first for LightRAG without blocking future Cognee and GraphRAG support?

7. What fields are essential now, and what fields should be deferred?

8. What should be included in `raw_trace` to avoid losing framework-specific information?

9. How can reports later show:

   * trace available;
   * trace partial;
   * trace missing;
   * number of retrieved nodes;
   * number of retrieved relationships;
   * number of retrieved chunks?

10. What tests can validate this without calling real LightRAG, Cognee, GraphRAG, OpenAI or Neo4j?

---

## Optional Research

Before proposing the final plan, check how other RAG or graph-RAG systems log retrieval traces, retrieved nodes, retrieved documents, graph context or agent context.

Search only for design inspiration.

Do not copy external code.

The final recommendation should be adapted to this project’s benchmark architecture.

Potential topics to investigate:

```text
RAG retrieval trace logging
GraphRAG retrieved context logging
LightRAG retrieved entities relationships context
Neo4j graph RAG retrieval audit
LangChain retrieved documents metadata logging
LlamaIndex retrieval trace source nodes
```

Summarize any useful ideas briefly and cite sources if web research is used.

---

## Implementation Constraints

Do not implement yet.

Do not run benchmark.

Do not run questions.

Do not call OpenAI.

Do not call LightRAG real.

Do not call Cognee.

Do not call GraphRAG.

Do not alter Neo4j.

Do not reindex.

Do not delete artifacts.

If database inspection is needed, use read-only queries only and ask before making schema changes.

---

## Expected Output of This Planning Phase

Return a plan with:

1. Critical assessment of this simplified design.
2. Suggested minimal schema.
3. Whether to use a new table or existing tables.
4. How to map LightRAG to the design.
5. How the same design can later support Cognee and GraphRAG.
6. Simplifications you recommend.
7. Risks.
8. Testing strategy with fakes/mocks.
9. Exact implementation order.
10. Clear statement of what will not be implemented yet.

No code changes should be made in this planning phase.
