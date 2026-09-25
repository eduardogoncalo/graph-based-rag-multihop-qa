from __future__ import annotations

import os
import time

from benchmark.core.schemas import RetrievalResult
from benchmark.methods.lightrag_neo4j.adapter import LightRAGNeo4jAdapter
from benchmark.methods.lightrag_neo4j.config_builder import (
    LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
    LIGHTRAG_NEO4J_METHOD_ID,
    READER_CONTEXT_CHUNKS,
    READER_CONTEXT_NATIVE,
    READER_CONTEXT_STRUCTURED,
    LightRAGReaderContext,
)
from benchmark.methods.lightrag_neo4j.document_scope import (
    NOT_ENOUGH_INFORMATION,
    apply_document_scope,
    build_scoped_system_prompt,
)
from benchmark.methods.lightrag_neo4j.output_parser import parse_query_output
from benchmark.storage.experiment_store import ExperimentStore


def retrieve(
    *,
    adapter: LightRAGNeo4jAdapter,
    query: str,
    top_k: int = 5,
    query_mode: str = "mix",
    experiment_store: ExperimentStore | None = None,
    run_id: str | None = None,
    target_document_id: str | None = None,
    method_id: str = LIGHTRAG_NEO4J_METHOD_ID,
    chunk_document_map: dict[str, str] | None = None,
    context_mode: LightRAGReaderContext | None = None,
) -> RetrievalResult:
    started = time.perf_counter()
    # Reader-context mode: explicit arg wins, else the adapter's config field
    # (env-resolved). The doc-scoped-strict variant builds its own final_context,
    # so it always parses as chunks regardless of the knob.
    resolved_context = context_mode or getattr(
        getattr(adapter, "config", None), "reader_context", READER_CONTEXT_CHUNKS
    )
    if method_id == LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID:
        raw_result = _retrieve_doc_scoped_strict(
            adapter=adapter,
            query=query,
            query_mode=query_mode,
            top_k=top_k,
            target_document_id=target_document_id,
            chunk_document_map=chunk_document_map,
        )
        parse_context = READER_CONTEXT_CHUNKS
    elif resolved_context == READER_CONTEXT_NATIVE:
        # LightRAG's OWN assembled context string (only_need_context=True): a single block to
        # the reader (like cognee's blob), no native generation, so llm_response never reaches
        # the reader. top_k parameterizable via LIGHTRAG_NATIVE_TOPK (default = passed top_k).
        raw_result = adapter.query_native_context(
            query=query, mode=query_mode, top_k=_resolve_native_topk(top_k)
        )
        parse_context = READER_CONTEXT_NATIVE
    elif resolved_context == READER_CONTEXT_STRUCTURED:
        # Full graph context (entities + relationships + chunks), no native
        # generation — the graph method's actual retrieved context.
        raw_result = adapter.query_data(query=query, mode=query_mode, top_k=top_k)
        parse_context = READER_CONTEXT_STRUCTURED
    else:
        raw_result = adapter.query(query=query, mode=query_mode, top_k=top_k)
        parse_context = READER_CONTEXT_CHUNKS
    query_latency_ms = (time.perf_counter() - started) * 1000
    result = parse_query_output(
        query=query,
        raw_response=raw_result.raw_response,
        query_mode=query_mode,
        latency_ms=query_latency_ms,
        top_k=top_k,
        method_id=method_id,
        context_mode=parse_context,
    )
    usage_metadata = adapter.usage_metadata(raw_result)
    if usage_metadata:
        result = result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "lightrag_usage": usage_metadata,
                    "lightrag_query_latency_ms": query_latency_ms,
                }
            }
        )
    cache_metadata = _cache_control_metadata(adapter)
    if cache_metadata:
        result = result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    **cache_metadata,
                }
            }
        )
    trace_payload = _extract_trace_payload(raw_result.raw_response)
    if trace_payload is not None:
        result = result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                        "retrieval_trace": {
                            "framework_id": "lightrag_neo4j",
                            "query_mode": query_mode,
                            "retrieval_strategy": f"graph_rag_{query_mode}",
                        **trace_payload,
                    },
                }
            }
        )
    if experiment_store is not None:
        experiment_store.persist_retrieval_result(
            result=result,
            dataset_id=adapter.workspace.dataset_id,
            dataset_version=adapter.workspace.dataset_version,
            top_k=top_k,
            run_id=run_id,
        )
    return result


def _resolve_native_topk(default_top_k: int) -> int | None:
    """top_k for ``native_context``, parameterizable and documented.

    ``LIGHTRAG_NATIVE_TOPK`` env:
      - unset            -> ``default_top_k`` (the caller's top_k, e.g. 40): PRESERVES
                            comparability with the historical chunks/structured runs.
      - "default"/"none"/"lib"/"libdefault" -> ``None``: LightRAG's OWN default retrieval
                            breadth (the real lib default, not pinned to 40).
      - an integer       -> that int.
    """
    raw = os.environ.get("LIGHTRAG_NATIVE_TOPK")
    if raw is None:
        return default_top_k
    value = raw.strip().lower()
    if value in {"", "default", "none", "lib", "libdefault", "lib_default"}:
        return None
    try:
        return int(value)
    except ValueError:
        return default_top_k


def _retrieve_doc_scoped_strict(
    *,
    adapter: LightRAGNeo4jAdapter,
    query: str,
    query_mode: str,
    top_k: int,
    target_document_id: str | None,
    chunk_document_map: dict[str, str] | None,
):
    if not target_document_id:
        raise ValueError("target_document_id is required for lightrag_neo4j_doc_scoped_strict")
    data_result = adapter.query_data(query=query, mode=query_mode, top_k=top_k)
    raw_data = data_result.raw_response if isinstance(data_result.raw_response, dict) else {}
    scoped = apply_document_scope(
        raw_data,
        target_document_id=target_document_id,
        chunk_document_map=chunk_document_map,
    )
    chunks_after_scope = scoped.scope_metadata["chunks_after_scope"]
    if chunks_after_scope:
        system_prompt = build_scoped_system_prompt(scoped.final_context)
        generation_result = adapter.generate(query=query, system_prompt=system_prompt)
        answer = _extract_llm_answer(generation_result.raw_response)
        usage = generation_result.usage
        generation_latency_ms = generation_result.generation_latency_ms
        generation_skipped = False
        generation_skip_reason = None
    else:
        answer = NOT_ENOUGH_INFORMATION
        usage = data_result.usage
        generation_latency_ms = data_result.generation_latency_ms
        generation_skipped = True
        generation_skip_reason = "empty_document_scoped_context"
    metadata = {
        **(raw_data.get("metadata") if isinstance(raw_data.get("metadata"), dict) else {}),
        "query_mode": query_mode,
        "method_variant": LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
        **scoped.scope_metadata,
    }
    if generation_skipped:
        metadata.update(
            {
                "generation_skipped": True,
                "generation_skip_reason": generation_skip_reason,
            }
        )
    trace_metadata = dict(scoped.scope_metadata)
    if generation_skipped:
        trace_metadata.update(
            {
                "generation_skipped": True,
                "generation_skip_reason": generation_skip_reason,
            }
        )
    raw_response = {
        **scoped.raw_data,
        "status": "success" if chunks_after_scope else "failure",
        "message": "Document-scoped strict query processed",
        "metadata": metadata,
        "llm_response": {
            "content": answer,
            "response_iterator": None,
            "is_streaming": False,
        },
        "retrieval_trace": {
            "final_context": {
                "type": "document_scoped_context",
                "text": scoped.final_context,
            },
            **trace_metadata,
        },
    }
    return type(data_result)(
        query=data_result.query,
        mode=query_mode,
        raw_response=raw_response,
        generation_latency_ms=generation_latency_ms,
        usage=usage,
    )


def _extract_llm_answer(raw_response: object) -> str:
    if isinstance(raw_response, str):
        return raw_response.strip()
    if isinstance(raw_response, dict):
        llm_response = raw_response.get("llm_response")
        if isinstance(llm_response, dict) and isinstance(llm_response.get("content"), str):
            return str(llm_response["content"]).strip()
        for key in ("answer", "response", "result", "output"):
            value = raw_response.get(key)
            if isinstance(value, str):
                return value.strip()
    return ""


def _extract_trace_payload(raw_response: object) -> dict[str, object] | None:
    if not isinstance(raw_response, dict):
        return None
    data = raw_response.get("data")
    if isinstance(data, dict):
        metadata = raw_response.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        explicit_trace = raw_response.get("retrieval_trace")
        explicit_trace = explicit_trace if isinstance(explicit_trace, dict) else {}
        final_context = explicit_trace.get("final_context")
        payload: dict[str, object] = {
            "nodes": data.get("entities") if isinstance(data.get("entities"), list) else [],
            "relationships": data.get("relationships")
            if isinstance(data.get("relationships"), list)
            else [],
            "chunks": data.get("chunks") if isinstance(data.get("chunks"), list) else [],
            "query_mode": metadata.get("query_mode"),
            "raw_trace": {
                "data": data,
                "metadata": metadata,
            },
        }
        if isinstance(final_context, dict):
            payload["final_context"] = final_context
        payload.update(
            {
                key: value
                for key, value in explicit_trace.items()
                if key not in {"nodes", "entities", "relationships", "chunks", "raw_trace"}
            }
        )
        payload["trace_status"] = (
            "complete"
            if payload["nodes"]
            and payload["relationships"]
            and payload["chunks"]
            and payload.get("final_context")
            else "partial"
        )
        return {key: value for key, value in payload.items() if value is not None}
    for key in ("retrieval_trace", "trace", "context_trace"):
        value = raw_response.get(key)
        if isinstance(value, dict):
            return value
    trace_keys = {
        "nodes",
        "entities",
        "relationships",
        "relations",
        "edges",
        "chunks",
        "context_chunks",
        "final_context",
        "context",
        "assembled_context",
    }
    if any(key in raw_response for key in trace_keys):
        return {key: value for key, value in raw_response.items() if key in trace_keys}
    return None


def _cache_control_metadata(adapter: LightRAGNeo4jAdapter) -> dict[str, object]:
    config = getattr(adapter, "config", None)
    disabled = bool(getattr(config, "disable_llm_cache_for_query", False))
    if not disabled:
        return {}
    source = "env" if _env_truthy("LIGHTRAG_DISABLE_LLM_CACHE") else "config"
    return {
        "llm_cache_disabled": True,
        "cache_control_source": source,
        "instrumentation_version": "lightrag_trace_v1",
    }


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})
