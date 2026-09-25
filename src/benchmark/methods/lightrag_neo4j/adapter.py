from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from benchmark.core.schemas import Document
from benchmark.ingestion.text_normalization import clean_rag_text
from benchmark.methods.lightrag_neo4j.config_builder import (
    LIGHTRAG_NEO4J_BACKING_METHOD_IDS,
    LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID,
    LIGHTRAG_NEO4J_METHOD_ID,
    LightRAGNeo4jConfig,
    LightRAGNeo4jWorkspace,
    build_config_file,
    ensure_workspace,
    validate_neo4j_env_names,
)
from benchmark.methods.lightrag_neo4j.usage import (
    LightRAGUsageSummary,
    LightRAGUsageTracker,
    extract_prompt_text,
    summary_to_metadata,
    usage_from_query_result,
)


@dataclass(frozen=True)
class LightRAGNeo4jIndexResult:
    document_count: int
    raw_response: dict[str, Any] | str | None = None


@dataclass(frozen=True)
class LightRAGNeo4jQueryResult:
    query: str
    mode: str
    raw_response: dict[str, Any] | list[Any] | str | None
    generation_latency_ms: float | None = None
    usage: LightRAGUsageSummary | None = None


class LightRAGNeo4jClient(Protocol):
    def index(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        documents: Sequence[Document],
    ) -> LightRAGNeo4jIndexResult:
        """Index documents with LightRAG configured for the isolated Neo4j backend."""

    def query(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int,
    ) -> LightRAGNeo4jQueryResult:
        """Query LightRAG using the isolated Neo4j backend."""

    def query_data(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int,
    ) -> LightRAGNeo4jQueryResult:
        """Retrieve LightRAG structured data without final answer generation."""

    def generate(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        system_prompt: str,
    ) -> LightRAGNeo4jQueryResult:
        """Generate an answer from a caller-provided prompt without retrieval."""

    def query_native_context(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int | None,
    ) -> LightRAGNeo4jQueryResult:
        """Return LightRAG's OWN assembled context string (only_need_context=True); no generation."""


class MissingLightRAGNeo4jClient:
    def index(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        documents: Sequence[Document],
    ) -> LightRAGNeo4jIndexResult:
        raise _missing_dependency_error()

    def query(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int,
    ) -> LightRAGNeo4jQueryResult:
        raise _missing_dependency_error()

    def query_data(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int,
    ) -> LightRAGNeo4jQueryResult:
        raise _missing_dependency_error()

    def generate(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        system_prompt: str,
    ) -> LightRAGNeo4jQueryResult:
        raise _missing_dependency_error()

    def query_native_context(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int | None,
    ) -> LightRAGNeo4jQueryResult:
        raise _missing_dependency_error()


@dataclass
class _ReusedRag:
    """A LightRAG instance kept alive across queries plus the mutable holder
    its dispatching ``llm_model_func`` reads per-call (so usage tracking and
    observed-prompt capture stay per-query while the heavy native vdbs load
    only once)."""

    rag: Any
    holder: dict[str, Any]


# Process-level cache of initialized LightRAG instances, keyed by working dir +
# workspace + relevant build flags. Opt-in via LIGHTRAG_REUSE_RAG so default
# behavior (build/init/finalize per query) is byte-identical to before. Without
# this, a batch run reloads the ~2.1 GB native index on every question and the
# transient json.load peak OOM-kills the process (see crash report 2026-06-22).
_REUSED_RAG_CACHE: dict[tuple[Any, ...], _ReusedRag] = {}


def reset_lightrag_rag_cache() -> None:
    """Finalize and drop every process-cached LightRAG instance.

    Used by tests and by callers that want to release the resident native
    index / Neo4j driver explicitly rather than at process exit.
    """
    while _REUSED_RAG_CACHE:
        _key, entry = _REUSED_RAG_CACHE.popitem()
        try:
            _finalize_lightrag_storages(entry.rag)
        except Exception:  # noqa: BLE001
            pass


def _rag_reuse_enabled() -> bool:
    value = os.environ.get("LIGHTRAG_REUSE_RAG")
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


def _make_reusable_llm_func(
    base_llm: Callable[..., Any],
    holder: dict[str, Any],
) -> Callable[..., Any]:
    """Stable LLM func baked once into a cached rag that, per call, routes
    through the *current* query's usage tracker and observed-prompt list (read
    from ``holder``). Mirrors the build-time wrapping order in ``_build_rag``:
    tracker wraps the prompt observer wraps the base."""

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        func = base_llm
        observed = holder.get("observed")
        if observed is not None:
            func = _observe_lightrag_prompt(func, observed_system_prompts=observed)
        tracker = holder.get("tracker")
        if tracker is not None:
            func = tracker.wrap(func)
        return func(*args, **kwargs)

    return dispatch


class RealLightRAGNeo4jClient:
    def __init__(
        self,
        *,
        lightrag_cls: Callable[..., Any] | None = None,
        query_param_cls: Callable[..., Any] | None = None,
        llm_model_func: Callable[..., Any] | None = None,
        embedding_func: Callable[..., Any] | None = None,
    ) -> None:
        self._lightrag_cls = lightrag_cls
        self._query_param_cls = query_param_cls
        self._llm_model_func = llm_model_func
        self._embedding_func = embedding_func

    def index(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        documents: Sequence[Document],
    ) -> LightRAGNeo4jIndexResult:
        with scoped_lightrag_neo4j_env(workspace=workspace, config=config):
            rag = self._build_rag(workspace)
            _initialize_lightrag_storages(rag)
            try:
                texts = [document.text for document in documents]
                ids = [document.document_id for document in documents]
                raw_response = rag.insert(texts, ids=ids)
            finally:
                _finalize_lightrag_storages(rag)
        return LightRAGNeo4jIndexResult(
            document_count=len(documents),
            raw_response=raw_response if raw_response is not None else {"status": "ok"},
        )

    def query(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int,
    ) -> LightRAGNeo4jQueryResult:
        usage_tracker = LightRAGUsageTracker()
        observed_system_prompts: list[str] = []
        with scoped_lightrag_neo4j_env(workspace=workspace, config=config):
            with self._rag_session(
                workspace,
                usage_tracker=usage_tracker,
                observed_system_prompts=observed_system_prompts,
                disable_llm_cache_for_query=config.disable_llm_cache_for_query,
            ) as rag:
                query_param_cls = self._resolve_query_param_cls()
                query_param = query_param_cls(mode=mode, top_k=top_k)
                started = time.perf_counter()
                if callable(getattr(rag, "query_llm", None)):
                    raw_response = rag.query_llm(query, param=query_param)
                else:
                    raw_response = rag.query(query, param=query_param)
                query_latency_ms = (time.perf_counter() - started) * 1000
        raw_response = _attach_observed_final_context(
            raw_response,
            observed_system_prompts=observed_system_prompts,
        )
        tracked_usage = usage_tracker.summary()
        generation_latency_ms = tracked_usage.generation_latency_ms or query_latency_ms
        usage = usage_from_query_result(
            query=query,
            raw_response=raw_response,
            tracked_usage=tracked_usage,
            generation_latency_ms=generation_latency_ms,
        )
        return LightRAGNeo4jQueryResult(
            query=query,
            mode=mode,
            raw_response=raw_response,
            generation_latency_ms=usage.generation_latency_ms,
            usage=usage,
        )

    def query_native_context(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int | None,
    ) -> LightRAGNeo4jQueryResult:
        """native_context: return LightRAG's OWN assembled context STRING (the ``kg_query_context``
        template for graph modes; ``naive_query_context`` = chunks for mode="naive"). Uses
        ``only_need_context=True``, so the lib SKIPS generation and hands the context string back in
        ``llm_response.content`` (no answer is produced). We relocate that string to
        ``raw_response['native_context']`` and DROP ``llm_response`` entirely — nothing generated can
        reach the reader. ``data`` (entities/relationships/chunks) is kept for audit + doc-id
        recovery. ``top_k=None`` uses LightRAG's own default breadth; an int (e.g. 40) preserves
        comparability with the historical chunks/structured runs."""
        usage_tracker = LightRAGUsageTracker()
        with scoped_lightrag_neo4j_env(workspace=workspace, config=config):
            with self._rag_session(
                workspace,
                usage_tracker=usage_tracker,
                observed_system_prompts=None,
                disable_llm_cache_for_query=config.disable_llm_cache_for_query,
            ) as rag:
                query_param_cls = self._resolve_query_param_cls()
                if top_k is None:
                    query_param = query_param_cls(mode=mode, only_need_context=True)
                else:
                    query_param = query_param_cls(mode=mode, top_k=top_k, only_need_context=True)
                if not callable(getattr(rag, "query_llm", None)):
                    raise RuntimeError("Installed LightRAG does not expose query_llm(...)")
                started = time.perf_counter()
                raw = rag.query_llm(query, param=query_param)
                query_latency_ms = (time.perf_counter() - started) * 1000
        # only_need_context => the lib returns the context string in llm_response.content (no
        # generation). Relocate it and drop llm_response so nothing generated can reach the reader.
        context_string = ""
        if isinstance(raw, str):
            context_string = raw
        elif isinstance(raw, dict):
            llm = raw.get("llm_response")
            if isinstance(llm, dict) and isinstance(llm.get("content"), str):
                context_string = llm["content"]
            elif isinstance(raw.get("content"), str):
                context_string = raw["content"]
        data = raw.get("data") if isinstance(raw, dict) and isinstance(raw.get("data"), dict) else {}
        base_meta = (
            raw.get("metadata") if isinstance(raw, dict) and isinstance(raw.get("metadata"), dict) else {}
        )
        clean_raw: dict[str, Any] = {
            "status": (raw.get("status") if isinstance(raw, dict) else None) or "ok",
            "native_context": context_string,
            "data": data,
            "metadata": {
                **base_meta,
                "reader_context_mode": "native_context",
                "only_need_context": True,
                "native_top_k": top_k,
            },
            # llm_response intentionally OMITTED — no generated text can leak into the reader.
        }
        tracked_usage = usage_tracker.summary()
        usage = usage_from_query_result(
            query=query,
            raw_response=raw,
            tracked_usage=tracked_usage,
            generation_latency_ms=tracked_usage.generation_latency_ms or query_latency_ms,
        )
        return LightRAGNeo4jQueryResult(
            query=query,
            mode=mode,
            raw_response=clean_raw,
            generation_latency_ms=usage.generation_latency_ms,
            usage=usage,
        )

    def query_data(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        mode: str,
        top_k: int,
    ) -> LightRAGNeo4jQueryResult:
        usage_tracker = LightRAGUsageTracker()
        with scoped_lightrag_neo4j_env(workspace=workspace, config=config):
            with self._rag_session(
                workspace,
                usage_tracker=usage_tracker,
                observed_system_prompts=None,
                disable_llm_cache_for_query=config.disable_llm_cache_for_query,
            ) as rag:
                query_param_cls = self._resolve_query_param_cls()
                query_param = query_param_cls(mode=mode, top_k=top_k)
                started = time.perf_counter()
                if not callable(getattr(rag, "query_data", None)):
                    raise RuntimeError("Installed LightRAG does not expose query_data(...)")
                raw_response = rag.query_data(query, param=query_param)
                query_latency_ms = (time.perf_counter() - started) * 1000
        tracked_usage = usage_tracker.summary()
        usage = usage_from_query_result(
            query=query,
            raw_response=raw_response,
            tracked_usage=tracked_usage,
            generation_latency_ms=tracked_usage.generation_latency_ms or query_latency_ms,
        )
        return LightRAGNeo4jQueryResult(
            query=query,
            mode=mode,
            raw_response=raw_response,
            generation_latency_ms=usage.generation_latency_ms,
            usage=usage,
        )

    def generate(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig,
        query: str,
        system_prompt: str,
    ) -> LightRAGNeo4jQueryResult:
        usage_tracker = LightRAGUsageTracker()
        observed_system_prompts: list[str] = []
        with scoped_lightrag_neo4j_env(workspace=workspace, config=config):
            with self._rag_session(
                workspace,
                usage_tracker=usage_tracker,
                observed_system_prompts=observed_system_prompts,
                disable_llm_cache_for_query=config.disable_llm_cache_for_query,
            ) as rag:
                query_param_cls = self._resolve_query_param_cls()
                query_param = query_param_cls(mode="bypass", top_k=0)
                started = time.perf_counter()
                if callable(getattr(rag, "query_llm", None)):
                    raw_response = rag.query_llm(query, param=query_param, system_prompt=system_prompt)
                else:
                    answer = rag.query(query, param=query_param, system_prompt=system_prompt)
                    raw_response = {
                        "status": "success",
                        "message": "Bypass mode response",
                        "data": {},
                        "metadata": {"query_mode": "bypass"},
                        "llm_response": {
                            "content": answer if isinstance(answer, str) else "",
                            "response_iterator": None,
                            "is_streaming": False,
                        },
                    }
                query_latency_ms = (time.perf_counter() - started) * 1000
        raw_response = _attach_observed_final_context(
            raw_response,
            observed_system_prompts=observed_system_prompts or [system_prompt],
        )
        tracked_usage = usage_tracker.summary()
        usage = usage_from_query_result(
            query=query,
            raw_response=raw_response,
            tracked_usage=tracked_usage,
            generation_latency_ms=tracked_usage.generation_latency_ms or query_latency_ms,
        )
        return LightRAGNeo4jQueryResult(
            query=query,
            mode="bypass",
            raw_response=raw_response,
            generation_latency_ms=usage.generation_latency_ms,
            usage=usage,
        )

    def _build_rag(
        self,
        workspace: LightRAGNeo4jWorkspace,
        *,
        usage_tracker: LightRAGUsageTracker | None = None,
        observed_system_prompts: list[str] | None = None,
        disable_llm_cache_for_query: bool = False,
    ) -> Any:
        llm_model_func = self._llm_model_func or _default_lightrag_llm_model_func()
        if observed_system_prompts is not None:
            llm_model_func = _observe_lightrag_prompt(
                llm_model_func,
                observed_system_prompts=observed_system_prompts,
            )
        if usage_tracker is not None:
            llm_model_func = usage_tracker.wrap(llm_model_func)
        embedding_func = self._embedding_func or _default_lightrag_embedding_func()
        kwargs: dict[str, Any] = {
            "working_dir": str(workspace.native_dir),
            "workspace": _lightrag_workspace_name(workspace),
            "graph_storage": "Neo4JStorage",
        }
        if llm_model_func is not None:
            kwargs["llm_model_func"] = llm_model_func
        if embedding_func is not None:
            kwargs["embedding_func"] = embedding_func
        if disable_llm_cache_for_query:
            kwargs["enable_llm_cache"] = False
        return self._resolve_lightrag_cls()(**kwargs)

    @contextmanager
    def _rag_session(
        self,
        workspace: LightRAGNeo4jWorkspace,
        *,
        usage_tracker: LightRAGUsageTracker | None,
        observed_system_prompts: list[str] | None,
        disable_llm_cache_for_query: bool,
    ) -> Iterator[Any]:
        """Yield a ready-to-query LightRAG instance.

        Default path: build → initialize → finalize per call (unchanged). When
        LIGHTRAG_REUSE_RAG is set, yield a process-cached instance instead and
        only swap in this call's usage tracker / observed-prompt list so the
        heavy native index loads exactly once per process.
        """
        if not _rag_reuse_enabled():
            rag = self._build_rag(
                workspace,
                usage_tracker=usage_tracker,
                observed_system_prompts=observed_system_prompts,
                disable_llm_cache_for_query=disable_llm_cache_for_query,
            )
            _initialize_lightrag_storages(rag)
            try:
                yield rag
            finally:
                _finalize_lightrag_storages(rag)
            return

        entry = self._get_or_build_reusable_rag(
            workspace,
            disable_llm_cache_for_query=disable_llm_cache_for_query,
        )
        entry.holder["tracker"] = usage_tracker
        entry.holder["observed"] = observed_system_prompts
        try:
            yield entry.rag
        finally:
            entry.holder["tracker"] = None
            entry.holder["observed"] = None

    def _get_or_build_reusable_rag(
        self,
        workspace: LightRAGNeo4jWorkspace,
        *,
        disable_llm_cache_for_query: bool,
    ) -> _ReusedRag:
        key = (
            str(workspace.native_dir),
            _lightrag_workspace_name(workspace),
            bool(disable_llm_cache_for_query),
            id(self._lightrag_cls),
            id(self._llm_model_func),
            id(self._embedding_func),
        )
        entry = _REUSED_RAG_CACHE.get(key)
        if entry is None:
            holder: dict[str, Any] = {"tracker": None, "observed": None}
            rag = self._build_reusable_rag(
                workspace,
                holder=holder,
                disable_llm_cache_for_query=disable_llm_cache_for_query,
            )
            _initialize_lightrag_storages(rag)
            entry = _ReusedRag(rag=rag, holder=holder)
            _REUSED_RAG_CACHE[key] = entry
        return entry

    def _build_reusable_rag(
        self,
        workspace: LightRAGNeo4jWorkspace,
        *,
        holder: dict[str, Any],
        disable_llm_cache_for_query: bool,
    ) -> Any:
        base_llm = self._llm_model_func or _default_lightrag_llm_model_func()
        embedding_func = self._embedding_func or _default_lightrag_embedding_func()
        kwargs: dict[str, Any] = {
            "working_dir": str(workspace.native_dir),
            "workspace": _lightrag_workspace_name(workspace),
            "graph_storage": "Neo4JStorage",
        }
        if base_llm is not None:
            kwargs["llm_model_func"] = _make_reusable_llm_func(base_llm, holder)
        if embedding_func is not None:
            kwargs["embedding_func"] = embedding_func
        if disable_llm_cache_for_query:
            kwargs["enable_llm_cache"] = False
        return self._resolve_lightrag_cls()(**kwargs)

    def _resolve_lightrag_cls(self) -> Callable[..., Any]:
        if self._lightrag_cls is not None:
            return self._lightrag_cls
        try:
            from lightrag import LightRAG
        except ImportError as exc:
            raise RuntimeError(
                "lightrag-hku is required for lightrag_neo4j live execution. "
                "Install lightrag-hku in the main project environment before running "
                "real LightRAG Neo4j indexing or querying."
            ) from exc
        return LightRAG

    def _resolve_query_param_cls(self) -> Callable[..., Any]:
        if self._query_param_cls is not None:
            return self._query_param_cls
        try:
            from lightrag import QueryParam
        except ImportError as exc:
            raise RuntimeError(
                "lightrag-hku is required for lightrag_neo4j live execution. "
                "Install lightrag-hku in the main project environment before running "
                "real LightRAG Neo4j queries."
            ) from exc
        return QueryParam


class LightRAGNeo4jAdapter:
    def __init__(
        self,
        *,
        workspace: LightRAGNeo4jWorkspace,
        config: LightRAGNeo4jConfig | None = None,
        client: LightRAGNeo4jClient | None = None,
    ) -> None:
        if workspace.method_id not in LIGHTRAG_NEO4J_BACKING_METHOD_IDS:
            raise ValueError(
                "LightRAG Neo4j adapter only supports method_id='lightrag_neo4j' "
                "or 'lightrag_neo4j_doc_scoped_strict'"
            )
        self.workspace = workspace
        self.config = config or LightRAGNeo4jConfig()
        validate_neo4j_env_names(self.config)
        self.client = client or RealLightRAGNeo4jClient()

    def prepare_inputs(self, documents: Sequence[Document]) -> list[Path]:
        ensure_workspace(self.workspace)
        paths: list[Path] = []
        manifest: list[dict[str, str]] = []
        for document in documents:
            path = self.workspace.input_dir / f"{document.document_id}.txt"
            path.write_text(clean_rag_text(document.text), encoding="utf-8")
            paths.append(path)
            manifest.append(
                {
                    "document_id": document.document_id,
                    "dataset_id": document.dataset_id,
                    "dataset_version": document.dataset_version,
                    "path": str(path),
                }
            )
        (self.workspace.input_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return paths

    def index(self, *, documents: Sequence[Document]) -> LightRAGNeo4jIndexResult:
        cleaned_documents = _clean_documents_for_rag(documents)
        self.prepare_inputs(cleaned_documents)
        build_config_file(self.workspace, config=self.config)
        result = self.client.index(
            workspace=self.workspace,
            config=self.config,
            documents=cleaned_documents,
        )
        self._preserve_raw("index", result.raw_response)
        if result.document_count != len(documents):
            raise RuntimeError("LightRAG Neo4j indexing reported an unexpected document count")
        return result

    def query(
        self,
        *,
        query: str,
        mode: str = "mix",
        top_k: int = 5,
    ) -> LightRAGNeo4jQueryResult:
        ensure_workspace(self.workspace)
        build_config_file(self.workspace, config=self.config)
        started = time.perf_counter()
        result = self.client.query(
            workspace=self.workspace,
            config=self.config,
            query=query,
            mode=mode,
            top_k=top_k,
        )
        query_latency_ms = (time.perf_counter() - started) * 1000
        if result.usage is None:
            usage = usage_from_query_result(
                query=query,
                raw_response=result.raw_response,
                tracked_usage=LightRAGUsageTracker().summary(),
                generation_latency_ms=result.generation_latency_ms or query_latency_ms,
            )
            result = LightRAGNeo4jQueryResult(
                query=result.query,
                mode=result.mode,
                raw_response=result.raw_response,
                generation_latency_ms=usage.generation_latency_ms,
                usage=usage,
            )
        self._preserve_raw(f"query_{mode}", result.raw_response)
        return result

    def query_native_context(
        self,
        *,
        query: str,
        mode: str = "mix",
        top_k: int | None = 40,
    ) -> LightRAGNeo4jQueryResult:
        ensure_workspace(self.workspace)
        build_config_file(self.workspace, config=self.config)
        started = time.perf_counter()
        result = self.client.query_native_context(
            workspace=self.workspace,
            config=self.config,
            query=query,
            mode=mode,
            top_k=top_k,
        )
        query_latency_ms = (time.perf_counter() - started) * 1000
        if result.usage is None:
            usage = usage_from_query_result(
                query=query,
                raw_response=result.raw_response,
                tracked_usage=LightRAGUsageTracker().summary(),
                generation_latency_ms=result.generation_latency_ms or query_latency_ms,
            )
            result = LightRAGNeo4jQueryResult(
                query=result.query,
                mode=result.mode,
                raw_response=result.raw_response,
                generation_latency_ms=usage.generation_latency_ms,
                usage=usage,
            )
        self._preserve_raw(f"native_context_{mode}", result.raw_response)
        return result

    def query_data(
        self,
        *,
        query: str,
        mode: str = "mix",
        top_k: int = 5,
    ) -> LightRAGNeo4jQueryResult:
        ensure_workspace(self.workspace)
        build_config_file(self.workspace, config=self.config)
        started = time.perf_counter()
        result = self.client.query_data(
            workspace=self.workspace,
            config=self.config,
            query=query,
            mode=mode,
            top_k=top_k,
        )
        query_latency_ms = (time.perf_counter() - started) * 1000
        if result.usage is None:
            usage = usage_from_query_result(
                query=query,
                raw_response=result.raw_response,
                tracked_usage=LightRAGUsageTracker().summary(),
                generation_latency_ms=result.generation_latency_ms or query_latency_ms,
            )
            result = LightRAGNeo4jQueryResult(
                query=result.query,
                mode=result.mode,
                raw_response=result.raw_response,
                generation_latency_ms=usage.generation_latency_ms,
                usage=usage,
            )
        self._preserve_raw(f"query_data_{mode}", result.raw_response)
        return result

    def generate(
        self,
        *,
        query: str,
        system_prompt: str,
    ) -> LightRAGNeo4jQueryResult:
        ensure_workspace(self.workspace)
        build_config_file(self.workspace, config=self.config)
        started = time.perf_counter()
        result = self.client.generate(
            workspace=self.workspace,
            config=self.config,
            query=query,
            system_prompt=system_prompt,
        )
        query_latency_ms = (time.perf_counter() - started) * 1000
        if result.usage is None:
            usage = usage_from_query_result(
                query=query,
                raw_response=result.raw_response,
                tracked_usage=LightRAGUsageTracker().summary(),
                generation_latency_ms=result.generation_latency_ms or query_latency_ms,
            )
            result = LightRAGNeo4jQueryResult(
                query=result.query,
                mode=result.mode,
                raw_response=result.raw_response,
                generation_latency_ms=usage.generation_latency_ms,
                usage=usage,
            )
        self._preserve_raw("generate_doc_scoped_strict", result.raw_response)
        return result

    @staticmethod
    def usage_metadata(result: LightRAGNeo4jQueryResult) -> dict[str, Any]:
        if result.usage is None:
            return {}
        return summary_to_metadata(result.usage)

    def _preserve_raw(self, stem: str, raw_response: object) -> None:
        ensure_workspace(self.workspace)
        path = self.workspace.raw_dir / f"{stem}_result.json"
        path.write_text(json.dumps(raw_response, sort_keys=True, default=str), encoding="utf-8")


def _missing_dependency_error() -> RuntimeError:
    return RuntimeError(
        "lightrag_neo4j live execution is not configured. This phase provides only the "
        "offline adapter boundary. Validate a LightRAG runtime with Neo4j storage support "
        "before running real Neo4j-backed ingestion or querying."
    )


def _attach_observed_final_context(
    raw_response: dict[str, Any] | list[Any] | str | None,
    *,
    observed_system_prompts: list[str],
) -> dict[str, Any] | list[Any] | str | None:
    if not isinstance(raw_response, dict) or not observed_system_prompts:
        return raw_response
    trace = raw_response.get("retrieval_trace")
    if not isinstance(trace, dict):
        trace = {}
    trace.setdefault(
        "final_context",
        {
            "type": "observed_system_prompt",
            "text": observed_system_prompts[-1],
        },
    )
    raw_response = dict(raw_response)
    raw_response["retrieval_trace"] = trace
    return raw_response


def _observe_lightrag_prompt(
    llm_model_func: Callable[..., Any] | None,
    *,
    observed_system_prompts: list[str],
) -> Callable[..., Any] | None:
    if llm_model_func is None:
        return None

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        system_prompt = kwargs.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt:
            observed_system_prompts.append(system_prompt)
        else:
            prompt_text = extract_prompt_text(args=args, kwargs=kwargs)
            if prompt_text:
                observed_system_prompts.append(prompt_text)
        return llm_model_func(*args, **kwargs)

    return wrapped


def _clean_documents_for_rag(documents: Sequence[Document]) -> list[Document]:
    return [
        document.model_copy(update={"text": clean_rag_text(document.text)})
        for document in documents
    ]


def _default_lightrag_llm_model_func() -> Callable[..., Any] | None:
    if os.environ.get("MODEL_PROVIDER", "").lower() != "openai":
        return None
    try:
        from lightrag.llm.openai import gpt_4o_mini_complete
    except ImportError as exc:
        raise RuntimeError(
            "lightrag_neo4j with MODEL_PROVIDER=openai requires LightRAG's OpenAI "
            "LLM helper."
        ) from exc
    return gpt_4o_mini_complete


def _default_lightrag_embedding_func() -> Callable[..., Any] | None:
    if os.environ.get("MODEL_PROVIDER", "").lower() != "openai":
        return None
    try:
        from lightrag.llm.openai import openai_embed
    except ImportError as exc:
        raise RuntimeError(
            "lightrag_neo4j with MODEL_PROVIDER=openai requires LightRAG's OpenAI "
            "embedding helper."
        ) from exc
    return openai_embed


def _initialize_lightrag_storages(rag: Any) -> None:
    initializer = getattr(rag, "initialize_storages", None)
    if initializer is None:
        return
    _run_lightrag_async(initializer())


def _finalize_lightrag_storages(rag: Any) -> None:
    finalizer = getattr(rag, "finalize_storages", None)
    if finalizer is None:
        return
    _run_lightrag_async(finalizer())


def _run_lightrag_async(awaitable: Any) -> None:
    try:
        from lightrag.utils import always_get_an_event_loop
    except ImportError as exc:
        raise RuntimeError("LightRAG async lifecycle helper is unavailable.") from exc
    loop = always_get_an_event_loop()
    loop.run_until_complete(awaitable)


@contextmanager
def scoped_lightrag_neo4j_env(
    *,
    workspace: LightRAGNeo4jWorkspace,
    config: LightRAGNeo4jConfig,
) -> Iterator[None]:
    validate_neo4j_env_names(config)
    values = _resolve_lightag_neo4j_env_values(workspace=workspace, config=config)
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _resolve_lightag_neo4j_env_values(
    *,
    workspace: LightRAGNeo4jWorkspace,
    config: LightRAGNeo4jConfig,
) -> dict[str, str]:
    # Each value prefers an Option C explicit override on the config; only when
    # absent does it fall back to the LightRAG-specific *_env variable. This
    # keeps a per-dataset isolated container (resolved from the port registry)
    # from being overridden by a global LIGHTRAG_NEO4J_URI.
    resolved = {
        "NEO4J_URI": config.neo4j_uri or os.environ.get(config.neo4j_uri_env),
        "NEO4J_USERNAME": config.neo4j_user or os.environ.get(config.neo4j_user_env),
        "NEO4J_PASSWORD": config.neo4j_password or os.environ.get(config.neo4j_password_env),
        "NEO4J_DATABASE": config.neo4j_database or os.environ.get(config.neo4j_database_env),
    }
    missing = [
        env_name
        for env_name, value in (
            (config.neo4j_uri_env, resolved["NEO4J_URI"]),
            (config.neo4j_user_env, resolved["NEO4J_USERNAME"]),
            (config.neo4j_password_env, resolved["NEO4J_PASSWORD"]),
            (config.neo4j_database_env, resolved["NEO4J_DATABASE"]),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "lightrag_neo4j requires LightRAG-specific Neo4j environment variables "
            f"(or explicit config overrides): {', '.join(missing)}"
        )
    return {
        **resolved,
        "NEO4J_WORKSPACE": _lightrag_workspace_name(workspace),
    }


def _lightrag_workspace_name(workspace: LightRAGNeo4jWorkspace) -> str:
    method_id = (
        LIGHTRAG_NEO4J_METHOD_ID
        if workspace.method_id == LIGHTRAG_NEO4J_DOC_SCOPED_STRICT_METHOD_ID
        else workspace.method_id
    )
    return f"{method_id}_{workspace.dataset_id}_{workspace.dataset_version}"
