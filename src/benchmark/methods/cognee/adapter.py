from __future__ import annotations

import importlib
import inspect
import logging
import os
import time
import traceback
from contextlib import contextmanager
from typing import Any, Protocol

from benchmark.core.schemas import RetrievalResult
from benchmark.methods.cognee.config_builder import (
    CogneeConfig,
    CogneeWorkspace,
    cognee_env,
    ensure_workspace,
    validate_environment_isolation,
)
from benchmark.methods.cognee.parser import parse_cognee_result

logger = logging.getLogger(__name__)


def _error_entry(api: str, exc: Exception) -> dict[str, str]:
    """Rich error record so a retrieval failure is diagnosable (the raw neo4j
    async error stringifies to an opaque object repr)."""
    return {
        "api": api,
        "error_type": type(exc).__name__,
        "error": repr(exc) or str(exc),
        "traceback": traceback.format_exc(),
    }


class CogneeClient(Protocol):
    async def remember(self, data: Any, *, dataset_name: str, **kwargs: Any) -> Any: ...

    async def recall(
        self,
        query: str,
        *,
        top_k: int,
        dataset_name: str,
        dataset_names: tuple[str, ...] | None = None,
        only_context: bool = True,
    ) -> Any: ...

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        dataset_name: str,
        dataset_names: tuple[str, ...] | None = None,
        only_context: bool = True,
    ) -> Any: ...

    async def add(self, data: Any, *, dataset_name: str, **kwargs: Any) -> Any: ...

    async def cognify(self, *, dataset_name: str, **kwargs: Any) -> Any: ...


class RealCogneeClient:
    def __init__(self) -> None:
        self.module = importlib.import_module("cognee")

    async def remember(self, data: Any, *, dataset_name: str, **kwargs: Any) -> Any:
        return await _maybe_await(
            self.module.remember(
                data,
                dataset_name=dataset_name,
                run_in_background=False,
                **kwargs,
            )
        )

    async def recall(
        self,
        query: str,
        *,
        top_k: int,
        dataset_name: str,
        dataset_names: tuple[str, ...] | None = None,
        only_context: bool = True,
    ) -> Any:
        candidate = getattr(self.module, "recall", None)
        if not callable(candidate):
            raise RuntimeError("Installed cognee package does not expose recall(...)")
        datasets = list(dataset_names or (dataset_name,))
        return await _maybe_await(
            candidate(
                query,
                datasets=datasets,
                top_k=top_k,
                only_context=only_context,
                auto_route=False,
                scope="graph",
            )
        )

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        dataset_name: str,
        dataset_names: tuple[str, ...] | None = None,
        only_context: bool = True,
    ) -> Any:
        candidate = getattr(self.module, "search", None)
        if not callable(candidate):
            raise RuntimeError("Installed cognee package does not expose search(...)")
        search_type = getattr(getattr(self.module, "SearchType", None), "CHUNKS", None)
        kwargs: dict[str, Any] = {
            "datasets": list(dataset_names or (dataset_name,)),
            "top_k": top_k,
            "only_context": only_context,
        }
        if search_type is not None:
            kwargs["query_type"] = search_type
        return await _maybe_await(candidate(query, **kwargs))

    async def add(self, data: Any, *, dataset_name: str, **kwargs: Any) -> Any:
        return await _maybe_await(self.module.add(data, dataset_name=dataset_name, **kwargs))

    async def cognify(self, *, dataset_name: str, **kwargs: Any) -> Any:
        return await _maybe_await(self.module.cognify(datasets=[dataset_name], **kwargs))


class CogneeAdapter:
    def __init__(
        self,
        *,
        workspace: CogneeWorkspace,
        config: CogneeConfig | None = None,
        client: CogneeClient | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config or CogneeConfig()
        self.client = client

    def retrieve(self, *, query: str, top_k: int | None = None) -> RetrievalResult:
        return _run_sync(self.retrieve_async(query=query, top_k=top_k))

    async def retrieve_async(self, *, query: str, top_k: int | None = None) -> RetrievalResult:
        ensure_workspace(self.workspace)
        resolved_top_k = top_k or self.config.top_k
        started = time.perf_counter()
        with scoped_cognee_env(workspace=self.workspace, config=self.config):
            validate_environment_isolation()  # env now scrubbed by scoped_cognee_env
            client = self.client or RealCogneeClient()
            raw_result = await self._retrieve_with_fallbacks(
                client=client,
                query=query,
                top_k=resolved_top_k,
            )
            if self.config.local_index is not None:
                # The local index stores chunks in sqlite+lancedb under the
                # workspace, NOT the Postgres chunk-mapping store; doc ids are
                # recovered from the [DOCUMENT_ID:] markers in the chunk text.
                # Skipping avoids a wrong-store DB connection on every query.
                chunk_mappings, mapping_error = [], ""
            else:
                chunk_mappings, mapping_error = self._load_chunk_mappings()
        result = parse_cognee_result(
            query=query,
            raw_result=raw_result,
            top_k=resolved_top_k,
            latency_ms=(time.perf_counter() - started) * 1000,
            chunk_mappings=chunk_mappings,
        )
        result.metadata["chunk_mapping_enrichment_available"] = bool(chunk_mappings)
        if mapping_error:
            result.metadata["chunk_mapping_enrichment_error"] = mapping_error
        return result

    async def remember_async(self, data: Any, **kwargs: Any) -> Any:
        ensure_workspace(self.workspace)
        with scoped_cognee_env(workspace=self.workspace, config=self.config):
            validate_environment_isolation()  # env now scrubbed by scoped_cognee_env
            client = self.client or RealCogneeClient()
            try:
                return await client.remember(data, dataset_name=self.config.dataset_name, **kwargs)
            except Exception:
                add_result = await client.add(data, dataset_name=self.config.dataset_name, **kwargs)
                cognify_result = await client.cognify(dataset_name=self.config.dataset_name, **kwargs)
                return {
                    "primary_api": "add+cognify",
                    "add_result": _jsonable(add_result),
                    "cognify_result": _jsonable(cognify_result),
                }

    async def add_and_cognify_async(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        ensure_workspace(self.workspace)
        with scoped_cognee_env(workspace=self.workspace, config=self.config):
            validate_environment_isolation()  # env now scrubbed by scoped_cognee_env
            client = self.client or RealCogneeClient()
            add_result = await client.add(data, dataset_name=self.config.dataset_name, **kwargs)
            cognify_result = await client.cognify(dataset_name=self.config.dataset_name, **kwargs)
            return {
                "primary_api": "add+cognify",
                "add_result": _jsonable(add_result),
                "cognify_result": _jsonable(cognify_result),
            }

    def remember(self, data: Any, **kwargs: Any) -> Any:
        return _run_sync(self.remember_async(data, **kwargs))

    async def _retrieve_with_fallbacks(
        self,
        *,
        client: CogneeClient,
        query: str,
        top_k: int,
    ) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        dataset_names = self.config.dataset_names or (self.config.dataset_name,)
        only_context = self.config.only_context
        try:
            raw = await client.recall(
                query,
                top_k=top_k,
                dataset_name=self.config.dataset_name,
                dataset_names=dataset_names,
                only_context=only_context,
            )
            return {
                "primary_api": "recall",
                "only_context": only_context,
                "dataset_names": list(dataset_names),
                "raw_trace": _jsonable(raw),
                "result": raw,
            }
        except Exception as exc:
            errors.append(_error_entry("recall", exc))
            logger.warning("cognee recall failed: %r", exc)
        try:
            raw = await client.search(
                query,
                top_k=top_k,
                dataset_name=self.config.dataset_name,
                dataset_names=dataset_names,
                only_context=only_context,
            )
            return {
                "primary_api": "search",
                "only_context": only_context,
                "dataset_names": list(dataset_names),
                "raw_trace": _jsonable(raw),
                "result": raw,
                "fallback_errors": errors,
            }
        except Exception as exc:
            errors.append(_error_entry("search", exc))
            # Lead the message with the error TYPES so the batch log's str(exc)[:120]
            # truncation is informative (the raw neo4j async error stringifies to an
            # opaque object repr). Full tracebacks go to the logger.
            summary = "; ".join(f"{e['api']}={e['error_type']}" for e in errors)
            logger.error(
                "cognee retrieval failed (recall+search): %s\nrecall_traceback:\n%s\nsearch_traceback:\n%s",
                summary,
                errors[0].get("traceback", "") if errors else "",
                errors[-1].get("traceback", "") if errors else "",
            )
            raise RuntimeError(
                f"Cognee retrieval failed (recall+search): {summary} :: {errors}"
            ) from exc

    def _load_chunk_mappings(self) -> tuple[list[dict[str, Any]], str]:
        try:
            from benchmark.methods.cognee.audit import load_cognee_chunk_mappings_from_postgres

            relational = self.config.relational_store
            return (
                load_cognee_chunk_mappings_from_postgres(
                    host=relational.host,
                    port=relational.port,
                    database=relational.database,
                    username=relational.username,
                    password=os.environ.get(relational.password_env, ""),
                ),
                "",
            )
        except Exception as exc:  # pragma: no cover - live database enrichment is best effort
            return [], str(exc)


@contextmanager
def scoped_cognee_env(*, workspace: CogneeWorkspace, config: CogneeConfig | None = None):
    import os

    from benchmark.methods.cognee.config_builder import FORBIDDEN_COGNEE_ENV_PREFIXES

    updates = cognee_env(workspace=workspace, config=config)
    # Scrub method-specific Neo4j envs (lightrag/graphrag) for the duration of the
    # cognee call so a stale value — the harness re-loads .env between questions —
    # cannot bleed into cognee's connection. Restored on exit. (cognee reads
    # NEO4J_*/GRAPH_DATABASE_*, which `updates` sets to the cognee target.)
    scrub_keys = [
        key
        for key in list(os.environ)
        if any(key.startswith(prefix) for prefix in FORBIDDEN_COGNEE_ENV_PREFIXES)
    ]
    previous = {key: os.environ.get(key) for key in list(updates) + scrub_keys}
    os.environ.update(updates)
    for key in scrub_keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


# One persistent event loop reused for every sync cognee call in this process.
# asyncio.run() creates+closes a fresh loop per call; cognee's cached Neo4j
# async driver gets stranded on the closed loop, leaking one pooled connection
# per query until the default 100-connection pool is exhausted (observed: every
# query past #100 fails with an neo4j async-concurrency error). Reusing a single
# loop keeps the driver valid so connections return to the pool — mirrors the
# lightrag LIGHTRAG_REUSE_RAG fix. The loop is never closed; it dies with the
# process.
_PERSISTENT_LOOP: Any | None = None


def _persistent_loop() -> Any:
    import asyncio

    global _PERSISTENT_LOOP
    if _PERSISTENT_LOOP is None or _PERSISTENT_LOOP.is_closed():
        _PERSISTENT_LOOP = asyncio.new_event_loop()
    return _PERSISTENT_LOOP


def _run_sync(coro: Any) -> Any:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = _persistent_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    raise RuntimeError("CogneeAdapter sync API cannot run inside an active event loop; use async API")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
