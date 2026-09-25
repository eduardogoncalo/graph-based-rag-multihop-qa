from __future__ import annotations

from typing import Any

from benchmark.core.settings import Settings, load_settings


def connect_postgres(settings: Settings | None = None) -> Any:
    resolved_settings = settings or load_settings()
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required for Postgres operations. Install project dependencies first."
        ) from exc
    return psycopg.connect(str(resolved_settings.database_url))
