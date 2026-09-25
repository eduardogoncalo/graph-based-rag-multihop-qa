from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


@dataclass(frozen=True)
class AppliedMigration:
    version: str
    path: Path


class Migrator:
    def __init__(
        self,
        connection: Any,
        migrations_dir: str | Path = DEFAULT_MIGRATIONS_DIR,
    ) -> None:
        self.connection = connection
        self.migrations_dir = Path(migrations_dir)

    def migration_files(self) -> list[Path]:
        return sorted(self.migrations_dir.glob("*.sql"))

    def apply(self) -> list[AppliedMigration]:
        applied: list[AppliedMigration] = []
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version text PRIMARY KEY,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            for path in self.migration_files():
                version = path.name
                cursor.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
                if cursor.fetchone():
                    continue
                cursor.execute(path.read_text(encoding="utf-8"))
                cursor.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
                applied.append(AppliedMigration(version=version, path=path))
        self.connection.commit()
        return applied
