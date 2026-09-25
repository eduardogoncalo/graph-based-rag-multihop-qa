from __future__ import annotations

from typing import Any


class FakeConnection:
    def __init__(
        self,
        rows: list[tuple] | None = None,
        fetchone_value: object | None = None,
    ) -> None:
        self.cursor_obj = FakeCursor(rows or [], fetchone_value=fetchone_value)
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1


class FakeCursor:
    def __init__(self, rows: list[tuple], fetchone_value: object | None = None) -> None:
        self.rows = rows
        self.fetchone_value = fetchone_value
        self.executed: list[tuple[str, object | None]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object | None = None) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> Any:
        return self.fetchone_value

    def fetchall(self) -> list[tuple]:
        return self.rows
