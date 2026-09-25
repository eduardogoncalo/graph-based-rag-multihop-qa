import sys
from types import SimpleNamespace

from benchmark.core.settings import Settings
from benchmark.storage.postgres import connect_postgres


def test_connect_postgres_reads_database_url(monkeypatch) -> None:
    calls: list[str] = []

    def fake_connect(database_url: str) -> str:
        calls.append(database_url)
        return "connection"

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=fake_connect))

    connection = connect_postgres(
        Settings(database_url="postgresql://user:pass@localhost:15432/test")
    )

    assert connection == "connection"
    assert calls == ["postgresql://user:pass@localhost:15432/test"]
