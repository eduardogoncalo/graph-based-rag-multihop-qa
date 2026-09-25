from pathlib import Path

from benchmark.storage.migrator import Migrator
from tests.fakes import FakeConnection


def test_migrator_applies_migrations_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "002_second.sql").write_text("CREATE TABLE second (id text);", encoding="utf-8")
    (tmp_path / "001_first.sql").write_text("CREATE TABLE first (id text);", encoding="utf-8")
    connection = FakeConnection()

    applied = Migrator(connection, migrations_dir=tmp_path).apply()

    assert [migration.version for migration in applied] == ["001_first.sql", "002_second.sql"]
    executed_sql = [sql for sql, _ in connection.cursor_obj.executed]
    assert executed_sql.index("CREATE TABLE first (id text);") < executed_sql.index(
        "CREATE TABLE second (id text);"
    )
    assert connection.commits == 1
