from pathlib import Path


def test_migration_sql_contains_required_tables() -> None:
    sql = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("src/benchmark/storage/migrations").glob("*.sql"))
    )

    for table in [
        "datasets",
        "documents",
        "chunks",
        "questions",
        "gold_evidence",
        "experiments",
        "runs",
        "retrieval_results",
        "retrieval_items",
        "retrieval_traces",
        "answers",
        "agent_messages",
        "evaluation_results",
    ]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
