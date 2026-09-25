from __future__ import annotations

import argparse
import importlib
import os
import re
from dataclasses import dataclass

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 15433
DEFAULT_USER = "benchmark"
BENCHMARK_DATABASE = "benchmark"
DEFAULT_DATABASE = "cognee_benchmark"
DEFAULT_SCHEMA = "cognee"
VECTOR_EXTENSION = "vector"
PASSWORD_ENV = "COGNEE_POSTGRES_PASSWORD"


@dataclass(frozen=True)
class BootstrapPlan:
    dry_run: bool
    execute: bool
    create_database: bool
    create_schema: bool
    create_vector_extension: bool
    host: str
    port: int
    username: str
    password: str
    database_name: str
    schema_name: str
    sql: tuple[str, ...]


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    print_plan(plan)
    if not plan.execute:
        return 0
    execute_plan(plan)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the isolated Cognee Postgres database/schema. Defaults to dry-run."
    )
    parser.add_argument("--dry-run", action="store_true", default=True, help="Print SQL only. This is the default.")
    parser.add_argument("--execute", action="store_true", help="Execute requested create operations.")
    parser.add_argument("--create-database", action="store_true", help="Create the Cognee database if missing.")
    parser.add_argument("--create-schema", action="store_true", help="Create the Cognee schema if missing.")
    parser.add_argument(
        "--create-vector-extension",
        action="store_true",
        help="Create the pgvector extension in the isolated Cognee database if missing.",
    )
    parser.add_argument("--database-name", default=DEFAULT_DATABASE)
    parser.add_argument("--schema-name", default=DEFAULT_SCHEMA)
    parser.add_argument("--host", default=os.environ.get("COGNEE_POSTGRES_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("COGNEE_POSTGRES_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--username", default=os.environ.get("COGNEE_POSTGRES_USER", DEFAULT_USER))
    return parser.parse_args(argv)


def build_plan(args: argparse.Namespace) -> BootstrapPlan:
    database_name = validate_identifier(args.database_name)
    schema_name = validate_identifier(args.schema_name)
    if args.create_vector_extension and database_name == BENCHMARK_DATABASE:
        raise ValueError("Cognee vector extension must not be created in the benchmark database")
    sql: list[str] = []
    if args.create_database:
        sql.append(f"CREATE DATABASE {database_name};")
    else:
        sql.append(f"-- optional: CREATE DATABASE {database_name};")
    if args.create_schema:
        sql.append(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
    else:
        sql.append(f"-- optional: CREATE SCHEMA IF NOT EXISTS {schema_name};")
    if args.create_vector_extension:
        sql.append(f"CREATE EXTENSION IF NOT EXISTS {VECTOR_EXTENSION};")
    execute = bool(args.execute and (args.create_database or args.create_schema or args.create_vector_extension))
    return BootstrapPlan(
        dry_run=not execute,
        execute=execute,
        create_database=bool(args.create_database),
        create_schema=bool(args.create_schema),
        create_vector_extension=bool(args.create_vector_extension),
        host=args.host,
        port=args.port,
        username=args.username,
        password=os.environ.get(PASSWORD_ENV, ""),
        database_name=database_name,
        schema_name=schema_name,
        sql=tuple(sql),
    )


def print_plan(plan: BootstrapPlan) -> None:
    mode = "execute" if plan.execute else "dry-run"
    print(f"Mode: {mode}")
    print(f"Host: {plan.host}:{plan.port}")
    print(f"User: {plan.username}")
    print(f"Cognee database: {plan.database_name}")
    print(f"Cognee schema: {plan.schema_name}")
    print("SQL:")
    for statement in plan.sql:
        print(statement)
    if not plan.execute:
        print("No changes were made. Re-run with --execute and explicit create flags after approval.")


def execute_plan(plan: BootstrapPlan) -> None:
    if not plan.password:
        raise RuntimeError(f"{PASSWORD_ENV} is required for execution")
    psycopg = importlib.import_module("psycopg")
    if plan.create_database:
        _create_database_if_missing(psycopg, plan)
    if plan.create_schema:
        _create_schema_if_missing(psycopg, plan)
    if plan.create_vector_extension:
        _create_vector_extension_if_missing(psycopg, plan)


def _create_database_if_missing(psycopg, plan: BootstrapPlan) -> None:
    with psycopg.connect(_dsn(plan, "postgres"), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = %s)", (plan.database_name,))
            exists = bool(cur.fetchone()[0])
            if not exists:
                cur.execute(f"CREATE DATABASE {plan.database_name}")


def _create_schema_if_missing(psycopg, plan: BootstrapPlan) -> None:
    with psycopg.connect(_dsn(plan, plan.database_name), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {plan.schema_name}")


def _create_vector_extension_if_missing(psycopg, plan: BootstrapPlan) -> None:
    if plan.database_name == BENCHMARK_DATABASE:
        raise RuntimeError("Refusing to create Cognee vector extension in the benchmark database")
    with psycopg.connect(_dsn(plan, plan.database_name), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {VECTOR_EXTENSION}")


def _dsn(plan: BootstrapPlan, database: str) -> str:
    return f"postgresql://{plan.username}:{plan.password}@{plan.host}:{plan.port}/{database}"


def validate_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe Postgres identifier: {value!r}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
