from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

GRAPH_EXPECTED_ENV_VARS = {
    "GRAPH_DATABASE_PROVIDER": "neo4j",
    "GRAPH_DATABASE_URL": "bolt://localhost:18689",
    "GRAPH_DATABASE_USERNAME": "neo4j",
    "GRAPH_DATABASE_NAME": "neo4j",
}
GRAPH_PASSWORD_ENV = "COGNEE_NEO4J_PASSWORD"
GRAPH_RUNTIME_PASSWORD_ENV = "GRAPH_DATABASE_PASSWORD"

POSTGRES_HOST_ENV = "COGNEE_POSTGRES_HOST"
POSTGRES_PORT_ENV = "COGNEE_POSTGRES_PORT"
POSTGRES_USER_ENV = "COGNEE_POSTGRES_USER"
POSTGRES_PASSWORD_ENV = "COGNEE_POSTGRES_PASSWORD"
POSTGRES_DB_ENV = "COGNEE_POSTGRES_DB"
POSTGRES_SCHEMA_ENV = "COGNEE_POSTGRES_SCHEMA"
VECTOR_PROVIDER_ENV = "COGNEE_VECTOR_PROVIDER"
VECTOR_DB_HOST_ENV = "COGNEE_VECTOR_DB_HOST"
VECTOR_DB_PORT_ENV = "COGNEE_VECTOR_DB_PORT"
VECTOR_DB_USER_ENV = "COGNEE_VECTOR_DB_USER"
VECTOR_DB_PASSWORD_ENV = "COGNEE_VECTOR_DB_PASSWORD"
VECTOR_DB_NAME_ENV = "COGNEE_VECTOR_DB_NAME"
VECTOR_DB_SCHEMA_ENV = "COGNEE_VECTOR_DB_SCHEMA"

POSTGRES_DEFAULT_HOST = "127.0.0.1"
POSTGRES_DEFAULT_PORT = 15433
POSTGRES_DEFAULT_USER = "benchmark"
POSTGRES_BENCHMARK_DB = "benchmark"
POSTGRES_COGNEE_DB = "cognee_benchmark"
POSTGRES_COGNEE_SCHEMA = "cognee"
VECTOR_PROVIDER = "pgvector"
VECTOR_EXTENSION = "vector"

COGNEE_PORTS = {18476: "Neo4j Browser", 18689: "Neo4j Bolt"}
POSTGRES_PORTS = {POSTGRES_DEFAULT_PORT: "Postgres"}
COGNEE_CONTAINER_HINTS = ("neo4j_cognee", "thesis_neo4j_cognee_1", "thesis-neo4j-cognee-1")
POSTGRES_CONTAINER_HINTS = ("thesis-postgres-openai-smoke", "postgres")
COMPOSE_FILE = Path("docker-compose.cognee.yml")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    severity: str = "warn"


@dataclass(frozen=True)
class PostgresSettings:
    host: str
    port: int
    username: str
    password: str
    cognee_database: str
    schema: str
    vector_provider: str
    vector_database: str
    vector_schema: str
    vector_password: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Cognee configuration without starting containers, indexing, or calling models."
    )
    parser.add_argument(
        "--check-neo4j",
        action="store_true",
        help="If Cognee Neo4j is running, attempt a read-only connectivity check.",
    )
    args = parser.parse_args()

    container_rows = _container_rows()
    postgres_settings = _postgres_settings()
    results = [
        _check_python(),
        _check_cognee_import(),
        _check_neo4j_driver_import(),
        _check_psycopg_import(),
        _check_compose_file(),
        *_check_ports(COGNEE_PORTS, COGNEE_CONTAINER_HINTS, container_rows),
        _check_cognee_container(container_rows),
        _check_cognee_graph_env(),
        _check_cognee_postgres_env(postgres_settings),
        _check_cognee_vector_env(postgres_settings),
        _check_env_isolation(),
        *_check_ports({postgres_settings.port: "Postgres"}, POSTGRES_CONTAINER_HINTS, container_rows),
        _check_postgres_container(container_rows),
        *_check_postgres_live(postgres_settings),
    ]
    if args.check_neo4j:
        results.extend(_check_neo4j_live_if_running(container_rows))

    for result in results:
        status = "OK" if result.ok else result.severity.upper()
        print(f"[{status}] {result.name}: {result.detail}")
    return 0


def _check_python() -> CheckResult:
    return CheckResult(
        name="python",
        ok=sys.version_info >= (3, 11),
        detail=f"{sys.version.split()[0]} at {sys.executable}",
    )


def _check_cognee_import() -> CheckResult:
    try:
        module = importlib.import_module("cognee")
    except ImportError as exc:
        return CheckResult("cognee package", False, f"not importable: {exc}")
    version = _package_version("cognee")
    return CheckResult(
        "cognee package",
        True,
        f"version={version}; module={getattr(module, '__file__', '<namespace>')}",
    )


def _check_neo4j_driver_import() -> CheckResult:
    try:
        module = importlib.import_module("neo4j")
    except ImportError as exc:
        return CheckResult(
            "Neo4j Python driver",
            False,
            f"not importable; install Cognee Neo4j extras or neo4j driver: {exc}",
        )
    version = _package_version("neo4j")
    return CheckResult(
        "Neo4j Python driver",
        True,
        f"version={version}; module={getattr(module, '__file__', '<namespace>')}",
    )


def _check_psycopg_import() -> CheckResult:
    try:
        module = importlib.import_module("psycopg")
    except ImportError as exc:
        return CheckResult("psycopg", False, f"not importable; live Postgres checks skipped: {exc}")
    version = _package_version("psycopg")
    return CheckResult("psycopg", True, f"version={version}; module={getattr(module, '__file__', '<namespace>')}")


def _check_compose_file() -> CheckResult:
    if not COMPOSE_FILE.exists():
        return CheckResult("Cognee compose", False, f"missing: {COMPOSE_FILE}")
    return CheckResult("Cognee compose", True, f"present: {COMPOSE_FILE}")


def _check_ports(
    ports: dict[int, str],
    expected_container_hints: tuple[str, ...],
    container_rows: list[str],
) -> list[CheckResult]:
    results = []
    for port, label in ports.items():
        open_port = _port_is_open("127.0.0.1", port)
        if not open_port:
            results.append(CheckResult(f"{label} port {port}", True, "closed/free"))
            continue
        owner = _port_owner(port, container_rows)
        if any(hint in owner for hint in expected_container_hints):
            results.append(CheckResult(f"{label} port {port}", True, f"open on expected container: {owner}"))
        else:
            results.append(
                CheckResult(f"{label} port {port}", False, f"open on non-expected process/container: {owner or 'unknown'}")
            )
    return results


def _check_cognee_container(container_rows: list[str]) -> CheckResult:
    matches = [row for row in container_rows if any(hint in row for hint in COGNEE_CONTAINER_HINTS)]
    if not matches:
        return CheckResult("Cognee Neo4j container", True, "not present/running yet; OK before smoke")
    return CheckResult("Cognee Neo4j container", True, " | ".join(matches))


def _check_postgres_container(container_rows: list[str]) -> CheckResult:
    matches = [row for row in container_rows if any(hint in row for hint in POSTGRES_CONTAINER_HINTS)]
    if not matches:
        return CheckResult("Postgres container", False, "not found in docker/podman ps -a")
    expected = [row for row in matches if "thesis-postgres-openai-smoke" in row]
    if expected:
        return CheckResult("Postgres container", True, " | ".join(expected))
    return CheckResult("Postgres container", True, f"found postgres-like container(s): {' | '.join(matches)}")


def _check_cognee_graph_env() -> CheckResult:
    missing = [name for name in GRAPH_EXPECTED_ENV_VARS if not os.environ.get(name)]
    mismatched = [
        f"{name}={os.environ.get(name)} expected {expected}"
        for name, expected in GRAPH_EXPECTED_ENV_VARS.items()
        if os.environ.get(name) and os.environ.get(name) != expected
    ]
    password_present = bool(os.environ.get(GRAPH_PASSWORD_ENV) or os.environ.get(GRAPH_RUNTIME_PASSWORD_ENV))
    if missing or mismatched or not password_present:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if not password_present:
            details.append(f"missing password env: {GRAPH_PASSWORD_ENV} or {GRAPH_RUNTIME_PASSWORD_ENV}")
        if mismatched:
            details.append(f"mismatched: {'; '.join(mismatched)}")
        return CheckResult("Cognee graph env", False, " | ".join(details))
    return CheckResult("Cognee graph env", True, "preferred graph vars are present without using LightRAG/GraphRAG envs")


def _check_cognee_postgres_env(settings: PostgresSettings) -> CheckResult:
    missing = []
    if not os.environ.get(POSTGRES_PASSWORD_ENV):
        missing.append(POSTGRES_PASSWORD_ENV)
    if missing:
        return CheckResult(
            "Cognee Postgres env",
            False,
            f"missing: {', '.join(missing)}; live database/schema checks need this env var",
        )
    return CheckResult(
        "Cognee Postgres env",
        True,
        f"host={settings.host}; port={settings.port}; user={settings.username}; db={settings.cognee_database}; schema={settings.schema}",
    )


def _check_cognee_vector_env(settings: PostgresSettings) -> CheckResult:
    if settings.vector_provider != VECTOR_PROVIDER:
        return CheckResult(
            "Vector store target",
            False,
            f"provider={settings.vector_provider}; expected {VECTOR_PROVIDER}",
        )
    if settings.vector_database == POSTGRES_BENCHMARK_DB:
        return CheckResult("Vector store target", False, "invalid: vector store points to benchmark database")
    password_source = VECTOR_DB_PASSWORD_ENV if os.environ.get(VECTOR_DB_PASSWORD_ENV) else POSTGRES_PASSWORD_ENV
    return CheckResult(
        "Vector store target",
        True,
        (
            f"provider={settings.vector_provider}; database={settings.vector_database}; "
            f"schema={settings.vector_schema}; password_env={password_source}"
        ),
    )


def _check_env_isolation() -> CheckResult:
    present = [
        name
        for name in os.environ
        if name.startswith("LIGHTRAG_NEO4J_") or name.startswith("GRAPHRAG_NEO4J_")
    ]
    if present:
        return CheckResult(
            "Cognee env isolation",
            True,
            f"method-specific env vars are present globally but ignored by Cognee config: {', '.join(sorted(present))}",
        )
    return CheckResult("Cognee env isolation", True, "no LightRAG/GraphRAG Neo4j env vars detected")


def _check_neo4j_live_if_running(container_rows: list[str]) -> list[CheckResult]:
    if not any(any(hint in row and "Up" in row for hint in COGNEE_CONTAINER_HINTS) for row in container_rows):
        return [CheckResult("Cognee Neo4j live check", True, "skipped; Cognee Neo4j container is not running")]
    if not _port_is_open("127.0.0.1", 18689):
        return [CheckResult("Cognee Neo4j live check", False, "container appears present but Bolt port 18689 is closed")]
    try:
        neo4j = importlib.import_module("neo4j")
    except ImportError as exc:
        return [CheckResult("Cognee Neo4j live check", False, f"driver missing: {exc}")]
    uri = os.environ.get("GRAPH_DATABASE_URL", GRAPH_EXPECTED_ENV_VARS["GRAPH_DATABASE_URL"])
    user = os.environ.get("GRAPH_DATABASE_USERNAME", GRAPH_EXPECTED_ENV_VARS["GRAPH_DATABASE_USERNAME"])
    password = os.environ.get(GRAPH_RUNTIME_PASSWORD_ENV, os.environ.get(GRAPH_PASSWORD_ENV, ""))
    database = os.environ.get("GRAPH_DATABASE_NAME", GRAPH_EXPECTED_ENV_VARS["GRAPH_DATABASE_NAME"])
    if not password:
        return [CheckResult("Cognee Neo4j live check", False, f"missing {GRAPH_PASSWORD_ENV}")]
    try:
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            ok = session.run("RETURN 1 AS ok").single()["ok"]
            name = session.run("CALL db.info() YIELD name RETURN name").single()["name"]
        driver.close()
    except Exception as exc:  # pragma: no cover - live check is optional.
        return [CheckResult("Cognee Neo4j live check", False, str(exc))]
    return [
        CheckResult("Cognee Neo4j connectivity", ok == 1, f"connected to {uri}"),
        CheckResult("Cognee Neo4j database", name == database, f"connected database: {name}"),
    ]


def _check_postgres_live(settings: PostgresSettings) -> list[CheckResult]:
    if not _port_is_open(settings.host, settings.port):
        return [
            CheckResult("Postgres live check", True, "skipped; port is closed"),
            CheckResult("benchmark database", False, "unknown; Postgres port is closed"),
            CheckResult("Cognee database", False, "unknown; Postgres port is closed"),
            CheckResult("Cognee schema", False, "unknown; Postgres port is closed"),
            CheckResult("pgvector extension in Cognee database", False, "unknown; Postgres port is closed"),
            CheckResult(
                "Action required before smoke",
                False,
                "yes: start/reuse Postgres and create database/schema/vector extension if missing",
            ),
        ]
    if not settings.password:
        return [
            CheckResult("Postgres live check", False, f"skipped; missing {POSTGRES_PASSWORD_ENV}"),
            CheckResult("Action required before smoke", False, f"yes: set {POSTGRES_PASSWORD_ENV} and create database/schema if missing"),
        ]
    try:
        psycopg = importlib.import_module("psycopg")
    except ImportError as exc:
        return [CheckResult("Postgres live check", False, f"skipped; psycopg missing: {exc}")]

    results: list[CheckResult] = []
    connection_db = _first_connectable_database(psycopg, settings, ("postgres", POSTGRES_BENCHMARK_DB))
    if connection_db is None:
        return [
            CheckResult("Postgres live check", False, "unable to connect to postgres or benchmark database"),
            CheckResult("Action required before smoke", False, "yes: verify Postgres credentials and target database/schema"),
        ]

    benchmark_exists = _database_exists(psycopg, settings, connection_db, POSTGRES_BENCHMARK_DB)
    cognee_db_exists = _database_exists(psycopg, settings, connection_db, settings.cognee_database)
    results.append(CheckResult("benchmark database", benchmark_exists, "exists" if benchmark_exists else "missing"))
    results.append(CheckResult("Cognee database", cognee_db_exists, "exists" if cognee_db_exists else "missing"))

    schema_database = settings.cognee_database if cognee_db_exists else POSTGRES_BENCHMARK_DB
    schema_exists = False
    if _can_connect(psycopg, settings, schema_database):
        schema_exists = _schema_exists(psycopg, settings, schema_database, settings.schema)
        schema_detail = f"{settings.schema} in database {schema_database}: {'exists' if schema_exists else 'missing'}"
    else:
        schema_detail = f"unknown; cannot connect to database {schema_database}"
    results.append(CheckResult("Cognee schema", schema_exists, schema_detail))
    vector_extension_exists = False
    if cognee_db_exists and _can_connect(psycopg, settings, settings.cognee_database):
        vector_extension_exists, vector_extension_available = _extension_status(
            psycopg,
            settings,
            settings.cognee_database,
            VECTOR_EXTENSION,
        )
        if vector_extension_exists:
            vector_detail = f"{VECTOR_EXTENSION}: exists in database {settings.cognee_database}"
        elif vector_extension_available:
            vector_detail = f"{VECTOR_EXTENSION}: missing in database {settings.cognee_database}; extension is available"
        else:
            vector_detail = f"{VECTOR_EXTENSION}: missing in database {settings.cognee_database}; extension not available"
    else:
        vector_extension_available = False
        vector_detail = f"unknown; cannot connect to database {settings.cognee_database}"
    results.append(CheckResult("pgvector extension in Cognee database", vector_extension_exists, vector_detail))
    missing_resources = []
    if not cognee_db_exists:
        missing_resources.append("database")
    if not schema_exists:
        missing_resources.append("schema")
    if not vector_extension_exists:
        missing_resources.append("vector extension")
    action_required = bool(missing_resources)
    detail = f"yes: create missing {'/'.join(missing_resources)} before smoke" if action_required else "no"
    results.append(CheckResult("Action required before smoke", not action_required, detail))
    return results


def _database_exists(psycopg, settings: PostgresSettings, database: str, name: str) -> bool:
    dsn = _postgres_dsn(settings, database)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = %s)", (name,))
            return bool(cur.fetchone()[0])


def _schema_exists(psycopg, settings: PostgresSettings, database: str, schema: str) -> bool:
    dsn = _postgres_dsn(settings, database)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = %s)", (schema,))
            return bool(cur.fetchone()[0])


def _extension_status(psycopg, settings: PostgresSettings, database: str, extension: str) -> tuple[bool, bool]:
    dsn = _postgres_dsn(settings, database)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = %s)", (extension,))
            exists = bool(cur.fetchone()[0])
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = %s)", (extension,))
            available = bool(cur.fetchone()[0])
            return exists, available


def _first_connectable_database(psycopg, settings: PostgresSettings, databases: tuple[str, ...]) -> str | None:
    for database in databases:
        if _can_connect(psycopg, settings, database):
            return database
    return None


def _can_connect(psycopg, settings: PostgresSettings, database: str) -> bool:
    try:
        with psycopg.connect(_postgres_dsn(settings, database)):
            return True
    except Exception:
        return False


def _postgres_dsn(settings: PostgresSettings, database: str) -> str:
    return (
        f"postgresql://{settings.username}:{settings.password}"
        f"@{settings.host}:{settings.port}/{database}"
    )


def _postgres_settings() -> PostgresSettings:
    vector_password = os.environ.get(VECTOR_DB_PASSWORD_ENV, os.environ.get(POSTGRES_PASSWORD_ENV, ""))
    return PostgresSettings(
        host=os.environ.get(POSTGRES_HOST_ENV, POSTGRES_DEFAULT_HOST),
        port=int(os.environ.get(POSTGRES_PORT_ENV, str(POSTGRES_DEFAULT_PORT))),
        username=os.environ.get(POSTGRES_USER_ENV, POSTGRES_DEFAULT_USER),
        password=os.environ.get(POSTGRES_PASSWORD_ENV, ""),
        cognee_database=os.environ.get(POSTGRES_DB_ENV, POSTGRES_COGNEE_DB),
        schema=os.environ.get(POSTGRES_SCHEMA_ENV, POSTGRES_COGNEE_SCHEMA),
        vector_provider=os.environ.get(VECTOR_PROVIDER_ENV, VECTOR_PROVIDER),
        vector_database=os.environ.get(VECTOR_DB_NAME_ENV, os.environ.get(POSTGRES_DB_ENV, POSTGRES_COGNEE_DB)),
        vector_schema=os.environ.get(VECTOR_DB_SCHEMA_ENV, os.environ.get(POSTGRES_SCHEMA_ENV, POSTGRES_COGNEE_SCHEMA)),
        vector_password=vector_password,
    )


def _container_rows() -> list[str]:
    for command in (["podman", "ps", "-a"], ["docker", "ps", "-a"]):
        try:
            result = subprocess.run(
                [*command, "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return [line for line in result.stdout.splitlines() if line.strip()]
    return []


def _port_owner(port: int, container_rows: list[str]) -> str:
    matches = [row for row in container_rows if f":{port}->" in row or f":{port}-" in row]
    return " | ".join(matches)


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
