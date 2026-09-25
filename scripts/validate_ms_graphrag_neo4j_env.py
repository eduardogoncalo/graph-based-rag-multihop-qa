from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass

REQUIRED_ENV_VARS = (
    "GRAPHRAG_NEO4J_URI",
    "GRAPHRAG_NEO4J_USER",
    "GRAPHRAG_NEO4J_PASSWORD",
    "GRAPHRAG_NEO4J_DATABASE",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Neo4jLiveChecks:
    connectivity: CheckResult
    database: CheckResult
    apoc: CheckResult
    gds: CheckResult


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the optional ms_graphrag_neo4j runtime without calling OpenAI."
    )
    parser.add_argument(
        "--check-neo4j",
        action="store_true",
        help="Also attempt a Neo4j connectivity check using GRAPHRAG_NEO4J_* env vars.",
    )
    args = parser.parse_args()

    results = [
        _check_python(),
        _check_import("ms_graphrag_neo4j", "target package"),
        _check_import("neo4j", "Neo4j Python driver"),
        _check_graphrag_env(),
        _check_forbidden_env_use(),
    ]
    if args.check_neo4j:
        live_checks = _check_neo4j_live()
        results.extend(
            [
                live_checks.connectivity,
                live_checks.database,
                live_checks.apoc,
                live_checks.gds,
            ]
        )

    for result in results:
        status = "OK" if result.ok else "WARN"
        print(f"[{status}] {result.name}: {result.detail}")

    # Missing optional packages/env vars are warnings in this validation phase. A failed explicit
    # Neo4j connectivity check is the only hard failure because the user requested that check.
    if args.check_neo4j and not all(result.ok for result in results[-4:]):
        return 1
    return 0


def _check_python() -> CheckResult:
    version = sys.version_info
    ok = version >= (3, 11)
    return CheckResult(
        name="python",
        ok=ok,
        detail=f"{sys.version.split()[0]} at {sys.executable}",
    )


def _check_import(module_name: str, label: str) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return CheckResult(
            name=label,
            ok=False,
            detail=f"not importable: {exc}",
        )
    module_path = getattr(module, "__file__", "<namespace>")
    return CheckResult(name=label, ok=True, detail=f"imported from {module_path}")


def _check_graphrag_env() -> CheckResult:
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        return CheckResult(
            name="GRAPHRAG_NEO4J env",
            ok=False,
            detail=f"missing: {', '.join(missing)}",
        )
    return CheckResult(
        name="GRAPHRAG_NEO4J env",
        ok=True,
        detail="all required GRAPHRAG_NEO4J_* variables are present",
    )


def _check_forbidden_env_use() -> CheckResult:
    if os.environ.get("LIGHTRAG_NEO4J_URI"):
        return CheckResult(
            name="LightRAG isolation",
            ok=True,
            detail="LIGHTRAG_NEO4J_URI is present but ignored by this validator",
        )
    if os.environ.get("NEO4J_URI"):
        return CheckResult(
            name="generic Neo4j env",
            ok=True,
            detail="NEO4J_URI is present but ignored; ms_graphrag_neo4j uses GRAPHRAG_NEO4J_URI",
        )
    return CheckResult(
        name="Neo4j env isolation",
        ok=True,
        detail="no generic or LightRAG Neo4j URI is needed for this method",
    )


def _check_neo4j_live() -> Neo4jLiveChecks:
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        failed = CheckResult(
            name="Neo4j connectivity",
            ok=False,
            detail=f"cannot connect because env vars are missing: {', '.join(missing)}",
        )
        skipped = CheckResult(name="Neo4j live check", ok=False, detail="skipped")
        return Neo4jLiveChecks(failed, skipped, skipped, skipped)
    try:
        neo4j = importlib.import_module("neo4j")
    except ImportError as exc:
        failed = CheckResult(name="Neo4j connectivity", ok=False, detail=f"driver missing: {exc}")
        skipped = CheckResult(name="Neo4j live check", ok=False, detail="skipped")
        return Neo4jLiveChecks(failed, skipped, skipped, skipped)

    uri = os.environ["GRAPHRAG_NEO4J_URI"]
    user = os.environ["GRAPHRAG_NEO4J_USER"]
    password = os.environ["GRAPHRAG_NEO4J_PASSWORD"]
    database = os.environ["GRAPHRAG_NEO4J_DATABASE"]
    try:
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            connectivity_value = session.run("RETURN 1 AS ok").single()["ok"]
            database_value = session.run("CALL db.info() YIELD name RETURN name").single()["name"]
            apoc_value = session.run("RETURN apoc.version() AS version").single()["version"]
            gds_value = session.run("RETURN gds.version() AS version").single()["version"]
        driver.close()
    except Exception as exc:  # pragma: no cover - live connectivity is opt-in.
        failed = CheckResult(name="Neo4j connectivity", ok=False, detail=str(exc))
        skipped = CheckResult(name="Neo4j live check", ok=False, detail="skipped")
        return Neo4jLiveChecks(failed, skipped, skipped, skipped)

    return Neo4jLiveChecks(
        connectivity=CheckResult(
            name="Neo4j connectivity",
            ok=connectivity_value == 1,
            detail=f"connected to {uri}",
        ),
        database=CheckResult(
            name="Neo4j database",
            ok=database_value == database,
            detail=f"connected database: {database_value}",
        ),
        apoc=CheckResult(name="APOC", ok=bool(apoc_value), detail=f"version {apoc_value}"),
        gds=CheckResult(name="GDS", ok=bool(gds_value), detail=f"version {gds_value}"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
