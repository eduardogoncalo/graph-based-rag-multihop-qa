#!/usr/bin/env python
"""Start (idempotently) an isolated Option C Neo4j container for the cognee method.

Mirrors how lightrag_neo4j stands up its per-dataset Neo4j (Option C): one
container + named volume, apoc/gds plugins, ports from the registry band. Safe to
re-run after a suspend/hibernate — if the container already exists it is started
(not recreated), so the graph data on its named volume is preserved and indexing
can resume. Runs in the MAIN .venv (uses the benchmark infra helper).

Never does down -v / prune; only ever touches the exact container/volume named.
"""
from __future__ import annotations

import argparse
import sys

from benchmark.infra.guard import exigir_alvo_permitido
from benchmark.infra.neo4j_containers import (
    Neo4jContainerHandle,
    Neo4jContainerSpec,
    container_exists,
    start_neo4j_container,
    wait_for_ready,
    _docker,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="container name")
    ap.add_argument("--volume", default=None, help="volume name (default <name>_data)")
    ap.add_argument("--http-port", type=int, required=True)
    ap.add_argument("--bolt-port", type=int, required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    # Antes de qualquer coisa: este script arranca containers, e um `start`
    # sobre um container do ambiente original põe de pé o Neo4j da dissertação.
    # As portas vêm da linha de comando e nada as validava.
    exigir_alvo_permitido(f"bolt://localhost:{args.bolt_port}", origem="--bolt-port")
    exigir_alvo_permitido(f"http://localhost:{args.http_port}", origem="--http-port")

    volume = args.volume or f"{args.name}_data"
    bolt_uri = f"bolt://localhost:{args.bolt_port}"

    if container_exists(args.name):
        # Reuse: start if stopped, then wait for readiness. Preserves graph data.
        _docker("start", args.name, check=False)
        wait_for_ready(bolt_uri, "neo4j", args.password, timeout=args.timeout)
        print(f"REUSED_EXISTING name={args.name} bolt={bolt_uri} volume={volume}")
        return 0

    spec = Neo4jContainerSpec(
        container_name=args.name,
        volume_name=volume,
        http_port=args.http_port,
        bolt_port=args.bolt_port,
        password=args.password,
    )
    handle: Neo4jContainerHandle = start_neo4j_container(spec, wait=True, timeout=args.timeout)
    print(
        f"STARTED name={handle.container_name} bolt={handle.bolt_uri} "
        f"http={handle.http_uri} volume={handle.volume_name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
