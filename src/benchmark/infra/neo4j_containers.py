from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from benchmark.core.naming import (
    neo4j_container_name,
    neo4j_volume_name,
    port_registry_key,
)

# Neutral launcher/manager for Option C: one isolated Neo4j Community container
# (+ named volume) per (method, dataset, version). No graph-method logic lives
# here. Destructive operations are limited to `docker rm -f <name>` and
# `docker volume rm <name>` for resources this module created — NEVER
# `down -v`, `prune`, or any bulk removal.
#
# `docker` is the podman shim on this host; every call goes through the
# `docker` binary so the same code works under Docker or podman.

DEFAULT_IMAGE = "docker.io/library/neo4j:5"
DEFAULT_USERNAME = "neo4j"
DOCKER_BIN = "docker"
DEFAULT_PORT_REGISTRY = Path("configs/infra/neo4j_ports.yaml")


class PortConflictError(RuntimeError):
    """Raised when a required host port is already in use or reserved."""


@dataclass(frozen=True)
class Neo4jContainerSpec:
    """Everything needed to launch one isolated Neo4j container."""

    container_name: str
    volume_name: str
    http_port: int
    bolt_port: int
    password: str
    username: str = DEFAULT_USERNAME
    image: str = DEFAULT_IMAGE

    @classmethod
    def for_dataset(
        cls,
        *,
        method_short: str,
        slug: str,
        http_port: int,
        bolt_port: int,
        password: str,
        username: str = DEFAULT_USERNAME,
        image: str = DEFAULT_IMAGE,
    ) -> Neo4jContainerSpec:
        return cls(
            container_name=neo4j_container_name(method_short, slug),
            volume_name=neo4j_volume_name(method_short, slug),
            http_port=http_port,
            bolt_port=bolt_port,
            password=password,
            username=username,
            image=image,
        )


@dataclass(frozen=True)
class Neo4jContainerHandle:
    """Connection details for a started container."""

    container_name: str
    volume_name: str
    bolt_uri: str
    http_uri: str
    username: str
    password: str


# --------------------------------------------------------------------------- #
# Environment / port probing
# --------------------------------------------------------------------------- #
def docker_available() -> bool:
    """True if a usable `docker` (or podman shim) CLI is on PATH and responds."""
    if shutil.which(DOCKER_BIN) is None:
        return False
    try:
        result = subprocess.run(
            [DOCKER_BIN, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already accepting TCP connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


# --------------------------------------------------------------------------- #
# Port registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Neo4jPortRegistry:
    """Parsed ``configs/infra/neo4j_ports.yaml``."""

    assignments: dict[str, dict[str, int]]
    reserved: frozenset[int]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PORT_REGISTRY) -> Neo4jPortRegistry:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        assignments = {
            key: {"http": int(value["http"]), "bolt": int(value["bolt"])}
            for key, value in (data.get("assignments") or {}).items()
        }
        reserved = frozenset(int(p) for p in (data.get("reserved") or []))
        return cls(assignments=assignments, reserved=reserved)

    def ports_for(self, method_short: str, slug: str) -> tuple[int, int]:
        key = port_registry_key(method_short, slug)
        if key not in self.assignments:
            raise KeyError(f"no port assignment for {key!r} in registry")
        entry = self.assignments[key]
        return entry["http"], entry["bolt"]


def resolve_ports(
    method_short: str,
    slug: str,
    *,
    registry: Neo4jPortRegistry | None = None,
    preflight: bool = True,
) -> tuple[int, int]:
    """Resolve the (http, bolt) ports for a (method, dataset) from the registry.

    When ``preflight`` is True (default), both ports are checked against the
    registry's ``reserved`` list and probed for a live listener; a conflict
    raises :class:`PortConflictError` instead of letting the container fail to
    bind.
    """
    registry = registry or Neo4jPortRegistry.load()
    http_port, bolt_port = registry.ports_for(method_short, slug)
    if preflight:
        preflight_ports(http_port, bolt_port, reserved=registry.reserved)
    return http_port, bolt_port


def preflight_ports(*ports: int, reserved: frozenset[int] = frozenset()) -> None:
    """Raise :class:`PortConflictError` if any port is reserved or in use."""
    for port in ports:
        if port in reserved:
            raise PortConflictError(f"port {port} is reserved and must not be used")
        if port_in_use(port):
            raise PortConflictError(f"port {port} is already in use on the host")


# --------------------------------------------------------------------------- #
# Container lifecycle
# --------------------------------------------------------------------------- #
def _docker(*args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [DOCKER_BIN, *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def container_exists(name: str) -> bool:
    result = _docker(
        "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}", check=False
    )
    return name in result.stdout.split()


def volume_exists(name: str) -> bool:
    result = _docker("volume", "ls", "--format", "{{.Name}}", check=False)
    return name in result.stdout.split()


def volume_mountpoint(name: str) -> str | None:
    """Return a volume's on-disk mountpoint, or None if it does not exist."""
    result = _docker(
        "volume", "inspect", name, "--format", "{{.Mountpoint}}", check=False
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def wait_for_ready(
    bolt_uri: str,
    username: str,
    password: str,
    *,
    timeout: int = 120,
    interval: float = 2.0,
) -> None:
    """Block until the Neo4j Bolt endpoint accepts an authenticated connection.

    Imports the driver lazily so importing this module never requires neo4j.
    """
    from neo4j import GraphDatabase  # lazy
    from neo4j.exceptions import Neo4jError, ServiceUnavailable

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            driver = GraphDatabase.driver(bolt_uri, auth=(username, password))
            try:
                driver.verify_connectivity()
                return
            finally:
                driver.close()
        except (ServiceUnavailable, Neo4jError, OSError) as exc:  # not ready yet
            last_error = exc
            time.sleep(interval)
    raise TimeoutError(
        f"Neo4j at {bolt_uri} not ready after {timeout}s: {last_error!r}"
    )


def start_neo4j_container(
    spec: Neo4jContainerSpec,
    *,
    wait: bool = True,
    timeout: int = 120,
    apoc_gds: bool = True,
) -> Neo4jContainerHandle:
    """Start one isolated Neo4j container and (optionally) wait for readiness.

    Refuses to start if a container of the same name already exists, so a stale
    container is never silently reused. Caller owns cleanup via
    :func:`remove_container`.
    """
    if container_exists(spec.container_name):
        raise RuntimeError(
            f"container {spec.container_name!r} already exists; remove it first"
        )

    env = [f"NEO4J_AUTH={spec.username}/{spec.password}"]
    if apoc_gds:
        env += [
            'NEO4J_PLUGINS=["apoc","graph-data-science"]',
            "NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*",
            "NEO4J_dbms_security_procedures_allowlist=apoc.*,gds.*",
        ]
    env_args: list[str] = []
    for item in env:
        env_args += ["-e", item]

    _docker(
        "run",
        "-d",
        "--name",
        spec.container_name,
        *env_args,
        "-p",
        f"{spec.http_port}:7474",
        "-p",
        f"{spec.bolt_port}:7687",
        "-v",
        f"{spec.volume_name}:/data",
        spec.image,
        timeout=timeout,
    )

    handle = Neo4jContainerHandle(
        container_name=spec.container_name,
        volume_name=spec.volume_name,
        bolt_uri=f"bolt://localhost:{spec.bolt_port}",
        http_uri=f"http://localhost:{spec.http_port}",
        username=spec.username,
        password=spec.password,
    )
    if wait:
        wait_for_ready(handle.bolt_uri, handle.username, handle.password, timeout=timeout)
    return handle


def remove_container(name: str, *, remove_volume: str | None = None) -> None:
    """Force-remove one container by name, optionally one named volume.

    Only ever touches the exact names passed in — no bulk pruning. Safe to call
    when the container/volume does not exist.
    """
    if container_exists(name):
        _docker("rm", "-f", name, check=False)
    if remove_volume and volume_exists(remove_volume):
        _docker("volume", "rm", remove_volume, check=False)


@contextmanager
def temporary_neo4j(
    spec: Neo4jContainerSpec,
    *,
    timeout: int = 120,
) -> Iterator[Neo4jContainerHandle]:
    """Start a container, yield its handle, and guarantee cleanup of ONLY it.

    Removes the exact container + volume it created (never anything else),
    even if the body raises.
    """
    handle = start_neo4j_container(spec, wait=True, timeout=timeout)
    try:
        yield handle
    finally:
        remove_container(handle.container_name, remove_volume=handle.volume_name)
