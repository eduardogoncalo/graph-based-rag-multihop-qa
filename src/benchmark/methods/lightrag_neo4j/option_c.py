from __future__ import annotations

import os
from dataclasses import dataclass, replace

from benchmark.core.naming import (
    method_short_tag,
    neo4j_container_name,
    neo4j_volume_name,
    slugify_dataset_version,
)
from benchmark.infra.guard import exigir_dataset_permitido
from benchmark.infra.neo4j_containers import (
    DEFAULT_IMAGE,
    Neo4jContainerSpec,
    Neo4jPortRegistry,
    resolve_ports,
)
from benchmark.methods.lightrag_neo4j.config_builder import (
    LIGHTRAG_NEO4J_METHOD_ID,
    LIGHTRAG_NEO4J_PASSWORD_ENV,
    LightRAGNeo4jConfig,
)

# Binds LightRAG to the Option C infrastructure: given (method, dataset,
# version) it resolves the isolated Neo4j container/volume/ports from the port
# registry + naming helpers and produces both a launch spec and a config with
# explicit connection overrides. No graph-method logic; no indexing.

LIGHTRAG_NEO4J_PASSWORD_DEFAULT = "benchmark_lightrag"
LIGHTRAG_NEO4J_USERNAME_DEFAULT = "neo4j"
LIGHTRAG_NEO4J_DATABASE_DEFAULT = "neo4j"


@dataclass(frozen=True)
class LightRAGOptionCBinding:
    """Resolved Option C placement for LightRAG on one dataset."""

    method_id: str
    dataset_id: str
    dataset_version: str
    slug: str
    method_short: str
    container_name: str
    volume_name: str
    http_port: int
    bolt_port: int
    bolt_uri: str
    http_uri: str
    username: str
    password: str
    database: str

    def container_spec(self, *, image: str = DEFAULT_IMAGE) -> Neo4jContainerSpec:
        return Neo4jContainerSpec(
            container_name=self.container_name,
            volume_name=self.volume_name,
            http_port=self.http_port,
            bolt_port=self.bolt_port,
            password=self.password,
            username=self.username,
            image=image,
        )

    def to_config(self, base: LightRAGNeo4jConfig | None = None) -> LightRAGNeo4jConfig:
        """Return a config carrying the resolved connection as explicit overrides.

        Env-var *names* stay the LightRAG-specific defaults (so the anti-
        anti-host validation still holds); the explicit values take precedence
        at runtime.
        """
        base = base or LightRAGNeo4jConfig(method_id=self.method_id)
        return replace(
            base,
            method_id=self.method_id,
            neo4j_uri=self.bolt_uri,
            neo4j_user=self.username,
            neo4j_password=self.password,
            neo4j_database=self.database,
        )


def resolve_lightrag_option_c(
    *,
    dataset_id: str,
    dataset_version: str,
    method_id: str = LIGHTRAG_NEO4J_METHOD_ID,
    registry: Neo4jPortRegistry | None = None,
    preflight: bool = True,
    password: str | None = None,
    username: str = LIGHTRAG_NEO4J_USERNAME_DEFAULT,
    database: str = LIGHTRAG_NEO4J_DATABASE_DEFAULT,
) -> LightRAGOptionCBinding:
    """Resolve the isolated Neo4j placement for LightRAG on a dataset.

    Ports come from ``configs/infra/neo4j_ports.yaml`` (never hardcoded here).
    ``preflight=True`` refuses reserved/busy ports before returning.
    """
    short = method_short_tag(method_id)  # "lightrag"
    slug = slugify_dataset_version(dataset_id, dataset_version)
    # Travão de nome. O `container_name` abaixo é global ao podman, e para os
    # datasets da fase experimental é literalmente o container da dissertação.
    # Inerte numa máquina sem o ambiente original — ver guard.exigir_dataset_permitido.
    exigir_dataset_permitido(
        slug, origem=f"LightRAG Option C for {dataset_id} {dataset_version}"
    )
    http_port, bolt_port = resolve_ports(
        short, slug, registry=registry, preflight=preflight
    )
    resolved_password = (
        password
        or os.environ.get(LIGHTRAG_NEO4J_PASSWORD_ENV)
        or LIGHTRAG_NEO4J_PASSWORD_DEFAULT
    )
    return LightRAGOptionCBinding(
        method_id=method_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        slug=slug,
        method_short=short,
        container_name=neo4j_container_name(short, slug),
        volume_name=neo4j_volume_name(short, slug),
        http_port=http_port,
        bolt_port=bolt_port,
        bolt_uri=f"bolt://localhost:{bolt_port}",
        http_uri=f"http://localhost:{http_port}",
        username=username,
        password=resolved_password,
        database=database,
    )
