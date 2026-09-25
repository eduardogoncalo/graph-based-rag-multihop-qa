from __future__ import annotations

import os

from benchmark.core.naming import method_short_tag, slugify_dataset_version
from benchmark.infra.guard import exigir_alvo_permitido, exigir_dataset_permitido
from benchmark.infra.neo4j_containers import Neo4jPortRegistry, resolve_ports
from benchmark.methods.cognee.config_builder import (
    COGNEE_METHOD_ID,
    COGNEE_NEO4J_PASSWORD_ENV,
    CogneeConfig,
    CogneeLocalIndexProfile,
)

# Binds cognee retrieval to the Option C infrastructure: given (dataset,
# version) it resolves the isolated Neo4j bolt port from the port registry and
# returns a CogneeConfig whose ``local_index`` points retrieval at the stores
# the indexer actually wrote (Option C Neo4j + sqlite + lancedb under the
# workspace). No indexing, no graph-method logic.

COGNEE_NEO4J_PASSWORD_DEFAULT = "benchmark_cognee"


def resolve_cognee_option_c(
    *,
    dataset_id: str,
    dataset_version: str,
    top_k: int = 5,
    registry: Neo4jPortRegistry | None = None,
    preflight: bool = False,
    password: str | None = None,
) -> CogneeConfig:
    """Build a CogneeConfig bound to the dataset's isolated local index.

    Ports come from ``configs/infra/neo4j_ports.yaml`` (never hardcoded). The
    bolt password defaults to ``$COGNEE_NEO4J_PASSWORD`` then the Option C
    literal used at index time. ``preflight=False`` because we connect to an
    already-running container rather than allocating a fresh port.

    Raises ``KeyError`` if the dataset has no Option C cognee assignment — the
    caller falls back to the default (Postgres/pgvector) CogneeConfig.
    """
    short = method_short_tag(COGNEE_METHOD_ID)  # "cognee"
    slug = slugify_dataset_version(dataset_id, dataset_version)
    # Travão de nome, e vem antes de resolver a porta. Desde 2026-08-10 os
    # datasets completos têm portas utilizáveis na banda 18xxx — o que é
    # necessário para quem recebe o pacote os poder indexar, e o que torna esta
    # verificação indispensável nesta máquina, onde o container de mesmo nome
    # guarda o índice da dissertação.
    exigir_dataset_permitido(
        slug, origem=f"local Cognee index for {dataset_id} {dataset_version}"
    )
    _http_port, bolt_port = resolve_ports(short, slug, registry=registry, preflight=preflight)
    resolved_password = (
        password
        or os.environ.get(COGNEE_NEO4J_PASSWORD_ENV)
        or COGNEE_NEO4J_PASSWORD_DEFAULT
    )
    # Travão. Este caminho liga-se de propósito a um container JÁ EM EXECUÇÃO,
    # e por isso corre com preflight desligado — o que também significa que a
    # recusa por porta reservada, que protege os outros métodos, não se aplica
    # aqui. Sem esta verificação, um slug do ambiente experimental original
    # resolveria para o grafo que sustenta a dissertação, e a indexação
    # escreveria lá dentro.
    graph_url = exigir_alvo_permitido(
        f"bolt://localhost:{bolt_port}",
        origem=f"local Cognee index for {slug} (configs/infra/neo4j_ports.yaml)",
    )
    return CogneeConfig(
        top_k=top_k,
        dataset_name=slug,
        local_index=CogneeLocalIndexProfile(
            graph_url=graph_url,
            graph_password=resolved_password,
        ),
    )


def resolve_cognee_config(
    *,
    dataset_id: str,
    dataset_version: str,
    top_k: int = 5,
) -> CogneeConfig:
    """Return the Option C local-index config when the dataset is registered,
    else the default Postgres/pgvector CogneeConfig."""
    try:
        return resolve_cognee_option_c(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            top_k=top_k,
        )
    except KeyError:
        return CogneeConfig(top_k=top_k)
