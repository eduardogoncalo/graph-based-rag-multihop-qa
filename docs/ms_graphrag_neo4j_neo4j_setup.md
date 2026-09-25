# ms_graphrag_neo4j Neo4j Setup

Phase 8A.2 prepares the isolated Neo4j service required by `ms_graphrag_neo4j`.

This service is only for the Neo4j-backed Microsoft GraphRAG-style method. Do not use it for
LightRAG. Planned `lightrag_neo4j` has a separate service, ports, password, and volume.

## Service

`docker-compose.yml` defines:

- service: `neo4j_graphrag`
- image: `docker.io/library/neo4j:5`
- browser: `http://localhost:7474`
- bolt: `bolt://localhost:7687`
- user: `neo4j`
- password: `benchmark_graphrag`
- database: `neo4j`
- volume: `neo4j_graphrag_data`
- plugins: APOC and Graph Data Science

The GraphRAG method must use only:

```bash
GRAPHRAG_NEO4J_URI=bolt://localhost:7687
GRAPHRAG_NEO4J_USER=neo4j
GRAPHRAG_NEO4J_PASSWORD=benchmark_graphrag
GRAPHRAG_NEO4J_DATABASE=neo4j
```

Do not use `NEO4J_URI` or `LIGHTRAG_NEO4J_URI` for `ms_graphrag_neo4j`.

## Start

After explicit approval to run Docker:

```bash
docker compose up -d neo4j_graphrag
```

If port `7474` or `7687` is already occupied, stop the old process/container or temporarily change
the port mapping for local testing. Do not silently change the benchmark default ports, because the
default `GRAPHRAG_NEO4J_URI` assumes `bolt://localhost:7687`.

## Browser Verification

Open:

```text
http://localhost:7474
```

Login:

```text
username: neo4j
password: benchmark_graphrag
```

## Bolt Verification

After installing the Neo4j Python driver in the approved runtime, validate connectivity explicitly:

```bash
export GRAPHRAG_NEO4J_URI=bolt://localhost:7687
export GRAPHRAG_NEO4J_USER=neo4j
export GRAPHRAG_NEO4J_PASSWORD=benchmark_graphrag
export GRAPHRAG_NEO4J_DATABASE=neo4j
python scripts/validate_ms_graphrag_neo4j_env.py --check-neo4j
```

The validator does not connect unless `--check-neo4j` is passed.

## APOC Verification

In Neo4j Browser:

```cypher
RETURN apoc.version() AS apoc_version;
```

Or with the validator:

```bash
python scripts/validate_ms_graphrag_neo4j_env.py --check-neo4j
```

The validator checks APOC as part of the explicit live connectivity check.

## GDS Verification

In Neo4j Browser:

```cypher
RETURN gds.version() AS gds_version;
```

Or with the validator:

```bash
python scripts/validate_ms_graphrag_neo4j_env.py --check-neo4j
```

The validator checks GDS as part of the explicit live connectivity check.

## Stop Safely

Stop the service without deleting volumes:

```bash
docker compose stop neo4j_graphrag
```

Do not run `docker compose down -v` unless explicitly approved, because that deletes named volumes.

## LightRAG Isolation

`neo4j_lightrag` is reserved for planned `lightrag_neo4j` and is not used in this phase.

Its default ports and volume are separate:

- browser: `http://localhost:7475`
- bolt: `bolt://localhost:7688`
- volume: `neo4j_lightrag_data`
