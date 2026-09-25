from __future__ import annotations

from pathlib import Path

import typer

from benchmark import __version__
from benchmark.core.config_loader import (
    load_dataset_config,
    load_experiment_config,
    load_method_config,
)
from benchmark.core.settings import load_settings
from benchmark.embeddings import create_embedding_provider
from benchmark.evaluation import evaluate_run
from benchmark.experiments import (
    compare_experiment,
    generate_report,
    load_question,
    run_agent_question,
)
from benchmark.ingestion import (
    AmostraDivergenteError,
    MusiqueRawDataMissingError,
    TwowikiRawDataMissingError,
    export_canonical_jsonl,
    load_musique,
    load_twowiki_hipporag,
    verificar_amostra_registada,
)
from benchmark.methods.lightrag import LightRAGAdapter
from benchmark.methods.lightrag import index as index_lightrag
from benchmark.methods.lightrag import retrieve as retrieve_lightrag
from benchmark.methods.lightrag.config_builder import LIGHTRAG_METHOD_ID
from benchmark.methods.lightrag.config_builder import build_workspace as build_lightrag_workspace
from benchmark.methods.lightrag.indexer import load_canonical_documents as load_lightrag_documents
from benchmark.methods.lightrag_neo4j import LightRAGNeo4jAdapter
from benchmark.methods.lightrag_neo4j import index as index_lightrag_neo4j
from benchmark.methods.lightrag_neo4j import retrieve as retrieve_lightrag_neo4j
from benchmark.methods.lightrag_neo4j.config_builder import (
    LIGHTRAG_NEO4J_METHOD_ID,
    LightRAGNeo4jConfig,
)
from benchmark.methods.lightrag_neo4j.config_builder import (
    build_workspace as build_lightrag_neo4j_workspace,
)
from benchmark.methods.lightrag_neo4j.indexer import (
    load_canonical_documents as load_lightrag_neo4j_documents,
)
from benchmark.methods.ms_graphrag import MicrosoftGraphRAGAdapter
from benchmark.methods.ms_graphrag import index as index_ms_graphrag
from benchmark.methods.ms_graphrag import retrieve as retrieve_ms_graphrag
from benchmark.methods.ms_graphrag.config_builder import MS_GRAPHRAG_METHOD_ID, build_workspace
from benchmark.methods.ms_graphrag.indexer import load_canonical_documents
from benchmark.methods.ms_graphrag_neo4j import GraphRAGNeo4jAdapter
from benchmark.methods.ms_graphrag_neo4j import index as index_ms_graphrag_neo4j
from benchmark.methods.ms_graphrag_neo4j import retrieve as retrieve_ms_graphrag_neo4j
from benchmark.methods.ms_graphrag_neo4j.config_builder import (
    MS_GRAPHRAG_NEO4J_METHOD_ID,
    GraphRAGNeo4jConfig,
)
from benchmark.methods.ms_graphrag_neo4j.config_builder import (
    build_workspace as build_graphrag_neo4j_workspace,
)
from benchmark.methods.ms_graphrag_neo4j.indexer import (
    load_canonical_documents as load_graphrag_neo4j_documents,
)
from benchmark.methods.vector_rag import build_retrieval_prompt
from benchmark.methods.vector_rag import index_chunks as index_vector_chunks
from benchmark.methods.vector_rag import retrieve as retrieve_vector
from benchmark.methods.vector_rag.indexer import load_canonical_chunks
from benchmark.storage import CanonicalStore, ExperimentStore, Migrator, connect_postgres
from benchmark.storage.pgvector_store import VECTOR_RAG_METHOD_ID, PgVectorStore

# Dataset loader dispatch. Each entry maps a dataset identifier to its
# canonical loader and the dataset-specific "raw data missing" error so the
# CLI can report a clean message when the raw file is absent. All loaders
# share the (config, *, sample_size) calling convention.
LOADERS = {
    "musique": (load_musique, MusiqueRawDataMissingError),
    # A amostra de smoke: mesmo carregador e mesmo ficheiro bruto do `musique`,
    # identificador próprio. É o identificador que a separa — canónico próprio,
    # containers próprios, portas próprias. Ver configs/datasets/musique_smoke_20.yaml.
    "musique_smoke_20": (load_musique, MusiqueRawDataMissingError),
    # 2Wiki = ficheiros de avaliação publicados pelo HippoRAG (decisão 2026-07-09)
    "twowiki": (load_twowiki_hipporag, TwowikiRawDataMissingError),
    # A amostra de smoke do 2Wiki, a 2026-08-10. Mesmo carregador e mesmos
    # ficheiros brutos do `twowiki`; o config traz o bloco `sample:`, que é o
    # que faz o carregador amostrar as perguntas E restringir o corpus às
    # passagens delas — sem isso seriam 6119 documentos a indexar.
    "twowiki_smoke_20": (load_twowiki_hipporag, TwowikiRawDataMissingError),
}

app = typer.Typer(help="Multi-Agent Memory Benchmark CLI.")
config_app = typer.Typer(help="Configuration utilities.")
db_app = typer.Typer(help="Database utilities.")
app.add_typer(config_app, name="config")
app.add_typer(db_app, name="db")


@app.command()
def version() -> None:
    typer.echo(__version__)


def _check_sample_fingerprint(
    *,
    config,
    canonical,
    sample_size: int | None,
    allow_divergent: bool,
) -> None:
    """Recusa escrever uma amostra que não seja a da dissertação.

    Corre entre carregar e exportar, e não depois: uma amostra divergente
    escrita em disco é indistinguível de uma boa para tudo o que vem a seguir —
    indexação, leitura, juiz — e a divergência só apareceria nos números, que é
    tarde de mais para se dar por ela.

    Três casos em que não há nada a verificar, e nenhum deles é erro:

    - ``--sample-size`` explícito. A amostra pequena diverge por construção,
      que é para o que ela serve. Diz-se em voz alta, porque um aviso calado
      seria a mesma armadilha ao contrário.
    - o dataset não tem impressão registada — uma amostra de smoke, um dataset
      novo. Não há nada a afirmar sobre ele.
    - ``--allow-divergent-sample``, que é a saída explícita para quem sabe o
      que está a fazer. Explícita de propósito: a alternativa era alguém apagar
      o ficheiro de impressão, e aí perdia-se a verificação para sempre.
    """
    if sample_size is not None:
        typer.echo(
            f"warning: --sample-size {sample_size} produces a sample of its own, so the "
            "fingerprint check does not apply. Numbers from here are "
            "NOT comparable with the thesis."
        )
        return

    question_ids = [question.question_id for question in canonical.questions]
    document_ids = [document.document_id for document in canonical.documents]

    try:
        impressao = verificar_amostra_registada(
            config.dataset_id,
            config.dataset_version,
            question_ids=question_ids,
            document_ids=document_ids,
        )
    except AmostraDivergenteError as exc:
        if not allow_divergent:
            typer.echo(f"Error: {exc}", err=True)
            typer.echo(
                "Nothing was written. To ingest anyway, "
                "--allow-divergent-sample.",
                err=True,
            )
            raise typer.Exit(1) from exc
        typer.echo(f"warning: {exc}")
        typer.echo("warning: ingested anyway because of --allow-divergent-sample.")
        return

    if impressao is None:
        typer.echo(
            f"no fingerprint recorded for {config.dataset_id} "
            f"{config.dataset_version}; the sample was not verified"
        )
        return

    # A frase é deliberadamente sobre a impressão registada, e não sobre a
    # dissertação: a amostra de smoke tem impressão própria, e não é a dela.
    typer.echo(
        f"fingerprint matches: {impressao.num_questions} questions and "
        f"{impressao.num_documents} documents, the same as recorded for "
        f"{impressao.slug}"
    )


@app.command()
def ingest(
    dataset: str = typer.Option(
        ...,
        help=f"Dataset identifier. Supported: {', '.join(sorted(LOADERS))}.",
    ),
    version: str = typer.Option(..., help="Dataset version, for example 'v1'."),
    sample_size: int | None = typer.Option(
        None,
        min=1,
        help="Maximum number of QA pairs to ingest.",
    ),
    allow_divergent_sample: bool = typer.Option(
        False,
        "--allow-divergent-sample",
        help=(
            "Ingest even when the sample does not match the recorded fingerprint. "
            "The resulting numbers are NOT comparable with the dissertation."
        ),
    ),
) -> None:
    if dataset not in LOADERS:
        raise typer.BadParameter(
            f"Supported datasets: {', '.join(sorted(LOADERS))}"
        )
    loader, missing_error = LOADERS[dataset]

    config = load_dataset_config(Path("configs/datasets") / f"{dataset}.yaml")
    if config.dataset_version != version:
        raise typer.BadParameter(
            f"Config for {dataset} has version {config.dataset_version!r}, not {version!r}"
        )

    try:
        canonical = loader(config, sample_size=sample_size)
    except missing_error as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    _check_sample_fingerprint(
        config=config,
        canonical=canonical,
        sample_size=sample_size,
        allow_divergent=allow_divergent_sample,
    )

    files = export_canonical_jsonl(canonical, config.canonical_path)
    typer.echo(
        "ingested "
        f"{len(canonical.documents)} documents, "
        f"{len(canonical.chunks)} chunks, "
        f"{len(canonical.questions)} questions, "
        f"{len(canonical.gold_evidence)} gold evidence records"
    )
    typer.echo(f"canonical output: {Path(config.canonical_path)}")
    for name, path in files.items():
        typer.echo(f"{name}: {path}")


@db_app.command("migrate")
def migrate_db() -> None:
    connection = _connect_postgres()
    applied = Migrator(connection).apply()
    if not applied:
        typer.echo("database already up to date")
        return
    for migration in applied:
        typer.echo(f"applied migration: {migration.version}")


@app.command()
def persist(
    dataset: str = typer.Option(..., help="Dataset identifier."),
    version: str = typer.Option(..., help="Dataset version."),
) -> None:
    dataset_config = load_dataset_config(Path("configs/datasets") / f"{dataset}.yaml")
    if dataset_config.dataset_version != version:
        raise typer.BadParameter(
            f"Config for {dataset} has version {dataset_config.dataset_version!r}, not {version!r}"
        )
    connection = _connect_postgres()
    summary = CanonicalStore(connection).persist_from_path(config=dataset_config)
    typer.echo(
        "persisted "
        f"{summary.documents} documents, "
        f"{summary.chunks} chunks, "
        f"{summary.questions} questions, "
        f"{summary.gold_evidence} gold evidence records"
    )


def _avisar_se_o_indice_foi_limitado(
    *, canonical_path: str, indexados: int, max_documents: int | None
) -> None:
    """Diz quantos documentos entraram, de quantos havia.

    Existe porque o contrário aconteceu: `max_documents: 5` viajou nos configs
    do `lightrag_neo4j` e do `ms_graphrag_neo4j` como valor de desenvolvimento,
    e a CLI aplicava-o **sem dizer nada**. Quem clonasse o pacote e corresse a
    indexação ficava com um índice de cinco documentos e uma linha de sucesso.

    A indexação oficial da dissertação nunca passou por aqui — o
    `scripts/twowiki_lightrag_index.py` força `max_documents=None` — por isso o
    limite nunca se notou em execução nenhuma.

    Um limite pedido de propósito continua a ser legítimo. O que deixa de ser
    possível é ele passar despercebido.
    """
    disponiveis = None
    caminho = Path(canonical_path) / "documents.jsonl"
    if caminho.is_file():
        with caminho.open(encoding="utf-8") as ficheiro:
            disponiveis = sum(1 for linha in ficheiro if linha.strip())

    if disponiveis is None:
        return
    if indexados >= disponiveis:
        typer.echo(f"documents indexed: {indexados} of {disponiveis} (all of them)")
        return
    typer.echo(
        f"WARNING: indexed {indexados} of {disponiveis} available documents. "
        f"The limit came from index.max_documents={max_documents} in the method "
        "config. Set it to null to index everything."
    )


@app.command()
def index(
    dataset: str = typer.Option(..., help="Dataset identifier."),
    version: str = typer.Option(..., help="Dataset version."),
    method: str = typer.Option(..., help="Method identifier."),
) -> None:
    if method not in {
        VECTOR_RAG_METHOD_ID,
        MS_GRAPHRAG_METHOD_ID,
        MS_GRAPHRAG_NEO4J_METHOD_ID,
        LIGHTRAG_METHOD_ID,
        LIGHTRAG_NEO4J_METHOD_ID,
    }:
        raise typer.BadParameter(
            "Supported methods: vector_rag, ms_graphrag, ms_graphrag_neo4j, "
            "lightrag, lightrag_neo4j"
        )
    dataset_config = load_dataset_config(Path("configs/datasets") / f"{dataset}.yaml")
    method_config = load_method_config(Path("configs/methods") / f"{method}.yaml")
    if dataset_config.dataset_version != version:
        raise typer.BadParameter(
            f"Config for {dataset} has version {dataset_config.dataset_version!r}, not {version!r}"
        )
    if method_config.method_id != method:
        raise typer.BadParameter(f"Method config must have method_id={method!r}")

    if method == MS_GRAPHRAG_METHOD_ID:
        try:
            documents = load_canonical_documents(dataset_config.canonical_path)
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        adapter = _ms_graphrag_adapter(dataset_config.dataset_id, dataset_config.dataset_version)
        try:
            summary = index_ms_graphrag(
                adapter=adapter,
                documents=documents,
                index_method=getattr(method_config, "index", {}).get("method", "standard"),
            )
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(
            "indexed "
            f"{summary.document_count} documents for "
            f"{summary.dataset_id}_{summary.dataset_version}/{summary.method_id}"
        )
        _avisar_se_o_indice_foi_limitado(
            canonical_path=dataset_config.canonical_path,
            indexados=summary.document_count,
            max_documents=None,
        )
        typer.echo(f"workspace: {summary.workspace_dir}")
        return

    if method == MS_GRAPHRAG_NEO4J_METHOD_ID:
        index_config = getattr(method_config, "index", {})
        max_documents = index_config.get(
            "max_documents",
            getattr(method_config, "max_documents", None),
        )
        try:
            documents = load_graphrag_neo4j_documents(
                dataset_config.canonical_path,
                max_documents=int(max_documents) if max_documents is not None else None,
            )
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        adapter = _ms_graphrag_neo4j_adapter(
            dataset_config.dataset_id,
            dataset_config.dataset_version,
            method_config=method_config,
        )
        try:
            summary = index_ms_graphrag_neo4j(adapter=adapter, documents=documents)
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(
            "indexed "
            f"{summary.document_count} documents for "
            f"{summary.dataset_id}_{summary.dataset_version}/{summary.method_id}"
        )
        _avisar_se_o_indice_foi_limitado(
            canonical_path=dataset_config.canonical_path,
            indexados=summary.document_count,
            max_documents=int(max_documents) if max_documents is not None else None,
        )
        typer.echo(f"workspace: {summary.workspace_dir}")
        return

    if method == LIGHTRAG_METHOD_ID:
        try:
            documents = load_lightrag_documents(dataset_config.canonical_path)
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        adapter = _lightrag_adapter(dataset_config.dataset_id, dataset_config.dataset_version)
        query_config = getattr(method_config, "query", {})
        try:
            summary = index_lightrag(
                adapter=adapter,
                documents=documents,
                query_mode=query_config.get("default_mode", "mix"),
                top_k=int(query_config.get("top_k", 5)),
            )
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(
            "indexed "
            f"{summary.document_count} documents for "
            f"{summary.dataset_id}_{summary.dataset_version}/{summary.method_id}"
        )
        _avisar_se_o_indice_foi_limitado(
            canonical_path=dataset_config.canonical_path,
            indexados=summary.document_count,
            max_documents=None,
        )
        typer.echo(f"workspace: {summary.workspace_dir}")
        return

    if method == LIGHTRAG_NEO4J_METHOD_ID:
        index_config = getattr(method_config, "index", {})
        max_documents = index_config.get(
            "max_documents",
            getattr(method_config, "max_documents", None),
        )
        try:
            documents = load_lightrag_neo4j_documents(
                dataset_config.canonical_path,
                max_documents=int(max_documents) if max_documents is not None else None,
            )
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        adapter = _lightrag_neo4j_adapter(
            dataset_config.dataset_id,
            dataset_config.dataset_version,
            method_config=method_config,
        )
        try:
            summary = index_lightrag_neo4j(adapter=adapter, documents=documents)
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(
            "indexed "
            f"{summary.document_count} documents for "
            f"{summary.dataset_id}_{summary.dataset_version}/{summary.method_id}"
        )
        _avisar_se_o_indice_foi_limitado(
            canonical_path=dataset_config.canonical_path,
            indexados=summary.document_count,
            max_documents=int(max_documents) if max_documents is not None else None,
        )
        typer.echo(f"workspace: {summary.workspace_dir}")
        return

    try:
        chunks = load_canonical_chunks(dataset_config.canonical_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    connection = _connect_postgres()
    provider = _embedding_provider(method_config)
    store = PgVectorStore(connection, dimension=provider.dimension)
    summary = index_vector_chunks(
        chunks=chunks,
        embedding_provider=provider,
        store=store,
        dataset_id=dataset_config.dataset_id,
        dataset_version=dataset_config.dataset_version,
    )
    typer.echo(
        "indexed "
        f"{summary.chunk_count} chunks for "
        f"{summary.dataset_id}_{summary.dataset_version}/{summary.method_id} "
        f"using {summary.embedding_model}"
    )


@app.command("retrieve")
def retrieve_cmd(
    dataset: str = typer.Option(..., "--dataset", help="Dataset identifier."),
    version: str = typer.Option(..., "--version", help="Dataset version."),
    method: str = typer.Option(
        ...,
        "--method",
        help=(
            "Method identifier. Supports vector_rag, ms_graphrag, "
            "ms_graphrag_neo4j, lightrag, and lightrag_neo4j."
        ),
    ),
    query: str = typer.Option(..., "--query", help="Retrieval query."),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Number of chunks to retrieve."),
    query_method: str = typer.Option("local", "--query-method", help="GraphRAG query method."),
    query_mode: str = typer.Option("mix", "--query-mode", help="LightRAG query mode."),
    persist_result: bool = typer.Option(
        False,
        "--persist/--no-persist",
        help="Persist retrieval result in shared tracking tables.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run ID for persistence."),
    show_prompt: bool = typer.Option(
        False,
        "--show-prompt",
        help="Print the assembled RAG prompt.",
    ),
) -> None:
    if method not in {
        VECTOR_RAG_METHOD_ID,
        MS_GRAPHRAG_METHOD_ID,
        MS_GRAPHRAG_NEO4J_METHOD_ID,
        LIGHTRAG_METHOD_ID,
        LIGHTRAG_NEO4J_METHOD_ID,
    }:
        raise typer.BadParameter(
            "Supported methods: vector_rag, ms_graphrag, ms_graphrag_neo4j, "
            "lightrag, lightrag_neo4j"
        )

    dataset_config = load_dataset_config(Path("configs/datasets") / f"{dataset}.yaml")
    method_config = load_method_config(Path("configs/methods") / f"{method}.yaml")
    if dataset_config.dataset_version != version:
        raise typer.BadParameter(
            f"Config for {dataset} has version {dataset_config.dataset_version!r}, not {version!r}"
        )

    framework_method = method in {
        MS_GRAPHRAG_METHOD_ID,
        MS_GRAPHRAG_NEO4J_METHOD_ID,
        LIGHTRAG_METHOD_ID,
        LIGHTRAG_NEO4J_METHOD_ID,
    }
    connection = _connect_postgres() if persist_result or not framework_method else None
    experiment_store = (
        ExperimentStore(connection) if persist_result and connection is not None else None
    )
    if method == MS_GRAPHRAG_METHOD_ID:
        adapter = _ms_graphrag_adapter(dataset_config.dataset_id, dataset_config.dataset_version)
        try:
            result = retrieve_ms_graphrag(
                adapter=adapter,
                query=query,
                top_k=top_k,
                query_method=query_method,
                experiment_store=experiment_store,
                run_id=run_id,
            )
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
    elif method == LIGHTRAG_METHOD_ID:
        adapter = _lightrag_adapter(dataset_config.dataset_id, dataset_config.dataset_version)
        try:
            result = retrieve_lightrag(
                adapter=adapter,
                query=query,
                top_k=top_k,
                query_mode=query_mode,
                experiment_store=experiment_store,
                run_id=run_id,
            )
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
    elif method == LIGHTRAG_NEO4J_METHOD_ID:
        adapter = _lightrag_neo4j_adapter(
            dataset_config.dataset_id,
            dataset_config.dataset_version,
            method_config=method_config,
        )
        try:
            result = retrieve_lightrag_neo4j(
                adapter=adapter,
                query=query,
                top_k=top_k,
                query_mode=query_mode,
                experiment_store=experiment_store,
                run_id=run_id,
            )
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
    elif method == MS_GRAPHRAG_NEO4J_METHOD_ID:
        adapter = _ms_graphrag_neo4j_adapter(
            dataset_config.dataset_id,
            dataset_config.dataset_version,
            method_config=method_config,
        )
        try:
            result = retrieve_ms_graphrag_neo4j(
                adapter=adapter,
                query=query,
                top_k=top_k,
                experiment_store=experiment_store,
                run_id=run_id,
            )
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
    else:
        if connection is None:
            connection = _connect_postgres()
        provider = _embedding_provider(method_config)
        store = PgVectorStore(connection, dimension=provider.dimension)
        result = retrieve_vector(
            query=query,
            dataset_id=dataset_config.dataset_id,
            dataset_version=dataset_config.dataset_version,
            embedding_provider=provider,
            store=store,
            top_k=top_k,
            experiment_store=experiment_store,
            run_id=run_id,
        )
    typer.echo(f"retrieved {len(result.items)} chunks in {result.latency_ms:.2f} ms")
    for item in result.items:
        snippet = item.text.replace("\n", " ")[:160]
        score = f"{item.score:.4f}" if item.score is not None else "n/a"
        typer.echo(f"{item.source_chunk_id}\tscore={score}\t{snippet}")
    if show_prompt:
        typer.echo("")
        typer.echo(build_retrieval_prompt(query, result.items))


@app.command("run-single")
def run_single_cmd(
    dataset: str = typer.Option(..., "--dataset", help="Dataset identifier."),
    version: str = typer.Option(..., "--version", help="Dataset version."),
    method: str = typer.Option(
        ...,
        "--method",
        help="Method identifier. Supports vector_rag and lightrag_neo4j.",
    ),
    question_id: str = typer.Option(..., "--question-id", help="Canonical question ID."),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Number of chunks to retrieve."),
    experiment_id: str | None = typer.Option(
        None,
        "--experiment-id",
        help="Optional experiment ID.",
    ),
) -> None:
    _run_question_command(
        dataset=dataset,
        version=version,
        method=method,
        question_id=question_id,
        top_k=top_k,
        experiment_id=experiment_id,
        agent_mode="single_agent",
    )


@app.command("run-agent")
def run_agent_cmd(
    dataset: str = typer.Option(..., "--dataset", help="Dataset identifier."),
    version: str = typer.Option(..., "--version", help="Dataset version."),
    method: str = typer.Option(
        ...,
        "--method",
        help="Method identifier. Supports vector_rag and lightrag_neo4j.",
    ),
    question_id: str = typer.Option(..., "--question-id", help="Canonical question ID."),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Number of chunks to retrieve."),
    experiment_id: str | None = typer.Option(
        None,
        "--experiment-id",
        help="Optional experiment ID.",
    ),
) -> None:
    _run_question_command(
        dataset=dataset,
        version=version,
        method=method,
        question_id=question_id,
        top_k=top_k,
        experiment_id=experiment_id,
        agent_mode="multi_agent",
    )


@app.command()
def evaluate(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to evaluate."),
    k: int = typer.Option(5, "--k", min=1, help="Evidence cutoff."),
) -> None:
    store = ExperimentStore(_connect_postgres())
    try:
        summary = evaluate_run(run_id=run_id, store=store, k=k)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for metric_name, metric_value in sorted(summary.metrics.items()):
        typer.echo(f"{metric_name}: {metric_value:.6f}")


@app.command()
def compare(
    experiment_id: str = typer.Option(..., "--experiment-id", help="Experiment ID to compare."),
) -> None:
    rows = compare_experiment(
        experiment_id=experiment_id,
        store=ExperimentStore(_connect_postgres()),
    )
    typer.echo(_format_comparison_table(rows))


@app.command()
def report(
    experiment_id: str = typer.Option(..., "--experiment-id", help="Experiment ID to report."),
) -> None:
    typer.echo(
        generate_report(
            experiment_id=experiment_id,
            store=ExperimentStore(_connect_postgres()),
        )
    )


@config_app.command("validate")
def validate_config(path: Path, kind: str = typer.Option("experiment")) -> None:
    loaders = {
        "dataset": load_dataset_config,
        "method": load_method_config,
        "experiment": load_experiment_config,
    }
    if kind not in loaders:
        raise typer.BadParameter(f"kind must be one of: {', '.join(loaders)}")

    config = loaders[kind](path)
    identifier = getattr(config, f"{kind}_id")
    typer.echo(f"valid {kind} config: {identifier}")


def _embedding_provider(method_config: object) -> object:
    try:
        return create_embedding_provider(method_config=method_config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _connect_postgres() -> object:
    try:
        return connect_postgres()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


def _ms_graphrag_adapter(dataset_id: str, dataset_version: str) -> MicrosoftGraphRAGAdapter:
    settings = load_settings()
    workspace = build_workspace(
        artifacts_dir=settings.artifacts_dir,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )
    return MicrosoftGraphRAGAdapter(workspace=workspace)


def _ms_graphrag_neo4j_adapter(
    dataset_id: str,
    dataset_version: str,
    *,
    method_config: object,
) -> GraphRAGNeo4jAdapter:
    settings = load_settings()
    workspace = build_graphrag_neo4j_workspace(
        artifacts_dir=settings.artifacts_dir,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )
    query_config = getattr(method_config, "query", {})
    index_config = getattr(method_config, "index", {})
    config = GraphRAGNeo4jConfig(
        method_id=method_config.method_id,
        neo4j_uri_env=getattr(method_config, "neo4j_uri_env", "GRAPHRAG_NEO4J_URI"),
        neo4j_user_env=getattr(method_config, "neo4j_user_env", "GRAPHRAG_NEO4J_USER"),
        neo4j_password_env=getattr(
            method_config,
            "neo4j_password_env",
            "GRAPHRAG_NEO4J_PASSWORD",
        ),
        neo4j_database_env=getattr(
            method_config,
            "neo4j_database_env",
            "GRAPHRAG_NEO4J_DATABASE",
        ),
        dependency_mode=getattr(method_config, "dependency_mode", "optional"),
        top_k=int(query_config.get("top_k", 5)),
        max_documents=index_config.get(
            "max_documents", getattr(method_config, "max_documents", None)
        ),
    )
    return GraphRAGNeo4jAdapter(workspace=workspace, config=config)


def _lightrag_adapter(dataset_id: str, dataset_version: str) -> LightRAGAdapter:
    settings = load_settings()
    workspace = build_lightrag_workspace(
        artifacts_dir=settings.artifacts_dir,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )
    return LightRAGAdapter(workspace=workspace)


def _lightrag_neo4j_adapter(
    dataset_id: str,
    dataset_version: str,
    *,
    method_config: object,
) -> LightRAGNeo4jAdapter:
    settings = load_settings()
    workspace = build_lightrag_neo4j_workspace(
        artifacts_dir=settings.artifacts_dir,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )
    query_config = getattr(method_config, "query", {})
    index_config = getattr(method_config, "index", {})
    config = LightRAGNeo4jConfig(
        method_id=method_config.method_id,
        neo4j_uri_env=getattr(method_config, "neo4j_uri_env", "LIGHTRAG_NEO4J_URI"),
        neo4j_user_env=getattr(method_config, "neo4j_user_env", "LIGHTRAG_NEO4J_USER"),
        neo4j_password_env=getattr(
            method_config,
            "neo4j_password_env",
            "LIGHTRAG_NEO4J_PASSWORD",
        ),
        neo4j_database_env=getattr(
            method_config,
            "neo4j_database_env",
            "LIGHTRAG_NEO4J_DATABASE",
        ),
        dependency_mode=getattr(method_config, "dependency_mode", "optional"),
        query_mode=query_config.get("default_mode", "mix"),
        top_k=int(query_config.get("top_k", 5)),
        max_documents=index_config.get(
            "max_documents", getattr(method_config, "max_documents", None)
        ),
    )
    return LightRAGNeo4jAdapter(workspace=workspace, config=config)


def _run_question_command(
    *,
    dataset: str,
    version: str,
    method: str,
    question_id: str,
    top_k: int,
    experiment_id: str | None,
    agent_mode: str,
) -> None:
    if method not in {VECTOR_RAG_METHOD_ID, LIGHTRAG_NEO4J_METHOD_ID}:
        raise typer.BadParameter("Supported methods: vector_rag, lightrag_neo4j")
    dataset_config = load_dataset_config(Path("configs/datasets") / f"{dataset}.yaml")
    if dataset_config.dataset_version != version:
        raise typer.BadParameter(
            f"Config for {dataset} has version {dataset_config.dataset_version!r}, not {version!r}"
        )
    connection = _connect_postgres()
    try:
        question = load_question(connection, question_id)
        answer = run_agent_question(
            connection=connection,
            dataset_id=dataset_config.dataset_id,
            dataset_version=dataset_config.dataset_version,
            method_id=method,
            experiment_id=experiment_id or f"{dataset}_{version}_{method}_{agent_mode}",
            question_id=question_id,
            question=question,
            agent_mode=agent_mode,
            top_k=top_k,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(answer.answer_text)
    if answer.citations:
        typer.echo(f"citations: {', '.join(answer.citations)}")


def _format_comparison_table(rows: object) -> str:
    lines = [
        "method_id\tagent_mode\tquestion_type\texact_match\tanswer_f1\t"
        "evidence_recall@5\tcitation_accuracy\tlatency_ms\ttotal_tokens\testimated_cost"
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.method_id,
                    row.agent_mode,
                    row.question_type,
                    f"{row.exact_match:.4f}",
                    f"{row.answer_f1:.4f}",
                    f"{row.evidence_recall_at_5:.4f}",
                    f"{row.citation_accuracy:.4f}",
                    f"{row.latency_ms:.2f}",
                    f"{row.total_tokens:.0f}",
                    f"{row.estimated_cost:.6f}",
                ]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    app()
