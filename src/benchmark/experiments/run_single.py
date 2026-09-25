from __future__ import annotations

from typing import Any

from benchmark.agents import (
    AgentGraphState,
    AgentWorkflow,
    FinalAnswer,
    build_agent_graph,
    create_agent_llm,
)
from benchmark.core.ids import deterministic_id
from benchmark.core.schemas import RetrievalResult
from benchmark.core.settings import load_settings
from benchmark.embeddings import EmbeddingProvider, create_embedding_provider
from benchmark.methods.baselines import (
    ORACLE_METHOD_ID,
    ZERO_SHOT_METHOD_ID,
    retrieve_gold_documents,
    retrieve_zero_shot,
)
from benchmark.methods.cognee import CogneeAdapter
from benchmark.methods.cognee import build_workspace as build_cognee_workspace
from benchmark.methods.cognee import retrieve as retrieve_cognee
from benchmark.methods.cognee.config_builder import COGNEE_METHOD_ID
from benchmark.methods.cognee.option_c import resolve_cognee_config
from benchmark.methods.hipporag2 import Hipporag2Adapter
from benchmark.methods.hipporag2 import build_workspace as build_hipporag2_workspace
from benchmark.methods.hipporag2 import retrieve as retrieve_hipporag2
from benchmark.methods.hipporag2.config_builder import HIPPORAG2_METHOD_ID
from benchmark.methods.lightrag_neo4j import retrieve as retrieve_lightrag_neo4j
from benchmark.methods.lightrag_neo4j.adapter import LightRAGNeo4jAdapter
from benchmark.methods.lightrag_neo4j.config_builder import (
    LIGHTRAG_NEO4J_METHOD_ID,
    LightRAGNeo4jConfig,
)
from benchmark.methods.lightrag_neo4j.config_builder import (
    build_workspace as build_lightrag_neo4j_workspace,
)
from benchmark.methods.ms_graphrag.adapter import MicrosoftGraphRAGAdapter
from benchmark.methods.ms_graphrag.config_builder import MS_GRAPHRAG_METHOD_ID
from benchmark.methods.ms_graphrag.config_builder import (
    build_workspace as build_ms_graphrag_workspace,
)
from benchmark.methods.ms_graphrag.retriever import retrieve as retrieve_ms_graphrag
from benchmark.methods.vector_rag import retrieve as retrieve_vector
from benchmark.storage.experiment_store import ExperimentStore
from benchmark.storage.pgvector_store import VECTOR_RAG_METHOD_ID, PgVectorStore


def load_question(connection: Any, question_id: str) -> str:
    with connection.cursor() as cursor:
        cursor.execute("SELECT question FROM questions WHERE question_id = %s", (question_id,))
        row = cursor.fetchone()
    if not row:
        raise ValueError(f"Question not found: {question_id}")
    return str(row[0])


def run_agent_question(
    *,
    connection: Any,
    dataset_id: str,
    dataset_version: str,
    method_id: str,
    experiment_id: str,
    question_id: str,
    question: str,
    agent_mode: str,
    top_k: int = 5,
    workflow: AgentWorkflow | None = None,
) -> FinalAnswer:
    experiment_store = ExperimentStore(connection)
    experiment_store.create_experiment(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        metadata={"phase": "phase_4_langgraph"},
    )
    run_id = deterministic_id(
        "run",
        [experiment_id, dataset_id, dataset_version, method_id, agent_mode, question_id],
    )
    experiment_store.create_run(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        method_id=method_id,
        agent_mode=agent_mode,
        run_id=run_id,
    )

    resolved_workflow = workflow or build_agent_graph(
        retriever=_build_retriever(connection=connection, top_k=top_k),
        llm=create_agent_llm(),
        experiment_store=experiment_store,
        top_k=top_k,
    )
    state = AgentGraphState(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        method_id=method_id,
        experiment_id=experiment_id,
        run_id=run_id,
        agent_mode=agent_mode,
        question_id=question_id,
        question=question,
        agent_messages=[],
    )
    return resolved_workflow.run(state)


# Cache de PROCESSO (não de closure): run_agent_question constrói um retriever
# novo por pergunta, e o subprocess --serve do hipporag2 carrega o índice inteiro
# em RAM — um adapter por pergunta vaza um servidor de vários GiB por query.
_HIPPORAG2_ADAPTER: Hipporag2Adapter | None = None


def _build_retriever(connection: Any, top_k: int):
    vector_provider: EmbeddingProvider | None = None

    def retrieve(state: AgentGraphState, query: str) -> RetrievalResult:
        nonlocal vector_provider
        global _HIPPORAG2_ADAPTER
        method_id = state["method_id"]
        if method_id == ZERO_SHOT_METHOD_ID:
            return retrieve_zero_shot(query=query)
        if method_id == ORACLE_METHOD_ID:
            return retrieve_gold_documents(
                connection=connection,
                query=query,
                question_id=state["question_id"],
                dataset_id=state["dataset_id"],
                dataset_version=state["dataset_version"],
            )
        if method_id == VECTOR_RAG_METHOD_ID:
            if vector_provider is None:
                vector_provider = create_embedding_provider()
            return retrieve_vector(
                query=query,
                dataset_id=state["dataset_id"],
                dataset_version=state["dataset_version"],
                embedding_provider=vector_provider,
                store=PgVectorStore(connection, dimension=vector_provider.dimension),
                top_k=top_k,
            )
        if method_id == LIGHTRAG_NEO4J_METHOD_ID:
            settings = load_settings()
            adapter = LightRAGNeo4jAdapter(
                workspace=build_lightrag_neo4j_workspace(
                    artifacts_dir=settings.artifacts_dir,
                    dataset_id=state["dataset_id"],
                    dataset_version=state["dataset_version"],
                ),
                config=LightRAGNeo4jConfig(),
            )
            return retrieve_lightrag_neo4j(
                adapter=adapter,
                query=query,
                top_k=top_k,
                query_mode="mix",
            )
        if method_id == COGNEE_METHOD_ID:
            settings = load_settings()
            adapter = CogneeAdapter(
                workspace=build_cognee_workspace(
                    artifacts_dir=settings.artifacts_dir,
                    dataset_id=state["dataset_id"],
                    dataset_version=state["dataset_version"],
                ),
                config=resolve_cognee_config(
                    dataset_id=state["dataset_id"],
                    dataset_version=state["dataset_version"],
                    top_k=top_k,
                ),
            )
            return retrieve_cognee(
                adapter=adapter,
                query=query,
                top_k=top_k,
            )
        if method_id == MS_GRAPHRAG_METHOD_ID:
            settings = load_settings()
            adapter = MicrosoftGraphRAGAdapter(
                workspace=build_ms_graphrag_workspace(
                    artifacts_dir=settings.artifacts_dir,
                    dataset_id=state["dataset_id"],
                    dataset_version=state["dataset_version"],
                ),
            )
            return retrieve_ms_graphrag(
                adapter=adapter,
                query=query,
                top_k=top_k,
                query_method="local",
            )
        if method_id == HIPPORAG2_METHOD_ID:
            if _HIPPORAG2_ADAPTER is None:
                settings = load_settings()
                _HIPPORAG2_ADAPTER = Hipporag2Adapter(
                    workspace=build_hipporag2_workspace(
                        artifacts_dir=settings.artifacts_dir,
                        dataset_id=state["dataset_id"],
                        dataset_version=state["dataset_version"],
                    ),
                )
            return retrieve_hipporag2(
                adapter=_HIPPORAG2_ADAPTER,
                query=query,
                top_k=top_k,
            )
        raise ValueError(
            "Supported methods: vector_rag, lightrag_neo4j, cognee, ms_graphrag, hipporag2"
        )

    return retrieve
