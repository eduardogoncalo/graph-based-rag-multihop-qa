from benchmark.core.schemas import RetrievalResult, RetrievalTrace, RetrievedItem
from benchmark.storage.experiment_store import ExperimentStore
from tests.fakes import FakeConnection


def test_experiment_store_creates_run() -> None:
    connection = FakeConnection()
    store = ExperimentStore(connection)

    run_id = store.create_run(
        experiment_id="exp_1",
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        method_id="vector_rag",
        agent_mode="single_agent",
    )

    assert run_id.startswith("run_")
    assert "INSERT INTO runs" in connection.cursor_obj.executed[0][0]
    assert connection.commits == 1


def test_experiment_store_persists_retrieval_result_and_items() -> None:
    connection = FakeConnection()
    result = RetrievalResult(
        method_id="vector_rag",
        query="law",
        latency_ms=2.5,
        items=[
            RetrievedItem(
                item_id="vector_rag:chunk_1",
                text="Delaware law",
                source_document_id="doc_1",
                source_chunk_id="chunk_1",
                score=0.8,
            )
        ],
    )

    retrieval_result_id = ExperimentStore(connection).persist_retrieval_result(
        result=result,
        dataset_id="musique_smoke_20",
        dataset_version="v1",
        top_k=1,
    )

    assert retrieval_result_id.startswith("retrieval_")
    assert "INSERT INTO retrieval_results" in connection.cursor_obj.executed[0][0]
    assert "INSERT INTO retrieval_items" in connection.cursor_obj.executed[1][0]


def test_experiment_store_persists_one_retrieval_trace_per_run() -> None:
    connection = FakeConnection()
    trace = RetrievalTrace(
        trace_id="trace_1",
        run_id="run_1",
        trace_status="complete",
        trace={
            "framework_id": "lightrag_neo4j",
            "nodes": [{"id": "entity_1"}],
            "relationships": [{"id": "rel_1"}],
            "chunks": [{"chunk_id": "chunk_1"}],
            "final_context": {"text": "context"},
            "counts": {"nodes": 1, "relationships": 1, "chunks": 1},
        },
        raw_trace={"raw": True},
        metadata={"instrumentation_version": "retrieval_trace_v1"},
    )

    trace_id = ExperimentStore(connection).persist_retrieval_trace(trace=trace)

    assert trace_id == "trace_1"
    sql = connection.cursor_obj.executed[0][0]
    assert "INSERT INTO retrieval_traces" in sql
    assert "ON CONFLICT (run_id)" in sql
    assert connection.commits == 1


def test_experiment_store_persists_answer_agent_message_and_evaluation() -> None:
    connection = FakeConnection()
    store = ExperimentStore(connection)

    answer_id = store.persist_answer(
        run_id="run_1",
        method_id="vector_rag",
        agent_mode="single_agent",
        answer_text="Delaware.",
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        estimated_cost=0.001,
    )
    message_id = store.persist_agent_message(
        run_id="run_1",
        agent_name="retriever",
        role="assistant",
        content="Retrieved chunk_1",
    )
    evaluation_id = store.persist_evaluation_result(
        run_id="run_1",
        metric_name="evidence_recall_at_5",
        metric_value=1.0,
    )

    assert answer_id.startswith("answer_")
    assert message_id.startswith("message_")
    assert evaluation_id.startswith("eval_")
    sql = "\n".join(statement for statement, _ in connection.cursor_obj.executed)
    assert "INSERT INTO answers" in sql
    assert "INSERT INTO agent_messages" in sql
    assert "INSERT INTO evaluation_results" in sql
