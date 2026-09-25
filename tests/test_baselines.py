from benchmark.agents.nodes import AgentNodeRunner, _retrieval_context_prompt
from benchmark.core.schemas import RetrievalResult
from benchmark.methods.baselines import (
    ORACLE_METHOD_ID,
    ZERO_SHOT_METHOD_ID,
    retrieve_gold_documents,
    retrieve_zero_shot,
)


def test_retrieve_zero_shot_returns_empty_result() -> None:
    result = retrieve_zero_shot(query="Who founded the company?")

    assert result.method_id == ZERO_SHOT_METHOD_ID
    assert result.items == []
    assert result.metadata["baseline"] == "closed_book"


def test_retrieve_gold_documents_builds_items_from_db_rows() -> None:
    connection = FakeConnection(
        rows=[
            ("doc_a", "Title A", "Body A."),
            ("doc_b", None, "Body B."),
        ]
    )

    result = retrieve_gold_documents(
        connection=connection,
        query="q?",
        question_id="q_1",
        dataset_id="musique",
        dataset_version="ans_v1.0_eval1k",
    )

    assert result.method_id == ORACLE_METHOD_ID
    assert [item.item_id for item in result.items] == ["doc_a", "doc_b"]
    assert result.items[0].text == "Title A\nBody A."
    assert result.items[1].text == "Body B."
    assert result.items[0].source_document_id == "doc_a"
    assert result.items[0].metadata["document_ids"] == ["doc_a"]
    assert connection.cursor_obj.params == ("q_1", "musique", "ans_v1.0_eval1k")


def test_retrieve_gold_documents_raises_when_no_golds() -> None:
    connection = FakeConnection(rows=[])
    try:
        retrieve_gold_documents(
            connection=connection,
            query="q?",
            question_id="q_missing",
            dataset_id="musique",
            dataset_version="ans_v1.0_eval1k",
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "q_missing" in str(exc)


def test_zero_shot_uses_closed_book_prompt() -> None:
    prompt = _retrieval_context_prompt(
        {
            "method_id": ZERO_SHOT_METHOD_ID,
            "question": "Who founded the company?",
        }
    )

    assert "own knowledge" in prompt
    assert "Retrieved chunks" not in prompt
    assert "ONLY the retrieved chunks" not in prompt


def test_oracle_uses_grounded_reader_prompt() -> None:
    prompt = _retrieval_context_prompt(
        {
            "method_id": ORACLE_METHOD_ID,
            "question": "Who founded the company?",
            "retrieval_result": retrieve_zero_shot(query="q").model_copy(
                update={"method_id": ORACLE_METHOD_ID}
            ),
        }
    )

    assert "ONLY the retrieved chunks" in prompt


def _oracle_prompt() -> str:
    return _retrieval_context_prompt(
        {
            "method_id": ORACLE_METHOD_ID,
            "question": "Who founded the company?",
            "retrieval_result": retrieve_zero_shot(query="q").model_copy(
                update={"method_id": ORACLE_METHOD_ID}
            ),
        }
    )


def test_reader_grounding_defaults_to_grounded(monkeypatch) -> None:
    monkeypatch.delenv("READER_GROUNDING", raising=False)
    prompt = _oracle_prompt()
    assert "ONLY the retrieved chunks" in prompt
    assert "insufficient instead of guessing" in prompt


def test_reader_grounding_v1_selects_free_reader(monkeypatch) -> None:
    monkeypatch.setenv("READER_GROUNDING", "v1")
    prompt = _oracle_prompt()
    # free reader: no ground-only restriction, no forced abstention
    assert "ONLY the retrieved chunks" not in prompt
    assert "insufficient instead of guessing" not in prompt
    assert "rely on your own knowledge" in prompt
    # scaffold (question + chunks block) is preserved so grounding is the
    # single variable that changes vs the grounded reader.
    assert "Retrieved chunks:" in prompt


def test_reader_grounding_off_aliases_map_to_free(monkeypatch) -> None:
    for value in ("off", "free", "0", "no", "none"):
        monkeypatch.setenv("READER_GROUNDING", value)
        assert "ONLY the retrieved chunks" not in _oracle_prompt()


def test_agent_graph_accepts_baseline_methods() -> None:
    from benchmark.agents import build_agent_graph

    def retriever(state, query: str) -> RetrievalResult:
        if state["method_id"] == ZERO_SHOT_METHOD_ID:
            return retrieve_zero_shot(query=query)
        return retrieve_gold_documents(
            connection=FakeConnection(rows=[("doc_a", "Title A", "Body A.")]),
            query=query,
            question_id=state["question_id"],
            dataset_id=state["dataset_id"],
            dataset_version=state["dataset_version"],
        )

    for method_id, expected_citations in [
        (ZERO_SHOT_METHOD_ID, []),
        (ORACLE_METHOD_ID, ["doc_a"]),
    ]:
        state = build_agent_graph(retriever=retriever).invoke(
            {
                "dataset_id": "musique",
                "dataset_version": "ans_v1.0_eval1k",
                "method_id": method_id,
                "experiment_id": "exp_1",
                "run_id": "run_1",
                "agent_mode": "single_agent",
                "question_id": "q_1",
                "question": "Who founded the company?",
                "agent_messages": [],
            }
        )
        assert state["final_answer"].citations == expected_citations


class FakeCursor:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.params = None

    def execute(self, sql, params) -> None:
        self.params = params

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class FakeConnection:
    def __init__(self, rows) -> None:
        self.cursor_obj = FakeCursor(rows)

    def cursor(self) -> FakeCursor:
        return self.cursor_obj
