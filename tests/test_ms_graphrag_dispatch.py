"""ms_graphrag is dispatchable through the agent graph (mirror of cognee's test)."""

from benchmark.agents import build_agent_graph
from benchmark.agents.nodes import _SUPPORTED_METHODS
from benchmark.core.schemas import RetrievalResult, RetrievedItem
from benchmark.methods.ms_graphrag.config_builder import MS_GRAPHRAG_METHOD_ID


def test_ms_graphrag_in_supported_methods() -> None:
    assert MS_GRAPHRAG_METHOD_ID in _SUPPORTED_METHODS


def test_agent_graph_accepts_ms_graphrag() -> None:
    def retriever(state, query: str) -> RetrievalResult:
        assert state["method_id"] == MS_GRAPHRAG_METHOD_ID
        return RetrievalResult(
            method_id=MS_GRAPHRAG_METHOD_ID,
            query=query,
            latency_ms=1.0,
            items=[
                RetrievedItem(
                    item_id="unit-uuid",
                    text="Sample text unit.",
                    source_chunk_id="7",
                    source_document_id="doc_6358038e054ac768",
                    metadata={"document_ids": ["doc_6358038e054ac768"]},
                )
            ],
        )

    graph = build_agent_graph(retriever=retriever)
    state = graph.invoke(
        {
            "dataset_id": "musique",
            "dataset_version": "ans_v1.0_eval1k",
            "method_id": MS_GRAPHRAG_METHOD_ID,
            "experiment_id": "exp_test",
            "run_id": "run_test",
            "agent_mode": "single_agent",
            "question_id": "q_test",
            "question": "Who founded the company?",
            "agent_messages": [],
        }
    )
    assert state["retrieval_result"].method_id == MS_GRAPHRAG_METHOD_ID
    assert state["final_answer"] is not None


def test_parser_document_ids_count_for_recall() -> None:
    """The parser's document_ids must register as a hit in the recall metric."""
    import json

    from benchmark.evaluation.retrieval_metrics import evidence_recall_at_k
    from benchmark.methods.ms_graphrag.output_parser import parse_query_output

    result = parse_query_output(
        query="q",
        query_method="local",
        latency_ms=1.0,
        top_k=20,
        stdout=json.dumps(
            {
                "context_text": "ctx",
                "items": [
                    {
                        "unit_id": "u1",
                        "unit_short_id": "1",
                        "document_id": "doc_6358038e054ac768",
                        "title": "doc_6358038e054ac768",
                        "text": "some text",
                        "in_context": True,
                    },
                    {
                        "unit_id": "u2",
                        "unit_short_id": "2",
                        "document_id": "doc_aaaabbbbccccdddd",
                        "title": "doc_aaaabbbbccccdddd",
                        "text": "other text",
                        "in_context": True,
                    },
                ],
                "stats": {"n_items": 2},
            }
        ),
    )

    assert result.items[0].metadata["document_ids"] == ["doc_6358038e054ac768"]
    recall = evidence_recall_at_k(
        result.items, ["doc_6358038e054ac768", "doc_ffff0000ffff0000"], k=20
    )
    assert recall == 0.5
