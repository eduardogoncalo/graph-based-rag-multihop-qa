from __future__ import annotations

from typing import Protocol

from benchmark.agents.schemas import AgentGraphState


class AgentLLM(Protocol):
    def generate(self, *, agent_name: str, prompt: str, state: AgentGraphState) -> str:
        """Generate deterministic or model-backed agent text."""


class FakeAgentLLM:
    def generate(self, *, agent_name: str, prompt: str, state: AgentGraphState) -> str:
        question = state.get("question", "")
        retrieval_result = state.get("retrieval_result")
        chunk_ids = []
        if retrieval_result is not None:
            chunk_ids = [
                item.source_chunk_id or item.item_id
                for item in retrieval_result.items
                if item.source_chunk_id or item.item_id
            ]
        if agent_name == "domain_reasoner":
            citations = ", ".join(chunk_ids) if chunk_ids else "no_chunks"
            return f"Draft answer for '{question}' using {citations}."
        if agent_name == "final_answer":
            draft = state.get("draft_answer", "")
            return draft.replace("Draft answer", "Final answer")
        return f"{agent_name}: {prompt}"
