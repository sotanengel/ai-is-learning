"""Strategy Injector: enriches system prompts with relevant learned Concepts."""

from __future__ import annotations

import random
from typing import Any

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.memory.schemas import Concept
from kolb_loop.memory.vector_store import VectorStore


def _format_concept(concept: Concept) -> str:
    return (
        f"[Learned: {concept.title}] "
        f"When: {concept.condition} → Do: {concept.action} "
        f"(Expected: {concept.expected_effect})"
    )


class StrategyInjector:
    """Retrieves top-K relevant concepts and injects them into the system prompt.

    A/B mode: ab_test_ratio fraction of calls get no injection (control group).
    Injected concept IDs are returned so the Evaluator can attribute outcomes.
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        vector_store: VectorStore,
        embedder_model: str,
        max_concepts: int = 3,
        min_similarity: float = 0.6,
        ab_test_ratio: float = 0.2,
    ) -> None:
        self._adapter = adapter
        self._vs = vector_store
        self._embedder_model = embedder_model
        self._max_concepts = max_concepts
        self._min_similarity = min_similarity
        self._ab_test_ratio = ab_test_ratio

    async def inject(
        self, messages: list[dict[str, Any]], concepts: list[Concept]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Return (augmented_messages, injected_concept_ids).

        If A/B control: returns original messages, empty list.
        """
        if random.random() < self._ab_test_ratio:
            return messages, []

        if not concepts:
            return messages, []

        # Build query from the latest user message
        user_text = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        if not user_text.strip():
            return messages, []

        try:
            embeddings = await self._adapter.embeddings([user_text], self._embedder_model)
            query_emb = embeddings[0] if embeddings else []
        except Exception:
            return messages, []

        if not query_emb:
            return messages, []

        results = self._vs.search(query_emb, top_k=self._max_concepts)
        concept_map = {c.id: c for c in concepts}
        injected: list[Concept] = []
        for res in results:
            cid = res.get("id", "")
            if cid in concept_map:
                injected.append(concept_map[cid])

        if not injected:
            return messages, []

        hint = "\n".join(_format_concept(c) for c in injected)
        augmented = list(messages)

        # Prepend or append to system message
        if augmented and augmented[0].get("role") == "system":
            augmented[0] = {
                **augmented[0],
                "content": augmented[0]["content"] + "\n\n" + hint,
            }
        else:
            augmented = [{"role": "system", "content": hint}] + augmented

        return augmented, [c.id for c in injected]

    async def inject_from_store(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Search the vector store directly without pre-loaded concepts.

        Returns (augmented_messages, injected_concept_ids).
        """
        if random.random() < self._ab_test_ratio:
            return messages, []

        user_text = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        if not user_text.strip():
            return messages, []

        try:
            embeddings = await self._adapter.embeddings([user_text], self._embedder_model)
            query_emb = embeddings[0] if embeddings else []
        except Exception:
            return messages, []

        results = self._vs.search(query_emb, top_k=self._max_concepts)
        if not results:
            return messages, []

        hints = []
        injected_ids: list[str] = []
        for res in results:
            title = res.get("title", "")
            if title:
                hints.append(f"[Learned concept: {title}]")
                injected_ids.append(res.get("id", ""))

        if not hints:
            return messages, []

        hint_text = "\n".join(hints)
        augmented = list(messages)
        if augmented and augmented[0].get("role") == "system":
            augmented[0] = {
                **augmented[0],
                "content": augmented[0]["content"] + "\n\n" + hint_text,
            }
        else:
            augmented = [{"role": "system", "content": hint_text}] + augmented

        return augmented, injected_ids
