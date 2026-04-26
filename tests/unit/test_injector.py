"""Tests for StrategyInjector (TDD)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.injector.strategy_injector import StrategyInjector
from kolb_loop.memory.schemas import Concept
from kolb_loop.memory.vector_store import VectorStore


def _make_concept(title: str = "Test concept", cid: str = "c1") -> Concept:
    c = Concept(title=title, condition="when X", action="do Y", expected_effect="Z")
    c.id = cid
    return c


def _make_adapter(embedding: list[float]) -> LLMAdapter:
    adapter = MagicMock(spec=LLMAdapter)
    adapter.embeddings = AsyncMock(return_value=[embedding])
    return adapter


async def test_inject_adds_system_message() -> None:
    vs = VectorStore(":memory:")
    concept = _make_concept()
    vs.upsert(concept.id, [1.0, 0.0, 0.0], {"title": concept.title, "status": "hypothesis"})

    adapter = _make_adapter([1.0, 0.0, 0.0])
    injector = StrategyInjector(adapter, vs, "bge-m3", ab_test_ratio=0.0)

    messages = [{"role": "user", "content": "what should I do?"}]
    augmented, ids = await injector.inject(messages, [concept])

    assert len(augmented) == 2
    assert augmented[0]["role"] == "system"
    assert "Test concept" in augmented[0]["content"]
    assert concept.id in ids


async def test_inject_appends_to_existing_system() -> None:
    vs = VectorStore(":memory:")
    concept = _make_concept()
    vs.upsert(concept.id, [1.0, 0.0], {"title": concept.title, "status": "hypothesis"})

    adapter = _make_adapter([1.0, 0.0])
    injector = StrategyInjector(adapter, vs, "bge-m3", ab_test_ratio=0.0)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "help me"},
    ]
    augmented, _ = await injector.inject(messages, [concept])

    assert augmented[0]["role"] == "system"
    assert "You are helpful." in augmented[0]["content"]
    assert "Test concept" in augmented[0]["content"]


async def test_inject_ab_test_no_injection() -> None:
    vs = VectorStore(":memory:")
    concept = _make_concept()
    vs.upsert(concept.id, [1.0, 0.0], {"title": concept.title, "status": "hypothesis"})

    adapter = _make_adapter([1.0, 0.0])
    # Force ab_test_ratio=1.0 → always skip injection
    injector = StrategyInjector(adapter, vs, "bge-m3", ab_test_ratio=1.0)

    messages = [{"role": "user", "content": "help"}]
    augmented, ids = await injector.inject(messages, [concept])

    assert augmented == messages
    assert ids == []


async def test_inject_empty_messages_skips() -> None:
    vs = VectorStore(":memory:")
    adapter = _make_adapter([0.1, 0.2])
    injector = StrategyInjector(adapter, vs, "bge-m3", ab_test_ratio=0.0)

    augmented, ids = await injector.inject([], [])
    assert augmented == []
    assert ids == []


async def test_inject_no_concepts_unchanged() -> None:
    vs = VectorStore(":memory:")
    adapter = _make_adapter([0.1, 0.2])
    injector = StrategyInjector(adapter, vs, "bge-m3", ab_test_ratio=0.0)

    messages = [{"role": "user", "content": "hi"}]
    augmented, ids = await injector.inject(messages, [])

    assert augmented == messages
    assert ids == []
