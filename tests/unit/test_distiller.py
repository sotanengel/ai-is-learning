"""Tests for ConceptDistiller and ConceptStore (TDD)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.distiller.concept_distiller import ConceptDistiller, ConceptStore, _parse_concept
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import ConceptStatus, Reflection, Verdict
from kolb_loop.memory.vector_store import VectorStore


def _make_reflection(verdict: Verdict = Verdict.FAILURE, causes: list[str] | None = None) -> Reflection:
    return Reflection(
        experience_id="exp-1",
        verdict=verdict,
        causes=causes or ["wrong approach"],
        improvement_hypotheses=["try better"],
    )


def _mock_adapter(content: str) -> LLMAdapter:
    adapter = MagicMock(spec=LLMAdapter)
    adapter.chat_completions = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]}
    )
    # Return one embedding vector per input text
    async def _embeddings(texts: list[str], model: str) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    adapter.embeddings = _embeddings
    return adapter


_CONCEPT_JSON = json.dumps({
    "category": "tool_use",
    "title": "Check logs before acting",
    "condition": "when diagnosing issues",
    "action": "read logs first",
    "expected_effect": "faster diagnosis",
})


def test_parse_concept_valid() -> None:
    concept = _parse_concept(_CONCEPT_JSON, ["ref-1"])
    assert concept is not None
    assert concept.title == "Check logs before acting"
    assert concept.category == "tool_use"
    assert "ref-1" in concept.source_reflection_ids


def test_parse_concept_invalid_returns_none() -> None:
    assert _parse_concept("not json", []) is None


def test_parse_concept_missing_field_returns_none() -> None:
    bad = json.dumps({"category": "general", "title": "no condition"})
    assert _parse_concept(bad, []) is None


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


async def test_distill_creates_concept(db: EpisodicDB) -> None:
    vs = VectorStore(":memory:")
    adapter = _mock_adapter(_CONCEPT_JSON)
    distiller = ConceptDistiller(adapter, db, vs, embedder_model="bge-m3")

    refs = [_make_reflection(), _make_reflection(Verdict.SUCCESS, ["good context"])]
    concepts = await distiller.distill(refs)

    assert len(concepts) >= 1
    assert concepts[0].title == "Check logs before acting"
    saved = db.list_concepts()
    assert len(saved) >= 1


async def test_distill_empty_reflections(db: EpisodicDB) -> None:
    vs = VectorStore(":memory:")
    adapter = _mock_adapter(_CONCEPT_JSON)
    distiller = ConceptDistiller(adapter, db, vs, embedder_model="bge-m3")
    concepts = await distiller.distill([])
    assert concepts == []


async def test_distill_dedup_skips_same_title(db: EpisodicDB) -> None:
    vs = VectorStore(":memory:")
    adapter = _mock_adapter(_CONCEPT_JSON)
    distiller = ConceptDistiller(adapter, db, vs, embedder_model="bge-m3")

    refs = [_make_reflection()]
    await distiller.distill(refs)
    # Second distill with same title should not create a duplicate
    concepts2 = await distiller.distill(refs)
    assert len(concepts2) == 0
    assert len(db.list_concepts()) == 1


def test_concept_store_promote(db: EpisodicDB) -> None:
    from kolb_loop.memory.schemas import Concept

    c = Concept(title="T", condition="c", action="a", expected_effect="e")
    db.save_concept(c)
    store = ConceptStore(db)

    assert store.promote(c.id) is True
    fetched = db.get_concept(c.id)
    assert fetched is not None
    assert fetched.status == ConceptStatus.VALIDATED


def test_concept_store_promote_nonexistent(db: EpisodicDB) -> None:
    store = ConceptStore(db)
    assert store.promote("nonexistent") is False


def test_concept_store_deprecate(db: EpisodicDB) -> None:
    from kolb_loop.memory.schemas import Concept

    c = Concept(title="T", condition="c", action="a", expected_effect="e")
    db.save_concept(c)
    store = ConceptStore(db)

    assert store.deprecate(c.id) is True
    fetched = db.get_concept(c.id)
    assert fetched is not None
    assert fetched.status == ConceptStatus.DEPRECATED
