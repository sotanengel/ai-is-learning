"""Tests for ReflectionEngine (TDD: tests before implementation)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Experience, Session, Verdict
from kolb_loop.reflection.engine import ReflectionEngine, _parse_reflection

# ---------------------------------------------------------------------------
# _parse_reflection unit tests
# ---------------------------------------------------------------------------


def test_parse_valid_json() -> None:
    raw = json.dumps(
        {
            "verdict": "failure",
            "causes": ["wrong tool called"],
            "improvement_hypotheses": ["use read tool instead"],
            "evidence_spans": [{"field": "response_message", "excerpt": "bash -c rm"}],
            "better_response": "I should have used the read tool.",
        }
    )
    ref = _parse_reflection(raw, "exp-1")
    assert ref.verdict == Verdict.FAILURE
    assert ref.causes == ["wrong tool called"]
    assert len(ref.evidence_spans) == 1
    assert ref.better_response == "I should have used the read tool."


def test_parse_json_embedded_in_text() -> None:
    raw = 'Here is my analysis:\n{"verdict": "success", "causes": [], "improvement_hypotheses": [], "evidence_spans": [], "better_response": null}'
    ref = _parse_reflection(raw, "exp-2")
    assert ref.verdict == Verdict.SUCCESS


def test_parse_invalid_json_returns_partial() -> None:
    ref = _parse_reflection("This is not JSON at all", "exp-3")
    assert ref.verdict == Verdict.PARTIAL
    assert ref.raw_llm_output == "This is not JSON at all"


def test_parse_missing_verdict_defaults_to_partial() -> None:
    raw = json.dumps({"causes": [], "improvement_hypotheses": [], "evidence_spans": []})
    ref = _parse_reflection(raw, "exp-4")
    assert ref.verdict == Verdict.PARTIAL


# ---------------------------------------------------------------------------
# ReflectionEngine integration (mocked LLM)
# ---------------------------------------------------------------------------


def _mock_adapter(response_content: str) -> LLMAdapter:
    adapter = MagicMock(spec=LLMAdapter)
    adapter.chat_completions = AsyncMock(
        return_value={
            "choices": [{"message": {"content": response_content}}]
        }
    )
    return adapter


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


@pytest.fixture
def experience(db: EpisodicDB) -> Experience:
    session = Session()
    db.save_session(session)
    exp = Experience(
        session_id=session.id,
        request_messages=[{"role": "user", "content": "delete all files"}],
        response_message={"role": "assistant", "content": "Sure, running rm -rf /"},
        model="qwen3:8b",
    )
    db.save_experience(exp)
    return exp


async def test_reflect_saves_to_db(db: EpisodicDB, experience: Experience) -> None:
    raw = json.dumps(
        {
            "verdict": "failure",
            "causes": ["dangerous command"],
            "improvement_hypotheses": ["refuse harmful requests"],
            "evidence_spans": [],
            "better_response": "I cannot help with that.",
        }
    )
    adapter = _mock_adapter(raw)
    engine = ReflectionEngine(adapter, db, model="qwen3:8b")

    ref = await engine.reflect(experience)

    assert ref.verdict == Verdict.FAILURE
    saved = db.get_reflections_for_experience(experience.id)
    assert len(saved) == 1
    assert saved[0].verdict == Verdict.FAILURE


async def test_reflect_batch_processes_unreflected(db: EpisodicDB) -> None:
    session = Session()
    db.save_session(session)
    for i in range(3):
        exp = Experience(
            session_id=session.id,
            request_messages=[{"role": "user", "content": f"msg {i}"}],
            model="m",
        )
        db.save_experience(exp)

    raw = json.dumps(
        {"verdict": "success", "causes": [], "improvement_hypotheses": [], "evidence_spans": []}
    )
    adapter = _mock_adapter(raw)
    engine = ReflectionEngine(adapter, db, model="m")

    refs = await engine.reflect_batch(limit=10)
    assert len(refs) == 3
    assert all(r.verdict == Verdict.SUCCESS for r in refs)


async def test_reflect_skips_already_reflected(db: EpisodicDB, experience: Experience) -> None:
    raw = json.dumps(
        {"verdict": "success", "causes": [], "improvement_hypotheses": [], "evidence_spans": []}
    )
    adapter = _mock_adapter(raw)
    engine = ReflectionEngine(adapter, db, model="m")

    # Reflect once
    await engine.reflect(experience)
    # Batch should now find 0 unreflected
    refs = await engine.reflect_batch(limit=10)
    assert len(refs) == 0
