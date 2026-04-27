"""Tests for KolbLoopOrchestrator (TDD)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import (
    Concept,
    Experience,
    Reflection,
    Session,
    Verdict,
)
from kolb_loop.orchestrator.orchestrator import CycleResult, KolbLoopOrchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


def _make_experience(db: EpisodicDB, model: str = "m") -> Experience:
    s = Session()
    db.save_session(s)
    exp = Experience(
        session_id=s.id,
        request_messages=[{"role": "user", "content": "hi"}],
        response_message={"role": "assistant", "content": "hello"},
        model=model,
    )
    db.save_experience(exp)
    return exp


def _mock_reflection_engine(reflections: list[Reflection] | None = None) -> MagicMock:
    engine = MagicMock()
    engine.reflect_batch = AsyncMock(return_value=reflections or [])
    engine.reflect = AsyncMock(
        side_effect=lambda exp: Reflection(experience_id=exp.id, verdict=Verdict.SUCCESS)
    )
    return engine


def _mock_distiller(concepts: list[Concept] | None = None) -> MagicMock:
    distiller = MagicMock()
    distiller.distill = AsyncMock(return_value=concepts or [])
    return distiller


def _mock_evaluator(deprecated: list[str] | None = None) -> MagicMock:
    evaluator = MagicMock()
    evaluator.record_trial = MagicMock()
    evaluator.update_concept_scores = MagicMock(return_value=deprecated or [])
    return evaluator


def _make_orchestrator(
    db: EpisodicDB,
    reflections: list[Reflection] | None = None,
    concepts: list[Concept] | None = None,
    deprecated: list[str] | None = None,
) -> KolbLoopOrchestrator:
    return KolbLoopOrchestrator(
        reflection_engine=_mock_reflection_engine(reflections),
        concept_distiller=_mock_distiller(concepts),
        evaluator=_mock_evaluator(deprecated),
        db=db,
    )


# ---------------------------------------------------------------------------
# run_cycle tests
# ---------------------------------------------------------------------------


async def test_run_cycle_empty_returns_zeros(db: EpisodicDB) -> None:
    orch = _make_orchestrator(db)
    result = await orch.run_cycle()

    assert isinstance(result, CycleResult)
    assert result.new_reflections == 0
    assert result.new_concepts == 0
    assert result.deprecated_concepts == 0


async def test_run_cycle_reflects_and_distills(db: EpisodicDB) -> None:
    exp = _make_experience(db)
    refs = [Reflection(experience_id=exp.id, verdict=Verdict.SUCCESS)]
    concepts = [
        Concept(
            title="T",
            condition="C",
            action="A",
            expected_effect="E",
        )
    ]
    orch = _make_orchestrator(db, reflections=refs, concepts=concepts)

    result = await orch.run_cycle()

    assert result.new_reflections == 1
    assert result.new_concepts == 1


async def test_run_cycle_skips_distillation_when_no_reflections(db: EpisodicDB) -> None:
    distiller = _mock_distiller()
    orch = KolbLoopOrchestrator(
        reflection_engine=_mock_reflection_engine([]),
        concept_distiller=distiller,
        evaluator=_mock_evaluator(),
        db=db,
    )
    await orch.run_cycle()

    distiller.distill.assert_not_called()


async def test_run_cycle_reports_deprecated(db: EpisodicDB) -> None:
    orch = _make_orchestrator(db, deprecated=["id-1", "id-2"])
    result = await orch.run_cycle()

    assert result.deprecated_concepts == 2


# ---------------------------------------------------------------------------
# on_experience tests
# ---------------------------------------------------------------------------


async def test_on_experience_records_trial(db: EpisodicDB) -> None:
    exp = _make_experience(db)
    evaluator = _mock_evaluator()
    orch = KolbLoopOrchestrator(
        reflection_engine=_mock_reflection_engine(),
        concept_distiller=_mock_distiller(),
        evaluator=evaluator,
        db=db,
    )

    await orch.on_experience(exp, injected_concept_ids=["c1"])

    evaluator.record_trial.assert_called_once_with(exp, ["c1"])


async def test_on_experience_no_concepts_records_empty(db: EpisodicDB) -> None:
    exp = _make_experience(db)
    evaluator = _mock_evaluator()
    orch = KolbLoopOrchestrator(
        reflection_engine=_mock_reflection_engine(),
        concept_distiller=_mock_distiller(),
        evaluator=evaluator,
        db=db,
    )

    await orch.on_experience(exp)

    evaluator.record_trial.assert_called_once_with(exp, [])


# ---------------------------------------------------------------------------
# distill_every_n tests
# ---------------------------------------------------------------------------


async def test_run_cycle_calls_eval_update(db: EpisodicDB) -> None:
    evaluator = _mock_evaluator()
    orch = KolbLoopOrchestrator(
        reflection_engine=_mock_reflection_engine(),
        concept_distiller=_mock_distiller(),
        evaluator=evaluator,
        db=db,
    )
    await orch.run_cycle()

    evaluator.update_concept_scores.assert_called_once()
