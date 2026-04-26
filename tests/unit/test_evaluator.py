"""Tests for Evaluator (TDD)."""

from __future__ import annotations

import pytest

from kolb_loop.evaluator.evaluator import Evaluator
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Concept, ConceptStatus, Experience, Session


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


def _make_exp(
    db: EpisodicDB, error: str | None = None, feedback: float | None = None
) -> Experience:
    session = Session()
    db.save_session(session)
    exp = Experience(
        session_id=session.id, request_messages=[], model="m", error=error, feedback_score=feedback
    )
    db.save_experience(exp)
    return exp


def _make_concept(db: EpisodicDB) -> Concept:
    c = Concept(title="T", condition="c", action="a", expected_effect="e")
    db.save_concept(c)
    return c


def test_record_injected_trial(db: EpisodicDB) -> None:
    ev = Evaluator(db)
    concept = _make_concept(db)
    exp = _make_exp(db)

    ev.record_trial(exp, [concept.id])
    stats = ev.get_trial_stats(concept.id)

    assert stats["trials"] == 1
    assert stats["injected_success_rate"] == 1.0


def test_record_baseline_trial(db: EpisodicDB) -> None:
    ev = Evaluator(db)
    concept = _make_concept(db)
    exp_inj = _make_exp(db)
    ev.record_trial(exp_inj, [concept.id])  # seed the concept in trials

    exp_base = _make_exp(db, error="timeout")
    ev.record_trial(exp_base, [])  # no injection → baseline

    stats = ev.get_trial_stats(concept.id)
    assert stats["baseline_success_rate"] == 0.0
    assert stats["trials"] == 2


def test_success_detection_error_is_failure(db: EpisodicDB) -> None:
    ev = Evaluator(db)
    concept = _make_concept(db)
    exp = _make_exp(db, error="backend error")
    ev.record_trial(exp, [concept.id])
    stats = ev.get_trial_stats(concept.id)
    assert stats["injected_success_rate"] == 0.0


def test_success_detection_feedback_positive(db: EpisodicDB) -> None:
    ev = Evaluator(db)
    concept = _make_concept(db)
    exp = _make_exp(db, feedback=0.8)
    ev.record_trial(exp, [concept.id])
    stats = ev.get_trial_stats(concept.id)
    assert stats["injected_success_rate"] == 1.0


def test_success_detection_feedback_negative(db: EpisodicDB) -> None:
    ev = Evaluator(db)
    concept = _make_concept(db)
    exp = _make_exp(db, feedback=-0.5)
    ev.record_trial(exp, [concept.id])
    stats = ev.get_trial_stats(concept.id)
    assert stats["injected_success_rate"] == 0.0


def test_update_concept_scores_deprecates_low_performer(db: EpisodicDB) -> None:
    ev = Evaluator(db, deprecate_threshold=0.4, min_trials=2)
    concept = _make_concept(db)

    # 10 injected failures, 10 baseline successes → lift very negative
    for _ in range(10):
        ev.record_trial(_make_exp(db, error="fail"), [concept.id])
    for _ in range(10):
        ev.record_trial(_make_exp(db), [])  # baseline success (no injection)
    # Seed baseline for concept tracking
    ev._trials.setdefault(concept.id, {"injected": [], "baseline": []})
    ev._trials[concept.id]["baseline"].extend([True] * 10)
    ev._trials[concept.id]["injected"].extend([False] * 10)

    deprecated = ev.update_concept_scores()
    assert concept.id in deprecated
    fetched = db.get_concept(concept.id)
    assert fetched is not None
    assert fetched.status == ConceptStatus.DEPRECATED


def test_update_concept_scores_validates_high_performer(db: EpisodicDB) -> None:
    ev = Evaluator(db, min_trials=2)
    concept = _make_concept(db)

    ev._trials[concept.id] = {
        "injected": [True] * 15,
        "baseline": [False] * 5,
    }
    ev.update_concept_scores()

    fetched = db.get_concept(concept.id)
    assert fetched is not None
    assert fetched.status == ConceptStatus.VALIDATED
    assert fetched.confidence >= 0.7


def test_update_skips_concepts_with_too_few_trials(db: EpisodicDB) -> None:
    ev = Evaluator(db, min_trials=10)
    concept = _make_concept(db)
    ev._trials[concept.id] = {"injected": [True], "baseline": []}

    ev.update_concept_scores()
    fetched = db.get_concept(concept.id)
    assert fetched is not None
    assert fetched.status == ConceptStatus.HYPOTHESIS  # unchanged
