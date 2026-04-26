"""Tests for data models and episodic DB (TDD: tests written before implementation)."""

import pytest

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import (
    Adapter,
    AdapterStatus,
    Concept,
    ConceptStatus,
    Experience,
    Reflection,
    SampleType,
    Session,
    TrainingSample,
    Verdict,
)
from kolb_loop.memory.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_experience_defaults() -> None:
    exp = Experience(
        session_id="s1",
        request_messages=[{"role": "user", "content": "hello"}],
        model="qwen3:8b",
    )
    assert exp.id
    assert exp.user_id == "default"
    assert exp.allow_training is False
    assert exp.feedback_score is None


def test_reflection_verdict_enum() -> None:
    ref = Reflection(
        experience_id="e1",
        verdict=Verdict.FAILURE,
        causes=["wrong tool"],
        improvement_hypotheses=["try different approach"],
    )
    assert ref.verdict == Verdict.FAILURE
    assert len(ref.causes) == 1


def test_concept_defaults() -> None:
    c = Concept(
        title="Test Concept",
        condition="when X",
        action="do Y",
        expected_effect="Z improves",
    )
    assert c.status == ConceptStatus.HYPOTHESIS
    assert c.confidence == 0.0
    assert c.trial_stats.trials == 0


def test_training_sample_sft() -> None:
    s = TrainingSample(
        type=SampleType.SFT,
        quality_score=0.85,
        prompt="What is X?",
        chosen="X is Y.",
    )
    assert s.rejected is None
    assert s.quality_score == 0.85


def test_adapter_defaults() -> None:
    a = Adapter(base_model="qwen3-8b-instruct")
    assert a.status == AdapterStatus.SHADOW
    assert a.traffic_pct == 0


# ---------------------------------------------------------------------------
# EpisodicDB CRUD tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


def test_save_and_get_experience(db: EpisodicDB) -> None:
    session = Session()
    db.save_session(session)

    exp = Experience(
        session_id=session.id,
        request_messages=[{"role": "user", "content": "hi"}],
        model="qwen3:8b",
    )
    db.save_experience(exp)

    fetched = db.get_experience(exp.id)
    assert fetched is not None
    assert fetched.id == exp.id
    assert fetched.model == "qwen3:8b"
    assert fetched.request_messages[0]["content"] == "hi"


def test_list_experiences(db: EpisodicDB) -> None:
    session = Session()
    db.save_session(session)

    for i in range(3):
        db.save_experience(
            Experience(
                session_id=session.id,
                request_messages=[{"role": "user", "content": str(i)}],
                model="qwen3:8b",
            )
        )

    exps = db.list_experiences(limit=10)
    assert len(exps) == 3


def test_feedback_update(db: EpisodicDB) -> None:
    session = Session()
    db.save_session(session)

    exp = Experience(
        session_id=session.id,
        request_messages=[],
        model="qwen3:8b",
    )
    db.save_experience(exp)
    db.update_feedback(exp.id, 0.9)

    fetched = db.get_experience(exp.id)
    assert fetched is not None
    assert fetched.feedback_score == pytest.approx(0.9)


def test_save_and_get_reflection(db: EpisodicDB) -> None:
    session = Session()
    db.save_session(session)
    exp = Experience(session_id=session.id, request_messages=[], model="m")
    db.save_experience(exp)

    ref = Reflection(
        experience_id=exp.id,
        verdict=Verdict.SUCCESS,
        causes=["good context"],
        improvement_hypotheses=[],
    )
    db.save_reflection(ref)

    refs = db.get_reflections_for_experience(exp.id)
    assert len(refs) == 1
    assert refs[0].verdict == Verdict.SUCCESS


def test_list_unreflected(db: EpisodicDB) -> None:
    session = Session()
    db.save_session(session)

    exp1 = Experience(session_id=session.id, request_messages=[], model="m")
    exp2 = Experience(session_id=session.id, request_messages=[], model="m")
    db.save_experience(exp1)
    db.save_experience(exp2)

    # Reflect only exp1
    db.save_reflection(Reflection(experience_id=exp1.id, verdict=Verdict.SUCCESS))

    unreflected = db.list_unreflected_experience_ids()
    assert exp2.id in unreflected
    assert exp1.id not in unreflected


def test_save_and_get_concept(db: EpisodicDB) -> None:
    concept = Concept(
        title="PR review order",
        condition="when reviewing PR",
        action="check git log first",
        expected_effect="fewer missed context",
    )
    db.save_concept(concept)

    fetched = db.get_concept(concept.id)
    assert fetched is not None
    assert fetched.title == "PR review order"
    assert fetched.status == ConceptStatus.HYPOTHESIS


def test_list_concepts_by_status(db: EpisodicDB) -> None:
    db.save_concept(
        Concept(
            title="A",
            condition="c",
            action="a",
            expected_effect="e",
            status=ConceptStatus.VALIDATED,
        )
    )
    db.save_concept(
        Concept(
            title="B",
            condition="c",
            action="a",
            expected_effect="e",
            status=ConceptStatus.HYPOTHESIS,
        )
    )

    validated = db.list_concepts(status="validated")
    assert len(validated) == 1
    assert validated[0].title == "A"


def test_save_training_sample_and_count(db: EpisodicDB) -> None:
    sample = TrainingSample(
        type=SampleType.DPO,
        quality_score=0.75,
        prompt="Q",
        chosen="Good answer",
        rejected="Bad answer",
    )
    db.save_training_sample(sample)
    assert db.count_training_samples() == 1


def test_save_adapter(db: EpisodicDB) -> None:
    adapter = Adapter(base_model="qwen3-8b-instruct")
    db.save_adapter(adapter)
    # No error means success


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------


def test_vector_store_upsert_and_search() -> None:
    vs = VectorStore(":memory:")
    emb = [0.1, 0.2, 0.3]
    vs.upsert("c1", emb, {"title": "concept 1", "status": "hypothesis"})
    vs.upsert("c2", [0.9, 0.8, 0.7], {"title": "concept 2", "status": "validated"})

    results = vs.search([0.1, 0.2, 0.3], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "c1"


def test_vector_store_delete() -> None:
    vs = VectorStore(":memory:")
    vs.upsert("c1", [1.0, 0.0], {"title": "t", "status": "hypothesis"})
    vs.delete("c1")

    results = vs.search([1.0, 0.0], top_k=5)
    assert not any(r["id"] == "c1" for r in results)


def test_vector_store_empty_search() -> None:
    vs = VectorStore(":memory:")
    results = vs.search([0.1, 0.2], top_k=5)
    assert results == []
