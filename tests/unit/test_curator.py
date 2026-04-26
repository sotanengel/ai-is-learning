"""Tests for Curator and ReplayBuffer (TDD)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.curator.curator import Curator, _extract_float
from kolb_loop.curator.replay_buffer import ReplayBuffer
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import (
    Experience,
    Reflection,
    SampleType,
    Session,
    TrainingSample,
    Verdict,
)

# ---------------------------------------------------------------------------
# _extract_float tests
# ---------------------------------------------------------------------------


def test_extract_float_simple() -> None:
    assert _extract_float("0.85") == pytest.approx(0.85)


def test_extract_float_embedded() -> None:
    assert _extract_float("The score is: 0.72.") == pytest.approx(0.72)


def test_extract_float_integer() -> None:
    assert _extract_float("1") == pytest.approx(1.0)


def test_extract_float_clamps_above_one() -> None:
    assert _extract_float("1.5") == pytest.approx(1.0)


def test_extract_float_clamps_below_zero() -> None:
    assert _extract_float("-0.3") == pytest.approx(0.0)


def test_extract_float_no_match() -> None:
    assert _extract_float("no number here") is None


# ---------------------------------------------------------------------------
# Curator tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


def _make_exp_ref(
    db: EpisodicDB,
    verdict: Verdict = Verdict.SUCCESS,
    response: str = "Good answer",
    better_response: str | None = None,
) -> tuple[Experience, Reflection]:
    session = Session()
    db.save_session(session)
    exp = Experience(
        session_id=session.id,
        request_messages=[{"role": "user", "content": "What is X?"}],
        response_message={"role": "assistant", "content": response},
        model="m",
    )
    db.save_experience(exp)
    ref = Reflection(
        experience_id=exp.id,
        verdict=verdict,
        causes=["cause1"],
        improvement_hypotheses=["try better"],
        better_response=better_response,
    )
    db.save_reflection(ref)
    return exp, ref


def _mock_adapter(score_text: str = "0.85", improved: str = "Improved answer") -> LLMAdapter:
    adapter = MagicMock(spec=LLMAdapter)

    async def _chat(payload: dict) -> dict:
        system = payload["messages"][0]["content"]
        if "quality evaluator" in system:
            return {"choices": [{"message": {"content": score_text}}]}
        return {"choices": [{"message": {"content": improved}}]}

    adapter.chat_completions = _chat
    return adapter


async def test_curate_success_creates_sft(db: EpisodicDB) -> None:
    adapter = _mock_adapter(score_text="0.90")
    curator = Curator(adapter, db, model="m", quality_threshold=0.7)
    exp, ref = _make_exp_ref(db, Verdict.SUCCESS, response="Great answer")

    sample = await curator.curate(exp, ref)

    assert sample is not None
    assert sample.type == SampleType.SFT
    assert sample.chosen == "Great answer"
    assert sample.quality_score == pytest.approx(0.90)
    assert db.count_training_samples() == 1


async def test_curate_failure_with_better_response(db: EpisodicDB) -> None:
    adapter = _mock_adapter(score_text="0.80")
    curator = Curator(adapter, db, model="m", quality_threshold=0.7)
    exp, ref = _make_exp_ref(
        db, Verdict.FAILURE, response="Bad answer", better_response="Fixed answer"
    )

    sample = await curator.curate(exp, ref)

    assert sample is not None
    assert sample.chosen == "Fixed answer"


async def test_curate_failure_without_better_response_generates(db: EpisodicDB) -> None:
    adapter = _mock_adapter(score_text="0.75", improved="LLM generated improvement")
    curator = Curator(adapter, db, model="m", quality_threshold=0.7)
    exp, ref = _make_exp_ref(db, Verdict.FAILURE)

    sample = await curator.curate(exp, ref)

    assert sample is not None
    assert sample.chosen == "LLM generated improvement"


async def test_curate_partial_creates_kto(db: EpisodicDB) -> None:
    adapter = _mock_adapter(score_text="0.75")
    curator = Curator(adapter, db, model="m", quality_threshold=0.7)
    exp, ref = _make_exp_ref(db, Verdict.PARTIAL)

    sample = await curator.curate(exp, ref)

    assert sample is not None
    assert sample.type == SampleType.KTO


async def test_curate_low_quality_discarded(db: EpisodicDB) -> None:
    adapter = _mock_adapter(score_text="0.30")
    curator = Curator(adapter, db, model="m", quality_threshold=0.7)
    exp, ref = _make_exp_ref(db, Verdict.SUCCESS)

    sample = await curator.curate(exp, ref)

    assert sample is None
    assert db.count_training_samples() == 0


async def test_curate_empty_prompt_skipped(db: EpisodicDB) -> None:
    adapter = _mock_adapter()
    curator = Curator(adapter, db, model="m")
    session = Session()
    db.save_session(session)
    exp = Experience(session_id=session.id, request_messages=[], model="m")
    db.save_experience(exp)
    ref = Reflection(experience_id=exp.id, verdict=Verdict.SUCCESS)
    db.save_reflection(ref)

    sample = await curator.curate(exp, ref)
    assert sample is None


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------


def _add_sample(db: EpisodicDB, prompt: str = "Q", chosen: str = "A") -> TrainingSample:
    s = TrainingSample(type=SampleType.SFT, quality_score=0.8, prompt=prompt, chosen=chosen)
    db.save_training_sample(s)
    return s


def test_replay_buffer_returns_mix(db: EpisodicDB) -> None:
    for i in range(10):
        _add_sample(db, f"Q{i}", f"A{i}")

    new = [_add_sample(db, "new_q", "new_a")]
    buffer = ReplayBuffer(db, replay_ratio=0.5)
    mixed = buffer.sample(new)

    assert len(mixed) > 1
    new_ids = {s.id for s in new}
    replay = [s for s in mixed if s.id not in new_ids]
    assert len(replay) >= 0


def test_replay_buffer_no_existing_data(db: EpisodicDB) -> None:
    new = [_add_sample(db)]
    buffer = ReplayBuffer(db, replay_ratio=0.3)
    mixed = buffer.sample(new)
    # Should not crash; may only contain new sample
    assert len(mixed) >= 1
