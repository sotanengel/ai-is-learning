"""Tests for ExperienceLogger."""

from kolb_loop.logger.experience_logger import ExperienceLogger
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Session


def _make_logger() -> tuple[EpisodicDB, ExperienceLogger]:
    db = EpisodicDB(":memory:")
    return db, ExperienceLogger(db)


def test_log_creates_experience() -> None:
    db, logger = _make_logger()
    session = Session()
    db.save_session(session)

    exp = logger.log(
        session_id=session.id,
        request_messages=[{"role": "user", "content": "Hello"}],
        model="qwen3:8b",
        response_message={"role": "assistant", "content": "Hi"},
        usage={"prompt_tokens": 5, "completion_tokens": 3},
    )

    fetched = db.get_experience(exp.id)
    assert fetched is not None
    assert fetched.model == "qwen3:8b"
    assert fetched.usage["prompt_tokens"] == 5


def test_log_error_experience() -> None:
    db, logger = _make_logger()
    session = Session()
    db.save_session(session)

    exp = logger.log(
        session_id=session.id,
        request_messages=[],
        model="m",
        error="timeout",
    )

    fetched = db.get_experience(exp.id)
    assert fetched is not None
    assert fetched.error == "timeout"
    assert fetched.response_message is None


def test_attach_feedback() -> None:
    db, logger = _make_logger()
    session = Session()
    db.save_session(session)

    exp = logger.log(session_id=session.id, request_messages=[], model="m")
    logger.attach_feedback(exp.id, 1.0)

    fetched = db.get_experience(exp.id)
    assert fetched is not None
    assert fetched.feedback_score == 1.0


def test_log_default_empty_fields() -> None:
    db, logger = _make_logger()
    exp = logger.log(session_id="s", request_messages=[], model="m")
    assert exp.tool_calls == []
    assert exp.usage == {}
    assert exp.metadata == {}
