"""Integration tests: proxy → experience logging → DB state."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from kolb_loop.ingress.proxy import ProxyHandler
from kolb_loop.logger.experience_logger import ExperienceLogger
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Session


def _fake_llm_response(content: str = "Hello!") -> dict[str, object]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _make_proxy_app(db: EpisodicDB, llm_response: dict[str, object] | None = None) -> TestClient:

    adapter = MagicMock()
    adapter.chat_completions = AsyncMock(return_value=llm_response or _fake_llm_response())
    adapter.aclose = AsyncMock()

    logger = ExperienceLogger(db)
    proxy = ProxyHandler(adapter, logger, fail_open=True)

    session = Session()
    db.save_session(session)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        return await proxy.handle(request)  # type: ignore[no-any-return]

    return TestClient(app)


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


def test_proxy_logs_experience_to_db(db: EpisodicDB) -> None:
    client = _make_proxy_app(db)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }

    resp = client.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"

    experiences = db.list_experiences()
    assert len(experiences) == 1
    exp = experiences[0]
    assert exp.model == "test-model"
    assert exp.request_messages[0]["content"] == "What is 2+2?"
    assert exp.response_message is not None
    assert exp.response_message["content"] == "Hello!"


def test_proxy_logs_error_on_llm_failure(db: EpisodicDB) -> None:

    adapter = MagicMock()
    adapter.chat_completions = AsyncMock(side_effect=RuntimeError("LLM down"))
    adapter.aclose = AsyncMock()

    logger = ExperienceLogger(db)
    proxy = ProxyHandler(adapter, logger, fail_open=True)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        return await proxy.handle(request)  # type: ignore[no-any-return]

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 502
    exps = db.list_experiences()
    assert len(exps) == 1
    assert exps[0].error == "LLM down"


def test_proxy_passes_session_id_header(db: EpisodicDB) -> None:
    client = _make_proxy_app(db)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "ping"}]},
        headers={"x-session-id": "test-session-123"},
    )

    assert resp.status_code == 200
    exps = db.list_experiences()
    assert exps[0].session_id == "test-session-123"


def test_proxy_multiple_requests_all_logged(db: EpisodicDB) -> None:
    client = _make_proxy_app(db)
    for i in range(3):
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": f"Q{i}"}]},
        )

    assert len(db.list_experiences()) == 3
