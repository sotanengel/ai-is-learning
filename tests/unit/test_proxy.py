"""Tests for ProxyHandler (unit - mocks the LLM adapter)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.ingress.proxy import ProxyHandler
from kolb_loop.logger.experience_logger import ExperienceLogger
from kolb_loop.memory.db import EpisodicDB


def _make_app(mock_response: dict[str, Any], fail_open: bool = True) -> FastAPI:
    db = EpisodicDB(":memory:")
    logger = ExperienceLogger(db)

    adapter = MagicMock(spec=LLMAdapter)
    adapter.chat_completions = AsyncMock(return_value=mock_response)

    proxy = ProxyHandler(adapter, logger, fail_open=fail_open)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await proxy.handle(request)

    return app


_MOCK_RESP: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def test_proxy_non_stream_success() -> None:
    app = _make_app(_MOCK_RESP)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3:8b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello there"


def test_proxy_logs_experience() -> None:
    db = EpisodicDB(":memory:")
    logger = ExperienceLogger(db)
    adapter = MagicMock(spec=LLMAdapter)
    adapter.chat_completions = AsyncMock(return_value=_MOCK_RESP)
    proxy = ProxyHandler(adapter, logger, fail_open=True)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def ep(request: Request) -> Response:
        return await proxy.handle(request)

    client = TestClient(app)
    client.post(
        "/v1/chat/completions",
        json={"model": "qwen3:8b", "messages": [{"role": "user", "content": "test"}]},
    )

    exps = db.list_experiences()
    assert len(exps) == 1
    assert exps[0].model == "qwen3:8b"


def test_proxy_fail_open_on_error() -> None:
    db = EpisodicDB(":memory:")
    logger = ExperienceLogger(db)
    adapter = MagicMock(spec=LLMAdapter)
    adapter.chat_completions = AsyncMock(side_effect=Exception("backend down"))
    proxy = ProxyHandler(adapter, logger, fail_open=True)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def ep(request: Request) -> Response:
        return await proxy.handle(request)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": []},
    )
    assert resp.status_code == 502
    exps = db.list_experiences()
    assert exps[0].error == "backend down"


def test_proxy_invalid_json() -> None:
    app = _make_app(_MOCK_RESP)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        content=b"not-json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
