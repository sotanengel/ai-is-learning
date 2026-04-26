"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request, Response

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.config import Settings, load_settings
from kolb_loop.ingress.proxy import ProxyHandler
from kolb_loop.logger.experience_logger import ExperienceLogger
from kolb_loop.memory.db import EpisodicDB


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = load_settings()

    db = EpisodicDB(settings.memory.episodic.path)
    adapter = LLMAdapter(
        base_url=settings.backends.main.base_url,
        api_key=settings.backends.main.api_key,
    )
    logger = ExperienceLogger(db)
    proxy = ProxyHandler(adapter, logger, fail_open=settings.sidecar.fail_open)

    app = FastAPI(title="Kolb Loop Sidecar", version="0.1.0")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await proxy.handle(request)

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": settings.backends.main.model, "object": "model"}],
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await adapter.aclose()
        db.close()

    return app
