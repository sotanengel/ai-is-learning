"""Custom REST API for accessing the Kolb Loop learning state."""

from __future__ import annotations

from fastapi import FastAPI, Query

from kolb_loop.memory.db import EpisodicDB


def create_api_app(db: EpisodicDB) -> FastAPI:
    app = FastAPI(title="Kolb Loop API", version="0.1.0")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/experiences")
    async def list_experiences(
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, object]:
        exps = db.list_experiences(limit=limit, offset=offset)
        return {
            "experiences": [
                {
                    "id": e.id,
                    "session_id": e.session_id,
                    "model": e.model,
                    "created_at": e.created_at.isoformat(),
                    "feedback_score": e.feedback_score,
                    "error": e.error,
                }
                for e in exps
            ]
        }

    @app.get("/api/concepts")
    async def list_concepts(
        status: str | None = Query(default=None),
    ) -> dict[str, object]:
        concepts = db.list_concepts(status=status)
        return {
            "concepts": [
                {
                    "id": c.id,
                    "title": c.title,
                    "category": c.category,
                    "condition": c.condition,
                    "action": c.action,
                    "expected_effect": c.expected_effect,
                    "status": c.status,
                    "confidence": c.confidence,
                    "support_count": c.support_count,
                }
                for c in concepts
            ]
        }

    @app.get("/api/stats")
    async def stats() -> dict[str, int]:
        exp_count = len(db.list_experiences(limit=10000))
        concept_count = len(db.list_concepts())
        sample_count = db.count_training_samples()
        return {
            "experiences": exp_count,
            "concepts": concept_count,
            "training_samples": sample_count,
        }

    return app
