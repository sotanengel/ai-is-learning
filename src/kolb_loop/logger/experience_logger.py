"""Records every inference call as a structured Experience in the episodic store."""

from __future__ import annotations

import uuid
from typing import Any

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Experience


class ExperienceLogger:
    def __init__(self, db: EpisodicDB) -> None:
        self._db = db

    def log(
        self,
        session_id: str,
        request_messages: list[dict[str, Any]],
        model: str,
        response_message: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        usage: dict[str, int] | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Experience:
        exp = Experience(
            id=str(uuid.uuid4()),
            session_id=session_id,
            request_messages=request_messages,
            model=model,
            response_message=response_message,
            tool_calls=tool_calls or [],
            usage=usage or {},
            error=error,
            metadata=metadata or {},
        )
        self._db.save_experience(exp)
        return exp

    def attach_feedback(self, experience_id: str, score: float) -> None:
        self._db.update_feedback(experience_id, score)
