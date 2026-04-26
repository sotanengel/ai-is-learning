"""Reflection Engine: produces structured self-critique for every Experience."""

from __future__ import annotations

import json
import re
from typing import Any

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import (
    EvidenceSpan,
    Experience,
    Reflection,
    Verdict,
)

_SYSTEM_PROMPT = """\
You are an AI self-critic. Analyse the given inference exchange and output a JSON object
with the following fields:
{
  "verdict": "success" | "partial" | "failure",
  "causes": ["<brief cause 1>", ...],
  "improvement_hypotheses": ["<hypothesis 1>", ...],
  "evidence_spans": [{"field": "<request_messages|response_message>", "excerpt": "<short quote>"}],
  "better_response": "<improved response text or null>"
}
Return ONLY the JSON object, no surrounding text.
"""


def _parse_reflection(raw: str, experience_id: str) -> Reflection:
    # Extract the first JSON object from the response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return Reflection(
            experience_id=experience_id,
            verdict=Verdict.PARTIAL,
            raw_llm_output=raw,
        )
    try:
        data: dict[str, Any] = json.loads(match.group())
        return Reflection(
            experience_id=experience_id,
            verdict=Verdict(data.get("verdict", "partial")),
            causes=data.get("causes", []),
            improvement_hypotheses=data.get("improvement_hypotheses", []),
            evidence_spans=[
                EvidenceSpan(**s) for s in data.get("evidence_spans", [])
            ],
            better_response=data.get("better_response"),
            raw_llm_output=raw,
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return Reflection(
            experience_id=experience_id,
            verdict=Verdict.PARTIAL,
            raw_llm_output=raw,
        )


def _build_user_message(exp: Experience) -> str:
    return json.dumps(
        {
            "request_messages": exp.request_messages,
            "response_message": exp.response_message,
            "model": exp.model,
            "error": exp.error,
            "feedback_score": exp.feedback_score,
        },
        ensure_ascii=False,
        indent=2,
    )


class ReflectionEngine:
    """Produces structured Reflection for a given Experience.

    Supports three trigger modes:
    - sync: called inline after each experience (blocking)
    - async_after_each: fire-and-forget via asyncio.create_task
    - batch: driven by an external scheduler
    """

    def __init__(self, adapter: LLMAdapter, db: EpisodicDB, model: str) -> None:
        self._adapter = adapter
        self._db = db
        self._model = model

    async def reflect(self, experience: Experience) -> Reflection:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(experience)},
            ],
            "temperature": 0.3,
        }
        resp = await self._adapter.chat_completions(payload)
        raw: str = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        reflection = _parse_reflection(raw, experience.id)
        self._db.save_reflection(reflection)
        return reflection

    async def reflect_batch(self, limit: int = 50) -> list[Reflection]:
        exp_ids = self._db.list_unreflected_experience_ids(limit=limit)
        results: list[Reflection] = []
        for exp_id in exp_ids:
            exp = self._db.get_experience(exp_id)
            if exp is None:
                continue
            ref = await self.reflect(exp)
            results.append(ref)
        return results
