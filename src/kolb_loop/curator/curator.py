"""Curator: generates SFT/DPO/KTO training samples from Experience+Reflection pairs."""

from __future__ import annotations

import re
from typing import Any

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import (
    Experience,
    Reflection,
    SampleType,
    TrainingSample,
    Verdict,
)

_JUDGE_SYSTEM = """\
You are a quality evaluator for training data.
Rate the following training sample on a scale from 0.0 to 1.0.
Consider: accuracy, clarity, helpfulness, and safety.
Reply with ONLY a float number, e.g. "0.85".
"""

_SFT_SYSTEM = """\
Given an AI interaction that went wrong, produce an improved response.
Return ONLY the improved response text, nothing else.
"""


def _extract_float(text: str) -> float | None:
    match = re.search(r"(-?\d+\.\d+|-?\d+)", text.strip())
    if match:
        val = float(match.group())
        return max(0.0, min(1.0, val))
    return None


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class Curator:
    """Converts Experience + Reflection pairs into training samples.

    Strategy:
    - success + high feedback  → SFT (actual response as 'chosen')
    - failure with better_response → SFT (improved response as 'chosen')
    - failure without better_response → ask LLM for improved response → SFT
    - partial → KTO (binary OK/NG)
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        db: EpisodicDB,
        model: str,
        quality_threshold: float = 0.7,
    ) -> None:
        self._adapter = adapter
        self._db = db
        self._model = model
        self._quality_threshold = quality_threshold

    async def curate(self, experience: Experience, reflection: Reflection) -> TrainingSample | None:
        prompt = _messages_to_prompt(experience.request_messages)
        if not prompt.strip():
            return None

        sample = await self._build_sample(prompt, experience, reflection)
        if sample is None:
            return None

        score = await self._score(sample)
        if score < self._quality_threshold:
            return None

        sample.quality_score = score
        self._db.save_training_sample(sample)
        return sample

    async def curate_batch(self, limit: int = 100) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        exps = self._db.list_experiences(limit=limit)
        for exp in exps:
            refs = self._db.get_reflections_for_experience(exp.id)
            if not refs:
                continue
            for ref in refs:
                sample = await self.curate(exp, ref)
                if sample:
                    samples.append(sample)
        return samples

    async def _build_sample(
        self, prompt: str, experience: Experience, reflection: Reflection
    ) -> TrainingSample | None:
        actual_output = ""
        if experience.response_message:
            actual_output = experience.response_message.get("content", "")

        if reflection.verdict == Verdict.SUCCESS:
            return TrainingSample(
                type=SampleType.SFT,
                quality_score=0.0,
                prompt=prompt,
                chosen=actual_output or "",
                source_experience_ids=[experience.id],
                source_reflection_ids=[reflection.id],
            )

        if reflection.verdict == Verdict.FAILURE:
            chosen = reflection.better_response or await self._generate_improvement(
                prompt, actual_output, reflection
            )
            if not chosen:
                return None
            return TrainingSample(
                type=SampleType.SFT,
                quality_score=0.0,
                prompt=prompt,
                chosen=chosen,
                source_experience_ids=[experience.id],
                source_reflection_ids=[reflection.id],
            )

        # PARTIAL → KTO
        return TrainingSample(
            type=SampleType.KTO,
            quality_score=0.0,
            prompt=prompt,
            chosen=actual_output or "",
            source_experience_ids=[experience.id],
            source_reflection_ids=[reflection.id],
        )

    async def _generate_improvement(self, prompt: str, actual: str, reflection: Reflection) -> str:
        hypotheses = "; ".join(reflection.improvement_hypotheses[:3])
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SFT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Original prompt:\n{prompt}\n\n"
                        f"Bad response:\n{actual}\n\n"
                        f"Improvement hints: {hypotheses}"
                    ),
                },
            ],
            "temperature": 0.3,
        }
        try:
            resp = await self._adapter.chat_completions(payload)
            return str(resp.get("choices", [{}])[0].get("message", {}).get("content", ""))
        except Exception:
            return ""

    async def _score(self, sample: TrainingSample) -> float:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Prompt: {sample.prompt}\nResponse: {sample.chosen}",
                },
            ],
            "temperature": 0.0,
        }
        try:
            resp = await self._adapter.chat_completions(payload)
            raw: str = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            return _extract_float(raw) or 0.0
        except Exception:
            return 0.0
