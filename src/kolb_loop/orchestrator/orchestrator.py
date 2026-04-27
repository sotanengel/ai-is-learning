"""KolbLoopOrchestrator: wires Experience → Reflection → Distillation → Evaluation."""

from __future__ import annotations

from pydantic import BaseModel

from kolb_loop.distiller.concept_distiller import ConceptDistiller
from kolb_loop.evaluator.evaluator import Evaluator
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Concept, Experience, Reflection
from kolb_loop.reflection.engine import ReflectionEngine


class CycleResult(BaseModel):
    new_reflections: int = 0
    new_concepts: int = 0
    deprecated_concepts: int = 0


class KolbLoopOrchestrator:
    """Central coordinator for the Kolb experiential learning loop.

    Typical usage:
    - Call on_experience() after each inference to record trial outcomes.
    - Call run_cycle() periodically (or via APScheduler) to batch-reflect,
      distill new concepts, and update evaluator scores.
    """

    def __init__(
        self,
        reflection_engine: ReflectionEngine,
        concept_distiller: ConceptDistiller,
        evaluator: Evaluator,
        db: EpisodicDB,
        reflection_batch_limit: int = 50,
    ) -> None:
        self._reflection_engine = reflection_engine
        self._concept_distiller = concept_distiller
        self._evaluator = evaluator
        self._db = db
        self._reflection_batch_limit = reflection_batch_limit

    async def on_experience(
        self,
        experience: Experience,
        injected_concept_ids: list[str] | None = None,
    ) -> None:
        """Record trial outcome for A/B evaluation after each inference."""
        self._evaluator.record_trial(experience, injected_concept_ids or [])

    async def run_cycle(self) -> CycleResult:
        """Full Kolb cycle: reflect batch → distill → update scores."""
        reflections: list[Reflection] = await self._reflection_engine.reflect_batch(
            self._reflection_batch_limit
        )

        concepts: list[Concept] = []
        if reflections:
            concepts = await self._concept_distiller.distill(reflections)

        deprecated = self._evaluator.update_concept_scores()

        return CycleResult(
            new_reflections=len(reflections),
            new_concepts=len(concepts),
            deprecated_concepts=len(deprecated),
        )
