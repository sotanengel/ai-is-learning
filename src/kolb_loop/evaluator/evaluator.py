"""Evaluator: compares injection vs non-injection outcomes, updates concept scores."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import ConceptStatus, Experience


def _is_success(exp: Experience) -> bool:
    if exp.error:
        return False
    if exp.feedback_score is not None:
        return exp.feedback_score > 0
    # No explicit signal: treat non-error as success
    return True


class Evaluator:
    """Tracks A/B trial outcomes and updates concept confidence scores.

    Call record_trial() after each inference to associate outcome with
    which concepts were injected. Call update_concept_scores() periodically
    to recalculate confidence and deprecate low-performing concepts.
    """

    def __init__(
        self,
        db: EpisodicDB,
        deprecate_threshold: float = 0.4,
        min_trials: int = 10,
    ) -> None:
        self._db = db
        self._deprecate_threshold = deprecate_threshold
        self._min_trials = min_trials
        # In-memory trial buffer: concept_id → {injected: [bool], baseline: [bool]}
        self._trials: dict[str, dict[str, list[bool]]] = {}

    def record_trial(
        self,
        experience: Experience,
        injected_concept_ids: list[str],
    ) -> None:
        outcome = _is_success(experience)
        if injected_concept_ids:
            for cid in injected_concept_ids:
                bucket = self._trials.setdefault(cid, {"injected": [], "baseline": []})
                bucket["injected"].append(outcome)
        else:
            # Control group: attribute to all known concepts as baseline
            for cid in self._trials:
                self._trials[cid]["baseline"].append(outcome)

    def get_trial_stats(self, concept_id: str) -> dict[str, Any]:
        bucket = self._trials.get(concept_id, {"injected": [], "baseline": []})
        inj = bucket["injected"]
        base = bucket["baseline"]
        return {
            "injected_success_rate": sum(inj) / len(inj) if inj else 0.0,
            "baseline_success_rate": sum(base) / len(base) if base else 0.0,
            "trials": len(inj) + len(base),
        }

    def update_concept_scores(self) -> list[str]:
        """Recompute confidence for all concepts. Returns list of deprecated IDs."""
        deprecated: list[str] = []
        concepts = self._db.list_concepts(status="hypothesis") + self._db.list_concepts(
            status="validated"
        )
        for concept in concepts:
            stats = self.get_trial_stats(concept.id)
            trials = stats["trials"]
            if trials < self._min_trials:
                continue

            lift = stats["injected_success_rate"] - stats["baseline_success_rate"]
            # Confidence: sigmoid-like mapping of lift to [0,1]
            confidence = max(0.0, min(1.0, 0.5 + lift))
            concept.confidence = confidence
            concept.trial_stats.injected_success_rate = stats["injected_success_rate"]
            concept.trial_stats.baseline_success_rate = stats["baseline_success_rate"]
            concept.trial_stats.trials = trials
            concept.updated_at = datetime.now(UTC)

            if confidence < self._deprecate_threshold:
                concept.status = ConceptStatus.DEPRECATED
                deprecated.append(concept.id)
            elif confidence >= 0.7 and concept.status == ConceptStatus.HYPOTHESIS:
                concept.status = ConceptStatus.VALIDATED

            self._db.save_concept(concept)
        return deprecated
