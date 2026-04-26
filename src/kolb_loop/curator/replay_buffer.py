"""Replay Buffer: prevents catastrophic forgetting by sampling diverse past data."""

from __future__ import annotations

import random
from typing import Any

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import SampleType, TrainingSample


class ReplayBuffer:
    """Samples a diverse subset of past TrainingSamples to mix into new training runs.

    Uses random sampling when embeddings are unavailable; clustering-based
    diversity sampling can be added later.
    """

    def __init__(self, db: EpisodicDB, replay_ratio: float = 0.3) -> None:
        self._db = db
        self._replay_ratio = replay_ratio

    def sample(self, new_samples: list[TrainingSample]) -> list[TrainingSample]:
        """Return new_samples + replay samples at the configured ratio."""
        n_replay = max(1, int(len(new_samples) * self._replay_ratio))
        all_samples = self._db._conn.execute(
            "SELECT id, type, quality_score, prompt, chosen, rejected, "
            "source_experience_ids, source_reflection_ids, created_at "
            "FROM training_samples ORDER BY RANDOM() LIMIT ?",
            [n_replay * 3],
        ).fetchall()

        existing_ids = {s.id for s in new_samples}
        candidates = [self._row_to_sample(r) for r in all_samples if r[0] not in existing_ids]

        replay = random.sample(candidates, min(n_replay, len(candidates)))
        return new_samples + replay

    def _row_to_sample(self, row: Any) -> TrainingSample:
        import json

        return TrainingSample(
            id=row[0],
            type=SampleType(row[1]),
            quality_score=row[2],
            prompt=row[3],
            chosen=row[4],
            rejected=row[5],
            source_experience_ids=json.loads(row[6]),
            source_reflection_ids=json.loads(row[7]),
            created_at=row[8],
        )
