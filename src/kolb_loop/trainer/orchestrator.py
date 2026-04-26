"""Training Orchestrator: triggers and manages QLoRA-SFT jobs."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Adapter, AdapterStatus, AdapterTrainingConfig, TrainingSample
from kolb_loop.trainer.adapter_registry import AdapterRegistry


class TriggerType(StrEnum):
    COUNT = "count"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class TrainResult(BaseModel):
    adapter_id: str
    artifact_path: str
    sample_count: int


class Trainer(Protocol):
    def train(
        self,
        samples: list[TrainingSample],
        config: AdapterTrainingConfig,
        output_dir: str,
    ) -> TrainResult: ...


class TrainingOrchestrator:
    def __init__(
        self,
        db: EpisodicDB,
        registry: AdapterRegistry,
        base_model: str,
        trainer: Trainer,
        count_threshold: int = 500,
        output_dir: str = "./adapters",
    ) -> None:
        self._db = db
        self._registry = registry
        self._base_model = base_model
        self._trainer = trainer
        self._count_threshold = count_threshold
        self._output_dir = output_dir

    def should_train(self, trigger: TriggerType = TriggerType.COUNT) -> bool:
        if trigger == TriggerType.MANUAL or trigger == TriggerType.SCHEDULED:
            return True
        return self._db.count_training_samples() >= self._count_threshold

    async def run(
        self,
        trigger: TriggerType = TriggerType.MANUAL,
        config: AdapterTrainingConfig | None = None,
    ) -> Adapter | None:
        samples = self._select_samples()
        if not samples:
            return None

        cfg = config or AdapterTrainingConfig(dataset_size=len(samples))
        cfg.dataset_size = len(samples)

        parent = self._registry.get_latest(self._base_model)
        parent_id = parent.id if parent else None

        result = self._trainer.train(
            samples=samples,
            config=cfg,
            output_dir=self._output_dir,
        )

        adapter = Adapter(
            base_model=self._base_model,
            parent_adapter_id=parent_id,
            training=cfg,
            status=AdapterStatus.SHADOW,
            artifact_path=result.artifact_path,
            source_experience_count=result.sample_count,
        )
        self._registry.register(adapter)
        return adapter

    def _select_samples(self) -> list[TrainingSample]:
        from kolb_loop.curator.replay_buffer import ReplayBuffer

        rows = self._db._conn.execute(
            "SELECT id, type, quality_score, prompt, chosen, rejected, "
            "source_experience_ids, source_reflection_ids, created_at "
            "FROM training_samples ORDER BY quality_score DESC"
        ).fetchall()

        import json

        from kolb_loop.memory.schemas import SampleType

        all_samples = [
            TrainingSample(
                id=r[0],
                type=SampleType(r[1]),
                quality_score=r[2],
                prompt=r[3],
                chosen=r[4],
                rejected=r[5],
                source_experience_ids=json.loads(r[6]),
                source_reflection_ids=json.loads(r[7]),
                created_at=r[8],
            )
            for r in rows
        ]
        if not all_samples:
            return []
        buffer = ReplayBuffer(self._db)
        return buffer.sample(all_samples)
