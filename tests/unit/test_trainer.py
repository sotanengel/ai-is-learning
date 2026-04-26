"""Tests for AdapterRegistry and TrainingOrchestrator (TDD)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import (
    Adapter,
    AdapterEvalResult,
    AdapterStatus,
    AdapterTrainingConfig,
    SampleType,
    TrainingSample,
)
from kolb_loop.trainer.adapter_registry import AdapterRegistry
from kolb_loop.trainer.orchestrator import TrainingOrchestrator, TrainResult, TriggerType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


@pytest.fixture
def registry(db: EpisodicDB) -> AdapterRegistry:
    return AdapterRegistry(db)


def _make_adapter(
    base_model: str = "llama3",
    status: AdapterStatus = AdapterStatus.SHADOW,
    domain_score: float = 0.5,
    parent_id: str | None = None,
) -> Adapter:
    adapter = Adapter(
        base_model=base_model,
        parent_adapter_id=parent_id,
        status=status,
        eval=AdapterEvalResult(domain_score=domain_score, passed=domain_score >= 0.6),
    )
    return adapter


def _add_samples(db: EpisodicDB, n: int) -> list[TrainingSample]:
    samples = []
    for i in range(n):
        s = TrainingSample(
            type=SampleType.SFT,
            quality_score=0.8,
            prompt=f"Q{i}",
            chosen=f"A{i}",
        )
        db.save_training_sample(s)
        samples.append(s)
    return samples


def _mock_trainer(result: TrainResult | None = None) -> MagicMock:
    trainer = MagicMock()
    trainer.train.return_value = result or TrainResult(
        adapter_id="new-id",
        artifact_path="/tmp/adapter",
        sample_count=10,
    )
    return trainer


# ---------------------------------------------------------------------------
# AdapterRegistry tests
# ---------------------------------------------------------------------------


def test_registry_register_and_get(registry: AdapterRegistry) -> None:
    adapter = _make_adapter()
    registry.register(adapter)

    fetched = registry.get(adapter.id)
    assert fetched is not None
    assert fetched.id == adapter.id
    assert fetched.base_model == "llama3"


def test_registry_get_missing_returns_none(registry: AdapterRegistry) -> None:
    assert registry.get("nonexistent-id") is None


def test_registry_list_by_status(registry: AdapterRegistry) -> None:
    shadow = _make_adapter(status=AdapterStatus.SHADOW)
    canary = _make_adapter(status=AdapterStatus.CANARY)
    registry.register(shadow)
    registry.register(canary)

    shadows = registry.list_adapters(status=AdapterStatus.SHADOW)
    assert len(shadows) == 1
    assert shadows[0].id == shadow.id


def test_registry_list_all(registry: AdapterRegistry) -> None:
    for _ in range(3):
        registry.register(_make_adapter())
    assert len(registry.list_adapters()) == 3


def test_registry_promote(registry: AdapterRegistry) -> None:
    adapter = _make_adapter()
    registry.register(adapter)

    registry.promote(adapter.id, AdapterStatus.CANARY, traffic_pct=10)

    fetched = registry.get(adapter.id)
    assert fetched is not None
    assert fetched.status == AdapterStatus.CANARY
    assert fetched.traffic_pct == 10


def test_registry_get_best(registry: AdapterRegistry) -> None:
    low = _make_adapter(domain_score=0.4)
    high = _make_adapter(domain_score=0.9)
    mid = _make_adapter(domain_score=0.7)
    for a in [low, high, mid]:
        registry.register(a)

    best = registry.get_best("llama3")
    assert best is not None
    assert best.id == high.id


def test_registry_get_best_no_adapters(registry: AdapterRegistry) -> None:
    assert registry.get_best("llama3") is None


def test_registry_get_latest(registry: AdapterRegistry) -> None:
    first = _make_adapter()
    registry.register(first)
    second = _make_adapter()
    registry.register(second)

    latest = registry.get_latest("llama3")
    assert latest is not None
    assert latest.id == second.id


# ---------------------------------------------------------------------------
# TrainingOrchestrator tests
# ---------------------------------------------------------------------------


def test_orchestrator_should_train_below_threshold(
    db: EpisodicDB, registry: AdapterRegistry
) -> None:
    _add_samples(db, 10)
    orch = TrainingOrchestrator(
        db=db,
        registry=registry,
        base_model="llama3",
        trainer=_mock_trainer(),
        count_threshold=500,
    )
    assert orch.should_train(TriggerType.COUNT) is False


def test_orchestrator_should_train_above_threshold(
    db: EpisodicDB, registry: AdapterRegistry
) -> None:
    _add_samples(db, 5)
    orch = TrainingOrchestrator(
        db=db,
        registry=registry,
        base_model="llama3",
        trainer=_mock_trainer(),
        count_threshold=5,
    )
    assert orch.should_train(TriggerType.COUNT) is True


def test_orchestrator_manual_always_trains(db: EpisodicDB, registry: AdapterRegistry) -> None:
    _add_samples(db, 1)
    orch = TrainingOrchestrator(
        db=db,
        registry=registry,
        base_model="llama3",
        trainer=_mock_trainer(),
        count_threshold=9999,
    )
    assert orch.should_train(TriggerType.MANUAL) is True


async def test_orchestrator_run_creates_adapter(db: EpisodicDB, registry: AdapterRegistry) -> None:
    _add_samples(db, 5)
    mock_trainer = _mock_trainer(
        TrainResult(adapter_id="x", artifact_path="/tmp/out", sample_count=5)
    )
    orch = TrainingOrchestrator(
        db=db,
        registry=registry,
        base_model="llama3",
        trainer=mock_trainer,
    )

    adapter = await orch.run(TriggerType.MANUAL)

    assert adapter is not None
    assert adapter.base_model == "llama3"
    assert adapter.artifact_path == "/tmp/out"
    assert adapter.source_experience_count == 5
    mock_trainer.train.assert_called_once()

    stored = registry.get(adapter.id)
    assert stored is not None


async def test_orchestrator_run_no_samples_returns_none(
    db: EpisodicDB, registry: AdapterRegistry
) -> None:
    orch = TrainingOrchestrator(
        db=db,
        registry=registry,
        base_model="llama3",
        trainer=_mock_trainer(),
    )
    adapter = await orch.run(TriggerType.MANUAL)
    assert adapter is None


async def test_orchestrator_run_uses_parent_adapter(
    db: EpisodicDB, registry: AdapterRegistry
) -> None:
    parent = _make_adapter(status=AdapterStatus.FULL)
    registry.register(parent)
    _add_samples(db, 3)

    orch = TrainingOrchestrator(
        db=db,
        registry=registry,
        base_model="llama3",
        trainer=_mock_trainer(),
    )
    adapter = await orch.run(TriggerType.MANUAL)

    assert adapter is not None
    assert adapter.parent_adapter_id == parent.id


async def test_orchestrator_run_uses_custom_config(
    db: EpisodicDB, registry: AdapterRegistry
) -> None:
    _add_samples(db, 3)
    mock_trainer = _mock_trainer()
    orch = TrainingOrchestrator(
        db=db,
        registry=registry,
        base_model="llama3",
        trainer=mock_trainer,
    )
    config = AdapterTrainingConfig(rank=32, epochs=5)
    await orch.run(TriggerType.MANUAL, config=config)

    call_config = mock_trainer.train.call_args[1]["config"]
    assert call_config.rank == 32
    assert call_config.epochs == 5
