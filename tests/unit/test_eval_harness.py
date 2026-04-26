"""Tests for EvalHarness and PromotionPipeline (TDD)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kolb_loop.eval_harness.harness import EvalConfig, EvalHarness, EvalReport
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Adapter, AdapterEvalResult, AdapterStatus
from kolb_loop.promotion.pipeline import PromotionPipeline, PromotionStage
from kolb_loop.trainer.adapter_registry import AdapterRegistry

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
    domain_score: float = 0.0,
) -> Adapter:
    return Adapter(
        base_model=base_model,
        status=status,
        eval=AdapterEvalResult(domain_score=domain_score),
    )


def _mock_eval_fn(
    domain_score: float = 0.75,
    general_bench_delta: float = 0.02,
    safety_regression: float = 0.0,
) -> MagicMock:
    fn = MagicMock()
    fn.return_value = EvalReport(
        domain_score=domain_score,
        general_bench_delta=general_bench_delta,
        safety_regression=safety_regression,
    )
    return fn


# ---------------------------------------------------------------------------
# EvalHarness tests
# ---------------------------------------------------------------------------


def test_eval_harness_passes_when_all_thresholds_met(registry: AdapterRegistry) -> None:
    adapter = _make_adapter()
    registry.register(adapter)
    eval_fn = _mock_eval_fn(domain_score=0.75, general_bench_delta=0.06, safety_regression=0.0)
    harness = EvalHarness(registry=registry, eval_fn=eval_fn)

    result = harness.evaluate(adapter)

    assert result.passed is True
    assert result.domain_score == pytest.approx(0.75)


def test_eval_harness_fails_on_low_domain_score(registry: AdapterRegistry) -> None:
    adapter = _make_adapter()
    registry.register(adapter)
    eval_fn = _mock_eval_fn(domain_score=0.02, general_bench_delta=0.06, safety_regression=0.0)
    harness = EvalHarness(registry=registry, eval_fn=eval_fn)

    result = harness.evaluate(adapter)

    assert result.passed is False


def test_eval_harness_fails_on_general_regression(registry: AdapterRegistry) -> None:
    adapter = _make_adapter()
    registry.register(adapter)
    eval_fn = _mock_eval_fn(domain_score=0.75, general_bench_delta=-0.05, safety_regression=0.0)
    harness = EvalHarness(registry=registry, eval_fn=eval_fn)

    result = harness.evaluate(adapter)

    assert result.passed is False


def test_eval_harness_fails_on_safety_regression(registry: AdapterRegistry) -> None:
    adapter = _make_adapter()
    registry.register(adapter)
    eval_fn = _mock_eval_fn(domain_score=0.75, general_bench_delta=0.06, safety_regression=0.01)
    harness = EvalHarness(registry=registry, eval_fn=eval_fn)

    result = harness.evaluate(adapter)

    assert result.passed is False


def test_eval_harness_updates_adapter_eval_in_registry(registry: AdapterRegistry) -> None:
    adapter = _make_adapter()
    registry.register(adapter)
    eval_fn = _mock_eval_fn(domain_score=0.8)
    harness = EvalHarness(registry=registry, eval_fn=eval_fn)

    harness.evaluate(adapter)

    stored = registry.get(adapter.id)
    assert stored is not None
    assert stored.eval.domain_score == pytest.approx(0.8)


def test_eval_harness_custom_thresholds(registry: AdapterRegistry) -> None:
    adapter = _make_adapter()
    registry.register(adapter)
    eval_fn = _mock_eval_fn(domain_score=0.03, general_bench_delta=0.0, safety_regression=0.0)
    config = EvalConfig(
        domain_lift_min=0.02, general_regression_max=-0.03, safety_regression_max=0.0
    )
    harness = EvalHarness(registry=registry, eval_fn=eval_fn, config=config)

    result = harness.evaluate(adapter)

    assert result.passed is True


# ---------------------------------------------------------------------------
# PromotionPipeline tests
# ---------------------------------------------------------------------------


async def test_promotion_pipeline_shadow_to_canary(registry: AdapterRegistry) -> None:
    adapter = _make_adapter(status=AdapterStatus.SHADOW)
    registry.register(adapter)

    pipeline = PromotionPipeline(registry=registry)
    updated = await pipeline.advance(adapter.id)

    assert updated is not None
    assert updated.status == AdapterStatus.CANARY
    assert updated.traffic_pct == 10


async def test_promotion_pipeline_canary_to_full(registry: AdapterRegistry) -> None:
    adapter = _make_adapter(status=AdapterStatus.CANARY)
    registry.register(adapter)

    pipeline = PromotionPipeline(registry=registry)
    updated = await pipeline.advance(adapter.id)

    assert updated is not None
    assert updated.status == AdapterStatus.FULL
    assert updated.traffic_pct == 100


async def test_promotion_pipeline_full_stays_full(registry: AdapterRegistry) -> None:
    adapter = _make_adapter(status=AdapterStatus.FULL)
    registry.register(adapter)

    pipeline = PromotionPipeline(registry=registry)
    updated = await pipeline.advance(adapter.id)

    assert updated is not None
    assert updated.status == AdapterStatus.FULL


async def test_promotion_pipeline_rollback(registry: AdapterRegistry) -> None:
    adapter = _make_adapter(status=AdapterStatus.CANARY)
    registry.register(adapter)

    pipeline = PromotionPipeline(registry=registry)
    updated = await pipeline.rollback(adapter.id)

    assert updated is not None
    assert updated.status == AdapterStatus.ROLLBACK
    assert updated.traffic_pct == 0


async def test_promotion_pipeline_missing_adapter_returns_none(registry: AdapterRegistry) -> None:
    pipeline = PromotionPipeline(registry=registry)
    result = await pipeline.advance("nonexistent-id")
    assert result is None


def test_promotion_pipeline_current_stage(registry: AdapterRegistry) -> None:
    shadow = _make_adapter(status=AdapterStatus.SHADOW)
    canary = _make_adapter(status=AdapterStatus.CANARY)
    registry.register(shadow)
    registry.register(canary)

    pipeline = PromotionPipeline(registry=registry)

    assert pipeline.current_stage(shadow.id) == PromotionStage.SHADOW
    assert pipeline.current_stage(canary.id) == PromotionStage.CANARY
    assert pipeline.current_stage("missing") is None
