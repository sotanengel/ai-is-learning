"""EvalHarness: 3-layer evaluation (domain / general bench / safety)."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel

from kolb_loop.memory.schemas import Adapter, AdapterEvalResult
from kolb_loop.trainer.adapter_registry import AdapterRegistry


class EvalReport(BaseModel):
    domain_score: float = 0.0
    general_bench_delta: float = 0.0
    safety_regression: float = 0.0
    passed: bool = False


class EvalConfig(BaseModel):
    domain_lift_min: float = 0.05
    general_regression_max: float = -0.03
    safety_regression_max: float = 0.0


class EvalFn(Protocol):
    def __call__(self, adapter: Adapter) -> EvalReport: ...


class EvalHarness:
    def __init__(
        self,
        registry: AdapterRegistry,
        eval_fn: EvalFn,
        config: EvalConfig | None = None,
    ) -> None:
        self._registry = registry
        self._eval_fn = eval_fn
        self._config = config or EvalConfig()

    def evaluate(self, adapter: Adapter) -> EvalReport:
        report = self._eval_fn(adapter)
        cfg = self._config
        passed = (
            report.domain_score >= cfg.domain_lift_min
            and report.general_bench_delta >= cfg.general_regression_max
            and report.safety_regression <= cfg.safety_regression_max
        )
        report.passed = passed

        self._registry._db._conn.execute(
            "UPDATE adapters SET eval = ? WHERE id = ?",
            [
                json.dumps(
                    AdapterEvalResult(
                        domain_score=report.domain_score,
                        general_bench_delta=report.general_bench_delta,
                        safety_regression=report.safety_regression,
                        passed=passed,
                    ).model_dump()
                ),
                adapter.id,
            ],
        )
        return report
