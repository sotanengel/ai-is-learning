"""Promotion Pipeline: advances adapters through Shadow → Canary → Full."""

from __future__ import annotations

from enum import StrEnum

from kolb_loop.memory.schemas import Adapter, AdapterStatus
from kolb_loop.trainer.adapter_registry import AdapterRegistry

_STAGE_TRAFFIC: dict[AdapterStatus, int] = {
    AdapterStatus.SHADOW: 0,
    AdapterStatus.CANARY: 10,
    AdapterStatus.FULL: 100,
    AdapterStatus.ROLLBACK: 0,
    AdapterStatus.DEPRECATED: 0,
}

_NEXT_STAGE: dict[AdapterStatus, AdapterStatus] = {
    AdapterStatus.SHADOW: AdapterStatus.CANARY,
    AdapterStatus.CANARY: AdapterStatus.FULL,
    AdapterStatus.FULL: AdapterStatus.FULL,
}


class PromotionStage(StrEnum):
    SHADOW = "shadow"
    CANARY = "canary"
    FULL = "full"
    ROLLBACK = "rollback"
    DEPRECATED = "deprecated"


class PromotionPipeline:
    def __init__(self, registry: AdapterRegistry) -> None:
        self._registry = registry

    async def advance(self, adapter_id: str) -> Adapter | None:
        adapter = self._registry.get(adapter_id)
        if adapter is None:
            return None
        next_status = _NEXT_STAGE.get(adapter.status, adapter.status)
        traffic = _STAGE_TRAFFIC[next_status]
        self._registry.promote(adapter_id, next_status, traffic_pct=traffic)
        return self._registry.get(adapter_id)

    async def rollback(self, adapter_id: str) -> Adapter | None:
        adapter = self._registry.get(adapter_id)
        if adapter is None:
            return None
        self._registry.promote(adapter_id, AdapterStatus.ROLLBACK, traffic_pct=0)
        return self._registry.get(adapter_id)

    def current_stage(self, adapter_id: str) -> PromotionStage | None:
        adapter = self._registry.get(adapter_id)
        if adapter is None:
            return None
        return PromotionStage(adapter.status.value)
