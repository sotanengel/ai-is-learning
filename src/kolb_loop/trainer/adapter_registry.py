"""Adapter Registry: versioned management of trained LoRA adapters."""

from __future__ import annotations

import json

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import (
    Adapter,
    AdapterEvalResult,
    AdapterStatus,
    AdapterTrainingConfig,
)


class AdapterRegistry:
    def __init__(self, db: EpisodicDB) -> None:
        self._db = db

    def register(self, adapter: Adapter) -> None:
        self._db.save_adapter(adapter)

    def get(self, adapter_id: str) -> Adapter | None:
        row = self._db._conn.execute("SELECT * FROM adapters WHERE id = ?", [adapter_id]).fetchone()
        if row is None:
            return None
        return self._row_to_adapter(row)

    def list_adapters(
        self,
        base_model: str | None = None,
        status: AdapterStatus | None = None,
    ) -> list[Adapter]:
        conditions = []
        params: list[str] = []
        if base_model:
            conditions.append("base_model = ?")
            params.append(base_model)
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._db._conn.execute(
            f"SELECT * FROM adapters {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        return [self._row_to_adapter(r) for r in rows]

    def promote(self, adapter_id: str, status: AdapterStatus, traffic_pct: int = 0) -> None:
        self._db._conn.execute(
            "UPDATE adapters SET status = ?, traffic_pct = ? WHERE id = ?",
            [status.value, traffic_pct, adapter_id],
        )

    def get_best(self, base_model: str) -> Adapter | None:
        row = self._db._conn.execute(
            "SELECT * FROM adapters WHERE base_model = ? "
            "ORDER BY CAST(json_extract(eval, '$.domain_score') AS DOUBLE) DESC LIMIT 1",
            [base_model],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_adapter(row)

    def get_latest(self, base_model: str) -> Adapter | None:
        row = self._db._conn.execute(
            "SELECT * FROM adapters WHERE base_model = ? ORDER BY created_at DESC LIMIT 1",
            [base_model],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_adapter(row)

    def _row_to_adapter(self, row: tuple) -> Adapter:  # type: ignore[type-arg]
        training_data = json.loads(row[3])
        eval_data = json.loads(row[4])
        return Adapter(
            id=row[0],
            base_model=row[1],
            parent_adapter_id=row[2],
            training=AdapterTrainingConfig(**training_data),
            eval=AdapterEvalResult(**eval_data),
            status=AdapterStatus(row[5]),
            traffic_pct=row[6],
            artifact_path=row[7],
            source_experience_count=row[8],
            created_at=row[9],
            promoted_at=row[10],
        )
