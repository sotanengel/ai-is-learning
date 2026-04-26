"""LanceDB semantic (vector) store for concepts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import lancedb

    _LANCEDB_AVAILABLE = True
except ImportError:
    _LANCEDB_AVAILABLE = False


class VectorStore:
    """Wraps LanceDB for concept semantic search.

    Falls back to a simple in-memory cosine similarity search when LanceDB
    is unavailable (e.g., in lightweight test environments).
    """

    _TABLE = "concepts"

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._use_lancedb = _LANCEDB_AVAILABLE and str(path) != ":memory:"
        self._memory: list[dict[str, Any]] = []

        if self._use_lancedb:
            self._db = lancedb.connect(self._path)
            self._table: Any = None

    def upsert(self, concept_id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        record = {"id": concept_id, "vector": embedding, **metadata}
        if self._use_lancedb:
            self._ensure_table(len(embedding))
            self._table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute([record])
        else:
            self._memory = [r for r in self._memory if r["id"] != concept_id]
            self._memory.append(record)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_lancedb and self._table is not None:
            results: list[dict[str, Any]] = self._table.search(query_embedding).limit(top_k).to_list()
            return results

        if not self._memory:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        scored = []
        for rec in self._memory:
            v = np.array(rec["vector"], dtype=np.float32)
            norm = np.linalg.norm(q) * np.linalg.norm(v)
            sim = float(np.dot(q, v) / norm) if norm > 0 else 0.0
            scored.append((sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def delete(self, concept_id: str) -> None:
        if self._use_lancedb and self._table is not None:
            self._table.delete(f"id = '{concept_id}'")
        else:
            self._memory = [r for r in self._memory if r["id"] != concept_id]

    def _ensure_table(self, dim: int) -> None:
        if self._table is not None:
            return
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("title", pa.string()),
                pa.field("status", pa.string()),
            ]
        )
        if self._TABLE in self._db.table_names():
            self._table = self._db.open_table(self._TABLE)
        else:
            self._table = self._db.create_table(self._TABLE, schema=schema)
