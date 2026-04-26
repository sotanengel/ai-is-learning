"""Distills clusters of Reflections into abstract Concepts via LLM."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np

from kolb_loop.adapter.openai_adapter import LLMAdapter
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Concept, ConceptStatus, Reflection
from kolb_loop.memory.vector_store import VectorStore

_DISTILL_SYSTEM = """\
You are an AI learning system. Given a cluster of self-critique notes,
extract a single abstract concept in JSON:
{
  "category": "<tool_use|reasoning|communication|safety|general>",
  "title": "<concise title>",
  "condition": "<when this concept applies>",
  "action": "<what to do>",
  "expected_effect": "<expected improvement>"
}
Return ONLY the JSON object.
"""


def _parse_concept(raw: str, source_ids: list[str]) -> Concept | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data: dict[str, Any] = json.loads(match.group())
        return Concept(
            category=data.get("category", "general"),
            title=data["title"],
            condition=data["condition"],
            action=data["action"],
            expected_effect=data["expected_effect"],
            source_reflection_ids=source_ids,
        )
    except (KeyError, json.JSONDecodeError):
        return None


class ConceptDistiller:
    """Cluster Reflections → distill abstract Concepts.

    Uses cosine similarity grouping when embeddings are available,
    otherwise groups all reflections into a single cluster.
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        db: EpisodicDB,
        vector_store: VectorStore,
        embedder_model: str,
        similarity_threshold: float = 0.75,
    ) -> None:
        self._adapter = adapter
        self._db = db
        self._vs = vector_store
        self._embedder_model = embedder_model
        self._sim_threshold = similarity_threshold

    async def distill(self, reflections: list[Reflection]) -> list[Concept]:
        if not reflections:
            return []

        texts = [
            f"{r.verdict}: " + "; ".join(r.causes) + " | " + "; ".join(r.improvement_hypotheses)
            for r in reflections
        ]

        try:
            embeddings = await self._adapter.embeddings(texts, self._embedder_model)
        except Exception:
            # No embedder available: treat all as one cluster
            embeddings = [[0.0] * 4 for _ in reflections]

        clusters = self._cluster(embeddings, reflections)
        concepts: list[Concept] = []
        for cluster_refs in clusters:
            concept = await self._distill_cluster(cluster_refs)
            if concept:
                concept = await self._dedup_check(concept)
                if concept:
                    self._db.save_concept(concept)
                    # Store embedding for future semantic search
                    if embeddings and len(embeddings) > 0:
                        idx = reflections.index(cluster_refs[0])
                        self._vs.upsert(
                            concept.id,
                            embeddings[idx],
                            {"title": concept.title, "status": concept.status},
                        )
                    concepts.append(concept)
        return concepts

    def _cluster(
        self, embeddings: list[list[float]], reflections: list[Reflection]
    ) -> list[list[Reflection]]:
        if len(reflections) <= 1:
            return [reflections]

        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = arr / norms

        assigned = [-1] * len(reflections)
        cluster_id = 0
        for i in range(len(reflections)):
            if assigned[i] != -1:
                continue
            assigned[i] = cluster_id
            for j in range(i + 1, len(reflections)):
                if assigned[j] != -1:
                    continue
                sim = float(np.dot(normalized[i], normalized[j]))
                if sim >= self._sim_threshold:
                    assigned[j] = cluster_id
            cluster_id += 1

        clusters: dict[int, list[Reflection]] = {}
        for idx, cid in enumerate(assigned):
            clusters.setdefault(cid, []).append(reflections[idx])
        return list(clusters.values())

    async def _distill_cluster(self, cluster: list[Reflection]) -> Concept | None:
        summaries = [
            f"verdict={r.verdict} causes={r.causes} hypotheses={r.improvement_hypotheses}"
            for r in cluster
        ]
        user_msg = "\n".join(f"- {s}" for s in summaries)
        payload: dict[str, Any] = {
            "model": self._embedder_model,
            "messages": [
                {"role": "system", "content": _DISTILL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
        }
        try:
            resp = await self._adapter.chat_completions(payload)
            raw: str = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            source_ids = [r.id for r in cluster]
            return _parse_concept(raw, source_ids)
        except Exception:
            return None

    async def _dedup_check(self, candidate: Concept) -> Concept | None:
        existing = self._db.list_concepts()
        for ex in existing:
            if ex.title.lower() == candidate.title.lower():
                return None  # Skip duplicate; could merge in future
        return candidate


class ConceptStore:
    """Convenience wrapper for concept CRUD with status transitions."""

    def __init__(self, db: EpisodicDB) -> None:
        self._db = db

    def promote(self, concept_id: str) -> bool:
        concept = self._db.get_concept(concept_id)
        if concept is None or concept.status != ConceptStatus.HYPOTHESIS:
            return False
        concept.status = ConceptStatus.VALIDATED
        concept.updated_at = datetime.now(UTC)
        self._db.save_concept(concept)
        return True

    def deprecate(self, concept_id: str) -> bool:
        concept = self._db.get_concept(concept_id)
        if concept is None:
            return False
        concept.status = ConceptStatus.DEPRECATED
        concept.updated_at = datetime.now(UTC)
        self._db.save_concept(concept)
        return True
