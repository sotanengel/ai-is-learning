"""DuckDB episodic store: Experience, Reflection, Concept, TrainingSample, Adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from kolb_loop.memory.schemas import (
    Adapter,
    Concept,
    Experience,
    Reflection,
    Session,
    TrainingSample,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          VARCHAR PRIMARY KEY,
    user_id     VARCHAR NOT NULL DEFAULT 'default',
    created_at  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS experiences (
    id                  VARCHAR PRIMARY KEY,
    session_id          VARCHAR NOT NULL,
    user_id             VARCHAR NOT NULL DEFAULT 'default',
    request_messages    JSON NOT NULL,
    response_message    JSON,
    model               VARCHAR NOT NULL,
    tool_calls          JSON NOT NULL DEFAULT '[]',
    usage               JSON NOT NULL DEFAULT '{}',
    error               VARCHAR,
    feedback_score      DOUBLE,
    metadata            JSON NOT NULL DEFAULT '{}',
    allow_training      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS reflections (
    id                      VARCHAR PRIMARY KEY,
    experience_id           VARCHAR NOT NULL,
    verdict                 VARCHAR NOT NULL,
    causes                  JSON NOT NULL DEFAULT '[]',
    improvement_hypotheses  JSON NOT NULL DEFAULT '[]',
    evidence_spans          JSON NOT NULL DEFAULT '[]',
    better_response         VARCHAR,
    raw_llm_output          VARCHAR NOT NULL DEFAULT '',
    created_at              TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    id                      VARCHAR PRIMARY KEY,
    category                VARCHAR NOT NULL DEFAULT 'general',
    title                   VARCHAR NOT NULL,
    condition               VARCHAR NOT NULL,
    action                  VARCHAR NOT NULL,
    expected_effect         VARCHAR NOT NULL,
    support_count           INTEGER NOT NULL DEFAULT 0,
    confidence              DOUBLE NOT NULL DEFAULT 0.0,
    trial_stats             JSON NOT NULL DEFAULT '{}',
    status                  VARCHAR NOT NULL DEFAULT 'hypothesis',
    source_reflection_ids   JSON NOT NULL DEFAULT '[]',
    created_at              TIMESTAMP NOT NULL,
    updated_at              TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS training_samples (
    id                      VARCHAR PRIMARY KEY,
    type                    VARCHAR NOT NULL,
    quality_score           DOUBLE NOT NULL,
    prompt                  VARCHAR NOT NULL,
    chosen                  VARCHAR NOT NULL,
    rejected                VARCHAR,
    source_experience_ids   JSON NOT NULL DEFAULT '[]',
    source_reflection_ids   JSON NOT NULL DEFAULT '[]',
    created_at              TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS adapters (
    id                      VARCHAR PRIMARY KEY,
    base_model              VARCHAR NOT NULL,
    parent_adapter_id       VARCHAR,
    training                JSON NOT NULL DEFAULT '{}',
    eval                    JSON NOT NULL DEFAULT '{}',
    status                  VARCHAR NOT NULL DEFAULT 'shadow',
    traffic_pct             INTEGER NOT NULL DEFAULT 0,
    artifact_path           VARCHAR NOT NULL DEFAULT '',
    source_experience_count INTEGER NOT NULL DEFAULT 0,
    created_at              TIMESTAMP NOT NULL,
    promoted_at             TIMESTAMP
);
"""


class EpisodicDB:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = duckdb.connect(str(path))
        self._conn.execute(_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def save_session(self, session: Session) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)",
            [session.id, session.user_id, session.created_at],
        )

    # ------------------------------------------------------------------
    # Experience
    # ------------------------------------------------------------------

    def save_experience(self, exp: Experience) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO experiences VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                exp.id,
                exp.session_id,
                exp.user_id,
                json.dumps(exp.request_messages),
                json.dumps(exp.response_message) if exp.response_message else None,
                exp.model,
                json.dumps(exp.tool_calls),
                json.dumps(exp.usage),
                exp.error,
                exp.feedback_score,
                json.dumps(exp.metadata),
                exp.allow_training,
                exp.created_at,
            ],
        )

    def get_experience(self, exp_id: str) -> Experience | None:
        row = self._conn.execute(
            "SELECT * FROM experiences WHERE id = ?", [exp_id]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_experience(row)

    def list_experiences(self, limit: int = 100, offset: int = 0) -> list[Experience]:
        rows = self._conn.execute(
            "SELECT * FROM experiences ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
        return [self._row_to_experience(r) for r in rows]

    def update_feedback(self, exp_id: str, score: float) -> None:
        self._conn.execute(
            "UPDATE experiences SET feedback_score = ? WHERE id = ?",
            [score, exp_id],
        )

    def _row_to_experience(self, row: Any) -> Experience:
        return Experience(
            id=row[0],
            session_id=row[1],
            user_id=row[2],
            request_messages=json.loads(row[3]),
            response_message=json.loads(row[4]) if row[4] else None,
            model=row[5],
            tool_calls=json.loads(row[6]),
            usage=json.loads(row[7]),
            error=row[8],
            feedback_score=row[9],
            metadata=json.loads(row[10]),
            allow_training=row[11],
            created_at=row[12],
        )

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def save_reflection(self, ref: Reflection) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO reflections VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                ref.id,
                ref.experience_id,
                ref.verdict.value,
                json.dumps(ref.causes),
                json.dumps(ref.improvement_hypotheses),
                json.dumps([s.model_dump() for s in ref.evidence_spans]),
                ref.better_response,
                ref.raw_llm_output,
                ref.created_at,
            ],
        )

    def get_reflections_for_experience(self, exp_id: str) -> list[Reflection]:
        rows = self._conn.execute(
            "SELECT * FROM reflections WHERE experience_id = ?", [exp_id]
        ).fetchall()
        return [self._row_to_reflection(r) for r in rows]

    def list_unreflected_experience_ids(self, limit: int = 50) -> list[str]:
        rows = self._conn.execute(
            """SELECT e.id FROM experiences e
               LEFT JOIN reflections r ON e.id = r.experience_id
               WHERE r.id IS NULL
               ORDER BY e.created_at ASC LIMIT ?""",
            [limit],
        ).fetchall()
        return [r[0] for r in rows]

    def _row_to_reflection(self, row: Any) -> Reflection:
        from kolb_loop.memory.schemas import EvidenceSpan, Verdict

        return Reflection(
            id=row[0],
            experience_id=row[1],
            verdict=Verdict(row[2]),
            causes=json.loads(row[3]),
            improvement_hypotheses=json.loads(row[4]),
            evidence_spans=[EvidenceSpan(**s) for s in json.loads(row[5])],
            better_response=row[6],
            raw_llm_output=row[7],
            created_at=row[8],
        )

    # ------------------------------------------------------------------
    # Concept
    # ------------------------------------------------------------------

    def save_concept(self, concept: Concept) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO concepts VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                concept.id,
                concept.category,
                concept.title,
                concept.condition,
                concept.action,
                concept.expected_effect,
                concept.support_count,
                concept.confidence,
                json.dumps(concept.trial_stats.model_dump()),
                concept.status.value,
                json.dumps(concept.source_reflection_ids),
                concept.created_at,
                concept.updated_at,
            ],
        )

    def get_concept(self, concept_id: str) -> Concept | None:
        row = self._conn.execute(
            "SELECT * FROM concepts WHERE id = ?", [concept_id]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_concept(row)

    def list_concepts(self, status: str | None = None) -> list[Concept]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM concepts WHERE status = ? ORDER BY confidence DESC",
                [status],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM concepts ORDER BY confidence DESC"
            ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    def _row_to_concept(self, row: Any) -> Concept:
        from kolb_loop.memory.schemas import ConceptStatus, TrialStats

        return Concept(
            id=row[0],
            category=row[1],
            title=row[2],
            condition=row[3],
            action=row[4],
            expected_effect=row[5],
            support_count=row[6],
            confidence=row[7],
            trial_stats=TrialStats(**json.loads(row[8])),
            status=ConceptStatus(row[9]),
            source_reflection_ids=json.loads(row[10]),
            created_at=row[11],
            updated_at=row[12],
        )

    # ------------------------------------------------------------------
    # TrainingSample / Adapter (v3)
    # ------------------------------------------------------------------

    def save_training_sample(self, sample: TrainingSample) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO training_samples VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                sample.id,
                sample.type.value,
                sample.quality_score,
                sample.prompt,
                sample.chosen,
                sample.rejected,
                json.dumps(sample.source_experience_ids),
                json.dumps(sample.source_reflection_ids),
                sample.created_at,
            ],
        )

    def count_training_samples(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0]  # type: ignore[index]
        )

    def save_adapter(self, adapter: Adapter) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO adapters VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [
                adapter.id,
                adapter.base_model,
                adapter.parent_adapter_id,
                json.dumps(adapter.training.model_dump()),
                json.dumps(adapter.eval.model_dump()),
                adapter.status.value,
                adapter.traffic_pct,
                adapter.artifact_path,
                adapter.source_experience_count,
                adapter.created_at,
                adapter.promoted_at,
            ],
        )
