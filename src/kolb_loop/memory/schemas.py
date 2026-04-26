"""Pydantic v2 schemas for all domain entities."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Session / Experience
# ---------------------------------------------------------------------------


class Experience(BaseModel):
    id: str = Field(default_factory=_new_id)
    session_id: str
    user_id: str = "default"
    request_messages: list[dict[str, Any]]
    response_message: dict[str, Any] | None = None
    model: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
    feedback_score: float | None = None  # -1 to 1
    metadata: dict[str, Any] = Field(default_factory=dict)
    allow_training: bool = False
    created_at: datetime = Field(default_factory=_now)


class Session(BaseModel):
    id: str = Field(default_factory=_new_id)
    user_id: str = "default"
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class EvidenceSpan(BaseModel):
    field: str
    excerpt: str


class Reflection(BaseModel):
    id: str = Field(default_factory=_new_id)
    experience_id: str
    verdict: Verdict
    causes: list[str] = Field(default_factory=list)
    improvement_hypotheses: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    better_response: str | None = None
    raw_llm_output: str = ""
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Concept
# ---------------------------------------------------------------------------


class ConceptStatus(StrEnum):
    HYPOTHESIS = "hypothesis"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"


class TrialStats(BaseModel):
    injected_success_rate: float = 0.0
    baseline_success_rate: float = 0.0
    trials: int = 0


class Concept(BaseModel):
    id: str = Field(default_factory=_new_id)
    category: str = "general"
    title: str
    condition: str
    action: str
    expected_effect: str
    support_count: int = 0
    confidence: float = 0.0
    trial_stats: TrialStats = Field(default_factory=TrialStats)
    status: ConceptStatus = ConceptStatus.HYPOTHESIS
    source_reflection_ids: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# v3: Training
# ---------------------------------------------------------------------------


class SampleType(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    KTO = "kto"
    CPT = "cpt"


class TrainingSample(BaseModel):
    id: str = Field(default_factory=_new_id)
    type: SampleType
    quality_score: float
    prompt: str
    chosen: str
    rejected: str | None = None
    source_experience_ids: list[str] = Field(default_factory=list)
    source_reflection_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class AdapterStatus(StrEnum):
    SHADOW = "shadow"
    CANARY = "canary"
    FULL = "full"
    ROLLBACK = "rollback"
    DEPRECATED = "deprecated"


class AdapterTrainingConfig(BaseModel):
    method: str = "qlora_sft"
    rank: int = 16
    alpha: int = 32
    lr: float = 2e-4
    epochs: int = 3
    dataset_size: int = 0
    replay_ratio: float = 0.3
    trainer: str = "trl.SFTTrainer"


class AdapterEvalResult(BaseModel):
    domain_score: float = 0.0
    general_bench_delta: float = 0.0
    safety_regression: float = 0.0
    passed: bool = False


class Adapter(BaseModel):
    id: str = Field(default_factory=_new_id)
    base_model: str
    parent_adapter_id: str | None = None
    training: AdapterTrainingConfig = Field(default_factory=AdapterTrainingConfig)
    eval: AdapterEvalResult = Field(default_factory=AdapterEvalResult)
    status: AdapterStatus = AdapterStatus.SHADOW
    traffic_pct: int = 0
    artifact_path: str = ""
    source_experience_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    promoted_at: datetime | None = None
