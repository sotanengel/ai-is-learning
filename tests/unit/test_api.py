"""Tests for custom REST API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from kolb_loop.ingress.api import create_api_app
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import (
    Concept,
    ConceptStatus,
    Experience,
    SampleType,
    Session,
    TrainingSample,
)


def _make_app(db: EpisodicDB) -> TestClient:
    app = create_api_app(db)
    return TestClient(app)


def _seed_experience(db: EpisodicDB) -> Experience:
    s = Session()
    db.save_session(s)
    exp = Experience(
        session_id=s.id,
        request_messages=[{"role": "user", "content": "Q"}],
        model="m",
    )
    db.save_experience(exp)
    return exp


def _seed_concept(db: EpisodicDB, title: str = "Test concept") -> Concept:
    c = Concept(
        title=title,
        condition="when X",
        action="do Y",
        expected_effect="result Z",
        status=ConceptStatus.VALIDATED,
    )
    db.save_concept(c)
    return c


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health_ok() -> None:
    db = EpisodicDB(":memory:")
    client = _make_app(db)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /api/experiences
# ---------------------------------------------------------------------------


def test_list_experiences_empty() -> None:
    db = EpisodicDB(":memory:")
    client = _make_app(db)
    resp = client.get("/api/experiences")
    assert resp.status_code == 200
    assert resp.json()["experiences"] == []


def test_list_experiences_returns_data() -> None:
    db = EpisodicDB(":memory:")
    _seed_experience(db)
    _seed_experience(db)
    client = _make_app(db)
    resp = client.get("/api/experiences")
    assert resp.status_code == 200
    assert len(resp.json()["experiences"]) == 2


def test_list_experiences_limit() -> None:
    db = EpisodicDB(":memory:")
    for _ in range(5):
        _seed_experience(db)
    client = _make_app(db)
    resp = client.get("/api/experiences?limit=3")
    assert resp.status_code == 200
    assert len(resp.json()["experiences"]) == 3


# ---------------------------------------------------------------------------
# /api/concepts
# ---------------------------------------------------------------------------


def test_list_concepts_empty() -> None:
    db = EpisodicDB(":memory:")
    client = _make_app(db)
    resp = client.get("/api/concepts")
    assert resp.status_code == 200
    assert resp.json()["concepts"] == []


def test_list_concepts_returns_data() -> None:
    db = EpisodicDB(":memory:")
    _seed_concept(db, "A")
    _seed_concept(db, "B")
    client = _make_app(db)
    resp = client.get("/api/concepts")
    assert len(resp.json()["concepts"]) == 2


def test_list_concepts_filter_by_status() -> None:
    db = EpisodicDB(":memory:")
    _seed_concept(db, "validated-one")
    hypo = Concept(
        title="hypo",
        condition="c",
        action="a",
        expected_effect="e",
        status=ConceptStatus.HYPOTHESIS,
    )
    db.save_concept(hypo)
    client = _make_app(db)
    resp = client.get("/api/concepts?status=validated")
    assert len(resp.json()["concepts"]) == 1
    assert resp.json()["concepts"][0]["title"] == "validated-one"


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------


def test_stats_returns_counts() -> None:
    db = EpisodicDB(":memory:")
    _seed_experience(db)
    _seed_experience(db)
    _seed_concept(db)
    db.save_training_sample(
        TrainingSample(type=SampleType.SFT, quality_score=0.8, prompt="q", chosen="a")
    )
    client = _make_app(db)
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["experiences"] == 2
    assert data["concepts"] == 1
    assert data["training_samples"] == 1
