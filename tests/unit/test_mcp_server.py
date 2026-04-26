"""Tests for KolbLoopMCPServer tool handlers (TDD)."""

from __future__ import annotations

import pytest

from kolb_loop.mcp_server.server import KolbLoopMCPServer
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Concept, Experience, Session, Verdict
from kolb_loop.memory.vector_store import VectorStore


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


@pytest.fixture
def server(db: EpisodicDB) -> KolbLoopMCPServer:
    vs = VectorStore(":memory:")
    return KolbLoopMCPServer(db, vs)


async def test_search_experiences_found(db: EpisodicDB, server: KolbLoopMCPServer) -> None:
    session = Session()
    db.save_session(session)
    exp = Experience(
        session_id=session.id,
        request_messages=[{"role": "user", "content": "delete database"}],
        model="m",
    )
    db.save_experience(exp)

    results = await server._search_experiences({"query": "delete", "limit": 5})
    assert len(results) == 1
    assert "delete" in results[0].text.lower() or exp.id[:8] in results[0].text


async def test_search_experiences_not_found(db: EpisodicDB, server: KolbLoopMCPServer) -> None:
    results = await server._search_experiences({"query": "nothing here", "limit": 5})
    assert "No matching" in results[0].text


async def test_recall_concepts_validated(db: EpisodicDB, server: KolbLoopMCPServer) -> None:
    from kolb_loop.memory.schemas import ConceptStatus

    c = Concept(
        title="Always check logs",
        condition="when debugging",
        action="read logs",
        expected_effect="faster resolution",
        status=ConceptStatus.VALIDATED,
    )
    db.save_concept(c)

    results = await server._recall_concepts({"context": "debug", "top_k": 3})
    assert "Always check logs" in results[0].text


async def test_recall_concepts_empty(db: EpisodicDB, server: KolbLoopMCPServer) -> None:
    results = await server._recall_concepts({"context": "anything", "top_k": 3})
    assert "No concepts" in results[0].text


async def test_submit_reflection_success(db: EpisodicDB, server: KolbLoopMCPServer) -> None:
    session = Session()
    db.save_session(session)
    exp = Experience(session_id=session.id, request_messages=[], model="m")
    db.save_experience(exp)

    results = await server._submit_reflection(
        {
            "experience_id": exp.id,
            "verdict": "success",
            "causes": ["well reasoned"],
            "improvement_hypotheses": [],
        }
    )
    assert "saved" in results[0].text.lower()
    refs = db.get_reflections_for_experience(exp.id)
    assert len(refs) == 1
    assert refs[0].verdict == Verdict.SUCCESS


async def test_submit_reflection_not_found(db: EpisodicDB, server: KolbLoopMCPServer) -> None:
    results = await server._submit_reflection(
        {"experience_id": "nonexistent", "verdict": "failure"}
    )
    assert "not found" in results[0].text.lower()
