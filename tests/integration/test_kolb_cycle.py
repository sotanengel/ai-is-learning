"""Integration tests: full Kolb Loop cycle (Experience → Reflection → Concept)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kolb_loop.distiller.concept_distiller import ConceptDistiller
from kolb_loop.evaluator.evaluator import Evaluator
from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.schemas import Experience, Session, Verdict
from kolb_loop.memory.vector_store import VectorStore
from kolb_loop.orchestrator.orchestrator import KolbLoopOrchestrator
from kolb_loop.reflection.engine import ReflectionEngine


@pytest.fixture
def db() -> EpisodicDB:
    return EpisodicDB(":memory:")


def _mock_llm_adapter(
    reflection_json: str | None = None,
    concept_json: str | None = None,
) -> MagicMock:
    default_reflection = (
        '{"verdict":"success","causes":["clear prompt"],'
        '"improvement_hypotheses":["keep it up"],"evidence_spans":[],"better_response":null}'
    )
    default_concept = (
        '{"category":"reasoning","title":"Be concise",'
        '"condition":"when asked for brief answer","action":"give short response",'
        '"expected_effect":"user satisfaction increases"}'
    )
    adapter = MagicMock()

    async def _chat(payload: dict[str, object]) -> dict[str, object]:
        messages = payload.get("messages", [])
        system = str(messages[0]["content"]) if messages else ""  # type: ignore[index]
        if "AI self-critic" in system:
            return {"choices": [{"message": {"content": reflection_json or default_reflection}}]}
        return {"choices": [{"message": {"content": concept_json or default_concept}}]}

    adapter.chat_completions = _chat

    async def _embeddings(texts: list[str], model: str) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    adapter.embeddings = _embeddings
    return adapter


def _seed_experience(db: EpisodicDB, content: str = "What is X?") -> Experience:
    s = Session()
    db.save_session(s)
    exp = Experience(
        session_id=s.id,
        request_messages=[{"role": "user", "content": content}],
        response_message={"role": "assistant", "content": "X is Y."},
        model="test-model",
    )
    db.save_experience(exp)
    return exp


async def test_reflection_stores_in_db(db: EpisodicDB) -> None:
    adapter = _mock_llm_adapter()
    engine = ReflectionEngine(adapter, db, model="m")
    exp = _seed_experience(db)

    ref = await engine.reflect(exp)

    assert ref.verdict == Verdict.SUCCESS
    stored = db.get_reflections_for_experience(exp.id)
    assert len(stored) == 1
    assert stored[0].id == ref.id


async def test_orchestrator_full_cycle_with_real_components(db: EpisodicDB) -> None:
    adapter = _mock_llm_adapter()
    vs = VectorStore(":memory:")
    engine = ReflectionEngine(adapter, db, model="m")
    distiller = ConceptDistiller(adapter, db, vs, embedder_model="emb")
    evaluator = Evaluator(db)
    orch = KolbLoopOrchestrator(engine, distiller, evaluator, db)

    # Seed 3 unreflected experiences
    for i in range(3):
        _seed_experience(db, f"Question {i}")

    result = await orch.run_cycle()

    assert result.new_reflections == 3
    assert result.new_concepts >= 1
    assert len(db.list_concepts()) >= 1


async def test_on_experience_then_cycle_updates_eval(db: EpisodicDB) -> None:
    adapter = _mock_llm_adapter()
    vs = VectorStore(":memory:")
    engine = ReflectionEngine(adapter, db, model="m")
    distiller = ConceptDistiller(adapter, db, vs, embedder_model="emb")
    evaluator = Evaluator(db, min_trials=1)
    orch = KolbLoopOrchestrator(engine, distiller, evaluator, db)

    exp = _seed_experience(db)
    await orch.on_experience(exp, injected_concept_ids=[])

    result = await orch.run_cycle()

    assert result.new_reflections == 1


async def test_reflection_batch_processes_unreflected_only(db: EpisodicDB) -> None:
    adapter = _mock_llm_adapter()
    engine = ReflectionEngine(adapter, db, model="m")

    exp1 = _seed_experience(db, "Q1")
    exp2 = _seed_experience(db, "Q2")

    # Manually reflect exp1 already
    await engine.reflect(exp1)

    # reflect_batch should only pick up exp2
    refs = await engine.reflect_batch(limit=10)
    assert len(refs) == 1
    assert refs[0].experience_id == exp2.id
