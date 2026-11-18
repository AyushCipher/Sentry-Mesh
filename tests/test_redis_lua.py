"""Lua-script paths against a REAL Redis (idempotency, MAXLEN trimming, atomic claim).

The in-memory backends mirror these semantics, but only a real server runs the
Lua.  Skipped unless ``SENTRYMESH_TEST_REDIS_URL`` points at a disposable Redis
(the tests FLUSHDB it):

    docker run -d --rm --name sentrymesh-test-redis -p 127.0.0.1:6390:6379 redis:7-alpine
    SENTRYMESH_TEST_REDIS_URL=redis://127.0.0.1:6390/0 pytest tests/test_redis_lua.py
"""

import asyncio
import os

import pytest

from sentrymesh.event_log import EventLog
from sentrymesh.hitl import ReviewQueue
from sentrymesh.models import Case, HumanDecision

URL = os.environ.get("SENTRYMESH_TEST_REDIS_URL", "")
pytestmark = pytest.mark.skipif(not URL, reason="SENTRYMESH_TEST_REDIS_URL not set")


async def _flush():
    import redis.asyncio as aioredis

    c = aioredis.from_url(URL)
    await c.flushdb()
    await c.aclose()


def test_redis_backend_is_really_in_use(run_async):
    async def scenario():
        await _flush()
        log = await EventLog.connect(URL, require_redis=True, maxlen=0)
        return log.backend

    assert run_async(scenario()) == "redis"


def test_idempotency_key_exactly_once_in_lua(run_async):
    async def scenario():
        await _flush()
        log = await EventLog.connect(URL, require_redis=True, maxlen=0)
        a = await log.append("c", "supervisor", "case_opened", {"n": 1}, idempotency_key="c:run1:case_opened")
        b = await log.append("c", "supervisor", "case_opened", {"n": 2}, idempotency_key="c:run1:case_opened")
        c = await log.append("c", "supervisor", "case_opened", {"n": 3}, idempotency_key="c:run2:case_opened")
        d = await log.append("c", "supervisor", "tool_call", {"n": 4})  # no key: always appends
        return a, b, c, d, await log.get_case_trace("c")

    a, b, c, d, trace = run_async(scenario())
    assert a.event_id == b.event_id != c.event_id
    assert [e.payload["n"] for e in trace] == [1, 3, 4]


def test_idempotency_under_concurrent_replay(run_async):
    async def scenario():
        await _flush()
        log = await EventLog.connect(URL, require_redis=True, maxlen=0)
        evs = await asyncio.gather(*(log.append("c", "supervisor", "verdict", {"i": i}, idempotency_key="c:r:verdict") for i in range(25)))
        return {e.event_id for e in evs}, await log.get_case_trace("c")

    ids, trace = run_async(scenario())
    assert len(ids) == 1 and len(trace) == 1


def test_maxlen_bounds_the_stream_on_real_redis(run_async):
    """XADD MAXLEN ~ is APPROXIMATE on a real server: it trims whole macro-nodes, so the
    stream ends up between the cap and a bit above it - never unbounded, never below the cap."""

    async def scenario():
        import redis.asyncio as aioredis

        await _flush()
        log = await EventLog.connect(URL, require_redis=True, maxlen=50)
        for i in range(600):
            await log.append("c", "supervisor", "tool_call", {"i": i})
        raw = aioredis.from_url(URL)
        n = await raw.xlen("sentrymesh:events")
        await raw.aclose()
        trace = await log.get_case_trace("c")
        return n, trace

    n, trace = run_async(scenario())
    assert 50 <= n < 300, f"stream length {n} not bounded by approximate MAXLEN"
    got = [e.payload["i"] for e in trace]
    assert got == sorted(got) and got[-1] == 599  # newest retained, order preserved
    assert len(got) == n  # per-case read tolerates trimmed index entries


def test_unbounded_when_maxlen_zero(run_async):
    async def scenario():
        await _flush()
        log = await EventLog.connect(URL, require_redis=True, maxlen=0)
        for i in range(300):
            await log.append("c", "supervisor", "tool_call", {"i": i})
        return len(await log.get_case_trace("c"))

    assert run_async(scenario()) == 300


async def _enqueue(q, case_id):
    from sentrymesh.agents.llm_client import LLMClient
    from sentrymesh.agents.supervisor_agent import SupervisorAgent
    from tests.test_hardening import StubSpecialist, f

    case = Case(case_id=case_id, category="x", summary="s")
    panel = [f("transaction_pattern", 90), f("identity_signal", 10), f("network_analysis", 10), f("historical_case", 10)]
    sup = SupervisorAgent(EventLog.in_memory(), LLMClient(backend="offline"))
    sup.specialists = [StubSpecialist(x) for x in panel]
    verdict = await sup.investigate(case)
    await q.enqueue(case, verdict)


def test_claim_is_atomic_on_real_redis(run_async):
    async def scenario():
        await _flush()
        log = await EventLog.connect(URL, require_redis=True, maxlen=0)
        q = await ReviewQueue.connect(log, URL)
        assert q.backend == "redis"
        await _enqueue(q, "T-RACE")
        results = await asyncio.gather(
            *(q.submit_decision(HumanDecision("T-RACE", f"rev{i}", "approve" if i % 2 else "decline")) for i in range(8)),
            return_exceptions=True,
        )
        return results, await log.get_case_trace("T-RACE"), await q.depth()

    results, trace, depth = run_async(scenario())
    assert sum(isinstance(r, dict) for r in results) == 1
    assert sum(isinstance(r, KeyError) for r in results) == 7
    assert len([e for e in trace if e.event_type == "human_decision"]) == 1
    assert depth == 0


def test_claim_missing_case_returns_nothing_and_failed_write_requeues(run_async):
    async def scenario():
        await _flush()
        log = await EventLog.connect(URL, require_redis=True, maxlen=0)
        q = await ReviewQueue.connect(log, URL)
        with pytest.raises(KeyError):
            await q.submit_decision(HumanDecision("NOPE", "alice", "approve"))
        await _enqueue(q, "T-BACK")

        async def boom(*a, **k):
            raise RuntimeError("disk full")

        log.append = boom
        with pytest.raises(RuntimeError):
            await q.submit_decision(HumanDecision("T-BACK", "alice", "approve"))
        return await q.depth(), await q.get("T-BACK")

    depth, snap = run_async(scenario())
    assert depth == 1 and snap is not None


def test_enqueue_twice_does_not_double_queue(run_async):
    async def scenario():
        await _flush()
        log = await EventLog.connect(URL, require_redis=True, maxlen=0)
        q = await ReviewQueue.connect(log, URL)
        await _enqueue(q, "T-ONCE")
        await _enqueue(q, "T-ONCE")
        return await q.depth()

    assert run_async(scenario()) == 1
