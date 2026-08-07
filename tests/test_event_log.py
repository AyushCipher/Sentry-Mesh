"""Event log: append/read ordering, trace reconstruction, index derivation."""

import asyncio

import pytest

from sentrymesh.event_log import EventLog, new_case_id


def test_append_returns_monotonic_ids(run_async):
    async def scenario():
        log = EventLog.in_memory()
        case = new_case_id()
        ids = [(await log.append(case, "supervisor", "tool_call", {"i": i})).event_id for i in range(25)]
        return ids

    ids = run_async(scenario())
    parsed = [tuple(int(p) for p in i.split("-")) for i in ids]
    assert parsed == sorted(parsed), "stream ids must be strictly increasing"
    assert len(set(ids)) == len(ids), "stream ids must be unique"


def test_case_trace_preserves_append_order(run_async):
    async def scenario():
        log = EventLog.in_memory()
        case = new_case_id()
        for i in range(10):
            await log.append(case, f"actor_{i % 3}", "finding", {"seq": i})
        return await log.get_case_trace(case)

    trace = run_async(scenario())
    assert [e.payload["seq"] for e in trace] == list(range(10))


def test_trace_is_scoped_to_one_case(run_async):
    async def scenario():
        log = EventLog.in_memory()
        a, b = new_case_id("a"), new_case_id("b")
        for i in range(4):
            await log.append(a, "supervisor", "tool_call", {"case": "a", "i": i})
            await log.append(b, "supervisor", "tool_call", {"case": "b", "i": i})
        return await log.get_case_trace(a), await log.get_case_trace(b)

    trace_a, trace_b = run_async(scenario())
    assert len(trace_a) == len(trace_b) == 4
    assert {e.payload["case"] for e in trace_a} == {"a"}
    assert {e.payload["case"] for e in trace_b} == {"b"}


def test_after_cursor_returns_only_newer_events(run_async):
    async def scenario():
        log = EventLog.in_memory()
        case = new_case_id()
        for i in range(6):
            await log.append(case, "supervisor", "finding", {"i": i})
        full = await log.get_case_trace(case)
        tail = await log.get_case_trace(case, after=full[2].event_id)
        return full, tail

    full, tail = run_async(scenario())
    assert len(full) == 6
    assert [e.payload["i"] for e in tail] == [3, 4, 5]


def test_case_index_is_derived_and_rebuildable(run_async):
    """The per-case index is a cache over the stream, not a second truth."""

    async def scenario():
        log = EventLog.in_memory()
        case = new_case_id()
        for i in range(5):
            await log.append(case, "supervisor", "tool_call", {"i": i})
        before = await log.get_case_trace(case)
        rebuilt = await log.rebuild_case_index()
        after = await log.get_case_trace(case)
        return before, rebuilt, after

    before, rebuilt, after = run_async(scenario())
    assert rebuilt == 5
    assert [e.event_id for e in before] == [e.event_id for e in after]


def test_rejects_unknown_event_type(run_async):
    async def scenario():
        log = EventLog.in_memory()
        with pytest.raises(ValueError, match="unknown event_type"):
            await log.append("c1", "supervisor", "gossip", {})

    run_async(scenario())


def test_rejects_missing_case_or_actor(run_async):
    async def scenario():
        log = EventLog.in_memory()
        with pytest.raises(ValueError, match="case_id is required"):
            await log.append("", "supervisor", "finding", {})
        with pytest.raises(ValueError, match="actor is required"):
            await log.append("c1", "", "finding", {})

    run_async(scenario())


def test_concurrent_appends_are_all_recorded_and_ordered(run_async):
    """Four specialists write concurrently; nothing may be lost or interleaved
    into an ambiguous order."""

    async def scenario():
        log = EventLog.in_memory()
        case = new_case_id()

        async def writer(name: str):
            for i in range(15):
                await log.append(case, name, "tool_call", {"actor": name, "i": i})

        await asyncio.gather(*(writer(f"specialist_{n}") for n in range(4)))
        return await log.get_case_trace(case)

    trace = run_async(scenario())
    assert len(trace) == 60
    ids = [tuple(int(p) for p in e.event_id.split("-")) for e in trace]
    assert ids == sorted(ids)
    # Per-actor ordering must survive interleaving.
    for n in range(4):
        seq = [e.payload["i"] for e in trace if e.actor == f"specialist_{n}"]
        assert seq == list(range(15))


def test_payloads_survive_a_round_trip(run_async):
    async def scenario():
        log = EventLog.in_memory()
        payload = {"nested": {"list": [1, 2, {"k": "v"}], "bool": True, "none": None}, "float": 1.5}
        await log.append("c-round-trip", "supervisor", "verdict", payload)
        return await log.get_case_trace("c-round-trip")

    trace = run_async(scenario())
    assert trace[0].payload["nested"]["list"][2]["k"] == "v"
    assert trace[0].payload["nested"]["none"] is None
    assert trace[0].payload["float"] == 1.5
