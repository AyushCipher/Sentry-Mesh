"""API surface: status codes and the full review-queue round trip.

Runs the real app against the in-memory backends, so the review flow exercised
here is the same code path the dashboard uses.
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

import backend.main as backend_main
from backend.main import app


@pytest.fixture
def client(monkeypatch):
    """Boot the app with in-memory backends and yield an async client factory."""
    from sentrymesh.orchestrator import SentryMesh

    original = SentryMesh.create

    async def in_memory_create(cls, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["in_memory"] = True
        return await original.__func__(cls, **kwargs)  # type: ignore[attr-defined]

    monkeypatch.setattr(SentryMesh, "create", classmethod(in_memory_create))

    def factory(scenario):
        async def run():
            async with backend_main.lifespan(app):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                    return await scenario(c)

        return asyncio.run(run())

    return factory


async def _investigate(c, seeded_case_id, case_id):
    response = await c.post("/api/cases", json={"seeded_case_id": seeded_case_id, "case_id": case_id})
    assert response.status_code == 202, response.text
    for _ in range(200):
        await asyncio.sleep(0.05)
        if case_id not in backend_main.RUNNING:
            return response.json()
    raise AssertionError(f"investigation of {case_id} did not finish in time")


def test_health_reports_runtime(client):
    body = client(lambda c: c.get("/health"))
    assert body.status_code == 200
    data = body.json()
    assert data["status"] == "ok"
    assert data["runtime"]["data"] == "synthetic"
    assert data["runtime"]["run_mode"] in {"anthropic", "openai_compat", "offline_deterministic"}


def test_case_bank_is_served_and_labelled_synthetic(client):
    data = client(lambda c: c.get("/api/case-bank")).json()
    assert data["data"] == "synthetic"
    assert 20 <= data["count"] <= 25, "the bank must stay within the specified size"
    assert {row["ground_truth"] for row in data["cases"]} >= {"fraud", "legitimate", "ambiguous"}


def test_unknown_case_returns_404(client):
    async def scenario(c):
        return await c.get("/api/cases/does-not-exist"), await c.get("/api/cases/does-not-exist/trace")

    status, trace = client(scenario)
    assert status.status_code == 404
    assert trace.status_code == 404


def test_submitting_an_unknown_seeded_case_returns_404(client):
    response = client(lambda c: c.post("/api/cases", json={"seeded_case_id": "SM-999"}))
    assert response.status_code == 404


def test_investigation_produces_a_verdict_and_a_trace(client):
    async def scenario(c):
        await _investigate(c, "SM-001", "T-API-1")
        status = await c.get("/api/cases/T-API-1")
        trace = await c.get("/api/cases/T-API-1/trace")
        return status.json(), trace.json()

    status, trace = client(scenario)
    assert status["status"].startswith(("closed_", "pending_review"))
    assert status["verdict"] is not None
    assert status["reconstructed_from"] == "event_log"
    assert len(status["findings"]) == 4, "all four specialists must report on every case"

    event_types = {e["event_type"] for e in trace["events"]}
    assert {"case_opened", "tool_call", "finding", "verdict"} <= event_types
    actors = {e["actor"] for e in trace["events"]}
    assert {"supervisor", "transaction_pattern", "identity_signal", "network_analysis", "historical_case"} <= actors


def test_duplicate_submission_of_a_running_case_conflicts(client):
    async def scenario(c):
        first = await c.post("/api/cases", json={"seeded_case_id": "SM-005", "case_id": "T-API-DUP"})
        second = await c.post("/api/cases", json={"seeded_case_id": "SM-005", "case_id": "T-API-DUP"})
        for _ in range(200):
            await asyncio.sleep(0.05)
            if "T-API-DUP" not in backend_main.RUNNING:
                break
        return first.status_code, second.status_code

    first, second = client(scenario)
    assert first == 202
    assert second == 409


def test_review_queue_round_trip_writes_a_human_decision(client):
    """Escalate a case, approve it, and confirm the loop closes in the log."""

    async def scenario(c):
        # SM-003 escalates under the current thresholds; assert rather than assume.
        await _investigate(c, "SM-003", "T-API-REVIEW")
        queue = (await c.get("/api/review-queue")).json()
        assert any(row["case_id"] == "T-API-REVIEW" for row in queue["cases"]), queue

        detail = await c.get("/api/review-queue/T-API-REVIEW")
        decision = await c.post(
            "/api/review-queue/T-API-REVIEW",
            json={"action": "approve", "reviewer": "test_reviewer", "notes": "verified by phone"},
        )
        after_queue = (await c.get("/api/review-queue")).json()
        after_case = (await c.get("/api/cases/T-API-REVIEW")).json()
        return detail.status_code, decision.json(), after_queue, after_case

    detail_status, decision, after_queue, after_case = client(scenario)
    assert detail_status == 200
    assert decision["status"] == "closed_approve"
    assert decision["removed_from_queue"] is True
    assert not any(row["case_id"] == "T-API-REVIEW" for row in after_queue["cases"])
    assert after_case["status"] == "closed_by_human_approve"
    assert after_case["human_decision"]["reviewer"] == "test_reviewer"


def test_request_detail_keeps_the_case_queued_and_reruns_a_specialist(client):
    async def scenario(c):
        await _investigate(c, "SM-003", "T-API-DETAIL")
        response = await c.post(
            "/api/review-queue/T-API-DETAIL",
            json={"action": "request_detail", "requested_specialist": "network_analysis"},
        )
        queue = (await c.get("/api/review-queue")).json()
        trace = (await c.get("/api/cases/T-API-DETAIL/trace")).json()
        return response.json(), queue, trace

    result, queue, trace = client(scenario)
    assert result["removed_from_queue"] is False
    assert result["status"] == "awaiting_detail"
    assert "detail_finding" in result, "requesting detail must actually re-run the specialist"
    assert any(row["case_id"] == "T-API-DETAIL" for row in queue["cases"])
    assert any(e["event_type"] == "human_decision" for e in trace["events"])


def test_review_of_a_case_not_in_the_queue_returns_404(client):
    response = client(lambda c: c.post("/api/review-queue/nope", json={"action": "approve"}))
    assert response.status_code == 404


def test_unknown_review_action_returns_400(client):
    async def scenario(c):
        await _investigate(c, "SM-003", "T-API-BADACTION")
        return await c.post("/api/review-queue/T-API-BADACTION", json={"action": "shred"})

    assert client(scenario).status_code == 400


def test_eval_report_endpoints_404_without_a_report(client, monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "_report_path", lambda: tmp_path / "missing.json")

    async def scenario(c):
        return await c.get("/api/eval/report"), await c.get("/api/eval/report/SM-001")

    report, case = client(scenario)
    assert report.status_code == 404
    assert case.status_code == 404


def test_queue_rebuilds_from_the_event_log_alone(client):
    """Wipe the queue and reconstruct it from the log - the property that makes
    a Redis-resident work list safe."""

    async def scenario(c):
        await _investigate(c, "SM-003", "T-API-REBUILD")
        before = (await c.get("/api/review-queue")).json()["depth"]
        await backend_main.mesh().queue.clear()
        emptied = (await c.get("/api/review-queue")).json()["depth"]
        rebuilt = (await c.post("/api/review-queue/_rebuild")).json()
        return before, emptied, rebuilt

    before, emptied, rebuilt = client(scenario)
    assert before >= 1
    assert emptied == 0
    assert rebuilt["restored"] == before
