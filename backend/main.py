"""Layer 5 - FastAPI backend.

Every read endpoint answers from the shared event log rather than from a
side-table the agents also happen to write, so the dashboard and an auditor see
the same history by construction.

Case submission returns immediately with ``202 Accepted`` and runs the
investigation in a background task. The WebSocket at ``/ws/cases/{case_id}``
tails the same log while the panel works, which is what makes the live
"watch it think" view a genuine read of agent state rather than a replay of a
canned animation.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentrymesh.case_bank import load_case, raw_case
from sentrymesh.config import RESULTS_DIR, SETTINGS
from sentrymesh.event_log import Event, new_case_id
from sentrymesh.models import Case, HumanDecision
from sentrymesh.orchestrator import SentryMesh

MESH: Optional[SentryMesh] = None
# Tracks in-flight investigations so a second submission of the same case
# cannot start a duplicate panel run.
RUNNING: Dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MESH
    MESH = await SentryMesh.create()
    try:
        yield
    finally:
        for task in list(RUNNING.values()):
            task.cancel()
        await MESH.close()


app = FastAPI(
    title="SentryMesh",
    version="1.0.0",
    description="Multi-agent fraud investigation and orchestration. All data is SYNTHETIC.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # v1 has no auth; the dashboard is a local demo surface.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def mesh() -> SentryMesh:
    if MESH is None:  # pragma: no cover - lifespan always runs in practice
        raise HTTPException(status_code=503, detail="SentryMesh is still starting up")
    return MESH


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CaseSubmission(BaseModel):
    case_id: Optional[str] = Field(default=None, description="Omit to generate one.")
    seeded_case_id: Optional[str] = Field(
        default=None, description="Load a case from the seeded synthetic bank instead of supplying data."
    )
    category: str = "unknown"
    summary: str = ""
    transaction_data: Dict[str, Any] = Field(default_factory=dict)
    device_data: Dict[str, Any] = Field(default_factory=dict)
    network_data: Dict[str, Any] = Field(default_factory=dict)


class ReviewSubmission(BaseModel):
    action: str = Field(description="approve | decline | request_detail")
    reviewer: str = "dashboard_reviewer"
    notes: str = ""
    requested_specialist: Optional[str] = None


def event_json(event: Event) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "case_id": event.case_id,
        "actor": event.actor,
        "event_type": event.event_type,
        "payload": event.payload,
        "timestamp": event.timestamp,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    m = mesh()
    return {
        "status": "ok",
        "event_log_reachable": await m.log.ping(),
        "review_queue_depth": await m.queue.depth(),
        "in_flight_investigations": len(RUNNING),
        "runtime": m.runtime(),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@app.post("/api/cases", status_code=202)
async def submit_case(submission: CaseSubmission, background: BackgroundTasks) -> Dict[str, Any]:
    m = mesh()

    if submission.seeded_case_id:
        case = load_case(submission.seeded_case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"no seeded case {submission.seeded_case_id}")
        if submission.case_id:
            case.case_id = submission.case_id
    else:
        case = Case(
            case_id=submission.case_id or new_case_id(),
            category=submission.category,
            summary=submission.summary,
            transaction_data=submission.transaction_data,
            device_data=submission.device_data,
            network_data=submission.network_data,
        )

    if case.case_id in RUNNING and not RUNNING[case.case_id].done():
        raise HTTPException(status_code=409, detail=f"case {case.case_id} is already under investigation")

    async def run() -> None:
        try:
            await m.investigate(case)
        except Exception as exc:  # noqa: BLE001 - a crashed run must still be in the trail
            await m.log.append(
                case.case_id, "supervisor", "error", {"stage": "investigation", "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            RUNNING.pop(case.case_id, None)

    RUNNING[case.case_id] = asyncio.create_task(run())
    return {
        "case_id": case.case_id,
        "status": "investigating",
        "trace_stream": f"/ws/cases/{case.case_id}",
        "run_mode": m.llm.run_mode,
    }


@app.get("/api/cases")
async def list_cases(limit: int = 100) -> Dict[str, Any]:
    m = mesh()
    ids = await m.log.known_case_ids()
    out: List[Dict[str, Any]] = []
    for case_id in ids[:limit]:
        status = await m.case_status(case_id)
        if status:
            out.append(
                {
                    "case_id": case_id,
                    "status": status["status"],
                    "summary": status["summary"],
                    "opened_at": status["opened_at"],
                    "combined_risk": (status["verdict"] or {}).get("combined_risk"),
                    "confidence": (status["verdict"] or {}).get("confidence"),
                }
            )
    return {"count": len(out), "cases": out}


@app.get("/api/cases/{case_id}")
async def get_case(case_id: str) -> Dict[str, Any]:
    status = await mesh().case_status(case_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"no events recorded for case {case_id}")
    status["in_flight"] = case_id in RUNNING and not RUNNING[case_id].done()
    return status


@app.get("/api/cases/{case_id}/trace")
async def get_trace(case_id: str) -> Dict[str, Any]:
    events = await mesh().trace(case_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"no events recorded for case {case_id}")
    return {"case_id": case_id, "event_count": len(events), "events": [event_json(e) for e in events]}


@app.websocket("/ws/cases/{case_id}")
async def stream_trace(websocket: WebSocket, case_id: str) -> None:
    """Stream a case's events as the panel produces them.

    Replays everything already recorded first, then tails. A client that
    connects late therefore sees the whole investigation, not just the tail -
    which matters because the panel often finishes faster than a human can
    click through to the page.
    """
    await websocket.accept()
    m = mesh()
    try:
        await websocket.send_text(json.dumps({"type": "runtime", "runtime": m.runtime()}))
        cursor: Optional[str] = None
        async for event in m.log.follow(case_id, after=cursor, idle_timeout=120.0):
            await websocket.send_text(json.dumps({"type": "event", "event": event_json(event)}, default=str))
            if event.event_type in {"verdict", "escalation"} and case_id not in RUNNING:
                break
        await websocket.send_text(json.dumps({"type": "complete", "case_id": case_id}))
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"}))
        except Exception:  # pragma: no cover
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------
@app.get("/api/review-queue")
async def review_queue() -> Dict[str, Any]:
    m = mesh()
    pending = await m.queue.pending()
    # The case payload is retained for specialist re-runs but is noise on the
    # wire and in the UI, so it is stripped from the list view.
    for row in pending:
        row.pop("case_payload", None)
    return {"depth": len(pending), "backend": m.queue.backend, "cases": pending}


@app.get("/api/review-queue/{case_id}")
async def review_case(case_id: str) -> Dict[str, Any]:
    row = await mesh().queue.get(case_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"case {case_id} is not in the review queue")
    row.pop("case_payload", None)
    return row


@app.post("/api/review-queue/_rebuild")
async def rebuild_queue() -> Dict[str, Any]:
    """Rebuild the queue from the event log. Proves the queue is derived state.

    Declared BEFORE the ``/{case_id}`` route: FastAPI matches in declaration
    order, so a literal path registered after a path parameter is unreachable -
    it would be swallowed as a review submission for a case called "_rebuild".
    """
    restored = await mesh().queue.rebuild_from_log()
    return {"restored": restored, "depth": await mesh().queue.depth()}


@app.post("/api/review-queue/{case_id}")
async def submit_review(case_id: str, submission: ReviewSubmission) -> Dict[str, Any]:
    decision = HumanDecision(
        case_id=case_id,
        reviewer=submission.reviewer,
        action=submission.action,
        notes=submission.notes,
        requested_specialist=submission.requested_specialist,
    )
    try:
        return await mesh().review(decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def _report_path() -> Path:
    return RESULTS_DIR / "harness_report.json"


@app.get("/api/eval/report")
async def eval_report() -> Dict[str, Any]:
    path = _report_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="no harness report yet - run `python eval/run_harness.py`")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@app.get("/api/eval/report/{case_id}")
async def eval_case_report(case_id: str) -> Dict[str, Any]:
    path = _report_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="no harness report yet - run `python eval/run_harness.py`")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    result = next((r for r in report.get("case_results", []) if r.get("case_id") == case_id), None)
    if result is None:
        raise HTTPException(status_code=404, detail=f"case {case_id} is not in the latest harness report")

    seeded = raw_case(case_id) or {}
    trace = await mesh().trace(result.get("run_case_id", case_id))
    return {
        "case_id": case_id,
        "result": result,
        "ground_truth": {
            "ground_truth": seeded.get("ground_truth"),
            "expected_verdict": seeded.get("expected_verdict"),
            "difficulty": seeded.get("difficulty"),
            "category": seeded.get("category"),
            "planted_signals": seeded.get("planted_signals", []),
            "summary": seeded.get("summary"),
        },
        "trace": [event_json(e) for e in trace],
        "trace_available": bool(trace),
    }


@app.get("/api/case-bank")
async def case_bank_index() -> Dict[str, Any]:
    """The seeded bank, for the dashboard's case picker.

    Ground-truth labels are included deliberately: this endpoint feeds the eval
    pages, which exist to compare against them. No agent ever reads it.
    """
    from sentrymesh.case_bank import case_files

    rows = []
    for path in case_files():
        data = raw_case(path.stem) or {}
        rows.append(
            {
                "case_id": data.get("case_id"),
                "category": data.get("category"),
                "ground_truth": data.get("ground_truth"),
                "expected_verdict": data.get("expected_verdict"),
                "difficulty": data.get("difficulty"),
                "summary": data.get("summary"),
                "planted_signals": data.get("planted_signals", []),
            }
        )
    return {"count": len(rows), "cases": rows, "data": "synthetic"}
