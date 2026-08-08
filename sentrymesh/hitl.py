"""Layer 4 - human-in-the-loop review.

STORAGE CHOICE
--------------
Two Redis structures, because the queue genuinely needs two things a single
structure cannot give:

* ``sentrymesh:review_queue`` - a **list**, holding case IDs in arrival order.
  Reviewers work a queue; FIFO order is part of the product, and a list is the
  only structure here that preserves it cheaply.
* ``sentrymesh:review_queue:meta`` - a **hash**, mapping case_id to the verdict
  snapshot the reviewer needs on screen.  The dashboard opens cases by ID, and
  scanning a list for a match would be O(n) per page load.

The queue is a *work list*, not a source of truth.  Nothing here is
authoritative: the escalation that put a case in the queue and the decision
that takes it out are both events in the shared log, and
:meth:`ReviewQueue.rebuild_from_log` reconstructs the entire queue from that log
alone - which is the property that makes it safe for the queue to be
Redis-resident rather than in Postgres.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sentrymesh.config import REVIEW_META_KEY, REVIEW_QUEUE_KEY, SETTINGS
from sentrymesh.event_log import EventLog
from sentrymesh.models import Case, HumanDecision, Verdict

REVIEW_ACTIONS = ("approve", "decline", "request_detail")


class _MemoryQueueBackend:
    name = "in_memory"

    def __init__(self) -> None:
        self._order: List[str] = []
        self._meta: Dict[str, str] = {}

    async def push(self, case_id: str, meta: str) -> None:
        if case_id not in self._meta:
            self._order.append(case_id)
        self._meta[case_id] = meta

    async def remove(self, case_id: str) -> None:
        self._order = [c for c in self._order if c != case_id]
        self._meta.pop(case_id, None)

    async def get(self, case_id: str) -> Optional[str]:
        return self._meta.get(case_id)

    async def update(self, case_id: str, meta: str) -> None:
        if case_id in self._meta:
            self._meta[case_id] = meta

    async def list_ids(self) -> List[str]:
        return list(self._order)

    async def clear(self) -> None:
        self._order.clear()
        self._meta.clear()


class _RedisQueueBackend:
    name = "redis"

    def __init__(self, client: Any) -> None:
        self._r = client

    async def push(self, case_id: str, meta: str) -> None:
        # HSETNX-style guard: only append to the ordered list the first time a
        # case is escalated, so a re-run cannot double-queue it.
        is_new = await self._r.hsetnx(REVIEW_META_KEY, case_id, meta)
        if is_new:
            await self._r.rpush(REVIEW_QUEUE_KEY, case_id)
        else:
            await self._r.hset(REVIEW_META_KEY, case_id, meta)

    async def remove(self, case_id: str) -> None:
        await self._r.lrem(REVIEW_QUEUE_KEY, 0, case_id)
        await self._r.hdel(REVIEW_META_KEY, case_id)

    async def get(self, case_id: str) -> Optional[str]:
        raw = await self._r.hget(REVIEW_META_KEY, case_id)
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    async def update(self, case_id: str, meta: str) -> None:
        if await self._r.hexists(REVIEW_META_KEY, case_id):
            await self._r.hset(REVIEW_META_KEY, case_id, meta)

    async def list_ids(self) -> List[str]:
        ids = await self._r.lrange(REVIEW_QUEUE_KEY, 0, -1)
        return [i.decode() if isinstance(i, bytes) else i for i in ids]

    async def clear(self) -> None:
        await self._r.delete(REVIEW_QUEUE_KEY, REVIEW_META_KEY)


class ReviewQueue:
    """Escalated cases waiting on a human decision."""

    def __init__(self, log: EventLog, backend: Any) -> None:
        self.log = log
        self._backend = backend

    @property
    def backend(self) -> str:
        return self._backend.name

    @classmethod
    async def connect(cls, log: EventLog, redis_url: Optional[str] = None) -> "ReviewQueue":
        url = redis_url or SETTINGS.redis_url
        try:
            import redis.asyncio as aioredis

            # Same timeout reasoning as EventLog.connect - see the comment there.
            client = aioredis.from_url(url, socket_connect_timeout=10, socket_timeout=10)
            await client.ping()
            return cls(log, _RedisQueueBackend(client))
        except Exception as exc:  # noqa: BLE001
            if SETTINGS.require_redis:
                raise RuntimeError(f"SENTRYMESH_REQUIRE_REDIS=1 but Redis at {url} is unreachable: {exc}") from exc
            return cls(log, _MemoryQueueBackend())

    @classmethod
    def in_memory(cls, log: EventLog) -> "ReviewQueue":
        return cls(log, _MemoryQueueBackend())

    # -- writes ------------------------------------------------------------
    async def enqueue(self, case: Case, verdict: Verdict) -> None:
        """Place an escalated case in front of a human.

        Called by the supervisor when :func:`sentrymesh.agents.supervisor_agent.decide`
        returns ``escalate``. The corresponding ``escalation`` event has already
        been written to the log by that point; this only materialises the work
        list.
        """
        await self._backend.push(case.case_id, json.dumps(self._snapshot(case, verdict), default=str))

    async def submit_decision(self, decision: HumanDecision) -> Dict[str, Any]:
        """Record a reviewer's decision and close (or hold) the case.

        ``approve``/``decline`` close the case and remove it from the queue.
        ``request_detail`` writes the request to the log but deliberately leaves
        the case queued: the loop is not closed until a human commits to an
        outcome, and a case that silently left the queue on a question would be
        a case nobody is accountable for.
        """
        if decision.action not in REVIEW_ACTIONS:
            raise ValueError(f"unknown review action {decision.action!r}; allowed: {REVIEW_ACTIONS}")

        raw = await self._backend.get(decision.case_id)
        if raw is None:
            raise KeyError(f"case {decision.case_id} is not in the review queue")
        snapshot = json.loads(raw)

        await self.log.append(
            decision.case_id,
            "human_reviewer",
            "human_decision",
            {
                **decision.to_dict(),
                "supervisor_decision": snapshot.get("decision"),
                "supervisor_combined_risk": snapshot.get("combined_risk"),
                "agreed_with_supervisor": _agrees(decision.action, snapshot.get("decision")),
                "queue_backend": self.backend,
            },
        )

        if decision.action == "request_detail":
            snapshot.setdefault("detail_requests", []).append(
                {"specialist": decision.requested_specialist, "notes": decision.notes, "reviewer": decision.reviewer}
            )
            snapshot["status"] = "awaiting_detail"
            await self._backend.update(decision.case_id, json.dumps(snapshot, default=str))
            return {"case_id": decision.case_id, "status": "awaiting_detail", "removed_from_queue": False}

        await self._backend.remove(decision.case_id)
        return {
            "case_id": decision.case_id,
            "status": f"closed_{decision.action}",
            "removed_from_queue": True,
        }

    async def clear(self) -> None:
        await self._backend.clear()

    # -- reads -------------------------------------------------------------
    async def pending(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for case_id in await self._backend.list_ids():
            raw = await self._backend.get(case_id)
            if raw:
                out.append(json.loads(raw))
        return out

    async def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        raw = await self._backend.get(case_id)
        return json.loads(raw) if raw else None

    async def depth(self) -> int:
        return len(await self._backend.list_ids())

    async def rebuild_from_log(self) -> int:
        """Rebuild the queue from the event log alone.

        Every case whose latest supervisor ``escalation`` has no subsequent
        closing ``human_decision`` belongs in the queue. Proves the queue is
        derived state and gives a real recovery path if Redis is lost.
        """
        await self._backend.clear()
        restored = 0
        for case_id in await self.log.known_case_ids():
            events = await self.log.get_case_trace(case_id)
            escalation = next((e for e in reversed(events) if e.event_type == "escalation"), None)
            if escalation is None:
                continue
            closed = any(
                e.event_type == "human_decision"
                and e.payload.get("action") in {"approve", "decline"}
                and e.event_id > escalation.event_id
                for e in events
            )
            if closed:
                continue
            verdict_event = next((e for e in reversed(events) if e.event_type == "verdict"), None)
            opened = next((e for e in events if e.event_type == "case_opened"), None)
            payload = verdict_event.payload if verdict_event else {}
            snapshot = {
                "case_id": case_id,
                "summary": (opened.payload.get("summary") if opened else "") or "",
                "category_hint": (opened.payload.get("category_hint") if opened else "") or "",
                "decision": payload.get("decision", "escalate"),
                "combined_risk": payload.get("combined_risk", 0.0),
                "confidence": payload.get("confidence", 0.0),
                "escalation_reason": escalation.payload.get("reason", ""),
                "rationale": payload.get("rationale", ""),
                "disagreement": payload.get("disagreement", {}),
                "findings": [],
                "escalated_at": escalation.timestamp,
                "status": "pending_review",
                "restored_from_log": True,
            }
            await self._backend.push(case_id, json.dumps(snapshot, default=str))
            restored += 1
        return restored

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _snapshot(case: Case, verdict: Verdict) -> Dict[str, Any]:
        return {
            "case_id": case.case_id,
            "summary": case.summary,
            "category_hint": case.category,
            "account_id": case.account_id,
            "decision": verdict.decision,
            "combined_risk": verdict.combined_risk,
            "confidence": verdict.confidence,
            "escalation_reason": verdict.escalation_reason,
            "rationale": verdict.rationale,
            "disagreement": verdict.disagreement.to_dict(),
            "findings": [
                {
                    "specialist": f.specialist,
                    "risk_score": f.risk_score,
                    "confidence": f.confidence,
                    "reasoning_summary": f.reasoning_summary,
                    "key_evidence": f.key_evidence,
                    "status": "ok" if f.ok else f"failed:{f.error_class}",
                }
                for f in verdict.findings
            ],
            "flagged_transaction": case.transaction_data.get("transaction", {}),
            # Retained so a reviewer's "request more detail" can genuinely
            # re-run a specialist. Ground-truth labels are deliberately not
            # copied here - a reviewer must never see them, and neither must a
            # re-run specialist.
            "case_payload": {
                "category": case.category,
                "summary": case.summary,
                "transaction_data": case.transaction_data,
                "device_data": case.device_data,
                "network_data": case.network_data,
            },
            "status": "pending_review",
            "detail_requests": [],
        }


def _agrees(action: str, supervisor_decision: Optional[str]) -> Optional[bool]:
    """Did the human land where the supervisor was leaning?

    Only meaningful once the supervisor actually recommended a direction; an
    escalation is by definition the supervisor declining to lean, so this
    returns None there rather than a misleading False.
    """
    if supervisor_decision == "auto_approve":
        return action == "approve"
    if supervisor_decision == "auto_decline":
        return action == "decline"
    return None
