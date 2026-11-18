"""Wiring for the five layers.

One object owns the event log, the LLM backend, the precedent index, the review
queue and the supervisor, so the API, the eval harness and the tests all run the
*same* system rather than three slightly different assemblies of it.

Note :meth:`SentryMesh.case_status` - it reconstructs a case's entire state from
the event log and nothing else. If that method can answer, the claim that the
log is the single source of truth is testable rather than aspirational.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sentrymesh.agents.llm_client import LLMClient
from sentrymesh.agents.supervisor_agent import SupervisorAgent
from sentrymesh.config import SETTINGS
from sentrymesh.event_log import Event, EventLog
from sentrymesh.hitl import ReviewQueue
from sentrymesh.models import Case, Finding, HumanDecision, Verdict
from sentrymesh.retrieval import build_index


class SentryMesh:
    def __init__(
        self,
        log: EventLog,
        queue: ReviewQueue,
        supervisor: SupervisorAgent,
        llm: LLMClient,
        precedent_index: Any,
    ) -> None:
        self.log = log
        self.queue = queue
        self.supervisor = supervisor
        self.llm = llm
        self.precedent_index = precedent_index

    @classmethod
    async def create(cls, *, llm: Optional[LLMClient] = None, in_memory: bool = False) -> "SentryMesh":
        log = EventLog.in_memory() if in_memory else await EventLog.connect()
        queue = ReviewQueue.in_memory(log) if in_memory else await ReviewQueue.connect(log)
        client = llm or LLMClient()
        index = await build_index()
        supervisor = SupervisorAgent(log, client, precedent_index=index, review_queue=queue)
        return cls(log, queue, supervisor, client, index)

    # -- runtime info -------------------------------------------------------
    def runtime(self) -> Dict[str, Any]:
        """What this process is actually running on. Surfaced by /health and
        stamped into every harness report so no result is ambiguous about it."""
        return {
            "event_log_backend": self.log.backend,
            "event_log_downgrade_reason": self.log.downgrade_reason,
            "review_queue_backend": self.queue.backend,
            "review_queue_downgrade_reason": self.queue.downgrade_reason,
            "retrieval_backend": getattr(self.precedent_index, "name", "unknown"),
            "llm_backend": self.llm.backend,
            "run_mode": self.llm.run_mode,
            "llm_provider": SETTINGS.provider_label(),
            "model": self.llm.model if self.llm.is_live else None,
            "thresholds": {
                "auto_decline_risk": SETTINGS.thresholds.auto_decline_risk,
                "auto_approve_risk": SETTINGS.thresholds.auto_approve_risk,
                "min_confidence": SETTINGS.thresholds.min_confidence,
            },
            "data": "synthetic",
        }

    # -- case flow ----------------------------------------------------------
    async def investigate(self, case: Case) -> Verdict:
        return await self.supervisor.investigate(case)

    async def trace(self, case_id: str) -> List[Event]:
        return await self.log.get_case_trace(case_id)

    async def case_status(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Rebuild a case's current state from the event log alone."""
        events = await self.log.get_case_trace(case_id)
        if not events:
            return None

        opened = next((e for e in events if e.event_type == "case_opened"), None)
        verdict = next((e for e in reversed(events) if e.event_type == "verdict"), None)
        escalation = next((e for e in reversed(events) if e.event_type == "escalation"), None)
        human = next(
            (
                e
                for e in reversed(events)
                if e.event_type == "human_decision" and e.payload.get("action") in {"approve", "decline"}
            ),
            None,
        )
        findings = [e for e in events if e.event_type == "finding"]
        errors = [e for e in events if e.event_type == "error"]

        if human is not None:
            status = f"closed_by_human_{human.payload.get('action')}"
        elif escalation is not None:
            status = "pending_review"
        elif verdict is not None:
            status = f"closed_{verdict.payload.get('decision')}"
        else:
            status = "in_progress"

        return {
            "case_id": case_id,
            "status": status,
            "opened_at": opened.timestamp if opened else events[0].timestamp,
            "summary": (opened.payload.get("summary") if opened else "") or "",
            "category_hint": (opened.payload.get("category_hint") if opened else "") or "",
            "run_mode": (opened.payload.get("run_mode") if opened else None),
            "verdict": verdict.payload if verdict else None,
            "escalation": escalation.payload if escalation else None,
            "human_decision": human.payload if human else None,
            "findings": [e.payload for e in findings],
            "errors": [e.payload for e in errors],
            "event_count": len(events),
            "reconstructed_from": "event_log",
        }

    # -- human review -------------------------------------------------------
    async def review(self, decision: HumanDecision) -> Dict[str, Any]:
        """Apply a human decision.

        ``request_detail`` genuinely re-runs the named specialist against the
        case rather than just logging the request, so "ask for more detail" is a
        real action with a new finding in the trace, not a UI affordance.
        """
        result = await self.queue.submit_decision(decision)

        if decision.action == "request_detail" and decision.requested_specialist:
            snapshot = await self.queue.get(decision.case_id)
            case = self._case_from_snapshot(decision.case_id, snapshot)
            specialist = next(
                (s for s in self.supervisor.specialists if s.name == decision.requested_specialist), None
            )
            if specialist is None:
                result["detail_error"] = f"no specialist named {decision.requested_specialist!r}"
            elif case is None:
                result["detail_error"] = "case payload is not retained in the queue snapshot"
            else:
                finding = await specialist.run(case)
                result["detail_finding"] = finding.to_dict()
                result["reaggregated"] = await self._reaggregate(decision.case_id, snapshot, finding)
        return result

    async def _reaggregate(self, case_id: str, snapshot: Dict[str, Any], new_finding: Finding) -> Dict[str, Any]:
        """Swap the re-run specialist's finding into the panel and re-derive the verdict.

        Deterministic (no LLM call). Writes a ``verdict`` event flagged
        ``reaggregated`` so the trace shows the refreshed numbers, and updates
        the queued snapshot - but never closes the case: only a human does.
        """
        panel: List[Finding] = []
        replaced = False
        for row in snapshot.get("findings", []):
            if row.get("specialist") == new_finding.specialist:
                panel.append(new_finding)
                replaced = True
            elif str(row.get("status", "ok")).startswith("failed:"):
                panel.append(Finding.failed(row["specialist"], "failed in original run", row["status"][7:]))
            else:
                panel.append(
                    Finding(
                        specialist=row["specialist"],
                        risk_score=float(row.get("risk_score", 0.0)),
                        confidence=float(row.get("confidence", 0.0)),
                        reasoning_summary=row.get("reasoning_summary", ""),
                        key_evidence=list(row.get("key_evidence", [])),
                    )
                )
        if not replaced:
            panel.append(new_finding)

        agg = self.supervisor.reaggregate(panel)
        await self.log.append(
            case_id,
            "supervisor",
            "verdict",
            {
                "decision": agg["decision"],
                "combined_risk": agg["combined_risk"],
                "confidence": agg["confidence"],
                "rationale": agg["rationale"],
                "disagreement": agg["disagreement"].to_dict(),
                "specialist_scores": {f.specialist: f.risk_score for f in panel},
                "specialist_confidence": {f.specialist: f.confidence for f in panel},
                "escalation_reason": agg["escalation_reason"],
                "reaggregated": True,
                "reaggregated_after": new_finding.specialist,
                "run_mode": "offline_deterministic",
            },
        )
        updated = {
            **snapshot,
            "decision": agg["decision"],
            "combined_risk": agg["combined_risk"],
            "confidence": agg["confidence"],
            "rationale": agg["rationale"],
            "escalation_reason": agg["escalation_reason"],
            "disagreement": agg["disagreement"].to_dict(),
            "findings": [
                {
                    "specialist": f.specialist,
                    "risk_score": f.risk_score,
                    "confidence": f.confidence,
                    "reasoning_summary": f.reasoning_summary,
                    "key_evidence": f.key_evidence,
                    "status": "ok" if f.ok else f"failed:{f.error_class}",
                }
                for f in panel
            ],
            "reaggregated": True,
        }
        await self.queue.update_snapshot(case_id, updated)
        return {
            "decision": agg["decision"],
            "combined_risk": agg["combined_risk"],
            "confidence": agg["confidence"],
            "escalation_reason": agg["escalation_reason"],
        }

    @staticmethod
    def _case_from_snapshot(case_id: str, snapshot: Optional[Dict[str, Any]]) -> Optional[Case]:
        if not snapshot or "case_payload" not in snapshot:
            return None
        return Case.from_dict({**snapshot["case_payload"], "case_id": case_id})

    async def close(self) -> None:
        await self.log.close()
        index_close = getattr(self.precedent_index, "close", None)
        if index_close is not None:
            await index_close()
