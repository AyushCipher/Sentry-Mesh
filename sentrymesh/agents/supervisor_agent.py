"""Layer 3 - the supervisor agent.

Responsibilities, in order:

1. **Route.**  Dispatch the case to the specialist panel and record the roster
   it dispatched as an event.  All four specialists run on every case - see
   :func:`SupervisorAgent.dispatch_roster` for why selective routing was
   rejected.
2. **Measure disagreement.**  :func:`detect_disagreement` is a pure, testable
   function over the four findings.  Disagreement is a first-class input, not a
   number that gets averaged away.
3. **Aggregate.**  The supervisor's own LLM call reads each finding *and* the
   disagreement report and reasons about what the conflict means, then returns
   a combined risk and an explicit confidence.
4. **Decide.**  :func:`decide` is a pure function over (risk, confidence,
   disagreement, findings, thresholds).  The model informs the numbers; code
   that a risk team can audit and unit-test turns those numbers into an action.

Splitting 3 from 4 is deliberate.  An LLM is good at "the network specialist and
the identity specialist cannot both be right here, and here is which one the
evidence favours"; it is a bad place to put a threshold that compliance will
one day need to explain in writing.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentrymesh.agents.historical_case_agent import HistoricalCaseAgent
from sentrymesh.agents.identity_signal_agent import IdentitySignalAgent
from sentrymesh.agents.llm_client import LLMClient, ToolCallRecord, tool
from sentrymesh.agents.network_analysis_agent import NetworkAnalysisAgent
from sentrymesh.agents.transaction_pattern_agent import TransactionPatternAgent
from sentrymesh.config import DecisionThresholds, SETTINGS
from sentrymesh.event_log import EventLog
from sentrymesh.models import Case, DisagreementReport, Finding, Verdict

# Relative weight of each signal category in the fallback aggregation.  Precedent
# is discounted because it is retrieval over a small corpus with mock
# embeddings; the two direct-evidence specialists carry the most weight.
SPECIALIST_WEIGHTS = {
    "transaction_pattern": 1.0,
    "identity_signal": 1.0,
    "network_analysis": 0.9,
    "historical_case": 0.7,
}

# A single high-confidence specialist screaming 95 must not be averaged down to
# 40 by three quiet colleagues. The aggregate keeps a floor tied to the loudest
# *confident* voice.
MAX_SIGNAL_BLEND = 0.35
MIN_CONFIDENCE_FOR_MAX_SIGNAL = 0.6


# ---------------------------------------------------------------------------
# Pure decision logic - unit-tested in tests/test_supervisor.py
# ---------------------------------------------------------------------------
def detect_disagreement(
    findings: Sequence[Finding], thresholds: Optional[DecisionThresholds] = None
) -> DisagreementReport:
    """Measure how much the specialist panel disagrees.

    Four independent checks, any of which makes disagreement 'significant':

    * **pairwise spread** - the widest gap between any two specialists
    * **standard deviation** - whether the whole panel is scattered
    * **specialist self-doubt** - any specialist reporting confidence below the
      floor is itself a disagreement signal (it disagrees with its own answer)
    * **split verdict** - specialists on opposite sides of the midpoint, each by
      a clear margin, meaning the panel disagrees on *direction* and not merely
      on magnitude

    Failed specialists are excluded from the score statistics but recorded, so a
    panel that lost half its members cannot look like a panel in agreement.
    """
    th = thresholds or SETTINGS.thresholds
    usable = [f for f in findings if f.ok]
    failed = [f for f in findings if not f.ok]

    reasons: List[str] = []
    dissenters: List[str] = []

    if len(usable) < 2:
        return DisagreementReport(
            pairwise_spread=0.0,
            risk_stdev=0.0,
            lowest_specialist_confidence=min((f.confidence for f in usable), default=0.0),
            split_verdict=False,
            dissenting_specialists=[f.specialist for f in failed],
            reasons=[f"only {len(usable)} of {len(findings)} specialists returned a usable finding"],
            significant=True,
        )

    scores = [f.risk_score for f in usable]
    spread = max(scores) - min(scores)
    stdev = statistics.pstdev(scores)
    lowest_conf = min(f.confidence for f in usable)

    high = [f for f in usable if f.risk_score >= th.split_midpoint + th.split_margin]
    low = [f for f in usable if f.risk_score <= th.split_midpoint - th.split_margin]
    split = bool(high and low)

    if spread >= th.max_pairwise_spread:
        top = max(usable, key=lambda f: f.risk_score)
        bottom = min(usable, key=lambda f: f.risk_score)
        reasons.append(
            f"{spread:.0f}-point spread between {top.specialist} ({top.risk_score:.0f}) "
            f"and {bottom.specialist} ({bottom.risk_score:.0f})"
        )
        dissenters.extend([top.specialist, bottom.specialist])
    if stdev >= th.max_risk_stdev:
        reasons.append(f"risk scores scattered (stdev {stdev:.1f} >= {th.max_risk_stdev})")
    if lowest_conf < th.min_specialist_confidence:
        weak = [f.specialist for f in usable if f.confidence < th.min_specialist_confidence]
        reasons.append(f"specialist self-reported low confidence: {', '.join(weak)} (< {th.min_specialist_confidence})")
        dissenters.extend(weak)
    if split:
        reasons.append(
            "split verdict: "
            + ", ".join(f"{f.specialist}={f.risk_score:.0f}" for f in high)
            + " vs "
            + ", ".join(f"{f.specialist}={f.risk_score:.0f}" for f in low)
        )
        dissenters.extend([f.specialist for f in high + low])
    if failed:
        reasons.append(f"{len(failed)} specialist(s) failed to report: {', '.join(f.specialist for f in failed)}")
        dissenters.extend([f.specialist for f in failed])

    return DisagreementReport(
        pairwise_spread=round(spread, 2),
        risk_stdev=round(stdev, 2),
        lowest_specialist_confidence=round(lowest_conf, 3),
        split_verdict=split,
        dissenting_specialists=sorted(set(dissenters)),
        reasons=reasons,
        significant=bool(reasons),
    )


def aggregate_scores(findings: Sequence[Finding]) -> Tuple[float, float]:
    """Confidence-weighted risk and a baseline panel confidence.

    Two pieces, blended:

    * a confidence-and-category weighted mean, so a specialist that admits it is
      unsure contributes less than one that is certain;
    * the highest risk reported by any *confident* specialist, blended in at
      :data:`MAX_SIGNAL_BLEND`, because fraud signals are disjunctive - one
      conclusive red flag is not cancelled by three clean categories.
    """
    usable = [f for f in findings if f.ok]
    if not usable:
        return 0.0, 0.0

    weights = [SPECIALIST_WEIGHTS.get(f.specialist, 1.0) * max(f.confidence, 0.05) for f in usable]
    weighted_mean = sum(f.risk_score * w for f, w in zip(usable, weights)) / sum(weights)

    confident = [f.risk_score for f in usable if f.confidence >= MIN_CONFIDENCE_FOR_MAX_SIGNAL]
    max_signal = max(confident) if confident else weighted_mean
    combined = (1 - MAX_SIGNAL_BLEND) * weighted_mean + MAX_SIGNAL_BLEND * max_signal

    # Panel confidence starts as the mean specialist confidence and is docked
    # for every specialist that never reported.
    coverage = len(usable) / max(len(findings), 1)
    panel_confidence = statistics.fmean([f.confidence for f in usable]) * coverage

    return round(max(0.0, min(100.0, combined)), 2), round(max(0.0, min(1.0, panel_confidence)), 3)


def decide(
    combined_risk: float,
    confidence: float,
    disagreement: DisagreementReport,
    findings: Sequence[Finding],
    thresholds: Optional[DecisionThresholds] = None,
) -> Tuple[str, Optional[str]]:
    """Turn numbers into an action. Pure, ordered, and auditable.

    Order matters and is deliberate: evidence availability is checked before
    disagreement, disagreement before confidence, and confidence before risk.
    A case can only be auto-resolved if it survives all three gates.
    """
    th = thresholds or SETTINGS.thresholds
    usable = [f for f in findings if f.ok]

    if len(usable) < 3:
        return "escalate", (
            f"insufficient specialist coverage: only {len(usable)} of {len(findings)} specialists "
            "returned a usable finding"
        )
    if disagreement.significant:
        return "escalate", "specialist disagreement: " + "; ".join(disagreement.reasons)
    if confidence < th.min_confidence:
        return "escalate", (
            f"aggregate confidence {confidence:.2f} below the {th.min_confidence:.2f} auto-resolution floor"
        )
    if combined_risk >= th.auto_decline_risk:
        return "auto_decline", None
    if combined_risk <= th.auto_approve_risk:
        return "auto_approve", None
    return "escalate", (
        f"combined risk {combined_risk:.1f} sits in the indeterminate band "
        f"({th.auto_approve_risk}-{th.auto_decline_risk}) where neither action is defensible"
    )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
SUPERVISOR_ASSESSMENT_TOOL = tool(
    "submit_assessment",
    "Submit your final aggregation of the specialist panel. Call this exactly once, after you "
    "have read every finding and the disagreement report.",
    {
        "combined_risk": {
            "type": "number",
            "description": "0-100 overall fraud risk. This is NOT an average of the specialist "
            "scores - weigh each finding by how confident and how conclusive it is, and do not let "
            "three quiet categories cancel one conclusive red flag.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 confidence in the combined risk. Lower it when specialists "
            "contradict each other, when a specialist is unsure, or when a specialist failed.",
        },
        "disagreement_interpretation": {
            "type": "string",
            "description": "If the specialists disagree, explain what the conflict most likely means "
            "(e.g. a shared-device ring with clean identity signals suggests a mule account rather "
            "than a compromised one). If they agree, say so explicitly.",
        },
        "rationale": {
            "type": "string",
            "description": "3-5 sentences a human reviewer could act on: what drove the score and "
            "what would change it.",
        },
        "recommended_action": {
            "type": "string",
            "enum": ["auto_approve", "auto_decline", "escalate"],
            "description": "Your recommendation. The final action is decided by policy thresholds, "
            "which may override you; a divergence is logged and is useful information.",
        },
    },
    required=["combined_risk", "confidence", "disagreement_interpretation", "rationale", "recommended_action"],
)


class SupervisorAgent:
    """Routes a case through the specialist panel and resolves or escalates it."""

    name = "supervisor"

    def __init__(
        self,
        log: EventLog,
        llm: Optional[LLMClient] = None,
        *,
        thresholds: Optional[DecisionThresholds] = None,
        precedent_index: Any = None,
        review_queue: Any = None,
    ) -> None:
        self.log = log
        self.llm = llm or LLMClient()
        self.thresholds = thresholds or SETTINGS.thresholds
        # Duck-typed: anything exposing `await enqueue(case, verdict)`.
        self.review_queue = review_queue
        self.specialists = [
            TransactionPatternAgent(log, self.llm),
            IdentitySignalAgent(log, self.llm),
            NetworkAnalysisAgent(log, self.llm),
            HistoricalCaseAgent(log, self.llm, index=precedent_index),
        ]

    def dispatch_roster(self, case: Case) -> List[str]:
        """Which specialists to run.

        Every specialist runs on every case, and that is a decision rather than
        a shortcut. Selective routing would mean a category could be scored
        without ever being looked at, which makes a clean result unfalsifiable:
        you could not tell "the network agent found nothing" from "the network
        agent was never asked". It would also make the eval's per-category
        false-positive analysis unattributable. The cost is four LLM calls per
        case instead of two; the benefit is that every verdict is backed by the
        same four-signal evidence base.
        """
        return [s.name for s in self.specialists]

    async def investigate(self, case: Case) -> Verdict:
        started = time.perf_counter()

        await self.log.append(
            case.case_id,
            self.name,
            "case_opened",
            {
                "category_hint": case.category,
                "summary": case.summary,
                "account_id": case.account_id,
                "run_mode": self.llm.run_mode,
                "synthetic_data": True,
            },
        )

        roster = self.dispatch_roster(case)
        await self.log.append(
            case.case_id,
            self.name,
            "tool_call",
            {
                "tool": "dispatch_specialists",
                "arguments": {"roster": roster, "execution": "parallel_asyncio_gather"},
                "result": {"dispatched": len(roster)},
                "note": "All specialists run on every case; see SupervisorAgent.dispatch_roster.",
            },
        )

        # Genuinely concurrent: each specialist owns its own LLM tool-use loop
        # and they share nothing but the append-only log, so there is no
        # ordering constraint between them. With the Anthropic backend this is
        # four in-flight HTTP requests, not four sequential ones.
        findings: List[Finding] = list(
            await asyncio.gather(*(s.run(case) for s in self.specialists))
        )

        disagreement = detect_disagreement(findings, self.thresholds)
        fallback_risk, fallback_confidence = aggregate_scores(findings)

        assessment, agg_error, agg_error_class = await self._assess(
            case, findings, disagreement, fallback_risk, fallback_confidence
        )

        combined_risk = assessment["combined_risk"]
        confidence = assessment["confidence"]
        decision, escalation_reason = decide(combined_risk, confidence, disagreement, findings, self.thresholds)

        run_errors = [
            {"actor": f.specialist, "error_class": f.error_class or "unknown_error", "error": f.error or ""}
            for f in findings
            if not f.ok
        ]
        if agg_error:
            run_errors.append({"actor": self.name, "error_class": agg_error_class or "unknown_error", "error": agg_error})

        verdict = Verdict(
            case_id=case.case_id,
            decision=decision,
            combined_risk=combined_risk,
            confidence=confidence,
            rationale=assessment["rationale"],
            disagreement=disagreement,
            findings=findings,
            escalation_reason=escalation_reason,
            supervisor_backend=self.llm.run_mode,
            duration_ms=int((time.perf_counter() - started) * 1000),
            run_errors=run_errors,
        )

        await self.log.append(
            case.case_id,
            self.name,
            "verdict",
            {
                "decision": decision,
                "combined_risk": combined_risk,
                "confidence": confidence,
                "rationale": assessment["rationale"],
                "disagreement_interpretation": assessment["disagreement_interpretation"],
                "model_recommended_action": assessment["recommended_action"],
                "policy_overrode_model": assessment["recommended_action"] != decision,
                "disagreement": disagreement.to_dict(),
                "specialist_scores": {f.specialist: f.risk_score for f in findings},
                "specialist_confidence": {f.specialist: f.confidence for f in findings},
                "deterministic_fallback_risk": fallback_risk,
                "run_mode": self.llm.run_mode,
                "run_errors": run_errors,
                "duration_ms": verdict.duration_ms,
            },
        )

        if decision == "escalate":
            await self.log.append(
                case.case_id,
                self.name,
                "escalation",
                {
                    "reason": escalation_reason,
                    "combined_risk": combined_risk,
                    "confidence": confidence,
                    "dissenting_specialists": disagreement.dissenting_specialists,
                    "queue": "human_review",
                },
            )
            if self.review_queue is not None:
                await self.review_queue.enqueue(case, verdict)

        return verdict

    # ------------------------------------------------------------------ LLM
    async def _assess(
        self,
        case: Case,
        findings: List[Finding],
        disagreement: DisagreementReport,
        fallback_risk: float,
        fallback_confidence: float,
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
        """Run the supervisor's own tool-use loop over the panel's output."""
        by_name = {f.specialist: f for f in findings}

        async def executor(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            if name == "get_specialist_finding":
                target = str(args.get("specialist", ""))
                finding = by_name.get(target)
                if finding is None:
                    return {"error": f"no specialist named {target!r}", "available": sorted(by_name)}
                return finding.to_dict()
            if name == "get_disagreement_report":
                return disagreement.to_dict()
            if name == "get_case_summary":
                return {
                    "case_id": case.case_id,
                    "summary": case.summary,
                    "account_id": case.account_id,
                    "flagged_transaction": case.transaction_data.get("transaction", {}),
                }
            raise ValueError(f"supervisor: unknown tool {name!r}")

        async def on_tool_call(record: ToolCallRecord) -> None:
            await self.log.append(
                case.case_id, self.name, "tool_call", {**record.to_dict(), "run_mode": self.llm.run_mode}
            )

        result = await self.llm.run_tool_loop(
            system=self._system_prompt(),
            user_prompt=self._user_prompt(case, findings, disagreement),
            tools=[
                tool(
                    "get_specialist_finding",
                    "Re-read one specialist's full structured finding, including its key evidence.",
                    {
                        "specialist": {
                            "type": "string",
                            "enum": sorted(by_name),
                            "description": "Which specialist's finding to retrieve.",
                        }
                    },
                    required=["specialist"],
                ),
                tool(
                    "get_disagreement_report",
                    "Retrieve the computed disagreement metrics for this panel: pairwise spread, "
                    "standard deviation, lowest specialist confidence, and split-verdict status.",
                    {},
                ),
                tool(
                    "get_case_summary",
                    "Retrieve the case summary and the flagged transaction.",
                    {},
                ),
                SUPERVISOR_ASSESSMENT_TOOL,
            ],
            executor=executor,
            final_tool="submit_assessment",
            offline_plan=lambda: [
                ("get_case_summary", {}),
                ("get_disagreement_report", {}),
                *[("get_specialist_finding", {"specialist": name}) for name in sorted(by_name)],
            ],
            offline_solver=lambda records: self._offline_assessment(
                findings, disagreement, fallback_risk, fallback_confidence
            ),
            on_tool_call=on_tool_call,
        )

        if result.error:
            # The supervisor's model call failing must not silently become a
            # confident verdict. Fall back to the deterministic aggregate, mark
            # the error, and let `decide` see the depressed confidence.
            await self.log.append(
                case.case_id,
                self.name,
                "error",
                {"stage": "aggregation", "error": result.error, "error_class": result.error_class},
            )
            assessment = self._offline_assessment(findings, disagreement, fallback_risk, fallback_confidence)
            assessment["confidence"] = round(min(assessment["confidence"], 0.5), 3)
            assessment["rationale"] = (
                f"Supervisor aggregation model call failed ({result.error_class}); fell back to the "
                "deterministic weighted aggregate with confidence capped so the case cannot be "
                "auto-resolved on a degraded assessment. " + assessment["rationale"]
            )
            return assessment, result.error, result.error_class

        structured = result.structured
        action = str(structured.get("recommended_action") or "escalate")
        return (
            {
                "combined_risk": _clamp(structured.get("combined_risk"), 0.0, 100.0, fallback_risk),
                "confidence": _clamp(structured.get("confidence"), 0.0, 1.0, fallback_confidence),
                "disagreement_interpretation": str(structured.get("disagreement_interpretation") or "").strip(),
                "rationale": str(structured.get("rationale") or "").strip() or "(no rationale returned)",
                "recommended_action": action if action in {"auto_approve", "auto_decline", "escalate"} else "escalate",
            },
            None,
            None,
        )

    def _system_prompt(self) -> str:
        th = self.thresholds
        return (
            "You are the supervising fraud investigator. Four independent specialists have each "
            "analysed one signal category of this case. Your job is to combine their findings into "
            "one risk score and one confidence value.\n\n"
            "Do NOT average the four risk scores. Averaging is how a conclusive single red flag "
            "gets buried under three quiet categories, and how four mediocre signals get inflated "
            "into a decline.\n\n"
            "Treat DISAGREEMENT as evidence in its own right. When specialists conflict, ask what "
            "pattern would produce exactly that conflict:\n"
            "- High network-ring score with clean identity and KYC signals often means a mule "
            "account operated by its real owner, not a compromised one.\n"
            "- High identity risk (new device, impossible travel) with a normal transaction profile "
            "often means an account takeover that has not yet cashed out.\n"
            "- Strong precedent against the live signals means either the precedent is stale or the "
            "live signals are a false alarm; say which you think it is.\n"
            "- A specialist reporting low confidence is telling you its category is uninformative "
            "here. Reweight, do not average it in at face value.\n\n"
            f"Policy thresholds applied after you answer: risk >= {th.auto_decline_risk} with "
            f"confidence >= {th.min_confidence} auto-declines; risk <= {th.auto_approve_risk} with "
            f"the same confidence auto-approves; anything else, any significant disagreement, or "
            "any confidence below that floor goes to a human reviewer. Escalating is not a failure "
            "- a false decline costs a real customer a real transaction, and a human reviewer is "
            "cheaper than either error.\n\n"
            "Call submit_assessment exactly once when you are done."
        )

    def _user_prompt(self, case: Case, findings: List[Finding], disagreement: DisagreementReport) -> str:
        panel = [
            {
                "specialist": f.specialist,
                "risk_score": f.risk_score,
                "confidence": f.confidence,
                "reasoning_summary": f.reasoning_summary,
                "key_evidence": f.key_evidence,
                "status": "ok" if f.ok else f"FAILED ({f.error_class})",
            }
            for f in findings
        ]
        return (
            f"Case {case.case_id} (SYNTHETIC evaluation data). Summary: {case.summary or 'n/a'}\n\n"
            "Specialist panel findings:\n"
            f"{json.dumps(panel, indent=2, default=str)}\n\n"
            "Computed disagreement report:\n"
            f"{json.dumps(disagreement.to_dict(), indent=2, default=str)}\n\n"
            "Use get_specialist_finding or get_case_summary if you need more detail, then call "
            "submit_assessment."
        )

    def _offline_assessment(
        self,
        findings: List[Finding],
        disagreement: DisagreementReport,
        fallback_risk: float,
        fallback_confidence: float,
    ) -> Dict[str, Any]:
        """Deterministic aggregation used when no LLM backend is configured."""
        confidence = fallback_confidence
        if disagreement.significant:
            # Each independent reason to doubt the panel costs confidence.
            confidence *= max(0.35, 1.0 - 0.18 * len(disagreement.reasons))
        confidence = round(max(0.0, min(1.0, confidence)), 3)

        if disagreement.significant:
            interpretation = (
                "Panel is not in agreement. " + "; ".join(disagreement.reasons) + ". "
                "Offline scorer cannot reason about *why* the categories conflict, so it lowers "
                "confidence rather than picking a side."
            )
        else:
            interpretation = (
                f"All {len([f for f in findings if f.ok])} reporting specialists agree in direction "
                f"(spread {disagreement.pairwise_spread}, stdev {disagreement.risk_stdev})."
            )

        recommended, _ = decide(fallback_risk, confidence, disagreement, findings, self.thresholds)
        detail = ", ".join(f"{f.specialist} {f.risk_score:.0f}@{f.confidence:.2f}" for f in findings)
        return {
            "combined_risk": fallback_risk,
            "confidence": confidence,
            "disagreement_interpretation": interpretation,
            "rationale": (
                "Offline deterministic aggregation (no LLM): confidence-weighted mean blended with "
                f"the strongest confident signal gives {fallback_risk:.1f}. Panel: {detail}. "
                f"Confidence {confidence:.2f} after disagreement penalty."
            ),
            "recommended_action": recommended,
        }


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        return round(max(low, min(high, float(value))), 3)
    except (TypeError, ValueError):
        return default
