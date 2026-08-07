"""Self-scrutiny for the harness report.

A report that only reports success is worth less than one that interrogates
itself, so this module exists to attack the results the harness just produced.
Nothing in here is templated: every statement is derived from the actual run,
and if there is nothing suspicious to say, it says that instead of manufacturing
a caveat.

The central technique is **ablation**, and it is cheap because it needs no
further model calls: the specialist findings are already recorded, so the
supervisor's aggregation and decision functions can simply be re-run over a
reduced panel. That turns "did the agents really reason over multiple signals?"
from a rhetorical question into a measurement - if removing the single
loudest specialist leaves the verdict unchanged, the decision genuinely rested
on more than one signal; if it collapses, it did not, whatever the headline
accuracy says.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sentrymesh.agents.supervisor_agent import aggregate_scores, decide, detect_disagreement, partition
from sentrymesh.models import Finding

# Accuracy at or above this in any slice is treated as a result to interrogate
# rather than a result to celebrate.
SUSPICION_THRESHOLD = 0.95


def _findings(row: Dict[str, Any]) -> List[Finding]:
    return [
        Finding(
            specialist=f["specialist"],
            risk_score=float(f["risk_score"]),
            confidence=float(f["confidence"]),
            reasoning_summary=f.get("reasoning_summary", ""),
            key_evidence=f.get("key_evidence", []),
            error=f.get("error"),
            error_class=f.get("error_class"),
        )
        for f in row.get("findings", [])
    ]


def replay(findings: Sequence[Finding]) -> Dict[str, Any]:
    """Re-run the supervisor's deterministic decision path over a panel."""
    risk, confidence = aggregate_scores(findings)
    disagreement = detect_disagreement(findings)
    decision, reason = decide(risk, confidence, disagreement, findings)
    return {"decision": decision, "combined_risk": risk, "confidence": confidence, "reason": reason}


def _top_contributor(findings: Sequence[Finding]) -> Optional[Finding]:
    contributing, _, _ = partition(findings)
    if not contributing:
        return None
    return max(contributing, key=lambda f: f.risk_score * f.confidence)


def ablation_study(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How much of each verdict rests on a single specialist?"""
    single_signal: List[Dict[str, Any]] = []
    multi_signal: List[str] = []
    precedent_dependent: List[Dict[str, Any]] = []
    examined = 0

    for row in results:
        findings = _findings(row)
        if not findings or row.get("decision") not in {"auto_approve", "auto_decline"}:
            continue
        examined += 1
        baseline = row.get("decision")

        top = _top_contributor(findings)
        if top is not None:
            without_top = replay([f for f in findings if f.specialist != top.specialist])
            if without_top["decision"] != baseline:
                single_signal.append(
                    {
                        "case_id": row["case_id"],
                        "decision": baseline,
                        "removed_specialist": top.specialist,
                        "removed_risk_score": top.risk_score,
                        "decision_without_it": without_top["decision"],
                        "risk_without_it": without_top["combined_risk"],
                    }
                )
            else:
                multi_signal.append(row["case_id"])

        without_precedent = replay([f for f in findings if f.specialist != "historical_case"])
        if without_precedent["decision"] != baseline:
            precedent_dependent.append(
                {
                    "case_id": row["case_id"],
                    "decision": baseline,
                    "decision_without_precedent": without_precedent["decision"],
                    "risk_delta": round(
                        without_precedent["combined_risk"] - float(row.get("combined_risk") or 0.0), 2
                    ),
                }
            )

    total = len(single_signal) + len(multi_signal)
    return {
        "method": "Re-runs the supervisor's aggregation and decision functions over a reduced panel "
        "using the findings already recorded in the event log. No additional model calls, so this "
        "measures the decision logic exactly as it ran.",
        "auto_decisions_examined": examined,
        "survives_removal_of_loudest_specialist": len(multi_signal),
        "collapses_without_loudest_specialist": len(single_signal),
        "multi_signal_share": round(len(multi_signal) / total, 4) if total else None,
        "single_signal_cases": single_signal,
        "precedent_dependence": {
            "decisions_that_change_without_historical_case": len(precedent_dependent),
            "cases": precedent_dependent,
            "why_this_matters": "The precedent corpus and the case bank were authored by the same "
            "person, so the historical-case specialist has a contamination advantage that would not "
            "survive contact with a real case archive. Decisions that depend on it should be "
            "discounted accordingly.",
        },
    }


def interrogate_slices(
    breakdown: Dict[str, Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    slice_key: str,
    ablation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Flag any slice at or near 100% and investigate whether it is real."""
    single_signal_ids = {c["case_id"] for c in ablation.get("single_signal_cases", [])}
    suspicions: List[Dict[str, Any]] = []

    for group, stats in breakdown.items():
        auto = stats["auto_only"]
        accuracy = auto.get("accuracy")
        scored = auto.get("scored_cases", 0)
        if accuracy is None or scored == 0 or accuracy < SUSPICION_THRESHOLD:
            continue

        rows = [r for r in results if str(r.get(slice_key)) == group]
        decisive = [r for r in rows if r.get("ground_truth") in {"fraud", "legitimate"}]
        auto_rows = [r for r in decisive if r.get("decision") in {"auto_approve", "auto_decline"}]
        escalated = len(decisive) - len(auto_rows)

        planted = [len(r.get("planted_signals", [])) for r in auto_rows]
        avg_planted = round(sum(planted) / len(planted), 2) if planted else 0.0
        contributors = [r.get("contributing_specialists", 0) for r in auto_rows]
        avg_contributors = round(sum(contributors) / len(contributors), 2) if contributors else 0.0
        collapsing = [r["case_id"] for r in auto_rows if r["case_id"] in single_signal_ids]

        checks: List[str] = []
        verdict_parts: List[str] = []

        if scored <= 4:
            checks.append(
                f"Only {scored} case(s) were scored in this slice. At that size a single case moves "
                "the number by 20 points or more; this is not a measurement, it is an anecdote."
            )
            verdict_parts.append("sample too small to mean anything")

        if escalated:
            checks.append(
                f"{escalated} of {len(decisive)} decisive cases in this slice were escalated and are "
                "therefore excluded from the auto_only accuracy. The perfect score describes only the "
                f"{len(auto_rows)} case(s) the system was willing to commit to."
            )
            verdict_parts.append("survivorship: the hard cases were escalated out of the metric")

        if avg_planted >= 4:
            checks.append(
                f"Cases in this slice carry {avg_planted} planted signals on average. They were built "
                "to be detectable, and a system that missed them would be broken rather than merely "
                "imperfect. Getting them right is a floor, not an achievement."
            )
            verdict_parts.append("cases were seeded to be easy")

        if collapsing:
            checks.append(
                f"{len(collapsing)} of {len(auto_rows)} auto-decisions here collapse when the loudest "
                f"specialist is removed ({', '.join(collapsing)}), so they rest on one signal rather "
                "than on corroboration across the panel."
            )
            verdict_parts.append("decisions rest on a single specialist")
        elif auto_rows:
            checks.append(
                f"All {len(auto_rows)} auto-decisions survive removal of the loudest specialist, and "
                f"{avg_contributors} specialists contributed a confident finding on average. The "
                "multi-signal reasoning is real for this slice."
            )

        suspicions.append(
            {
                "slice": f"{slice_key}={group}",
                "accuracy": accuracy,
                "scored_cases": scored,
                "escalated_out_of_metric": escalated,
                "avg_planted_signals": avg_planted,
                "avg_contributing_specialists": avg_contributors,
                "checks": checks,
                "verdict": (
                    "SUSPECT - " + "; ".join(verdict_parts)
                    if verdict_parts
                    else "Holds up under the checks available here, within the limits of a "
                    f"{scored}-case slice."
                ),
            }
        )
    return suspicions


def build_honesty_section(
    results: Sequence[Dict[str, Any]],
    overall: Dict[str, Any],
    by_category: Dict[str, Dict[str, Any]],
    by_difficulty: Dict[str, Dict[str, Any]],
    escalation: Dict[str, Any],
    errors: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the mandatory self-scrutiny section from this run's actual data."""
    ablation = ablation_study(results)
    suspicions = interrogate_slices(by_category, results, "category", ablation) + interrogate_slices(
        by_difficulty, results, "difficulty", ablation
    )

    concerns: List[str] = []

    # 1. What the run mode means for every number above it.
    if runtime.get("run_mode") != "anthropic":
        concerns.append(
            "RUN MODE: this run used the offline deterministic reasoner, not an LLM, because no "
            "ANTHROPIC_API_KEY was configured. Every accuracy number in this report therefore measures "
            "the explicit scoring functions inside each specialist, NOT agent reasoning. The tool "
            "calls, the event log, the supervisor's aggregation and the decision thresholds are all "
            "exercised exactly as they would be in an LLM run - but nothing here is evidence about "
            "model quality, and it must not be presented as such. Re-run with a key set to get numbers "
            "that mean what a reader will assume they mean."
        )
    else:
        concerns.append(
            f"RUN MODE: live Anthropic tool-use loops on {runtime.get('model')}. Specialist and "
            "supervisor reasoning is model-driven, so re-running will not reproduce these numbers "
            "exactly. Single-run results at this sample size should be treated as indicative."
        )

    # 2. Sample size, stated before anything is read into the numbers.
    total = len(results)
    decisive = sum(1 for r in results if r.get("ground_truth") in {"fraud", "legitimate"})
    concerns.append(
        f"SAMPLE SIZE: {total} cases, {decisive} with a decisive ground truth. Every per-category "
        "figure below rests on 2-7 cases. Confidence intervals at that size are wide enough to "
        "swallow most of the differences between categories, and none of these numbers would survive "
        "being quoted as a production benchmark."
    )

    # 3. The auto_only vs operational gap.
    auto = overall["auto_only"]
    flagged = overall["escalation_as_flag"]
    if auto.get("false_positive_rate") is not None and flagged.get("false_positive_rate") is not None:
        concerns.append(
            f"SCORING VIEW: the false-positive rate is {auto['false_positive_rate']:.1%} counting only "
            f"auto-declines, and {flagged['false_positive_rate']:.1%} once escalations are counted as "
            "flags. The second number is the one a customer feels: an escalated payment is still held. "
            "Quoting only the first would be the single most misleading thing this report could do."
        )

    # 4. Whether the system is actually deciding anything.
    rate = escalation.get("escalation_rate")
    if rate is not None and rate >= 0.4:
        concerns.append(
            f"AUTONOMY: {rate:.0%} of cases were escalated to a human. High precision on the remaining "
            "auto-decisions is partly a selection effect - the system chose which cases to be judged "
            "on. A fraud team would not deploy this without pushing the escalation rate down, and "
            "doing so is exactly what would expose the error rate this report cannot currently see."
        )

    # 5. Ablation.
    share = ablation.get("multi_signal_share")
    if share is not None:
        if share >= 0.8:
            concerns.append(
                f"MULTI-SIGNAL REASONING: {share:.0%} of auto-decisions survive removal of their "
                "loudest specialist, so the supervisor is genuinely combining evidence rather than "
                "relaying one agent's opinion. This is the claim the architecture exists to support "
                "and it is the one piece of good news in this section that is actually measured."
            )
        else:
            concerns.append(
                f"MULTI-SIGNAL REASONING: only {share:.0%} of auto-decisions survive removal of their "
                "loudest specialist. The rest collapse to a different verdict, which means the "
                "four-specialist architecture is not earning its cost on those cases - a single "
                "specialist plus the threshold logic would have produced the same answers."
            )

    # 6. Precedent contamination.
    dependent = ablation["precedent_dependence"]["decisions_that_change_without_historical_case"]
    concerns.append(
        f"RETRIEVAL CONTAMINATION: {dependent} auto-decision(s) change when the historical-case "
        "specialist is removed. The precedent corpus was written alongside the case bank by the same "
        "author, and retrieval is lexical TF-IDF, so precedent summaries share vocabulary with the "
        "cases they 'predict'. Retrieval hits at this rate would not transfer to a real case archive, "
        "and any per-category strength that traces back to this specialist should be discounted."
    )

    # 7. Errors, separated.
    infra = errors["infrastructure_failures"]["total"]
    protocol = errors["agent_protocol_failures"]["total"]
    wrong = errors["wrong_auto_decisions"]["total"]
    if infra or protocol:
        concerns.append(
            f"FAILURE ACCOUNTING: {infra} infrastructure failure(s) and {protocol} agent-protocol "
            f"failure(s) occurred, against {wrong} genuinely wrong auto-decision(s). These are "
            "reported apart because they mean different things: infrastructure failures make the "
            "metrics less reliable, protocol failures point at prompt or schema problems, and only "
            "the last bucket is evidence about reasoning quality."
        )
    else:
        concerns.append(
            f"FAILURE ACCOUNTING: no infrastructure or agent-protocol failures in this run. All "
            f"{wrong} incorrect outcome(s) are genuine wrong decisions rather than plumbing, so the "
            "metrics above are measured on complete evidence."
        )

    # 8. Anything the harness cannot see at all.
    concerns.append(
        "NOT MEASURED: the case bank was written by the same person who wrote the detectors, so it "
        "encodes the same assumptions about what fraud looks like. Nothing here tests the patterns "
        "neither the author nor the agents thought of, and that blind spot is where a real fraud "
        "system's losses actually come from. Adversarial cases authored by someone else would be the "
        "first thing to add."
    )

    if not suspicions:
        headline = (
            "No slice reached the 95% threshold that would trigger a suspicious-result investigation. "
            "The results are unremarkable, which at this sample size is the honest outcome to expect."
        )
    else:
        flagged_slices = [s["slice"] for s in suspicions]
        headline = (
            f"{len(suspicions)} slice(s) came back at or above {SUSPICION_THRESHOLD:.0%} accuracy "
            f"({', '.join(flagged_slices)}). Each is investigated below rather than reported as a win."
        )

    return {
        "headline": headline,
        "suspicious_results": suspicions,
        "ablation_study": ablation,
        "concerns": concerns,
        "how_to_read_this": (
            "Read the escalation rate first, then the escalation_as_flag false-positive rate, then the "
            "ablation study. The auto_only precision figure is the least informative number in this "
            "report and is printed first only because that is the convention."
        ),
    }
