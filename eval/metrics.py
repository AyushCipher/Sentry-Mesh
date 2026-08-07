"""Scoring for the SentryMesh eval harness.

TWO SCORING VIEWS, BOTH REPORTED
--------------------------------
An escalated case is not a prediction, but it is also not free. Reporting only
one view of it is how fraud evals flatter themselves, so this module computes
both and the report shows them side by side:

``auto_only``
    Escalations are excluded entirely; only auto-approve/auto-decline count as
    predictions. This measures the quality of the decisions the system actually
    commits to on its own. It is the *flattering* view, because a system that
    escalates everything it finds hard scores perfectly here.

``escalation_as_flag``
    Escalations count as "flagged". This is closer to what a customer and a
    review team experience: an escalated payment is held, and a human has to be
    paid to look at it. It is the *operational* view, and it is where an
    escalate-everything strategy shows its true false-positive cost.

The gap between the two numbers is the most informative thing on the page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

FRAUD_LABELS = {"fraud"}
LEGIT_LABELS = {"legitimate"}


@dataclass
class ConfusionMatrix:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    abstained: int = 0  # escalations, in the auto_only view

    @property
    def precision(self) -> Optional[float]:
        denom = self.true_positive + self.false_positive
        return round(self.true_positive / denom, 4) if denom else None

    @property
    def recall(self) -> Optional[float]:
        denom = self.true_positive + self.false_negative
        return round(self.true_positive / denom, 4) if denom else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return round(2 * p * r / (p + r), 4)

    @property
    def false_positive_rate(self) -> Optional[float]:
        """Share of legitimate cases wrongly flagged.

        In a real fraud system this number costs revenue and customers, and it
        is the one most often left out of a demo report. It is never omitted
        here, including when it is inconvenient.
        """
        denom = self.false_positive + self.true_negative
        return round(self.false_positive / denom, 4) if denom else None

    @property
    def accuracy(self) -> Optional[float]:
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        return round((self.true_positive + self.true_negative) / total, 4) if total else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "accuracy": self.accuracy,
            "scored_cases": self.true_positive + self.true_negative + self.false_positive + self.false_negative,
        }


def score(results: Sequence[Dict[str, Any]], *, escalation_as_flag: bool) -> ConfusionMatrix:
    """Build a confusion matrix over cases with a decisive ground truth.

    Ambiguous cases carry no fraud/legitimate label and are deliberately not
    scored here - forcing them into the matrix would be inventing a right
    answer. They are evaluated separately, on whether the system escalated them.
    """
    cm = ConfusionMatrix()
    for row in results:
        truth = row.get("ground_truth")
        if truth not in FRAUD_LABELS | LEGIT_LABELS:
            continue
        decision = row.get("decision")
        is_fraud = truth in FRAUD_LABELS

        if decision == "escalate":
            if not escalation_as_flag:
                cm.abstained += 1
                continue
            flagged = True
        elif decision == "auto_decline":
            flagged = True
        elif decision == "auto_approve":
            flagged = False
        else:  # a run that never produced a verdict is not a prediction
            cm.abstained += 1
            continue

        if is_fraud and flagged:
            cm.true_positive += 1
        elif is_fraud and not flagged:
            cm.false_negative += 1
        elif not is_fraud and flagged:
            cm.false_positive += 1
        else:
            cm.true_negative += 1
    return cm


def group_by(results: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        out.setdefault(str(row.get(key, "unknown")), []).append(row)
    return out


def scored_breakdown(results: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    """Per-group confusion matrices under both scoring views."""
    out: Dict[str, Dict[str, Any]] = {}
    for group, rows in sorted(group_by(results, key).items()):
        out[group] = {
            "cases": len(rows),
            "auto_only": score(rows, escalation_as_flag=False).to_dict(),
            "escalation_as_flag": score(rows, escalation_as_flag=True).to_dict(),
            "decisions": tally(rows, "decision"),
            "exact_verdict_matches": sum(1 for r in rows if r.get("verdict_matches_expected")),
        }
    return out


def tally(results: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in results:
        value = str(row.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


@dataclass
class EscalationAnalysis:
    """Did escalation actually track genuine ambiguity?

    A system that escalates uniformly is not exercising judgement even if its
    escalation rate looks reasonable, so this measures whether escalation
    concentrates on the cases that were *built* to be undecidable.
    """

    total_cases: int = 0
    escalated: int = 0
    auto_resolved: int = 0
    ambiguous_total: int = 0
    ambiguous_escalated: int = 0
    decisive_total: int = 0
    decisive_escalated: int = 0
    escalation_rate: Optional[float] = None
    ambiguous_escalation_rate: Optional[float] = None
    decisive_escalation_rate: Optional[float] = None
    lift: Optional[float] = None
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyse_escalation(results: Sequence[Dict[str, Any]]) -> EscalationAnalysis:
    a = EscalationAnalysis(total_cases=len(results))
    for row in results:
        escalated = row.get("decision") == "escalate"
        a.escalated += int(escalated)
        a.auto_resolved += int(row.get("decision") in {"auto_approve", "auto_decline"})
        if row.get("ground_truth") == "ambiguous":
            a.ambiguous_total += 1
            a.ambiguous_escalated += int(escalated)
        else:
            a.decisive_total += 1
            a.decisive_escalated += int(escalated)

    if a.total_cases:
        a.escalation_rate = round(a.escalated / a.total_cases, 4)
    if a.ambiguous_total:
        a.ambiguous_escalation_rate = round(a.ambiguous_escalated / a.ambiguous_total, 4)
    if a.decisive_total:
        a.decisive_escalation_rate = round(a.decisive_escalated / a.decisive_total, 4)
    if a.ambiguous_escalation_rate is not None and a.decisive_escalation_rate:
        a.lift = round(a.ambiguous_escalation_rate / a.decisive_escalation_rate, 3)

    # Interpretation, stated plainly rather than left for the reader to infer.
    if a.ambiguous_total == 0:
        a.findings.append("No ambiguous cases in the bank, so escalation targeting cannot be measured at all.")
    elif a.ambiguous_escalation_rate is not None and a.ambiguous_escalation_rate < 0.5:
        a.findings.append(
            f"Only {a.ambiguous_escalated}/{a.ambiguous_total} deliberately-ambiguous cases were escalated. "
            "The cases built to be undecidable are being auto-resolved, which is a real defect: the "
            "escalation path is not catching what it exists to catch."
        )
    else:
        a.findings.append(
            f"All {a.ambiguous_escalated}/{a.ambiguous_total} deliberately-ambiguous cases were escalated."
        )

    if a.lift is not None and a.lift < 1.2 and a.decisive_escalation_rate and a.decisive_escalation_rate > 0.4:
        a.findings.append(
            f"Escalation lift on ambiguous cases is only {a.lift}x over decisive cases, while "
            f"{a.decisive_escalation_rate:.0%} of decisive cases escalate too. Escalation is close to "
            "uniform, so correctly escalating the ambiguous cases is weak evidence of judgement - a "
            "system that escalated everything would score the same on that measure."
        )
    elif a.lift is not None:
        a.findings.append(
            f"Ambiguous cases escalate {a.lift}x as often as decisive ones "
            f"({a.ambiguous_escalation_rate:.0%} vs {a.decisive_escalation_rate:.0%})."
        )

    if a.escalation_rate is not None and a.escalation_rate > 0.4:
        a.findings.append(
            f"Overall escalation rate is {a.escalation_rate:.0%}. In production that is roughly "
            f"{a.escalated} human reviews per {a.total_cases} flagged payments, which is a staffing "
            "cost, not a free safety margin. Auto-decision quality below should be read against it."
        )
    return a


def classify_run_errors(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Separate infrastructure failures from genuine agent mistakes.

    Conflating "the API rate-limited us" with "the agent reached the wrong
    verdict" turns a report into a demo output. They are counted apart here and
    never summed into a generic "failed" bucket.
    """
    from sentrymesh.agents.llm_client import INFRASTRUCTURE_ERRORS

    by_class: Dict[str, int] = {}
    affected_cases: Dict[str, List[str]] = {}
    for row in results:
        for err in row.get("run_errors", []):
            cls = err.get("error_class", "unknown_error")
            by_class[cls] = by_class.get(cls, 0) + 1
            affected_cases.setdefault(cls, []).append(row.get("case_id", "?"))

    infra = {k: v for k, v in by_class.items() if k in INFRASTRUCTURE_ERRORS}
    agent = {k: v for k, v in by_class.items() if k not in INFRASTRUCTURE_ERRORS}

    wrong = [
        r
        for r in results
        if r.get("ground_truth") in FRAUD_LABELS | LEGIT_LABELS
        and r.get("decision") in {"auto_approve", "auto_decline"}
        and not r.get("prediction_correct")
    ]
    missed = [r for r in results if r.get("decision") == "escalate" and r.get("ground_truth") != "ambiguous"]

    return {
        "infrastructure_failures": {
            "total": sum(infra.values()),
            "by_class": infra,
            "affected_cases": {k: sorted(set(v)) for k, v in affected_cases.items() if k in infra},
            "meaning": "Transport or provider failures. These say nothing about agent reasoning quality "
            "and any metric computed over a run containing them is measured on degraded evidence.",
        },
        "agent_protocol_failures": {
            "total": sum(agent.values()),
            "by_class": agent,
            "affected_cases": {k: sorted(set(v)) for k, v in affected_cases.items() if k not in infra},
            "meaning": "The agent ran but did not produce a usable structured result (bad tool call, "
            "turn limit, tool execution error). A reasoning-adjacent failure, distinct from both "
            "infrastructure and from being wrong.",
        },
        "wrong_auto_decisions": {
            "total": len(wrong),
            "cases": [
                {
                    "case_id": r["case_id"],
                    "ground_truth": r["ground_truth"],
                    "decision": r["decision"],
                    "combined_risk": r.get("combined_risk"),
                    "confidence": r.get("confidence"),
                }
                for r in wrong
            ],
            "meaning": "The system ran correctly and committed to the wrong answer. This is the only "
            "bucket that measures reasoning quality, and it is the one that matters.",
        },
        "escalated_decisive_cases": {
            "total": len(missed),
            "cases": [r["case_id"] for r in missed],
            "meaning": "Cases with a clear ground truth that the system would not commit to. Not wrong, "
            "but not autonomous either - each one is a human review the system failed to save.",
        },
    }
