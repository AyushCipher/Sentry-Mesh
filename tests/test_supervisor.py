"""Supervisor: disagreement detection, aggregation and escalation thresholds.

The LLM is mocked throughout. The eval harness is the end-to-end test; what is
worth unit-testing here is the pure decision logic, because that is the part a
risk team would need to defend line by line.
"""

import pytest

from sentrymesh.agents.supervisor_agent import (
    ELEVATED_RISK_FLOOR,
    NEUTRAL_EVIDENCE,
    aggregate_scores,
    decide,
    detect_disagreement,
    partition,
)
from sentrymesh.config import DecisionThresholds
from sentrymesh.models import Finding

TH = DecisionThresholds()


def f(name, risk, confidence=0.8, **kwargs):
    return Finding(specialist=name, risk_score=risk, confidence=confidence, reasoning_summary="", **kwargs)


def panel(tx=10.0, ident=10.0, net=10.0, hist=10.0, confidence=0.8):
    return [
        f("transaction_pattern", tx, confidence),
        f("identity_signal", ident, confidence),
        f("network_analysis", net, confidence),
        f("historical_case", hist, confidence),
    ]


# ---------------------------------------------------------------- partition
def test_low_confidence_finding_is_an_abstention_not_a_failure():
    findings = panel()
    findings[3] = f("historical_case", 45.0, 0.25)
    contributing, abstained, failed = partition(findings, TH)
    assert [x.specialist for x in abstained] == ["historical_case"]
    assert len(contributing) == 3
    assert failed == []


def test_failed_specialist_is_not_an_abstention():
    findings = panel()
    findings[2] = Finding.failed("network_analysis", "boom", "llm_timeout")
    contributing, abstained, failed = partition(findings, TH)
    assert [x.specialist for x in failed] == ["network_analysis"]
    assert abstained == []
    assert len(contributing) == 3


# ---------------------------------------------------- disagreement detection
def test_orthogonal_spread_alone_is_not_disagreement():
    """Identity at 80 with network at 10 is a takeover with no ring - a coherent
    story, not a contradiction. A pure spread rule would escalate it."""
    report = detect_disagreement(panel(tx=60, ident=80, net=10, hist=58), TH)
    assert report.pairwise_spread == 70.0  # the diagnostic is still reported
    assert not report.significant, report.reasons


def test_single_elevated_specialist_is_uncorroborated_and_escalates():
    report = detect_disagreement(panel(tx=10, ident=85, net=5, hist=8), TH)
    assert report.significant
    assert any("uncorroborated" in r for r in report.reasons)
    assert "identity_signal" in report.dissenting_specialists


def test_two_elevated_specialists_corroborate_each_other():
    report = detect_disagreement(panel(tx=10, ident=85, net=5, hist=72), TH)
    assert not report.significant, report.reasons


def test_precedent_contradicting_live_evidence_is_disagreement():
    report = detect_disagreement(panel(tx=8, ident=6, net=5, hist=90), TH)
    assert report.significant
    assert any("precedent contradicts" in r for r in report.reasons)


def test_abstention_never_makes_disagreement_significant():
    findings = panel(tx=12, ident=9, net=7)
    findings[3] = f("historical_case", 95.0, 0.1)  # loud but unconfident
    report = detect_disagreement(findings, TH)
    assert report.abstaining_specialists == ["historical_case"]
    assert not report.significant, report.reasons


def test_failed_specialist_always_counts_as_missing_evidence():
    findings = panel(tx=12, ident=9, net=7, hist=11)
    findings[1] = Finding.failed("identity_signal", "rate limited", "llm_rate_limit")
    report = detect_disagreement(findings, TH)
    assert report.significant
    assert "identity_signal" in report.failed_specialists


def test_fewer_than_two_contributors_is_significant():
    findings = [f("transaction_pattern", 40.0, 0.9), f("identity_signal", 30.0, 0.1)]
    report = detect_disagreement(findings, TH)
    assert report.significant


# ------------------------------------------------------------- aggregation
def test_quiet_specialists_cannot_cancel_one_conclusive_signal():
    """The defect this replaced: a mean let three zeros bury a confident 95."""
    risk, _ = aggregate_scores(panel(tx=0, ident=95, net=0, hist=0), TH)
    assert risk >= 70.0, f"one conclusive signal must survive aggregation, got {risk}"


def test_evidence_accumulates_across_specialists():
    one = aggregate_scores(panel(tx=60, ident=0, net=0, hist=0), TH)[0]
    two = aggregate_scores(panel(tx=60, ident=60, net=0, hist=0), TH)[0]
    assert two > one, "two independent signals must outweigh one"


def test_scores_below_the_neutral_floor_contribute_nothing():
    risk, _ = aggregate_scores(panel(tx=NEUTRAL_EVIDENCE, ident=NEUTRAL_EVIDENCE, net=0, hist=0), TH)
    assert risk == 0.0


def test_clean_panel_scores_zero():
    assert aggregate_scores(panel(0, 0, 0, 0), TH)[0] == 0.0


def test_abstentions_cost_less_confidence_than_failures():
    with_abstention = panel()
    with_abstention[3] = f("historical_case", 45.0, 0.2)
    with_failure = panel()
    with_failure[3] = Finding.failed("historical_case", "boom", "llm_timeout")
    _, conf_abstain = aggregate_scores(with_abstention, TH)
    _, conf_fail = aggregate_scores(with_failure, TH)
    assert conf_abstain > conf_fail


def test_empty_panel_is_zero_risk_and_zero_confidence():
    assert aggregate_scores([], TH) == (0.0, 0.0)


# ----------------------------------------------------------------- decide
def test_high_risk_high_confidence_auto_declines():
    findings = panel(tx=75, ident=80, net=72, hist=70)
    decision, reason = decide(88.0, 0.85, detect_disagreement(findings, TH), findings, TH)
    assert decision == "auto_decline"
    assert reason is None


def test_low_risk_high_confidence_auto_approves():
    findings = panel(5, 5, 5, 5)
    decision, reason = decide(8.0, 0.85, detect_disagreement(findings, TH), findings, TH)
    assert decision == "auto_approve"
    assert reason is None


def test_low_confidence_always_escalates_however_extreme_the_risk():
    findings = panel(90, 88, 91, 89)
    decision, reason = decide(95.0, 0.4, detect_disagreement(findings, TH), findings, TH)
    assert decision == "escalate"
    assert "confidence" in (reason or "")


def test_significant_disagreement_escalates_even_at_high_confidence():
    findings = panel(tx=5, ident=90, net=5, hist=5)  # uncorroborated
    disagreement = detect_disagreement(findings, TH)
    decision, reason = decide(80.0, 0.95, disagreement, findings, TH)
    assert decision == "escalate"
    assert "disagreement" in (reason or "")


def test_indeterminate_risk_band_escalates():
    findings = panel(50, 48, 52, 49)
    decision, reason = decide(50.0, 0.9, detect_disagreement(findings, TH), findings, TH)
    assert decision == "escalate"
    assert "indeterminate band" in (reason or "")


def test_two_failed_specialists_escalate_on_incomplete_evidence():
    findings = panel(80, 82, 0, 0)
    findings[2] = Finding.failed("network_analysis", "timeout", "llm_timeout")
    findings[3] = Finding.failed("historical_case", "timeout", "llm_timeout")
    decision, reason = decide(85.0, 0.9, detect_disagreement(findings, TH), findings, TH)
    assert decision == "escalate"
    assert "failed to run" in (reason or "")


def test_too_few_contributors_escalates():
    findings = [f("transaction_pattern", 80.0, 0.9)] + [
        f(n, 40.0, 0.1) for n in ("identity_signal", "network_analysis", "historical_case")
    ]
    decision, reason = decide(80.0, 0.9, detect_disagreement(findings, TH), findings, TH)
    assert decision == "escalate"


@pytest.mark.parametrize(
    "risk,expected",
    [(TH.auto_decline_risk, "auto_decline"), (TH.auto_approve_risk, "auto_approve")],
)
def test_thresholds_are_inclusive_at_the_boundary(risk, expected):
    findings = panel(60, 60, 60, 60) if expected == "auto_decline" else panel(5, 5, 5, 5)
    decision, _ = decide(risk, 0.9, detect_disagreement(findings, TH), findings, TH)
    assert decision == expected


def test_elevated_floor_constant_sits_inside_the_indeterminate_band():
    """A specialist can be 'elevated' without the case being auto-declinable;
    that gap is what the uncorroborated-signal rule exists to catch."""
    assert TH.auto_approve_risk < ELEVATED_RISK_FLOOR < TH.auto_decline_risk
