"""Shared data shapes for cases, specialist findings and supervisor verdicts.

Everything modelled here is SYNTHETIC evaluation data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

VERDICTS = ("auto_approve", "auto_decline", "escalate")
GROUND_TRUTHS = ("fraud", "legitimate", "ambiguous")
DIFFICULTIES = ("easy", "medium", "hard")
CATEGORIES = (
    "card_testing",
    "account_takeover",
    "synthetic_identity",
    "money_muling",
    "legitimate",
)


@dataclass
class Case:
    """One flagged case under investigation."""

    case_id: str
    category: str = "unknown"
    summary: str = ""
    transaction_data: Dict[str, Any] = field(default_factory=dict)
    device_data: Dict[str, Any] = field(default_factory=dict)
    network_data: Dict[str, Any] = field(default_factory=dict)
    # Evaluation-only fields; never shown to any agent.
    ground_truth: Optional[str] = None
    expected_verdict: Optional[str] = None
    difficulty: Optional[str] = None
    synthetic: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Case":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def agent_view(self) -> Dict[str, Any]:
        """The case as an agent sees it - ground truth is stripped.

        Leaking ``ground_truth`` or ``expected_verdict`` into a prompt would
        make every eval number meaningless, so the redaction happens here, once,
        rather than being re-remembered at each of the five agent call sites.
        """
        return {
            "case_id": self.case_id,
            "summary": self.summary,
            "transaction_data": self.transaction_data,
            "device_data": self.device_data,
            "network_data": self.network_data,
            "data_disclaimer": "All values are synthetic evaluation data.",
        }

    @property
    def account_id(self) -> str:
        return str(
            self.transaction_data.get("account_id")
            or self.network_data.get("focus_account")
            or "unknown-account"
        )


@dataclass
class Finding:
    """A single specialist's structured output."""

    specialist: str
    risk_score: float
    confidence: float
    reasoning_summary: str
    key_evidence: List[str] = field(default_factory=list)
    tool_calls: int = 0
    duration_ms: int = 0
    backend: str = ""
    error: Optional[str] = None
    error_class: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return self.error is None

    @staticmethod
    def failed(specialist: str, error: str, error_class: str, backend: str = "") -> "Finding":
        """A specialist that could not run.

        It reports zero confidence rather than a neutral risk score, so the
        supervisor treats a broken specialist as missing evidence instead of as
        evidence of innocence.
        """
        return Finding(
            specialist=specialist,
            risk_score=0.0,
            confidence=0.0,
            reasoning_summary=f"specialist failed to produce a finding: {error}",
            key_evidence=[],
            backend=backend,
            error=error,
            error_class=error_class,
        )


@dataclass
class DisagreementReport:
    """Deterministic, testable measurement of how much the panel disagrees."""

    pairwise_spread: float
    risk_stdev: float
    lowest_specialist_confidence: float
    split_verdict: bool
    dissenting_specialists: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    significant: bool = False
    # Specialists that reported but declared their category uninformative.
    # Tracked separately from dissent: an abstention is a specialist saying
    # "I have nothing", not "I disagree".
    abstaining_specialists: List[str] = field(default_factory=list)
    failed_specialists: List[str] = field(default_factory=list)
    contributing_specialists: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """The supervisor's decision for a case."""

    case_id: str
    decision: str  # auto_approve | auto_decline | escalate
    combined_risk: float
    confidence: float
    rationale: str
    disagreement: DisagreementReport
    findings: List[Finding] = field(default_factory=list)
    escalation_reason: Optional[str] = None
    supervisor_backend: str = ""
    duration_ms: int = 0
    run_errors: List[Dict[str, str]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HumanDecision:
    case_id: str
    reviewer: str
    action: str  # approve | decline | request_detail
    notes: str = ""
    requested_specialist: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
