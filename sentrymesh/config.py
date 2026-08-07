"""Central configuration for SentryMesh.

Every tunable lives here so that the decision thresholds - the part of the
system a reviewer is most likely to interrogate - are in one auditable place
rather than scattered through the agents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time so tests never depend on a .env
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv missing is not an error
    pass


REPO_ROOT = Path(__file__).resolve().parent.parent
CASE_BANK_DIR = REPO_ROOT / "eval" / "case_bank"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
PRECEDENT_CORPUS = REPO_ROOT / "eval" / "precedent_cases.json"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DecisionThresholds:
    """Supervisor decision boundaries.

    These are deliberately plain numbers rather than an LLM free-text judgement:
    the LLM reasons about *why* specialists disagree, but whether a case is
    auto-resolved or escalated is decided by code that a risk team could audit,
    unit-test and tune without re-prompting a model.
    """

    # Combined risk at or above this, with sufficient confidence -> auto-decline.
    auto_decline_risk: float = 70.0
    # Combined risk at or below this, with sufficient confidence -> auto-approve.
    auto_approve_risk: float = 35.0
    # Below this aggregate confidence the case always goes to a human.
    min_confidence: float = 0.70

    # --- disagreement detection -------------------------------------------
    # Largest pairwise gap between any two *contributing* specialist scores.
    max_pairwise_spread: float = 40.0
    # Population standard deviation across the contributing risk scores.
    max_risk_stdev: float = 22.0
    # A specialist reporting confidence below this has ABSTAINED: it is telling
    # the supervisor its category carries no information on this case, which is
    # a different thing from contradicting the panel. Abstentions are excluded
    # from the spread/stdev statistics and from the weighted aggregate, and cost
    # a small amount of panel confidence. Conflating abstention with dissent was
    # the single biggest source of spurious escalation in early runs - a
    # retrieval agent that found no precedent would drag every case to a human.
    abstention_confidence: float = 0.40
    # A "split verdict": specialists on both sides of this line, each by at
    # least `split_margin`, means the panel does not agree on the direction.
    split_midpoint: float = 50.0
    split_margin: float = 15.0

    @classmethod
    def from_env(cls) -> "DecisionThresholds":
        return cls(
            auto_decline_risk=_env_float("SENTRYMESH_AUTO_DECLINE_RISK", 70.0),
            auto_approve_risk=_env_float("SENTRYMESH_AUTO_APPROVE_RISK", 35.0),
            min_confidence=_env_float("SENTRYMESH_MIN_CONFIDENCE", 0.70),
        )


@dataclass(frozen=True)
class Settings:
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    require_redis: bool = field(default_factory=lambda: _env_bool("SENTRYMESH_REQUIRE_REDIS"))

    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    require_pgvector: bool = field(default_factory=lambda: _env_bool("SENTRYMESH_REQUIRE_PGVECTOR"))

    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip())
    model: str = field(default_factory=lambda: os.getenv("SENTRYMESH_MODEL", "claude-sonnet-4-5"))
    max_tokens: int = field(default_factory=lambda: _env_int("SENTRYMESH_MAX_TOKENS", 2048))
    llm_backend: str = field(default_factory=lambda: os.getenv("SENTRYMESH_LLM_BACKEND", "auto").strip().lower())

    # Hard cap on tool-use turns per agent, so a confused model cannot loop.
    max_agent_turns: int = 8

    thresholds: DecisionThresholds = field(default_factory=DecisionThresholds.from_env)

    def resolved_backend(self) -> str:
        """Return the LLM backend that will actually be used: 'anthropic' | 'offline'."""
        if self.llm_backend == "anthropic":
            return "anthropic"
        if self.llm_backend == "offline":
            return "offline"
        return "anthropic" if self.anthropic_api_key else "offline"


SETTINGS = Settings()

# Redis key namespace.  Kept in one place so the dashboard, harness and agents
# can never drift apart on where the shared state lives.
STREAM_KEY = "sentrymesh:events"
CASE_INDEX_KEY = "sentrymesh:case:{case_id}:events"
CASE_STATE_KEY = "sentrymesh:case:{case_id}:state"
REVIEW_QUEUE_KEY = "sentrymesh:review_queue"
REVIEW_META_KEY = "sentrymesh:review_queue:meta"

SPECIALIST_NAMES = (
    "transaction_pattern",
    "identity_signal",
    "network_analysis",
    "historical_case",
)
