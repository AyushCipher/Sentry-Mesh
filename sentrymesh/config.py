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


def default_require_redis() -> bool:
    """Fail closed: Redis is required unless explicitly relaxed.

    ``SENTRYMESH_REQUIRE_REDIS`` wins when set. Otherwise the default is
    required, and only the dev flag ``SENTRYMESH_DEV=1`` allows the silent
    in-memory fallback (a downgrade that loses the audit log on restart).
    """
    if os.getenv("SENTRYMESH_REQUIRE_REDIS", "").strip():
        return _env_bool("SENTRYMESH_REQUIRE_REDIS")
    return not _env_bool("SENTRYMESH_DEV")


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
    # NOTE: raw pairwise spread / risk stdev are computed and reported as trace
    # diagnostics but deliberately have NO threshold: the specialists score
    # orthogonal categories, and a spread rule escalated 21 of 23 seeded cases
    # (see `detect_disagreement`). The former `max_pairwise_spread` and
    # `max_risk_stdev` settings were dead config and have been removed.
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

    # --- bounding of the supervisor LLM's numbers ---------------------------
    # The model's combined_risk is never allowed below
    # max(deterministic fallback risk, k * loudest confident specialist risk).
    risk_floor_factor: float = 0.9
    # A specialist counts as "confident" for the floor above at this confidence.
    floor_confidence: float = 0.70
    # If the model's risk differs from the deterministic fallback by more than
    # this many points, the model is not trusted and the case is escalated.
    max_model_risk_divergence: float = 30.0

    @classmethod
    def from_env(cls) -> "DecisionThresholds":
        return cls(
            auto_decline_risk=_env_float("SENTRYMESH_AUTO_DECLINE_RISK", 70.0),
            auto_approve_risk=_env_float("SENTRYMESH_AUTO_APPROVE_RISK", 35.0),
            min_confidence=_env_float("SENTRYMESH_MIN_CONFIDENCE", 0.70),
            risk_floor_factor=_env_float("SENTRYMESH_RISK_FLOOR_FACTOR", 0.9),
            max_model_risk_divergence=_env_float("SENTRYMESH_MAX_MODEL_RISK_DIVERGENCE", 30.0),
        )


@dataclass(frozen=True)
class Settings:
    # 127.0.0.1, not "localhost", and deliberately so: on Windows and on many
    # dual-stack Linux boxes "localhost" resolves to ::1 first, and a Redis
    # listening only on IPv4 makes the asyncio client burn its entire connect
    # timeout on the IPv6 attempt before falling back. That reads as "Redis is
    # down" and silently downgrades the event log to in-memory.
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
    require_redis: bool = field(default_factory=default_require_redis)

    # Optional cap on the shared stream length (XADD MAXLEN ~ n). 0 = unbounded.
    stream_maxlen: int = field(default_factory=lambda: max(0, _env_int("SENTRYMESH_STREAM_MAXLEN", 0)))

    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    require_pgvector: bool = field(default_factory=lambda: _env_bool("SENTRYMESH_REQUIRE_PGVECTOR"))

    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip())
    model: str = field(default_factory=lambda: os.getenv("SENTRYMESH_MODEL", "claude-sonnet-5"))
    max_tokens: int = field(default_factory=lambda: _env_int("SENTRYMESH_MAX_TOKENS", 2048))
    llm_backend: str = field(default_factory=lambda: os.getenv("SENTRYMESH_LLM_BACKEND", "auto").strip().lower())

    # OpenAI-compatible providers (Groq, Gemini's compat endpoint, GitHub Models,
    # Ollama, OpenRouter, Together...). "OpenAI-compatible" is a request format,
    # not the company: setting these does not involve OpenAI and needs no OpenAI
    # key. One adapter, and the provider is an environment variable.
    llm_base_url: str = field(default_factory=lambda: os.getenv("SENTRYMESH_LLM_BASE_URL", "").strip())
    llm_api_key: str = field(default_factory=lambda: os.getenv("SENTRYMESH_LLM_API_KEY", "").strip())

    # Hard cap on tool-use turns per agent, so a confused model cannot loop.
    max_agent_turns: int = 8

    # How many specialists may be in flight at once. 4 = full parallelism, the
    # default and the point of the fan-out. Free-tier providers meter tokens per
    # minute, and four concurrent specialists can exceed that budget in a single
    # burst - setting this to 1 or 2 trades wall-clock for staying under the
    # limit without changing any agent behaviour.
    max_concurrent_specialists: int = field(
        default_factory=lambda: max(1, _env_int("SENTRYMESH_MAX_CONCURRENT_SPECIALISTS", 4))
    )

    # Extra retries for provider rate limits, on top of the SDK's own. Free
    # tiers reset per minute, so the backoff below is seconds-scale rather than
    # the sub-second the SDK defaults to.
    rate_limit_retries: int = field(default_factory=lambda: _env_int("SENTRYMESH_RATE_LIMIT_RETRIES", 3))

    # Adaptive dynamic roster: when True, supervisor runs an initial fast triage
    # to dispatch only necessary specialists instead of fanning out to all 4 unconditionally.
    adaptive_roster: bool = field(default_factory=lambda: _env_bool("SENTRYMESH_ADAPTIVE_ROSTER", False))

    # Dense semantic embeddings: when True, uses sentence-transformers / dense embeddings if available.
    dense_embeddings: bool = field(default_factory=lambda: _env_bool("SENTRYMESH_DENSE_EMBEDDINGS", False))

    thresholds: DecisionThresholds = field(default_factory=DecisionThresholds.from_env)

    def resolved_backend(self) -> str:
        """Which LLM backend will actually run: 'anthropic' | 'openai_compat' | 'offline'.

        On ``auto``, an explicitly configured OpenAI-compatible base URL wins over
        an Anthropic key: setting one is a deliberate act, whereas an
        ANTHROPIC_API_KEY often lingers in a shell from unrelated work, and
        silently billing that key because it happened to be exported would be a
        genuinely bad surprise.
        """
        if self.llm_backend in {"anthropic", "openai_compat", "offline"}:
            return self.llm_backend
        if self.llm_base_url:
            return "openai_compat"
        return "anthropic" if self.anthropic_api_key else "offline"

    def provider_label(self) -> str:
        """Human-readable provider for reports and /health.

        The host, never the key. Every number in the harness report is a number
        about a specific provider and model, so the report has to say which.
        """
        backend = self.resolved_backend()
        if backend != "openai_compat":
            return backend
        try:
            from urllib.parse import urlparse

            host = urlparse(self.llm_base_url).hostname or self.llm_base_url
        except Exception:  # noqa: BLE001
            host = self.llm_base_url
        return f"openai_compat:{host}"


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

MODEL_COST_PER_MILLION_TOKENS = {
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "default": {"input": 1.00, "output": 3.00},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost based on token counts and model pricing."""
    rates = MODEL_COST_PER_MILLION_TOKENS.get(
        model,
        MODEL_COST_PER_MILLION_TOKENS["default"],
    )
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0
    return round(cost, 6)

