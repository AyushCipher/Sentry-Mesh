"""Specialist 1 - transaction pattern analysis.

Signal category: velocity, amount anomaly relative to *this account's own*
history, merchant-category drift and time-of-day pattern.  Deliberately blind
to device, network and precedent signals - keeping the specialists genuinely
independent is what makes their disagreement informative to the supervisor.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from sentrymesh.agents.base import SpecialistAgent
from sentrymesh.agents.llm_client import ToolCallRecord, tool
from sentrymesh.models import Case

# Categories a fraud team would treat as elevated-risk for card-not-present
# abuse.  MOCK - in production this comes from the scheme's MCC risk table,
# refreshed per-issuer, not from a constant in the repo.
ELEVATED_RISK_CATEGORIES = {
    "digital_goods": 0.7,
    "crypto_exchange": 0.9,
    "gift_cards": 0.85,
    "prepaid_reload": 0.8,
    "money_transfer": 0.85,
    "online_gambling": 0.7,
    "electronics": 0.45,
}


def parse_ts(value: Any) -> datetime:
    text = str(value or "").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class TransactionPatternAgent(SpecialistAgent):
    name = "transaction_pattern"
    focus = (
        "transaction velocity, amount anomalies against this account's own baseline, "
        "merchant-category drift and time-of-day patterns"
    )

    @property
    def system_prompt(self) -> str:
        return (
            "You are a transaction-pattern fraud analyst on a payments risk team. You examine one "
            "signal category only: how this transaction compares to the account's own transactional "
            "behaviour.\n\n"
            "How to weigh what you find:\n"
            "- A high amount is only suspicious relative to the account's own baseline. A 900 USD "
            "charge on an account that routinely spends 800 is not an anomaly.\n"
            "- Rapid low-value attempts against digital-goods or similar merchants are the classic "
            "card-testing shape, especially with declines interleaved.\n"
            "- A thin history means you cannot compute a meaningful baseline. Say so and lower your "
            "confidence rather than inventing one.\n"
            "- Off-hours activity is weak evidence on its own and strong evidence combined with a "
            "velocity spike.\n\n"
            "You do not see device, network or historical-precedent signals; other specialists own "
            "those. Do not speculate about them. Score only what your tools show you."
        )

    @property
    def investigation_tools(self) -> Sequence[Dict[str, Any]]:
        return [
            tool(
                "get_transaction_history",
                "Retrieve the account's recent transaction history and the transaction under "
                "investigation, plus account age.",
                {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum historical transactions to return (default 50).",
                    }
                },
            ),
            tool(
                "compute_velocity_stats",
                "Compute transaction counts in a trailing window, the amount z-score of the flagged "
                "transaction against the account's history, decline rate and distinct-merchant count.",
                {
                    "window_minutes": {
                        "type": "integer",
                        "description": "Trailing window in minutes, ending at the flagged transaction (default 60).",
                    }
                },
            ),
            tool(
                "flag_merchant_category",
                "Check whether a merchant category is elevated-risk and whether this account has "
                "transacted in it before.",
                {
                    "merchant_category": {
                        "type": "string",
                        "description": "Category to check. Defaults to the flagged transaction's category.",
                    }
                },
                required=[],
            ),
        ]

    # -- tool implementations ----------------------------------------------
    async def run_tool(self, case: Case, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_transaction_history":
            return self._history(case, int(args.get("limit") or 50))
        if name == "compute_velocity_stats":
            return self._velocity(case, int(args.get("window_minutes") or 60))
        if name == "flag_merchant_category":
            return self._merchant(case, args.get("merchant_category"))
        raise ValueError(f"{self.name}: unknown tool {name!r}")

    def _history(self, case: Case, limit: int) -> Dict[str, Any]:
        td = case.transaction_data
        history = sorted(td.get("history", []), key=lambda t: parse_ts(t.get("timestamp")), reverse=True)
        return {
            "account_id": td.get("account_id"),
            "account_age_days": td.get("account_age_days"),
            "flagged_transaction": td.get("transaction", {}),
            "history_count": len(history),
            "history": history[:limit],
            "note": "Synthetic evaluation data.",
        }

    def _velocity(self, case: Case, window_minutes: int) -> Dict[str, Any]:
        td = case.transaction_data
        txn = td.get("transaction", {})
        history: List[Dict[str, Any]] = list(td.get("history", []))
        now = parse_ts(txn.get("timestamp"))
        window_start = now.timestamp() - window_minutes * 60

        in_window = [h for h in history if parse_ts(h.get("timestamp")).timestamp() >= window_start]
        in_24h = [h for h in history if parse_ts(h.get("timestamp")).timestamp() >= now.timestamp() - 86400]
        amounts = [float(h.get("amount", 0.0)) for h in history]

        mean = statistics.fmean(amounts) if amounts else 0.0
        stdev = statistics.pstdev(amounts) if len(amounts) > 1 else 0.0
        amount = float(txn.get("amount", 0.0))
        if stdev > 0.01:
            z = (amount - mean) / stdev
        elif mean > 0:
            z = (amount - mean) / max(mean, 1.0)
        else:
            z = 0.0

        declines = [h for h in in_24h if str(h.get("status", "")).lower() in {"declined", "failed"}]
        hours = [parse_ts(h.get("timestamp")).hour for h in history]
        hour_mode = statistics.mode(hours) if hours else None

        return {
            "window_minutes": window_minutes,
            "transactions_in_window": len(in_window) + 1,
            "transactions_in_24h": len(in_24h) + 1,
            "distinct_merchants_in_window": len({h.get("merchant") for h in in_window if h.get("merchant")}),
            "declines_in_24h": len(declines),
            "decline_rate_24h": round(len(declines) / len(in_24h), 3) if in_24h else 0.0,
            "flagged_amount": amount,
            "history_amount_mean": round(mean, 2),
            "history_amount_stdev": round(stdev, 2),
            "amount_z_score": round(z, 2),
            "flagged_hour_utc": now.hour,
            "typical_hour_utc": hour_mode,
            "baseline_sample_size": len(amounts),
            "baseline_reliable": len(amounts) >= 8,
        }

    def _merchant(self, case: Case, category: Any) -> Dict[str, Any]:
        td = case.transaction_data
        txn = td.get("transaction", {})
        category = str(category or txn.get("merchant_category") or "unknown")
        seen = [h.get("merchant_category") for h in td.get("history", [])]
        return {
            "merchant_category": category,
            "elevated_risk": category in ELEVATED_RISK_CATEGORIES,
            "category_risk_weight": ELEVATED_RISK_CATEGORIES.get(category, 0.1),
            "account_has_used_category_before": category in seen,
            "times_used_by_account": seen.count(category),
            "account_top_categories": sorted({c for c in seen if c})[:8],
            "channel": txn.get("channel"),
            "source": "MOCK - replace with the card scheme's MCC risk table in production",
        }

    # -- offline fallback ---------------------------------------------------
    def offline_plan(self, case: Case) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            ("get_transaction_history", {"limit": 50}),
            ("compute_velocity_stats", {"window_minutes": 60}),
            ("flag_merchant_category", {}),
        ]

    def offline_analyze(self, case: Case, records: List[ToolCallRecord]) -> Dict[str, Any]:
        r = self.results_by_tool(records)
        vel = r.get("compute_velocity_stats", {})
        mer = r.get("flag_merchant_category", {})

        risk = 0.0
        evidence: List[str] = []

        count = int(vel.get("transactions_in_window", 1))
        if count >= 10:
            risk += 34
            evidence.append(f"{count} transactions in a {vel.get('window_minutes')}-minute window")
        elif count >= 5:
            risk += 20
            evidence.append(f"{count} transactions in a {vel.get('window_minutes')}-minute window")
        elif count >= 3:
            risk += 8

        z = float(vel.get("amount_z_score", 0.0))
        if abs(z) >= 3:
            risk += 26
            evidence.append(f"amount z-score {z} against the account's own baseline")
        elif abs(z) >= 2:
            risk += 15
            evidence.append(f"amount z-score {z} against the account's own baseline")

        decline_rate = float(vel.get("decline_rate_24h", 0.0))
        if decline_rate >= 0.4:
            risk += 22
            evidence.append(f"{vel.get('declines_in_24h')} declines in 24h (rate {decline_rate})")
        elif decline_rate >= 0.2:
            risk += 11

        if mer.get("elevated_risk") and not mer.get("account_has_used_category_before"):
            risk += 18
            evidence.append(f"first-ever spend in elevated-risk category {mer.get('merchant_category')}")
        elif mer.get("elevated_risk"):
            risk += 7

        flagged_hour = vel.get("flagged_hour_utc")
        typical_hour = vel.get("typical_hour_utc")
        if isinstance(flagged_hour, int) and isinstance(typical_hour, int):
            gap = min(abs(flagged_hour - typical_hour), 24 - abs(flagged_hour - typical_hour))
            if gap >= 7 and count >= 3:
                risk += 8
                evidence.append(f"activity at {flagged_hour:02d}:00 UTC vs typical {typical_hour:02d}:00 UTC")

        confidence = 0.82 if vel.get("baseline_reliable") else 0.45
        if not evidence:
            evidence.append("velocity, amount, category and timing all within the account's normal range")

        return {
            "risk_score": max(0.0, min(100.0, risk)),
            "confidence": confidence,
            "reasoning_summary": (
                "Offline deterministic scorer (no LLM): combined velocity, amount z-score, decline "
                f"rate and merchant-category drift into a risk of {min(100.0, risk):.0f}. Baseline "
                f"sample size {vel.get('baseline_sample_size')} "
                f"({'reliable' if vel.get('baseline_reliable') else 'too thin for a strong baseline'})."
            ),
            "key_evidence": evidence,
        }
