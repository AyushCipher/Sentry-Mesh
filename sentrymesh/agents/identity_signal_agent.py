"""Specialist 2 - identity and device signals.

Signal category: device-fingerprint consistency, geolocation vs. billing
address, and KYC flag status.  Blind to transaction amounts, ring structure and
precedent.
"""

from __future__ import annotations

from datetime import timezone
from typing import Any, Dict, List, Sequence, Tuple

from sentrymesh.agents.base import SpecialistAgent
from sentrymesh.agents.llm_client import ToolCallRecord, tool
from sentrymesh.agents.transaction_pattern_agent import parse_ts
from sentrymesh.models import Case

# MOCK - a production system reads these from a device-intelligence vendor
# (fingerprint reuse graph, emulator/bot detection) and a KYC provider.  The
# synthetic case bank carries the same fields those vendors would return.
KYC_RISK = {
    "verified": 0.0,
    "verified_enhanced": 0.0,
    "pending": 0.35,
    "expired": 0.5,
    "unverified": 0.7,
    "failed": 0.95,
}


class IdentitySignalAgent(SpecialistAgent):
    name = "identity_signal"
    focus = "device fingerprint consistency, geolocation vs billing address, and KYC status"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an identity and device-intelligence fraud analyst. You examine one signal "
            "category only: whether the person behind this transaction is plausibly the account "
            "holder, judged from device and identity evidence.\n\n"
            "How to weigh what you find:\n"
            "- A brand-new device on a long-lived account is routine; people buy phones. A new "
            "device combined with a fingerprint that keeps mutating is not.\n"
            "- Geographic distance alone is weak: travel exists, and so do VPNs used by ordinary "
            "privacy-conscious customers. Distance plus a proxy/hosting ASN plus a country the "
            "account has never touched is a different matter.\n"
            "- A failed KYC check is strong evidence. A pending one usually just means paperwork.\n"
            "- One device shared across many unrelated accounts is a strong signal, but the ring "
            "structure itself belongs to the network specialist - report the device fact, do not "
            "try to reconstruct the ring.\n\n"
            "Be specific about which of these you actually observed. If the identity evidence is "
            "thin or points both ways, report lower confidence rather than splitting the difference "
            "on the risk score."
        )

    @property
    def investigation_tools(self) -> Sequence[Dict[str, Any]]:
        return [
            tool(
                "get_device_history",
                "Retrieve the device fingerprint for this session, how long it has been seen, how "
                "many accounts have used it, and recent sessions on it.",
                {
                    "include_sessions": {
                        "type": "boolean",
                        "description": "Include the per-session history list (default true).",
                    }
                },
            ),
            tool(
                "check_geo_mismatch",
                "Compare the session's IP geolocation against the account's billing address, "
                "including proxy/hosting-ASN detection and whether the country is new for this account.",
                {},
            ),
            tool(
                "get_kyc_status",
                "Retrieve the account's KYC verification status and any outstanding KYC flags.",
                {},
            ),
        ]

    # -- tool implementations ----------------------------------------------
    async def run_tool(self, case: Case, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_device_history":
            return self._device(case, bool(args.get("include_sessions", True)))
        if name == "check_geo_mismatch":
            return self._geo(case)
        if name == "get_kyc_status":
            return self._kyc(case)
        raise ValueError(f"{self.name}: unknown tool {name!r}")

    def _device(self, case: Case, include_sessions: bool) -> Dict[str, Any]:
        dd = case.device_data
        txn_ts = parse_ts(case.transaction_data.get("transaction", {}).get("timestamp"))
        first_seen = dd.get("device_first_seen")
        age_days = None
        if first_seen:
            age_days = max(0, int((txn_ts - parse_ts(first_seen).astimezone(timezone.utc)).total_seconds() // 86400))
        out = {
            "device_id": dd.get("device_id"),
            "device_first_seen": first_seen,
            "device_age_days": age_days,
            "device_known_accounts": dd.get("device_known_accounts", 1),
            "fingerprint_hash": dd.get("browser_fingerprint_hash"),
            "fingerprint_changes_30d": dd.get("fingerprint_changes_30d", 0),
            "user_agent": dd.get("user_agent"),
            "emulator_signals": dd.get("emulator_signals", False),
            "account_age_days": case.transaction_data.get("account_age_days"),
            "source": "MOCK - replace with a real device-fingerprint provider in production",
        }
        if include_sessions:
            out["recent_sessions"] = dd.get("device_history", [])[:20]
        return out

    def _geo(self, case: Case) -> Dict[str, Any]:
        dd = case.device_data
        ip_country = dd.get("ip_country")
        billing_country = dd.get("billing_country")
        prior = dd.get("account_prior_countries", [])
        return {
            "ip": dd.get("ip"),
            "ip_country": ip_country,
            "ip_city": dd.get("geo_city"),
            "ip_asn": dd.get("ip_asn"),
            "is_proxy_or_hosting": bool(dd.get("is_proxy", False)),
            "billing_country": billing_country,
            "billing_city": dd.get("billing_city"),
            "distance_km": dd.get("distance_km"),
            "country_mismatch": bool(ip_country and billing_country and ip_country != billing_country),
            "country_new_for_account": bool(ip_country and ip_country not in prior),
            "account_prior_countries": prior,
            "impossible_travel": bool(dd.get("impossible_travel", False)),
            "source": "MOCK - replace with a real IP-intelligence provider in production",
        }

    def _kyc(self, case: Case) -> Dict[str, Any]:
        dd = case.device_data
        status = str(dd.get("kyc_status", "unknown"))
        return {
            "kyc_status": status,
            "kyc_risk_weight": KYC_RISK.get(status, 0.4),
            "kyc_flags": dd.get("kyc_flags", []),
            "kyc_last_reviewed": dd.get("kyc_last_reviewed"),
            "identity_document_type": dd.get("identity_document_type"),
            "source": "MOCK - replace with a real KYC provider in production",
        }

    # -- offline fallback ---------------------------------------------------
    def offline_plan(self, case: Case) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            ("get_device_history", {"include_sessions": True}),
            ("check_geo_mismatch", {}),
            ("get_kyc_status", {}),
        ]

    def offline_analyze(self, case: Case, records: List[ToolCallRecord]) -> Dict[str, Any]:
        r = self.results_by_tool(records)
        dev = r.get("get_device_history", {})
        geo = r.get("check_geo_mismatch", {})
        kyc = r.get("get_kyc_status", {})

        risk = 0.0
        evidence: List[str] = []

        changes = int(dev.get("fingerprint_changes_30d", 0) or 0)
        if changes >= 5:
            risk += 22
            evidence.append(f"{changes} fingerprint changes in 30 days")
        elif changes >= 3:
            risk += 11

        known_accounts = int(dev.get("device_known_accounts", 1) or 1)
        if known_accounts >= 5:
            risk += 20
            evidence.append(f"device seen on {known_accounts} distinct accounts")
        elif known_accounts >= 3:
            risk += 10

        if dev.get("emulator_signals"):
            risk += 15
            evidence.append("emulator/automation signals present on the session")

        device_age = dev.get("device_age_days")
        account_age = dev.get("account_age_days")
        if isinstance(device_age, int) and device_age <= 1 and isinstance(account_age, int) and account_age > 180:
            risk += 12
            evidence.append(f"brand-new device (age {device_age}d) on a {account_age}-day-old account")

        if geo.get("impossible_travel"):
            risk += 20
            evidence.append("impossible-travel between consecutive sessions")
        if geo.get("country_mismatch") and geo.get("country_new_for_account"):
            risk += 14
            evidence.append(
                f"IP country {geo.get('ip_country')} differs from billing {geo.get('billing_country')} "
                "and is new for this account"
            )
        elif geo.get("country_mismatch"):
            risk += 5
        if geo.get("is_proxy_or_hosting"):
            risk += 12
            evidence.append(f"session IP on proxy/hosting ASN {geo.get('ip_asn')}")

        kyc_weight = float(kyc.get("kyc_risk_weight", 0.4) or 0.0)
        risk += kyc_weight * 25
        if kyc_weight >= 0.7:
            evidence.append(f"KYC status '{kyc.get('kyc_status')}' with flags {kyc.get('kyc_flags')}")

        signals_present = sum(
            1 for x in (changes >= 3, known_accounts >= 3, geo.get("country_mismatch"), kyc_weight >= 0.35) if x
        )
        confidence = 0.85 if signals_present >= 2 else (0.62 if signals_present == 1 else 0.7)
        if dev.get("device_id") is None:
            confidence = min(confidence, 0.4)

        if not evidence:
            evidence.append("device, geolocation and KYC all consistent with the account holder")

        return {
            "risk_score": max(0.0, min(100.0, risk)),
            "confidence": confidence,
            "reasoning_summary": (
                "Offline deterministic scorer (no LLM): weighed fingerprint churn, device sharing, "
                f"geo/billing mismatch and KYC status into a risk of {min(100.0, risk):.0f} from "
                f"{signals_present} concurrent identity signal(s)."
            ),
            "key_evidence": evidence,
        }
