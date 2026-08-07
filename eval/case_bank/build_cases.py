"""Deterministic generator for the seeded case bank.

ALL DATA PRODUCED HERE IS SYNTHETIC.  No real account, card, device or person is
represented, and nothing is derived from real financial records.

Why a generator instead of 22 hand-written JSON files: each case is authored at
the level of *which signals are planted* (a card-testing burst with 9 declines,
a device shared across 7 accounts, a 6-account ring over two identifier types),
and the generator fills in a plausible transaction history around them from a
fixed seed.  That keeps the intent of each case readable and reviewable - the
SPECS table below is effectively the eval's answer key and its threat model in
one place - while still producing histories long enough for the velocity and
z-score tools to have something real to chew on.

Re-run with ``python eval/case_bank/build_cases.py``; output is byte-identical
for a given seed, so a regenerated bank never silently changes the eval.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
SEED = 20260115
REFERENCE = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

NORMAL_CATEGORIES = ["groceries", "restaurants", "transport", "utilities", "retail", "subscriptions", "fuel"]
MERCHANTS = {
    "groceries": ["Northfield Market", "GreenLeaf Grocer", "CityFresh"],
    "restaurants": ["Tanner & Co", "Blue Fig Cafe", "Riverside Diner"],
    "transport": ["MetroRail", "CityCab", "Harbour Ferries"],
    "utilities": ["Vantage Power", "Clearwater Utilities"],
    "retail": ["Harlow Home", "Kestrel Apparel", "Orchard Supply"],
    "subscriptions": ["StreamNine", "ReadDaily", "CloudVault"],
    "fuel": ["Ridgeway Fuel", "Beacon Petrol"],
    "digital_goods": ["PixelKeys Store", "GameCred Direct", "AppVault"],
    "gift_cards": ["GiftLane", "CardMint"],
    "crypto_exchange": ["Vertex Digital", "CoinRidge"],
    "prepaid_reload": ["TopUpNow", "ReloadHub"],
    "money_transfer": ["SwiftRemit", "PayBridge"],
    "electronics": ["Volt & Wire", "Circuit Bay"],
    "online_gambling": ["LuckyPine", "BetHarbour"],
    "travel": ["Aerolink", "StayWell Hotels"],
    "healthcare": ["Meadow Clinic", "PharmaPoint"],
}


def ts(offset_minutes: float) -> str:
    return (REFERENCE + timedelta(minutes=offset_minutes)).isoformat().replace("+00:00", "Z")


def merchant_for(rng: random.Random, category: str) -> str:
    return rng.choice(MERCHANTS.get(category, ["Generic Merchant"]))


def build_history(
    rng: random.Random,
    *,
    count: int,
    base_amount: float,
    amount_sd: float,
    span_hours: float,
    categories: List[str],
    declines: int = 0,
    typical_hour: int = 14,
    burst_minutes: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build a plausible prior-transaction history ending just before the flagged txn.

    ``burst_minutes`` compresses the whole history into a short window, which is
    what produces a genuine velocity spike for the transaction agent to find
    rather than a number asserted in the case file.
    """
    rows: List[Dict[str, Any]] = []
    for i in range(count):
        if burst_minutes is not None:
            offset = -burst_minutes * (i + 1) / count
        else:
            # Spread across the window, snapped near the account's typical hour.
            day = -(i * span_hours / max(count, 1)) / 24.0
            base_dt = REFERENCE + timedelta(days=day)
            jitter = rng.uniform(-1.5, 1.5)
            target = base_dt.replace(hour=typical_hour, minute=0, second=0) + timedelta(hours=jitter)
            offset = (target - REFERENCE).total_seconds() / 60.0
        category = rng.choice(categories)
        amount = max(1.0, rng.gauss(base_amount, amount_sd))
        rows.append(
            {
                "txn_id": f"T-{rng.randrange(10**6, 10**7)}",
                "amount": round(amount, 2),
                "currency": "USD",
                "merchant": merchant_for(rng, category),
                "merchant_category": category,
                "timestamp": ts(offset - 1),
                "status": "approved",
            }
        )
    for row in rng.sample(rows, min(declines, len(rows))):
        row["status"] = "declined"
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def build_network(
    rng: random.Random,
    focus: str,
    *,
    ring_size: int,
    types: List[str],
    dense: bool = False,
    identifier_pool: int = 2,
) -> Dict[str, Any]:
    """Build the shared-identifier edge list.

    ``identifier_pool`` is what separates a ring from a crowd: a small pool with
    a large ring_size means many accounts converging on few identifiers, which
    is the structure the network specialist scores on.
    """
    others = [f"ACC-{rng.randrange(20000, 99999)}" for _ in range(max(0, ring_size - 1))]
    identifiers = {
        kind: [f"{kind[:3].upper()}-{rng.randrange(1000, 9999)}" for _ in range(identifier_pool)] for kind in types
    }
    edges: List[Dict[str, Any]] = []
    for other in others:
        for kind in types:
            if rng.random() < (0.95 if kind == types[0] else 0.55):
                edges.append(
                    {
                        "account_a": focus,
                        "account_b": other,
                        "shared": kind,
                        "identifier": rng.choice(identifiers[kind]),
                        "first_seen_together": ts(-rng.uniform(1440, 20000)),
                    }
                )
    if dense:
        for i, a in enumerate(others):
            for b in others[i + 1 :]:
                if rng.random() < 0.55:
                    kind = rng.choice(types)
                    edges.append(
                        {
                            "account_a": a,
                            "account_b": b,
                            "shared": kind,
                            "identifier": rng.choice(identifiers[kind]),
                            "first_seen_together": ts(-rng.uniform(1440, 20000)),
                        }
                    )
    return {
        "focus_account": focus,
        "edges": edges,
        "shared_identifiers": identifiers,
        "note": "Synthetic evaluation data.",
    }


# ---------------------------------------------------------------------------
# The case bank.  This table is the eval's answer key AND its threat model.
# `planted_signals` is recorded in each file so the harness report can state
# exactly which evidence a case was built to contain - which is what makes
# "did the agents reason, or was the case just easy?" an answerable question.
# ---------------------------------------------------------------------------
SPECS: List[Dict[str, Any]] = [
    # ---------------------------------------------------- card testing (fraud)
    {
        "case_id": "SM-001",
        "category": "card_testing",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "easy",
        "summary": "Rapid sequence of sub-$5 digital-goods authorisations on a 3-day-old account, most declined.",
        "planted_signals": ["velocity burst", "high decline rate", "elevated-risk category", "proxy IP", "device shared across accounts"],
        "account_age_days": 3,
        "txn": {"amount": 1.99, "merchant_category": "digital_goods", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 22, "base_amount": 2.4, "amount_sd": 1.1, "span_hours": 1, "categories": ["digital_goods"], "declines": 13, "burst_minutes": 34},
        "device": {"known_accounts": 8, "fp_changes": 9, "proxy": True, "ip_country": "RO", "billing_country": "US", "kyc": "unverified", "kyc_flags": ["document_not_provided"], "distance_km": 8100, "device_age_days": 2, "emulator": True},
        "network": {"ring_size": 5, "types": ["device", "payment_method"], "dense": True, "identifier_pool": 2},
    },
    {
        "case_id": "SM-002",
        "category": "card_testing",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "easy",
        "summary": "Eighteen $0.50-$3 gift-card authorisations in 20 minutes across three merchants, half declined.",
        "planted_signals": ["velocity burst", "high decline rate", "gift cards", "new device", "unverified KYC"],
        "account_age_days": 6,
        "txn": {"amount": 0.99, "merchant_category": "gift_cards", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 18, "base_amount": 1.6, "amount_sd": 0.9, "span_hours": 1, "categories": ["gift_cards", "digital_goods"], "declines": 9, "burst_minutes": 20},
        "device": {"known_accounts": 4, "fp_changes": 6, "proxy": True, "ip_country": "NL", "billing_country": "US", "kyc": "unverified", "kyc_flags": ["document_not_provided"], "distance_km": 6200, "device_age_days": 0},
        "network": {"ring_size": 4, "types": ["payment_method"], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-003",
        "category": "card_testing",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "medium",
        "summary": "Low-value probing spread over 6 hours rather than a tight burst, on an account with a short but clean history.",
        "planted_signals": ["moderate velocity", "elevated decline rate", "digital goods", "fingerprint churn"],
        "account_age_days": 41,
        "txn": {"amount": 3.5, "merchant_category": "digital_goods", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 14, "base_amount": 3.2, "amount_sd": 1.4, "span_hours": 6, "categories": ["digital_goods", "subscriptions"], "declines": 6, "burst_minutes": 340},
        "device": {"known_accounts": 3, "fp_changes": 5, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "pending", "kyc_flags": [], "distance_km": 320, "device_age_days": 9},
        "network": {"ring_size": 3, "types": ["device"], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-004",
        "category": "card_testing",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "hard",
        "summary": "Card testing camouflaged inside an otherwise normal spending profile; only the decline pattern gives it away.",
        "planted_signals": ["decline clustering", "category first-use", "no network links", "clean KYC"],
        "account_age_days": 620,
        "txn": {"amount": 4.25, "merchant_category": "digital_goods", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 26, "base_amount": 46.0, "amount_sd": 22.0, "span_hours": 700, "categories": NORMAL_CATEGORIES, "declines": 11, "burst_minutes": None},
        "device": {"known_accounts": 1, "fp_changes": 4, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 40, "device_age_days": 210},
        "network": {"ring_size": 1, "types": [], "dense": False, "identifier_pool": 1},
    },
    # ------------------------------------------------ account takeover (fraud)
    {
        "case_id": "SM-005",
        "category": "account_takeover",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "easy",
        "summary": "Long-tenured account: new device, impossible travel from a hosting ASN, then a 12x-baseline electronics purchase.",
        "planted_signals": ["impossible travel", "new device on old account", "hosting ASN", "amount z-score spike"],
        "account_age_days": 1450,
        "txn": {"amount": 2480.0, "merchant_category": "electronics", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 30, "base_amount": 78.0, "amount_sd": 34.0, "span_hours": 900, "categories": NORMAL_CATEGORIES, "declines": 1, "burst_minutes": None},
        "device": {"known_accounts": 2, "fp_changes": 7, "proxy": True, "ip_country": "VN", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 13400, "device_age_days": 0, "impossible_travel": True},
        "network": {"ring_size": 2, "types": ["device"], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-006",
        "category": "account_takeover",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "medium",
        "summary": "Credential-stuffed account draining to money transfer after a device change; transaction size itself is unremarkable.",
        "planted_signals": ["new device", "country new for account", "money transfer first-use", "moderate amount"],
        "account_age_days": 880,
        "txn": {"amount": 320.0, "merchant_category": "money_transfer", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 24, "base_amount": 140.0, "amount_sd": 70.0, "span_hours": 800, "categories": NORMAL_CATEGORIES, "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 3, "fp_changes": 6, "proxy": True, "ip_country": "GH", "billing_country": "GB", "kyc": "verified", "kyc_flags": [], "distance_km": 5100, "device_age_days": 1},
        "network": {"ring_size": 4, "types": ["device", "ip_range"], "dense": False, "identifier_pool": 2},
    },
    {
        "case_id": "SM-007",
        "category": "account_takeover",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "medium",
        "summary": "Overnight session from an unfamiliar country followed by three high-value retail charges outside the account's pattern.",
        "planted_signals": ["off-hours activity", "geo mismatch", "amount z-score spike", "velocity uplift"],
        "account_age_days": 1100,
        "txn": {"amount": 1150.0, "merchant_category": "retail", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 28, "base_amount": 62.0, "amount_sd": 28.0, "span_hours": 850, "categories": NORMAL_CATEGORIES, "declines": 1, "burst_minutes": None, "typical_hour": 13},
        "device": {"known_accounts": 2, "fp_changes": 5, "proxy": False, "ip_country": "BG", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 7800, "device_age_days": 3},
        "network": {"ring_size": 2, "types": ["ip_range"], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-008",
        "category": "account_takeover",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "hard",
        "summary": "Takeover from inside the victim's own country and city; the only anomalies are a fresh device and a category the account never used.",
        "planted_signals": ["new device only", "no geo mismatch", "category first-use", "clean network"],
        "account_age_days": 1680,
        "txn": {"amount": 890.0, "merchant_category": "crypto_exchange", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 30, "base_amount": 95.0, "amount_sd": 40.0, "span_hours": 900, "categories": NORMAL_CATEGORIES, "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 1, "fp_changes": 6, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 25, "device_age_days": 0},
        "network": {"ring_size": 1, "types": [], "dense": False, "identifier_pool": 1},
    },
    # --------------------------------------------- synthetic identity (fraud)
    {
        "case_id": "SM-009",
        "category": "synthetic_identity",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "medium",
        "summary": "Two-week-old account, failed KYC on a mismatched document, immediately maxing out on prepaid reloads.",
        "planted_signals": ["failed KYC", "thin history", "prepaid reload", "device shared across accounts"],
        "account_age_days": 14,
        "txn": {"amount": 480.0, "merchant_category": "prepaid_reload", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 6, "base_amount": 120.0, "amount_sd": 90.0, "span_hours": 300, "categories": ["prepaid_reload", "retail"], "declines": 1, "burst_minutes": None},
        "device": {"known_accounts": 6, "fp_changes": 3, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "failed", "kyc_flags": ["document_mismatch", "ssn_not_verified"], "distance_km": 900, "device_age_days": 12},
        "network": {"ring_size": 7, "types": ["device", "phone", "email_pattern"], "dense": True, "identifier_pool": 2},
    },
    {
        "case_id": "SM-010",
        "category": "synthetic_identity",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "medium",
        "summary": "Fabricated identity built up with small legitimate-looking spend for a month before a large crypto cash-out.",
        "planted_signals": ["bust-out amount spike", "unverified KYC", "email-pattern ring", "crypto category"],
        "account_age_days": 34,
        "txn": {"amount": 3200.0, "merchant_category": "crypto_exchange", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 20, "base_amount": 31.0, "amount_sd": 12.0, "span_hours": 700, "categories": ["subscriptions", "retail", "restaurants"], "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 4, "fp_changes": 2, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "unverified", "kyc_flags": ["ssn_not_verified"], "distance_km": 60, "device_age_days": 30},
        "network": {"ring_size": 6, "types": ["email_pattern", "payment_method"], "dense": True, "identifier_pool": 2},
    },
    {
        "case_id": "SM-011",
        "category": "synthetic_identity",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "hard",
        "summary": "Patient synthetic identity: KYC passed on fabricated documents, 8 months of tenure, no ring links yet.",
        "planted_signals": ["KYC passed", "long tenure", "isolated in graph", "amount spike only"],
        "account_age_days": 245,
        "txn": {"amount": 4100.0, "merchant_category": "money_transfer", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 24, "base_amount": 58.0, "amount_sd": 26.0, "span_hours": 800, "categories": ["subscriptions", "retail", "groceries"], "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 2, "fp_changes": 1, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 15, "device_age_days": 200},
        "network": {"ring_size": 2, "types": ["ip_range"], "dense": False, "identifier_pool": 1},
    },
    # -------------------------------------------------- money muling (fraud)
    {
        "case_id": "SM-012",
        "category": "money_muling",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "medium",
        "summary": "Nine accounts converging on two devices and one payment instrument, all forwarding funds to the same transfer service.",
        "planted_signals": ["large dense ring", "identifier convergence", "money transfer", "clean identity signals"],
        "account_age_days": 190,
        "txn": {"amount": 940.0, "merchant_category": "money_transfer", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 18, "base_amount": 610.0, "amount_sd": 180.0, "span_hours": 700, "categories": ["money_transfer", "retail"], "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 9, "fp_changes": 1, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 30, "device_age_days": 180},
        "network": {"ring_size": 9, "types": ["device", "payment_method", "phone"], "dense": True, "identifier_pool": 2},
    },
    {
        "case_id": "SM-013",
        "category": "money_muling",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "medium",
        "summary": "Mule account with a verified owner: identity signals are entirely clean, the ring structure is not.",
        "planted_signals": ["6-account ring", "two identifier types", "verified KYC", "normal amounts"],
        "account_age_days": 410,
        "txn": {"amount": 720.0, "merchant_category": "money_transfer", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 22, "base_amount": 540.0, "amount_sd": 200.0, "span_hours": 800, "categories": ["money_transfer", "groceries", "transport"], "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 5, "fp_changes": 0, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified_enhanced", "kyc_flags": [], "distance_km": 12, "device_age_days": 400},
        "network": {"ring_size": 6, "types": ["device", "payment_method"], "dense": True, "identifier_pool": 2},
    },
    {
        "case_id": "SM-014",
        "category": "money_muling",
        "ground_truth": "fraud",
        "expected_verdict": "decline",
        "difficulty": "hard",
        "summary": "Small three-account mule cell using only a shared payment instrument; deliberately below the obvious ring threshold.",
        "planted_signals": ["small ring", "single identifier type", "clean identity", "repeat transfers"],
        "account_age_days": 300,
        "txn": {"amount": 610.0, "merchant_category": "money_transfer", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 20, "base_amount": 480.0, "amount_sd": 160.0, "span_hours": 750, "categories": ["money_transfer", "retail"], "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 3, "fp_changes": 1, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 20, "device_age_days": 290},
        "network": {"ring_size": 3, "types": ["payment_method"], "dense": True, "identifier_pool": 1},
    },
    # --------------------------------------------------------- legitimate
    {
        "case_id": "SM-015",
        "category": "legitimate",
        "ground_truth": "legitimate",
        "expected_verdict": "approve",
        "difficulty": "easy",
        "summary": "Routine grocery purchase on a four-year-old account from its usual device and city.",
        "planted_signals": ["no anomalies"],
        "account_age_days": 1520,
        "txn": {"amount": 84.2, "merchant_category": "groceries", "hour_offset": 0, "channel": "card_present"},
        "history": {"count": 30, "base_amount": 79.0, "amount_sd": 21.0, "span_hours": 900, "categories": NORMAL_CATEGORIES, "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 1, "fp_changes": 0, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 4, "device_age_days": 700},
        "network": {"ring_size": 1, "types": [], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-016",
        "category": "legitimate",
        "ground_truth": "legitimate",
        "expected_verdict": "approve",
        "difficulty": "medium",
        "summary": "Genuine holiday abroad: foreign IP and a large hotel charge, but travel is gradual and the device is the customer's own.",
        "planted_signals": ["geo mismatch (benign)", "amount above baseline", "known device", "verified KYC"],
        "account_age_days": 2100,
        "txn": {"amount": 640.0, "merchant_category": "travel", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 28, "base_amount": 96.0, "amount_sd": 55.0, "span_hours": 880, "categories": NORMAL_CATEGORIES + ["travel"], "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 1, "fp_changes": 1, "proxy": False, "ip_country": "ES", "billing_country": "GB", "kyc": "verified_enhanced", "kyc_flags": [], "distance_km": 1400, "device_age_days": 640, "prior_countries": ["GB", "ES", "FR", "IT"]},
        "network": {"ring_size": 1, "types": [], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-017",
        "category": "legitimate",
        "ground_truth": "legitimate",
        "expected_verdict": "approve",
        "difficulty": "medium",
        "summary": "Shared household: three family accounts on one residential IP range, ordinary supermarket spend.",
        "planted_signals": ["ip_range-only links (benign)", "no device sharing", "stable profile"],
        "account_age_days": 990,
        "txn": {"amount": 122.4, "merchant_category": "groceries", "hour_offset": 0, "channel": "card_present"},
        "history": {"count": 26, "base_amount": 110.0, "amount_sd": 35.0, "span_hours": 860, "categories": NORMAL_CATEGORIES, "declines": 1, "burst_minutes": None},
        "device": {"known_accounts": 1, "fp_changes": 0, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 2, "device_age_days": 500},
        "network": {"ring_size": 4, "types": ["ip_range"], "dense": True, "identifier_pool": 1},
    },
    {
        "case_id": "SM-018",
        "category": "legitimate",
        "ground_truth": "legitimate",
        "expected_verdict": "approve",
        "difficulty": "hard",
        "summary": "Privacy-conscious customer on a commercial VPN buying a laptop; VPN plus a big-ticket electronics charge looks bad and is not.",
        "planted_signals": ["proxy ASN (benign)", "amount spike", "long tenure", "no ring", "verified KYC"],
        "account_age_days": 1810,
        "txn": {"amount": 1890.0, "merchant_category": "electronics", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 30, "base_amount": 88.0, "amount_sd": 42.0, "span_hours": 900, "categories": NORMAL_CATEGORIES, "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 1, "fp_changes": 1, "proxy": True, "ip_country": "US", "billing_country": "US", "kyc": "verified_enhanced", "kyc_flags": [], "distance_km": 380, "device_age_days": 900, "prior_countries": ["US", "CA", "DE"]},
        "network": {"ring_size": 1, "types": [], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-019",
        "category": "legitimate",
        "ground_truth": "legitimate",
        "expected_verdict": "approve",
        "difficulty": "hard",
        "summary": "New customer's first week: thin history and a new device, but every signal that would indicate fraud is absent.",
        "planted_signals": ["thin baseline", "new device", "clean KYC", "no ring", "modest amount"],
        "account_age_days": 5,
        "txn": {"amount": 64.0, "merchant_category": "retail", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 4, "base_amount": 52.0, "amount_sd": 18.0, "span_hours": 100, "categories": ["retail", "groceries"], "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 1, "fp_changes": 0, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 8, "device_age_days": 5},
        "network": {"ring_size": 1, "types": [], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-020",
        "category": "legitimate",
        "ground_truth": "legitimate",
        "expected_verdict": "approve",
        "difficulty": "medium",
        "summary": "Payday burst: eleven small transactions in one afternoon, all at merchants the account uses regularly.",
        "planted_signals": ["velocity spike (benign)", "familiar merchants", "no declines", "stable device"],
        "account_age_days": 1240,
        "txn": {"amount": 38.5, "merchant_category": "restaurants", "hour_offset": 0, "channel": "card_present"},
        "history": {"count": 11, "base_amount": 34.0, "amount_sd": 12.0, "span_hours": 1, "categories": ["restaurants", "groceries", "transport", "fuel"], "declines": 0, "burst_minutes": 55},
        "device": {"known_accounts": 1, "fp_changes": 0, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 3, "device_age_days": 800},
        "network": {"ring_size": 1, "types": [], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-021",
        "category": "legitimate",
        "ground_truth": "legitimate",
        "expected_verdict": "approve",
        "difficulty": "medium",
        "summary": "Small business owner topping up prepaid cards for staff - an elevated-risk category used entirely legitimately.",
        "planted_signals": ["elevated-risk category (benign)", "repeat use of category", "verified business KYC", "no ring"],
        "account_age_days": 1450,
        "txn": {"amount": 300.0, "merchant_category": "prepaid_reload", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 24, "base_amount": 285.0, "amount_sd": 60.0, "span_hours": 880, "categories": ["prepaid_reload", "utilities", "retail"], "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 2, "fp_changes": 0, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified_enhanced", "kyc_flags": [], "distance_km": 6, "device_age_days": 1000},
        "network": {"ring_size": 2, "types": ["ip_range"], "dense": False, "identifier_pool": 1},
    },
    # ---------------------------------------------------------- ambiguous
    {
        "case_id": "SM-022",
        "category": "account_takeover",
        "ground_truth": "ambiguous",
        "expected_verdict": "escalate",
        "difficulty": "hard",
        "summary": "New device from a new country on a long-lived account, but the amount and merchant fit the customer perfectly. Genuinely undecidable from the available signals.",
        "planted_signals": ["identity risk vs clean transaction profile", "designed specialist conflict"],
        "account_age_days": 1300,
        "txn": {"amount": 118.0, "merchant_category": "retail", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 26, "base_amount": 112.0, "amount_sd": 30.0, "span_hours": 870, "categories": NORMAL_CATEGORIES, "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 3, "fp_changes": 6, "proxy": True, "ip_country": "PL", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 7300, "device_age_days": 0},
        "network": {"ring_size": 1, "types": [], "dense": False, "identifier_pool": 1},
    },
    {
        "case_id": "SM-023",
        "category": "money_muling",
        "ground_truth": "ambiguous",
        "expected_verdict": "escalate",
        "difficulty": "hard",
        "summary": "Four accounts sharing a device in what could be a mule cell or a shared family tablet; identity and transaction signals are spotless.",
        "planted_signals": ["network risk vs clean identity and transactions", "designed specialist conflict"],
        "account_age_days": 760,
        "txn": {"amount": 260.0, "merchant_category": "retail", "hour_offset": 0, "channel": "card_not_present"},
        "history": {"count": 24, "base_amount": 240.0, "amount_sd": 70.0, "span_hours": 840, "categories": NORMAL_CATEGORIES, "declines": 0, "burst_minutes": None},
        "device": {"known_accounts": 4, "fp_changes": 0, "proxy": False, "ip_country": "US", "billing_country": "US", "kyc": "verified", "kyc_flags": [], "distance_km": 5, "device_age_days": 700},
        "network": {"ring_size": 4, "types": ["device", "ip_range"], "dense": True, "identifier_pool": 1},
    },
]


def build_case(spec: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(f"{SEED}-{spec['case_id']}")
    account_id = f"ACC-{rng.randrange(10000, 19999)}"
    hist_cfg = dict(spec["history"])
    typical_hour = hist_cfg.pop("typical_hour", 14)
    history = build_history(rng, typical_hour=typical_hour, **hist_cfg)

    txn_cfg = spec["txn"]
    transaction = {
        "txn_id": f"T-{rng.randrange(10**6, 10**7)}",
        "amount": txn_cfg["amount"],
        "currency": "USD",
        "merchant": merchant_for(rng, txn_cfg["merchant_category"]),
        "merchant_category": txn_cfg["merchant_category"],
        "timestamp": ts(txn_cfg.get("hour_offset", 0)),
        "channel": txn_cfg["channel"],
    }

    dev = spec["device"]
    device_id = f"DEV-{rng.randrange(1000, 9999)}"
    device_history = [
        {
            "device_id": device_id,
            "account_id": account_id if i == 0 else f"ACC-{rng.randrange(20000, 99999)}",
            "last_seen": ts(-rng.uniform(10, 4000)),
            "user_agent": rng.choice(
                ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2)", "Mozilla/5.0 (X11; Linux x86_64)"]
            ),
        }
        for i in range(min(int(dev.get("known_accounts", 1)), 8))
    ]

    device_data = {
        "device_id": device_id,
        "device_first_seen": ts(-1440 * float(dev.get("device_age_days", 30))),
        "device_known_accounts": dev.get("known_accounts", 1),
        "browser_fingerprint_hash": f"fp_{rng.getrandbits(48):012x}",
        "fingerprint_changes_30d": dev.get("fp_changes", 0),
        "user_agent": device_history[0]["user_agent"],
        "emulator_signals": bool(dev.get("emulator", False)),
        "device_history": device_history,
        "ip": f"{rng.randrange(23, 220)}.{rng.randrange(0, 255)}.{rng.randrange(0, 255)}.{rng.randrange(1, 254)}",
        "ip_country": dev.get("ip_country", "US"),
        "geo_city": dev.get("geo_city", "unspecified"),
        "ip_asn": "AS-COMMERCIAL-VPN" if dev.get("proxy") else f"AS{rng.randrange(1000, 60000)}-RETAIL-ISP",
        "is_proxy": bool(dev.get("proxy", False)),
        "billing_country": dev.get("billing_country", "US"),
        "billing_city": dev.get("billing_city", "unspecified"),
        "distance_km": dev.get("distance_km", 0),
        "impossible_travel": bool(dev.get("impossible_travel", False)),
        "account_prior_countries": dev.get("prior_countries", [dev.get("billing_country", "US")]),
        "kyc_status": dev.get("kyc", "verified"),
        "kyc_flags": dev.get("kyc_flags", []),
        "kyc_last_reviewed": ts(-1440 * rng.uniform(30, 400)),
        "identity_document_type": rng.choice(["passport", "national_id", "drivers_licence"]),
    }

    net_cfg = spec["network"]
    network_data = build_network(
        rng,
        account_id,
        ring_size=net_cfg["ring_size"],
        types=net_cfg["types"],
        dense=net_cfg.get("dense", False),
        identifier_pool=net_cfg.get("identifier_pool", 2),
    )

    return {
        "case_id": spec["case_id"],
        "category": spec["category"],
        "ground_truth": spec["ground_truth"],
        "expected_verdict": spec["expected_verdict"],
        "difficulty": spec["difficulty"],
        "summary": spec["summary"],
        "planted_signals": spec["planted_signals"],
        "synthetic": True,
        "data_disclaimer": "SYNTHETIC evaluation data. Not derived from any real account, card, device or person.",
        "generator": {"script": "eval/case_bank/build_cases.py", "seed": SEED},
        "transaction_data": {
            "account_id": account_id,
            "account_age_days": spec["account_age_days"],
            "transaction": transaction,
            "history": history,
        },
        "device_data": device_data,
        "network_data": network_data,
    }


def main() -> None:
    written = 0
    for spec in SPECS:
        case = build_case(spec)
        path = HERE / f"{spec['case_id']}.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(case, handle, indent=2, sort_keys=False)
            handle.write("\n")
        written += 1
    counts: Dict[str, int] = {}
    for spec in SPECS:
        counts[spec["ground_truth"]] = counts.get(spec["ground_truth"], 0) + 1
    print(f"wrote {written} synthetic cases to {HERE}")
    print("ground truth distribution:", counts)


if __name__ == "__main__":
    main()
