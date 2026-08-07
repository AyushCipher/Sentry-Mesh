"""Specialist 4 - historical precedent.

Signal category: how cases that *look like this one* were actually resolved by
the team in the past.  Retrieval runs over a corpus of past **resolved** cases
via pgvector cosine similarity (or the equivalent in-process index when no
pgvector database is reachable - see :mod:`sentrymesh.retrieval`).

This specialist is the one most able to embarrass the others, and that is the
point: if four accounts with this exact shape were reviewed last quarter and
all four were legitimate, that belongs in the verdict.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentrymesh.agents.base import SpecialistAgent
from sentrymesh.agents.llm_client import LLMClient, ToolCallRecord, tool
from sentrymesh.event_log import EventLog
from sentrymesh.models import Case
from sentrymesh.retrieval import build_index

# Below this cosine similarity a "nearest" precedent is not really a precedent.
# Retrieval that always returns its top-k regardless of quality is how a
# retrieval agent ends up confidently citing an irrelevant case.
RELEVANCE_FLOOR = 0.25

# Similarity at which a precedent is treated as fully grounded. With hashed
# TF-IDF over a 22-precedent corpus, real matches land around 0.3-0.45, so this
# scales the offline score honestly instead of letting a weak match speak with
# the authority of a strong one.
STRONG_SIMILARITY = 0.6


class HistoricalCaseAgent(SpecialistAgent):
    name = "historical_case"
    focus = "precedent from past resolved cases with similar signal profiles"

    def __init__(self, log: EventLog, llm: LLMClient, index: Any = None) -> None:
        super().__init__(log, llm)
        self._index = index
        self._index_lock = asyncio.Lock()

    async def index(self) -> Any:
        if self._index is None:
            async with self._index_lock:
                if self._index is None:
                    self._index = await build_index()
        return self._index

    @property
    def system_prompt(self) -> str:
        return (
            "You are a fraud analyst whose job is precedent: what happened last time the team saw "
            "a case shaped like this one.\n\n"
            "How to weigh what you find:\n"
            "- Similarity scores matter. A top match at 0.2 similarity is not a precedent, it is "
            f"the least-bad row in the index. Treat anything below {RELEVANCE_FLOOR} as no precedent "
            "at all and report low confidence.\n"
            "- Precedent that cuts against fraud is as important as precedent for it. If similar "
            "cases were resolved as legitimate, that lowers the risk score - say so.\n"
            "- Look at what the reviewer actually concluded, not just the label. Use "
            "get_case_resolution on your best matches.\n"
            "- Mixed precedent (similar cases resolved both ways) is a real finding: report a "
            "middling risk score with LOW confidence, which tells the supervisor this case may "
            "need a human.\n\n"
            "You cannot see the live transaction, device or network analysis. Base your score on "
            "retrieved precedent only."
        )

    @property
    def investigation_tools(self) -> Sequence[Dict[str, Any]]:
        return [
            tool(
                "search_similar_cases",
                "Cosine-similarity search over past RESOLVED cases. Returns matches with their "
                "similarity score and how each was resolved.",
                {
                    "query": {
                        "type": "string",
                        "description": "Free-text description of the case's signal profile to match on. "
                        "Include the fraud pattern you suspect and the concrete signals present.",
                    },
                    "k": {"type": "integer", "description": "Number of matches to return (default 5, max 10)."},
                },
                required=["query"],
            ),
            tool(
                "get_case_resolution",
                "Retrieve the full resolution record - outcome, reviewer and notes - for one past case.",
                {"case_id": {"type": "string", "description": "The precedent case_id to look up."}},
                required=["case_id"],
            ),
        ]

    # -- tool implementations ----------------------------------------------
    async def run_tool(self, case: Case, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        index = await self.index()
        if name == "search_similar_cases":
            query = str(args.get("query") or self.default_query(case))
            k = max(1, min(10, int(args.get("k") or 5)))
            matches = await index.search(query, k)
            relevant = [m for m in matches if m.get("similarity", 0.0) >= RELEVANCE_FLOOR]
            return {
                "query": query,
                "retrieval_backend": getattr(index, "name", "unknown"),
                "relevance_floor": RELEVANCE_FLOOR,
                "matches_returned": len(matches),
                "matches_above_floor": len(relevant),
                "matches": matches,
                "warning": (
                    None
                    if relevant
                    else "no precedent cleared the relevance floor - treat this as an absence of precedent"
                ),
            }
        if name == "get_case_resolution":
            row = await index.get(str(args.get("case_id", "")))
            if row is None:
                return {"error": f"no resolved case with id {args.get('case_id')!r}"}
            return row
        raise ValueError(f"{self.name}: unknown tool {name!r}")

    @staticmethod
    def default_query(case: Case) -> str:
        """Build a signal-profile query string from the raw case data.

        Used as the offline query and as a fallback when the model calls
        ``search_similar_cases`` without one.
        """
        txn = case.transaction_data.get("transaction", {})
        dd = case.device_data
        nd = case.network_data
        bits = [
            case.summary,
            f"amount {txn.get('amount')} {txn.get('currency', '')}",
            f"merchant category {txn.get('merchant_category')}",
            f"channel {txn.get('channel')}",
            f"account age {case.transaction_data.get('account_age_days')} days",
            f"kyc {dd.get('kyc_status')}",
            f"device shared with {dd.get('device_known_accounts')} accounts",
            f"ip country {dd.get('ip_country')} billing {dd.get('billing_country')}",
            f"proxy {bool(dd.get('is_proxy'))}",
            f"{len(nd.get('edges', []))} shared identifier links",
        ]
        return " ".join(str(b) for b in bits if b)

    # -- offline fallback ---------------------------------------------------
    def offline_plan(self, case: Case) -> List[Tuple[str, Dict[str, Any]]]:
        return [("search_similar_cases", {"query": self.default_query(case), "k": 5})]

    def offline_analyze(self, case: Case, records: List[ToolCallRecord]) -> Dict[str, Any]:
        r = self.results_by_tool(records)
        search = r.get("search_similar_cases", {})
        matches: List[Dict[str, Any]] = [
            m for m in search.get("matches", []) if m.get("similarity", 0.0) >= RELEVANCE_FLOOR
        ]

        if not matches:
            return {
                "risk_score": 45.0,
                "confidence": 0.25,
                "reasoning_summary": (
                    "Offline deterministic scorer (no LLM): no past case cleared the "
                    f"{RELEVANCE_FLOOR} relevance floor, so there is no usable precedent. Reporting "
                    "a neutral score with low confidence rather than guessing."
                ),
                "key_evidence": ["no precedent above the relevance floor"],
            }

        weight_total = sum(m["similarity"] for m in matches)
        fraud_weight = sum(m["similarity"] for m in matches if m.get("resolution") in {"declined", "confirmed_fraud"})
        fraud_share = fraud_weight / weight_total if weight_total else 0.0

        top = matches[0]
        # Pull the score toward neutral in proportion to how weak the best match
        # is. A unanimous fraud precedent at 0.28 similarity is not the same
        # evidence as a unanimous fraud precedent at 0.8, and reporting both as
        # risk 100 was making this specialist the loudest voice on the panel
        # while being the least grounded one.
        strength = min(1.0, top["similarity"] / STRONG_SIMILARITY)
        risk = 50.0 + (fraud_share - 0.5) * 100.0 * strength
        outcomes = {m.get("resolution") for m in matches}
        mixed = len(outcomes) > 1

        # Confidence tracks both how similar the precedent is and whether it
        # agrees with itself. Mixed precedent is deliberately low-confidence -
        # that is the signal that pushes an ambiguous case to a human.
        confidence = min(0.9, 0.35 + top["similarity"])
        if mixed:
            confidence = min(confidence, 0.5)
        if len(matches) == 1:
            confidence = min(confidence, 0.55)

        evidence = [
            f"{m['case_id']} ({m.get('category')}) similarity {m['similarity']} -> {m.get('resolution')}"
            for m in matches[:4]
        ]

        return {
            "risk_score": max(0.0, min(100.0, risk)),
            "confidence": round(confidence, 2),
            "reasoning_summary": (
                f"Offline deterministic scorer (no LLM): {len(matches)} precedent(s) above the "
                f"relevance floor, {fraud_share:.0%} of similarity weight resolved as fraud. "
                + ("Precedent is mixed, so confidence is capped." if mixed else "Precedent is consistent.")
                + f" Retrieval backend: {search.get('retrieval_backend')}."
            ),
            "key_evidence": evidence,
        }
