"""Specialist 3 - network / ring analysis.

Signal category: accounts converging on shared identifiers (devices, payment
instruments, IP ranges).  Builds a real NetworkX graph from the case's edge
list and measures the structure around the account under investigation.

Ring detection here is structural, not a lookup: the same edge list that
produces a benign "two accounts share a household IP" shape also produces a
"nine accounts funnel through one device and one BIN" shape, and the scoring
distinguishes them by component size, density and identifier diversity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx

from sentrymesh.agents.base import SpecialistAgent
from sentrymesh.agents.llm_client import ToolCallRecord, tool
from sentrymesh.models import Case

# Identifier types weighted by how much sharing them actually implies collusion.
# A shared residential IP range is common (families, offices, carrier NAT);
# a shared device fingerprint or payment instrument is much less innocent.
IDENTIFIER_WEIGHTS = {
    "device": 1.0,
    "payment_method": 0.9,
    "email_pattern": 0.7,
    "phone": 0.7,
    "ip_range": 0.45,
}


class NetworkAnalysisAgent(SpecialistAgent):
    name = "network_analysis"
    focus = "shared identifiers between accounts and fraud-ring structure around the focus account"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a fraud-network analyst. You examine one signal category only: the graph of "
            "accounts linked to this one by shared identifiers.\n\n"
            "How to weigh what you find:\n"
            "- Sharing an IP range is weak evidence. Households, offices and carrier NAT produce "
            "this constantly. Two accounts on one residential range is normal.\n"
            "- Sharing a device fingerprint or a payment instrument across several unrelated "
            "accounts is much stronger, and several identifier *types* overlapping at once is "
            "stronger still.\n"
            "- Size matters, but so does shape: a dense cluster where many accounts converge on a "
            "few identifiers is a ring. A long thin chain usually is not.\n"
            "- An isolated account with no links is a genuine clean signal. Report it as low risk "
            "with high confidence rather than as 'no evidence'.\n\n"
            "You do not see transaction amounts, KYC status or precedent; other specialists own "
            "those. Report the structure you actually measured."
        )

    @property
    def investigation_tools(self) -> Sequence[Dict[str, Any]]:
        return [
            tool(
                "get_shared_identifiers",
                "List the identifiers (devices, payment methods, IP ranges) that the focus account "
                "shares with other accounts, and which accounts share each one.",
                {
                    "identifier_type": {
                        "type": "string",
                        "description": "Optional filter: device | payment_method | ip_range | email_pattern | phone.",
                    }
                },
                required=[],
            ),
            tool(
                "compute_ring_score",
                "Build the shared-identifier graph and measure the structure around the focus "
                "account: connected-component size, density, degree, identifier-type diversity and "
                "a 0-100 structural ring score.",
                {
                    "max_hops": {
                        "type": "integer",
                        "description": "How many hops out from the focus account to include (default 2).",
                    }
                },
                required=[],
            ),
        ]

    # -- graph construction -------------------------------------------------
    def _graph(self, case: Case) -> nx.Graph:
        graph = nx.Graph()
        focus = case.network_data.get("focus_account") or case.account_id
        graph.add_node(focus, focus=True)
        for edge in case.network_data.get("edges", []):
            a, b = edge.get("account_a"), edge.get("account_b")
            if not a or not b:
                continue
            kind = edge.get("shared", "unknown")
            if graph.has_edge(a, b):
                graph[a][b]["shared"].add(kind)
                graph[a][b]["identifiers"].add(str(edge.get("identifier", "")))
            else:
                graph.add_edge(a, b, shared={kind}, identifiers={str(edge.get("identifier", ""))})
        return graph

    async def run_tool(self, case: Case, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_shared_identifiers":
            return self._shared(case, args.get("identifier_type"))
        if name == "compute_ring_score":
            return self._ring(case, int(args.get("max_hops") or 2))
        raise ValueError(f"{self.name}: unknown tool {name!r}")

    def _shared(self, case: Case, identifier_type: Any) -> Dict[str, Any]:
        focus = case.network_data.get("focus_account") or case.account_id
        rows: List[Dict[str, Any]] = []
        for edge in case.network_data.get("edges", []):
            kind = edge.get("shared")
            if identifier_type and kind != identifier_type:
                continue
            if focus not in (edge.get("account_a"), edge.get("account_b")):
                continue
            other = edge.get("account_b") if edge.get("account_a") == focus else edge.get("account_a")
            rows.append(
                {
                    "identifier_type": kind,
                    "identifier": edge.get("identifier"),
                    "shared_with_account": other,
                    "weight": IDENTIFIER_WEIGHTS.get(str(kind), 0.5),
                    "first_seen_together": edge.get("first_seen_together"),
                }
            )
        by_type: Dict[str, int] = {}
        for row in rows:
            by_type[str(row["identifier_type"])] = by_type.get(str(row["identifier_type"]), 0) + 1
        return {
            "focus_account": focus,
            "direct_links": len(rows),
            "distinct_linked_accounts": len({r["shared_with_account"] for r in rows}),
            "links_by_identifier_type": by_type,
            "links": rows[:60],
            "note": "Synthetic evaluation data.",
        }

    def _ring(self, case: Case, max_hops: int) -> Dict[str, Any]:
        graph = self._graph(case)
        focus = case.network_data.get("focus_account") or case.account_id
        if focus not in graph:
            return {"error": f"focus account {focus} not present in the graph"}

        reachable = nx.single_source_shortest_path_length(graph, focus, cutoff=max_hops)
        sub = graph.subgraph(reachable.keys())
        size = sub.number_of_nodes()
        edges = sub.number_of_edges()
        density = round(nx.density(sub), 3) if size > 1 else 0.0
        degree = sub.degree(focus)

        types: set = set()
        identifiers: set = set()
        for _, _, data in sub.edges(data=True):
            types |= set(data.get("shared", set()))
            identifiers |= {i for i in data.get("identifiers", set()) if i}

        weighted = sum(IDENTIFIER_WEIGHTS.get(t, 0.5) for t in types)
        # Convergence: many accounts funnelling through few identifiers is the
        # ring shape.  Few accounts on many identifiers is just a busy user.
        convergence = round(size / max(len(identifiers), 1), 2)

        score = 0.0
        if size >= 8:
            score += 32
        elif size >= 5:
            score += 22
        elif size >= 3:
            score += 12
        elif size == 2:
            score += 4

        score += min(24.0, weighted * 12)
        if len(types) >= 3:
            score += 14
        elif len(types) >= 2:
            score += 8
        if density >= 0.5 and size >= 4:
            score += 12
        if convergence >= 2.0 and size >= 4:
            score += 14
        if degree >= 5:
            score += 8

        # A component whose only overlap is an IP range should not look like a
        # ring no matter how many accounts a carrier NAT sweeps in.
        if types and types <= {"ip_range"}:
            score = min(score, 25.0)

        try:
            cycles = len(nx.cycle_basis(sub))
        except Exception:  # noqa: BLE001 - defensive; cycle_basis is total on simple graphs
            cycles = 0

        return {
            "focus_account": focus,
            "max_hops": max_hops,
            "component_size": size,
            "component_edges": edges,
            "density": density,
            "focus_degree": degree,
            "identifier_types": sorted(types),
            "distinct_identifiers": len(identifiers),
            "convergence_ratio": convergence,
            "independent_cycles": cycles,
            "ring_score": round(max(0.0, min(100.0, score)), 1),
            "linked_accounts": sorted(n for n in sub.nodes if n != focus)[:40],
        }

    # -- offline fallback ---------------------------------------------------
    def offline_plan(self, case: Case) -> List[Tuple[str, Dict[str, Any]]]:
        return [("get_shared_identifiers", {}), ("compute_ring_score", {"max_hops": 2})]

    def offline_analyze(self, case: Case, records: List[ToolCallRecord]) -> Dict[str, Any]:
        r = self.results_by_tool(records)
        shared = r.get("get_shared_identifiers", {})
        ring = r.get("compute_ring_score", {})

        if ring.get("error"):
            return {
                "risk_score": 0.0,
                "confidence": 0.2,
                "reasoning_summary": f"Offline deterministic scorer: {ring['error']}",
                "key_evidence": [],
            }

        risk = float(ring.get("ring_score", 0.0))
        size = int(ring.get("component_size", 1))
        types = ring.get("identifier_types", [])

        evidence: List[str] = []
        if size > 1:
            evidence.append(f"{size}-account component, density {ring.get('density')}, types {types}")
        if float(ring.get("convergence_ratio", 0)) >= 2.0 and size >= 4:
            evidence.append(
                f"{size} accounts converging on {ring.get('distinct_identifiers')} identifiers "
                f"(convergence {ring.get('convergence_ratio')})"
            )
        if int(shared.get("distinct_linked_accounts", 0)) >= 3:
            evidence.append(f"focus account directly linked to {shared['distinct_linked_accounts']} accounts")
        if not evidence:
            evidence.append("no shared identifiers with any other account")

        if size <= 1:
            confidence = 0.86
        elif len(types) >= 2:
            confidence = 0.83
        else:
            confidence = 0.6

        return {
            "risk_score": max(0.0, min(100.0, risk)),
            "confidence": confidence,
            "reasoning_summary": (
                "Offline deterministic scorer (no LLM): structural ring score "
                f"{ring.get('ring_score')} from a {size}-account component spanning identifier types "
                f"{types}, density {ring.get('density')}, convergence {ring.get('convergence_ratio')}."
            ),
            "key_evidence": evidence,
        }
