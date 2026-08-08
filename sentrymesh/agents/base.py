"""Base class for the four specialist agents.

Every specialist is an independent LLM tool-use loop over one signal category.
The contract each subclass implements is small on purpose:

* ``system_prompt``   - the analyst's brief
* ``investigation_tools`` - the Anthropic tool schemas it may call
* ``run_tool``        - real execution of those tools against the case data
* ``offline_plan`` / ``offline_analyze`` - the labelled no-API-key fallback

Everything else - the loop, the event-log writes, timing, error classification
and finding validation - lives here so the four specialists cannot drift apart
in how they record their work.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

from sentrymesh.agents.llm_client import (
    FINDING_TOOL,
    LLMClient,
    ToolCallRecord,
)
from sentrymesh.event_log import EventLog
from sentrymesh.models import Case, Finding


def clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


class SpecialistAgent(ABC):
    """One independent analyst over one category of fraud signal."""

    name: str = "specialist"
    focus: str = ""

    def __init__(self, log: EventLog, llm: LLMClient) -> None:
        self.log = log
        self.llm = llm

    # -- subclass contract --------------------------------------------------
    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def investigation_tools(self) -> Sequence[Dict[str, Any]]: ...

    @abstractmethod
    async def run_tool(self, case: Case, name: str, args: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def offline_plan(self, case: Case) -> List[Tuple[str, Dict[str, Any]]]: ...

    @abstractmethod
    def offline_analyze(self, case: Case, records: List[ToolCallRecord]) -> Dict[str, Any]: ...

    # -- shared machinery ---------------------------------------------------
    def user_prompt(self, case: Case) -> str:
        """The opening brief - deliberately WITHOUT the case data.

        An earlier version pasted the whole case JSON in here alongside the
        tools. Every specialist then answered from the prompt and called zero
        tools: the tools were decorative, and the "agents investigate" claim was
        not true of a single run. Handing over only the case identity forces the
        evidence to come through the tools, which is also what makes each
        specialist's trace an actual record of what it looked at.

        It cuts the prompt from thousands of tokens to a few dozen as a side
        effect, which is what keeps four parallel specialists inside a
        free-tier per-minute token budget.
        """
        return (
            f"Investigate flagged case {case.case_id} (account {case.account_id}).\n"
            f"Analyst brief: {case.summary or 'no summary provided'}\n"
            f"Your remit is: {self.focus}\n\n"
            "You have NOT been given the case data. Retrieve what you need with your "
            "investigation tools, then call submit_finding exactly once.\n\n"
            "Score only your own signal category. If your category shows nothing unusual, "
            "say so with a low risk score - a clean finding is as valuable as a dirty one, "
            "and inflating a weak signal costs a legitimate customer their transaction. "
            "All data is SYNTHETIC evaluation data."
        )

    async def run(self, case: Case) -> Finding:
        """Investigate the case and return a structured finding.

        Every tool call and the finding itself are appended to the shared event
        log before this returns, so the specialist holds no state the audit
        trail does not have.
        """
        started = time.perf_counter()

        async def executor(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            return await self.run_tool(case, name, args)

        async def on_tool_call(record: ToolCallRecord) -> None:
            await self.log.append(
                case.case_id,
                self.name,
                "tool_call",
                {**record.to_dict(), "run_mode": self.llm.run_mode},
            )

        result = await self.llm.run_tool_loop(
            system=self.system_prompt,
            user_prompt=self.user_prompt(case),
            tools=[*self.investigation_tools, FINDING_TOOL],
            executor=executor,
            final_tool="submit_finding",
            offline_plan=lambda: self.offline_plan(case),
            offline_solver=lambda records: self.offline_analyze(case, records),
            on_tool_call=on_tool_call,
        )

        duration_ms = int((time.perf_counter() - started) * 1000)

        if result.error:
            finding = Finding.failed(self.name, result.error, result.error_class or "unknown_error", result.backend)
            finding.duration_ms = duration_ms
            finding.tool_calls = len(result.tool_calls)
            await self.log.append(
                case.case_id,
                self.name,
                "error",
                {
                    "error": result.error,
                    "error_class": result.error_class,
                    "run_mode": self.llm.run_mode,
                    "tool_calls_before_failure": len(result.tool_calls),
                },
            )
            return finding

        finding = self._to_finding(result.structured, result, duration_ms)
        await self.log.append(
            case.case_id,
            self.name,
            "finding",
            {
                **finding.to_dict(),
                "run_mode": self.llm.run_mode,
                "turns": result.turns,
                "tools_used": [r.tool for r in result.tool_calls],
                "model_narration": result.narration[:1200],
            },
        )
        return finding

    def _to_finding(self, structured: Dict[str, Any], result: Any, duration_ms: int) -> Finding:
        evidence = structured.get("key_evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        return Finding(
            specialist=self.name,
            risk_score=clamp(structured.get("risk_score"), 0.0, 100.0, 50.0),
            confidence=clamp(structured.get("confidence"), 0.0, 1.0, 0.3),
            reasoning_summary=str(structured.get("reasoning_summary") or "").strip() or "(no summary returned)",
            key_evidence=[str(e) for e in evidence][:10],
            tool_calls=len(result.tool_calls),
            duration_ms=duration_ms,
            backend=result.backend,
        )

    # -- offline helper -----------------------------------------------------
    @staticmethod
    def results_by_tool(records: List[ToolCallRecord]) -> Dict[str, Dict[str, Any]]:
        return {r.tool: (r.result or {}) for r in records if r.error is None}
