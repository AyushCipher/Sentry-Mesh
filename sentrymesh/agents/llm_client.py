"""LLM tool-use loop shared by every specialist and the supervisor.

Two backends, and the system is explicit about which one produced any given
result:

``anthropic``
    A real Anthropic tool-use loop.  The model chooses which investigation
    tools to call, in what order, and when it has seen enough; the loop ends
    when it calls the terminal ``submit_*`` tool that carries the structured
    output.  Nothing about the routing or the analysis is hardcoded.

``offline_deterministic``
    Used only when no ``ANTHROPIC_API_KEY`` is configured.  It still executes
    every one of the agent's real tools against the real case data, and still
    writes every tool call to the event log - but the *judgement* comes from
    the agent's own explicit scoring function instead of a model.  This is a
    labelled fallback so the harness can run on a machine with no API key.  It
    is recorded in every event payload, in every finding, and in the harness
    report as ``run_mode="offline_deterministic"``, because eval numbers
    produced this way are numbers about the scoring functions, not about an
    LLM.  Never present offline numbers as agent-reasoning numbers.

Error classification lives here on purpose: the eval harness is required to
distinguish "the API rate-limited us" from "the agent reached a wrong verdict",
and it can only do that if the transport layer labels its own failures.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from sentrymesh.config import SETTINGS

ToolExecutor = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]
OfflineSolver = Callable[[List["ToolCallRecord"]], Dict[str, Any]]
OfflinePlanner = Callable[[], List[Tuple[str, Dict[str, Any]]]]

# Error classes the harness reports separately.  An infrastructure failure and a
# wrong-but-working agent are different findings and must not share a bucket.
ERR_RATE_LIMIT = "llm_rate_limit"
ERR_TIMEOUT = "llm_timeout"
ERR_AUTH = "llm_auth"
ERR_CONNECTION = "llm_connection"
ERR_SERVER = "llm_server_error"
ERR_BAD_REQUEST = "llm_bad_request"
ERR_PROTOCOL = "agent_protocol_error"  # model never produced the structured output
ERR_TOOL = "tool_execution_error"
ERR_UNKNOWN = "unknown_error"

INFRASTRUCTURE_ERRORS = frozenset(
    {ERR_RATE_LIMIT, ERR_TIMEOUT, ERR_AUTH, ERR_CONNECTION, ERR_SERVER, ERR_BAD_REQUEST}
)


def classify_llm_error(exc: BaseException) -> str:
    """Map an exception onto one of the reportable error classes."""
    try:
        import anthropic
    except Exception:  # pragma: no cover - SDK always present in practice
        return ERR_UNKNOWN

    if isinstance(exc, anthropic.RateLimitError):
        return ERR_RATE_LIMIT
    if isinstance(exc, anthropic.APITimeoutError) or isinstance(exc, asyncio.TimeoutError):
        return ERR_TIMEOUT
    if isinstance(exc, anthropic.AuthenticationError):
        return ERR_AUTH
    if isinstance(exc, anthropic.APIConnectionError):
        return ERR_CONNECTION
    if isinstance(exc, anthropic.InternalServerError):
        return ERR_SERVER
    if isinstance(exc, anthropic.BadRequestError):
        return ERR_BAD_REQUEST
    if isinstance(exc, anthropic.APIStatusError):
        return ERR_SERVER if getattr(exc, "status_code", 500) >= 500 else ERR_BAD_REQUEST
    return ERR_UNKNOWN


@dataclass
class ToolCallRecord:
    """One tool invocation, as it is written to the event log."""

    tool: str
    arguments: Dict[str, Any]
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class LLMRunResult:
    structured: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    turns: int = 0
    backend: str = ""
    model: str = ""
    narration: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    error_class: Optional[str] = None


class LLMClient:
    """Runs a bounded tool-use loop and returns the terminal structured output."""

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.backend = backend or SETTINGS.resolved_backend()
        self.model = model or SETTINGS.model
        self.max_tokens = max_tokens or SETTINGS.max_tokens
        self.max_turns = max_turns or SETTINGS.max_agent_turns
        self._api_key = api_key or SETTINGS.anthropic_api_key
        self._client: Any = None

    @property
    def run_mode(self) -> str:
        return "anthropic" if self.backend == "anthropic" else "offline_deterministic"

    def _anthropic(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key, max_retries=3)
        return self._client

    async def run_tool_loop(
        self,
        *,
        system: str,
        user_prompt: str,
        tools: Sequence[Dict[str, Any]],
        executor: ToolExecutor,
        final_tool: str,
        offline_plan: OfflinePlanner,
        offline_solver: OfflineSolver,
        on_tool_call: Optional[Callable[[ToolCallRecord], Awaitable[None]]] = None,
    ) -> LLMRunResult:
        """Drive the agent to a structured answer.

        ``final_tool`` is the tool whose arguments *are* the structured answer;
        calling it terminates the loop.  Forcing structure through a tool schema
        rather than parsing free text is what makes the finding shape reliable
        enough for the supervisor to aggregate.
        """
        if self.backend == "anthropic":
            return await self._run_anthropic(
                system=system,
                user_prompt=user_prompt,
                tools=tools,
                executor=executor,
                final_tool=final_tool,
                on_tool_call=on_tool_call,
            )
        return await self._run_offline(
            executor=executor,
            offline_plan=offline_plan,
            offline_solver=offline_solver,
            on_tool_call=on_tool_call,
        )

    # ------------------------------------------------------------------ real
    async def _run_anthropic(
        self,
        *,
        system: str,
        user_prompt: str,
        tools: Sequence[Dict[str, Any]],
        executor: ToolExecutor,
        final_tool: str,
        on_tool_call: Optional[Callable[[ToolCallRecord], Awaitable[None]]],
    ) -> LLMRunResult:
        result = LLMRunResult(backend="anthropic", model=self.model)
        client = self._anthropic()
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        narration: List[str] = []

        for turn in range(1, self.max_turns + 1):
            result.turns = turn
            # On the last permitted turn, force the terminal tool so the loop
            # cannot end without a structured answer just because the model
            # wanted one more look at the data.
            tool_choice: Dict[str, Any] = (
                {"type": "tool", "name": final_tool} if turn == self.max_turns else {"type": "auto"}
            )
            try:
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=messages,
                    tools=list(tools),
                    tool_choice=tool_choice,
                )
            except Exception as exc:  # noqa: BLE001 - classified, not swallowed
                result.error = f"{type(exc).__name__}: {exc}"
                result.error_class = classify_llm_error(exc)
                return result

            usage = getattr(response, "usage", None)
            if usage is not None:
                result.input_tokens += getattr(usage, "input_tokens", 0) or 0
                result.output_tokens += getattr(usage, "output_tokens", 0) or 0

            tool_uses = []
            for block in response.content:
                if getattr(block, "type", None) == "text" and block.text.strip():
                    narration.append(block.text.strip())
                elif getattr(block, "type", None) == "tool_use":
                    tool_uses.append(block)

            if not tool_uses:
                result.error = "model produced no tool call and no structured finding"
                result.error_class = ERR_PROTOCOL
                result.narration = "\n".join(narration)
                return result

            # Terminal tool ends the loop with the structured answer.
            for block in tool_uses:
                if block.name == final_tool:
                    result.structured = dict(block.input or {})
                    result.narration = "\n".join(narration)
                    return result

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_uses:
                record = await self._invoke(executor, block.name, dict(block.input or {}))
                result.tool_calls.append(record)
                if on_tool_call:
                    await on_tool_call(record)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(record.result if record.error is None else {"error": record.error}, default=str),
                        "is_error": record.error is not None,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        result.error = f"exceeded {self.max_turns} turns without calling {final_tool}"
        result.error_class = ERR_PROTOCOL
        result.narration = "\n".join(narration)
        return result

    # --------------------------------------------------------------- offline
    async def _run_offline(
        self,
        *,
        executor: ToolExecutor,
        offline_plan: OfflinePlanner,
        offline_solver: OfflineSolver,
        on_tool_call: Optional[Callable[[ToolCallRecord], Awaitable[None]]],
    ) -> LLMRunResult:
        result = LLMRunResult(backend="offline_deterministic", model="none")
        for tool_name, args in offline_plan():
            record = await self._invoke(executor, tool_name, args)
            result.tool_calls.append(record)
            if on_tool_call:
                await on_tool_call(record)
        result.turns = 1
        try:
            result.structured = offline_solver(result.tool_calls)
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            result.error_class = ERR_UNKNOWN
        return result

    # ----------------------------------------------------------------- utils
    @staticmethod
    async def _invoke(executor: ToolExecutor, name: str, args: Dict[str, Any]) -> ToolCallRecord:
        started = time.perf_counter()
        record = ToolCallRecord(tool=name, arguments=args)
        try:
            record.result = await executor(name, args)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            record.error = f"{type(exc).__name__}: {exc}"
        record.duration_ms = int((time.perf_counter() - started) * 1000)
        return record


def tool(name: str, description: str, properties: Dict[str, Any], required: Sequence[str] = ()) -> Dict[str, Any]:
    """Build an Anthropic tool schema without repeating boilerplate."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(required),
        },
    }


FINDING_TOOL = tool(
    "submit_finding",
    "Submit your final structured finding for this case. Call this exactly once, "
    "after you have gathered enough evidence from the investigation tools.",
    {
        "risk_score": {
            "type": "number",
            "description": "0-100 risk from YOUR signal category only. Do not speculate about "
            "signals another specialist owns.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 confidence in your own risk score. Report low confidence when "
            "the evidence in your category is thin or contradictory - low confidence is useful "
            "information to the supervisor, not a failure.",
        },
        "reasoning_summary": {
            "type": "string",
            "description": "2-4 sentences explaining how the evidence produced this score.",
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific data points that drove the score, quoted from tool results.",
        },
    },
    required=["risk_score", "confidence", "reasoning_summary", "key_evidence"],
)
