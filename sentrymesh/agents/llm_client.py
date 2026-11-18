"""LLM tool-use loop shared by every specialist and the supervisor.

Two backends, and the system is explicit about which one produced any given
result:

``anthropic``
    A real Anthropic tool-use loop.  The model chooses which investigation
    tools to call, in what order, and when it has seen enough; the loop ends
    when it calls the terminal ``submit_*`` tool that carries the structured
    output.  Nothing about the routing or the analysis is hardcoded.

``openai_compat``
    The same loop against any provider that speaks the OpenAI chat-completions
    format - Groq, Google Gemini's compatibility endpoint, GitHub Models,
    Ollama, OpenRouter, Together.  "OpenAI-compatible" describes a request
    shape, not the company: this path never calls OpenAI and needs no OpenAI
    key.  Switching providers is a change to ``SENTRYMESH_LLM_BASE_URL``, not to
    this file, which is the entire reason the adapter is written against the
    format rather than against one vendor's SDK.

    The agent contract is unchanged - the same tool schemas, the same terminal
    ``submit_finding``, the same event-log writes - so the specialists and the
    supervisor are identical code on either backend.  Only the wire translation
    in :meth:`LLMClient._to_openai_tools` and the response parsing differ.

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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from sentrymesh.config import SETTINGS, estimate_cost_usd
from sentrymesh.safety import fence_untrusted

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
    """Map an exception onto one of the reportable error classes.

    Both SDKs are checked because the harness must separate infrastructure
    failures from wrong verdicts regardless of which provider produced them - an
    unclassified error would land in the generic bucket the report exists to
    avoid.
    """
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return ERR_RATE_LIMIT
        if isinstance(exc, openai.APITimeoutError):
            return ERR_TIMEOUT
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return ERR_AUTH
        if isinstance(exc, openai.APIConnectionError):
            return ERR_CONNECTION
        if isinstance(exc, openai.InternalServerError):
            return ERR_SERVER
        if isinstance(exc, (openai.BadRequestError, openai.NotFoundError)):
            return ERR_BAD_REQUEST
        if isinstance(exc, openai.APIStatusError):
            return ERR_SERVER if getattr(exc, "status_code", 500) >= 500 else ERR_BAD_REQUEST
    except ImportError:  # pragma: no cover - optional dependency
        pass

    try:
        import anthropic
    except Exception:  # pragma: no cover - SDK always present in practice
        return ERR_TIMEOUT if isinstance(exc, asyncio.TimeoutError) else ERR_UNKNOWN

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
    cost_usd: float = 0.0
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
        base_url: Optional[str] = None,
    ) -> None:
        self.backend = backend or SETTINGS.resolved_backend()
        self.model = model or SETTINGS.model
        self.max_tokens = max_tokens or SETTINGS.max_tokens
        self.max_turns = max_turns or SETTINGS.max_agent_turns
        self.base_url = base_url or SETTINGS.llm_base_url
        if api_key is not None:
            self._api_key = api_key
        elif self.backend == "openai_compat":
            # Ollama and other local servers accept any non-empty key.
            self._api_key = SETTINGS.llm_api_key or "not-required"
        else:
            self._api_key = SETTINGS.anthropic_api_key
        self._client: Any = None

    @property
    def run_mode(self) -> str:
        if self.backend in {"anthropic", "openai_compat"}:
            return self.backend
        return "offline_deterministic"

    @property
    def is_live(self) -> bool:
        """True when a real model is producing the reasoning."""
        return self.backend in {"anthropic", "openai_compat"}

    def _anthropic(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key, max_retries=3)
        return self._client

    def _openai(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.base_url or None,
                max_retries=3,
                timeout=120.0,
            )
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
        if self.backend == "openai_compat":
            return await self._run_openai_compat(
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
                    result.cost_usd = estimate_cost_usd(self.model, result.input_tokens, result.output_tokens)
                    return result

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_uses:
                record = await self._invoke(executor, block.name, dict(block.input or {}))
                result.tool_calls.append(record)
                if on_tool_call:
                    await on_tool_call(record)
                payload = record.result if record.error is None else {"error": record.error}
                content_str = json.dumps(payload, default=str)
                if len(content_str) > 2500:
                    content_str = content_str[:2500] + "... (truncated for context efficiency)"
                content_str = fence_untrusted(content_str)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content_str,
                        "is_error": record.error is not None,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        result.error = f"exceeded {self.max_turns} turns without calling {final_tool}"
        result.error_class = ERR_PROTOCOL
        result.narration = "\n".join(narration)
        result.cost_usd = estimate_cost_usd(self.model, result.input_tokens, result.output_tokens)
        return result

    # -------------------------------------------------- openai-compatible
    @staticmethod
    def _to_openai_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate Anthropic tool schemas to the OpenAI function format.

        The only structural differences are the wrapper object and the key name
        ``input_schema`` -> ``parameters``; the JSON Schema itself is identical,
        so the agents keep one set of tool definitions for both backends.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    async def _run_openai_compat(
        self,
        *,
        system: str,
        user_prompt: str,
        tools: Sequence[Dict[str, Any]],
        executor: ToolExecutor,
        final_tool: str,
        on_tool_call: Optional[Callable[[ToolCallRecord], Awaitable[None]]],
    ) -> LLMRunResult:
        result = LLMRunResult(backend="openai_compat", model=self.model)
        client = self._openai()
        openai_tools = self._to_openai_tools(tools)
        # System is a message here rather than a separate parameter - the one
        # shape difference that matters for the conversation, not just the tools.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        narration: List[str] = []

        for turn in range(1, self.max_turns + 1):
            result.turns = turn
            tool_choice: Any = (
                {"type": "function", "function": {"name": final_tool}}
                if turn == self.max_turns
                else "auto"
            )
            try:
                response = await self._with_rate_limit_retry(
                    lambda: client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice=tool_choice,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - classified, not swallowed
                result.error = f"{type(exc).__name__}: {exc}"
                result.error_class = classify_llm_error(exc)
                # Some providers reject a malformed tool call with a 400 whose
                # body carries `tool_use_failed` - the model wrote the call as
                # text instead of emitting a structured one. That is the model
                # fumbling the protocol, not a bad request from us, and
                # classifying it as a client error would file a reasoning-
                # adjacent failure under infrastructure in the harness report.
                if "tool_use_failed" in str(exc):
                    result.error_class = ERR_PROTOCOL
                result.narration = "\n".join(narration)
                return result

            usage = getattr(response, "usage", None)
            if usage is not None:
                result.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                result.output_tokens += getattr(usage, "completion_tokens", 0) or 0

            if not response.choices:
                result.error = "provider returned no choices"
                result.error_class = ERR_PROTOCOL
                return result

            message = response.choices[0].message
            if getattr(message, "content", None):
                narration.append(str(message.content).strip())

            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                result.error = "model produced no tool call and no structured finding"
                result.error_class = ERR_PROTOCOL
                result.narration = "\n".join(narration)
                return result

            # Terminal tool ends the loop with the structured answer. Arguments
            # arrive as a JSON *string* here (Anthropic returns a parsed object),
            # and smaller models sometimes emit malformed JSON - that is a
            # protocol failure, not a wrong verdict, and is classified as one.
            for call in tool_calls:
                if call.function.name == final_tool:
                    try:
                        result.structured = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError as exc:
                        result.error = f"{final_tool} arguments were not valid JSON: {exc}"
                        result.error_class = ERR_PROTOCOL
                    result.narration = "\n".join(narration)
                    result.cost_usd = estimate_cost_usd(self.model, result.input_tokens, result.output_tokens)
                    return result

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                record = await self._invoke(executor, call.function.name, args)
                result.tool_calls.append(record)
                if on_tool_call:
                    await on_tool_call(record)
                payload = record.result if record.error is None else {"error": record.error}
                content_str = json.dumps(payload, default=str)
                if len(content_str) > 2500:
                    content_str = content_str[:2500] + "... (truncated for context efficiency)"
                content_str = fence_untrusted(content_str)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content_str,
                    }
                )

        result.error = f"exceeded {self.max_turns} turns without calling {final_tool}"
        result.error_class = ERR_PROTOCOL
        result.narration = "\n".join(narration)
        result.cost_usd = estimate_cost_usd(self.model, result.input_tokens, result.output_tokens)
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
    async def _with_rate_limit_retry(self, call: Callable[[], Awaitable[Any]]) -> Any:
        """Retry a provider call across a per-minute rate-limit window.

        The SDKs already retry 429s, but with sub-second backoff aimed at
        per-second limits. Free tiers meter *tokens per minute*, and their 429
        bodies say things like "try again in 12.4s" - a budget the SDK's backoff
        never waits out. Honour the provider's own retry hint when it gives one.
        """
        delay = 5.0
        last: BaseException | None = None
        for attempt in range(SETTINGS.rate_limit_retries + 1):
            try:
                return await call()
            except Exception as exc:  # noqa: BLE001
                if classify_llm_error(exc) != ERR_RATE_LIMIT or attempt == SETTINGS.rate_limit_retries:
                    raise
                last = exc
                await asyncio.sleep(min(_retry_after_seconds(exc) or delay, 30.0))
                delay *= 2
        raise last  # type: ignore[misc]  # unreachable: loop either returns or raises

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


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Pull the provider's own wait hint out of a 429, if it offered one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            raw = headers.get(key)
            if raw:
                try:
                    return float(str(raw).rstrip("s"))
                except ValueError:
                    continue
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    return float(match.group(1)) + 1.0 if match else None


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
