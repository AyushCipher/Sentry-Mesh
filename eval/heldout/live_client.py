"""Resilient wrapper around the OpenAI-compatible transport used for live runs.

Adds, WITHOUT touching sentrymesh/ itself:

* on-disk response cache (append-only JSONL; a re-run or a crash resumes for free),
* exponential backoff with jitter on 429 / 5xx / timeouts (honours Retry-After),
* a global concurrency limit on in-flight provider calls,
* a hard cap on live calls and on estimated spend (raises BudgetExceeded),
* passthrough of Gemini ``extra_content`` (thought signatures) across tool turns.

The wrapper is a drop-in for ``openai.AsyncOpenAI`` as far as LLMClient is
concerned (only ``client.chat.completions.create`` is used).  The cache key
includes a ``repeat`` tag so that repeat runs are independent samples rather
than cache replays.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import openai
from openai.types.chat import ChatCompletion

from sentrymesh.agents.llm_client import LLMClient
from sentrymesh.config import SETTINGS


class BudgetExceeded(Exception):
    """Hard call or spend cap reached. Never retried."""


class DailyQuotaExhausted(BudgetExceeded):
    """Provider says the per-DAY quota is gone (retry hint is hours). Waiting out backoff cannot help."""


# ASSUMED paid-tier-equivalent pricing (USD per million tokens) for cost
# estimation, a guess for a "flash-lite"-class model. The key in use is on a
# provider FREE tier (quota errors say so), so actual billed spend is $0; this
# number only exists to enforce a spend cap as if it were billed.
ASSUMED_PRICE_IN = 0.10
ASSUMED_PRICE_OUT = 0.40


class Budget:
    def __init__(self, max_calls: int, max_cost_usd: float) -> None:
        self.max_calls, self.max_cost = max_calls, max_cost_usd
        self.live_calls = 0
        self.cache_hits = 0
        self.retries = 0
        self.failed_calls = 0
        self.in_tokens = 0
        self.out_tokens = 0  # includes hidden reasoning tokens (total - prompt)

    @property
    def cost(self) -> float:
        return (self.in_tokens * ASSUMED_PRICE_IN + self.out_tokens * ASSUMED_PRICE_OUT) / 1e6

    def check(self) -> None:
        if self.live_calls >= self.max_calls:
            raise BudgetExceeded(f"live call cap {self.max_calls} reached")
        if self.cost >= self.max_cost:
            raise BudgetExceeded(f"estimated spend cap ${self.max_cost:.2f} reached (${self.cost:.3f})")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "live_calls": self.live_calls, "cache_hits": self.cache_hits, "retries": self.retries,
            "failed_calls": self.failed_calls, "input_tokens": self.in_tokens, "output_tokens_incl_reasoning": self.out_tokens,
            "estimated_cost_usd": round(self.cost, 4),
            "assumed_price_per_million": {"input": ASSUMED_PRICE_IN, "output": ASSUMED_PRICE_OUT},
            "caps": {"max_calls": self.max_calls, "max_cost_usd": self.max_cost},
        }


class _Completions:
    def __init__(self, outer: "ResilientClient") -> None:
        self.o = outer

    async def create(self, **kw: Any) -> Any:
        return await self.o._create(**kw)


class ResilientClient:
    def __init__(self, *, cache_path: Path, budget: Budget, concurrency: int, repeat_tag: str,
                 max_retries: int = 8, max_backoff: float = 60.0) -> None:
        self._sdk = openai.AsyncOpenAI(api_key=SETTINGS.llm_api_key or "x", base_url=SETTINGS.llm_base_url or None,
                                       max_retries=0, timeout=180.0)
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.budget, self.tag = budget, repeat_tag
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries, self.max_backoff = max_retries, max_backoff
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                        self.cache[row["k"]] = row["v"]
                    except Exception:  # noqa: BLE001 - a torn last line is fine
                        pass
        self._extra: Dict[str, Any] = {}  # tool_call id -> extra_content
        self._wlock = asyncio.Lock()
        self.errors: Dict[str, int] = {}

    def _key(self, kw: Dict[str, Any]) -> str:
        blob = json.dumps({"tag": self.tag, "model": kw.get("model"), "messages": kw.get("messages"), "tools": kw.get("tools"),
                           "tool_choice": kw.get("tool_choice"), "max_tokens": kw.get("max_tokens")}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _inject_extra(self, messages: Any) -> Any:
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                m = {**m, "tool_calls": [({**c, "extra_content": self._extra[c["id"]]} if c.get("id") in self._extra else c)
                                          for c in m["tool_calls"]]}
            out.append(m)
        return out

    def _remember_extra(self, resp: ChatCompletion) -> None:
        for ch in resp.choices or []:
            for tc in getattr(ch.message, "tool_calls", None) or []:
                extra = getattr(tc, "extra_content", None) or (getattr(tc, "model_extra", None) or {}).get("extra_content")
                if extra:
                    self._extra[tc.id] = extra

    async def _create(self, **kw: Any) -> ChatCompletion:
        key = self._key(kw)
        hit = self.cache.get(key)
        if hit is not None:
            self.budget.cache_hits += 1
            resp = ChatCompletion.model_validate(hit)
            self._remember_extra(resp)
            return resp
        send = {**kw, "messages": self._inject_extra(kw["messages"])}
        attempt = 0
        while True:
            self.budget.check()
            try:
                async with self.sem:
                    self.budget.check()
                    self.budget.live_calls += 1
                    resp = await self._sdk.chat.completions.create(**send)
                break
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
                self.budget.failed_calls += 1
                name = type(exc).__name__
                if isinstance(exc, openai.RateLimitError) and (
                        "PerDay" in str(exc) or "per day" in str(exc) or (_retry_hint(exc) or 0) > 600):
                    self.errors["daily_quota_exhausted"] = self.errors.get("daily_quota_exhausted", 0) + 1
                    raise DailyQuotaExhausted("provider per-day quota exhausted") from exc
                self.errors[name] = self.errors.get(name, 0) + 1
                attempt += 1
                if attempt > self.max_retries:
                    raise
                self.budget.retries += 1
                hint = _retry_hint(exc)
                wait = min(self.max_backoff, max(hint or 0.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0, 1.5))
                await asyncio.sleep(wait)
            except openai.APIStatusError as exc:
                self.budget.failed_calls += 1
                self.errors[f"{type(exc).__name__}:{exc.status_code}"] = self.errors.get(f"{type(exc).__name__}:{exc.status_code}", 0) + 1
                raise
        usage = getattr(resp, "usage", None)
        if usage is not None:
            p = getattr(usage, "prompt_tokens", 0) or 0
            t = getattr(usage, "total_tokens", 0) or 0
            c = getattr(usage, "completion_tokens", 0) or 0
            self.budget.in_tokens += p
            self.budget.out_tokens += max(c, t - p)
        self._remember_extra(resp)
        dumped = resp.model_dump(mode="json")
        self.cache[key] = dumped
        async with self._wlock:
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": key, "v": dumped}) + "\n")
        return resp


def _retry_hint(exc: BaseException) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        raw = headers.get("retry-after")
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    m = re.search(r"retry in ([\d.]+)s", str(exc)) or re.search(r"try again in ([\d.]+)s", str(exc))
    return float(m.group(1)) + 1.0 if m else None


class ResilientLLMClient(LLMClient):
    """LLMClient whose openai_compat transport is a ResilientClient."""

    def __init__(self, resilient: ResilientClient, **kw: Any) -> None:
        super().__init__(backend="openai_compat", **kw)
        self._client = resilient

    async def _with_rate_limit_retry(self, call):  # the resilient client already retries
        return await call()
