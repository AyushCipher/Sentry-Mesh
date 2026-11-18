"""Shared test fixtures.

The environment is pinned BEFORE any sentrymesh import, because
``sentrymesh.config.SETTINGS`` is built at import time and would otherwise pick
up whatever is in the developer's `.env`.

Pinning the offline backend here is not a convenience - it is what keeps the
suite a test of *this code*. With a real provider configured, these tests call
a live API: they become slow (the suite went from 3 seconds to 91 minutes),
they cost money or burn a rate-limit budget, they fail on network problems that
have nothing to do with the change under test, and their assertions about
specific verdicts turn into assertions about what a model felt like saying that
afternoon. The eval harness is where a real provider belongs; the unit and API
tests deliberately run against the deterministic reasoner.
"""

import asyncio
import os
import sys
from pathlib import Path

# Must precede the sentrymesh import below - order is load-bearing.
os.environ["SENTRYMESH_LLM_BACKEND"] = "offline"
# Redis is fail-closed by default; the unit suite runs in-memory, so it is a dev run.
os.environ["SENTRYMESH_DEV"] = "1"
os.environ.pop("SENTRYMESH_REQUIRE_REDIS", None)
os.environ.pop("SENTRYMESH_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("SENTRYMESH_LLM_BASE_URL", None)
os.environ.pop("SENTRYMESH_LLM_API_KEY", None)

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _assert_offline_backend():
    """Fail loudly if anything re-enables a live backend mid-suite."""
    from sentrymesh.config import SETTINGS

    assert SETTINGS.resolved_backend() == "offline", (
        "tests must run on the offline deterministic reasoner; "
        f"got {SETTINGS.resolved_backend()!r}"
    )
    yield


@pytest.fixture
def run_async():
    """Run a coroutine to completion.

    The suite avoids pytest-asyncio's event-loop fixtures deliberately: the
    system under test creates its own connections per run, and a shared loop
    across tests hides ordering bugs that show up in the real per-request loop.
    """

    def _run(coro):
        return asyncio.run(coro)

    return _run
