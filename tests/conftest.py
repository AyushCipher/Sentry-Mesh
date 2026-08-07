import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
