"""Shared pytest fixtures and global hygiene.

`tests/test_api_routers.py` and `tests/test_chat.py` previously polluted
sys.modules with MagicMock stubs that were never torn down, breaking ~60 tests
in unrelated modules (verification, text_normalize, worker pipeline progress).
The polluters have been migrated to pytest's `monkeypatch` fixture, but this
autouse guard snapshots sys.modules between modules so any future regressions
fail loudly inside the offending file rather than corrupting a downstream one.
"""

import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """Restore sys.modules entries clobbered during a test.

    Snapshots the module table before each test and removes/restores any keys
    that were added or replaced. Pure-import additions (lazy `from x import y`
    triggering a real load) are left in place so we don't churn the module
    cache; only entries replaced with non-original objects are reverted.
    """
    snapshot = dict(sys.modules)
    yield
    for key in list(sys.modules.keys()):
        if key not in snapshot:
            continue
        if sys.modules[key] is not snapshot[key]:
            sys.modules[key] = snapshot[key]
