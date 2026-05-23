"""Workspace-wide pytest fixtures.

Isolation goals:

* The MS4 Double Agent module exposes module-level singletons
  (``default_blackboard`` and ``default_runner``) that point at the
  contained ``machine_spirit_4/runtime/double_agent.sqlite3``. If a
  test happens to call into ``Ms4HermesRunner.chat`` (or anything else
  that touches the Face Lobe wiring), it would otherwise write into
  the production database. Redirect both singletons to a tmp path for
  every test.
* The MS3 ``contained.py`` guard raises ``SystemExit`` if the active
  interpreter is not the MS4 ``.venv``. Pytest may import the gateway
  modules under coverage from a different interpreter; we already set
  ``MS4_ALLOW_UNCONTAINED_RUNTIME=1`` via the ``_called_from_pytest``
  bypass inside the guard, but doing it again here makes the intent
  explicit and survives plugin reorderings.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_double_agent_blackboard(tmp_path, monkeypatch):
    os.environ.setdefault("MS4_ALLOW_UNCONTAINED_RUNTIME", "1")
    from machine_spirit_4.double_agent import (
        Blackboard,
        JobRunner,
        _reset_default_blackboard_for_tests,
        _reset_default_runner_for_tests,
    )

    isolated_db = tmp_path / "double_agent.sqlite3"
    isolated_board = Blackboard(isolated_db)
    _reset_default_blackboard_for_tests(isolated_db)
    isolated_runner = JobRunner(
        blackboard=isolated_board,
        chat_runner_factory=lambda: (lambda **_: {"text": "fake"}),
    )
    _reset_default_runner_for_tests(isolated_runner)
    try:
        yield
    finally:
        try:
            isolated_runner.shutdown(wait=False)
        finally:
            _reset_default_runner_for_tests(None)
            _reset_default_blackboard_for_tests(None)
