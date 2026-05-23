"""End-to-end subprocess-backend tests.

These tests prove the actual fix for the phase-1 cancellation
limitation: running each worker in a child Python process so the
parent can OS-terminate it. They spawn real Python subprocesses so
they're a touch slower than the thread-backend tests but they exercise
the real production code path.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from machine_spirit_4.double_agent import (
    Blackboard,
    JobEnvelope,
    JobRunner,
)
from machine_spirit_4.double_agent import safety


# Make the fake-chat-runner module importable from inside the subprocess.
@pytest.fixture(autouse=True)
def _make_tests_importable(monkeypatch):
    tests_root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", tests_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    # Also set MS4_ALLOW_UNCONTAINED_RUNTIME so the subprocess can boot under
    # an arbitrary python (the test runner's own venv).
    monkeypatch.setenv("MS4_ALLOW_UNCONTAINED_RUNTIME", "1")


def _envelope(conv="conv-sub", revision=1) -> JobEnvelope:
    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id=conv,
        conversation_revision_id=revision,
        background_lobe_type="deep_chat",
        user_visible_goal="Subprocess worker smoke test.",
        internal_goal="Smoke test the subprocess worker path.",
    )
    env.validate()
    return env


def _await_state(runner: JobRunner, job_id: str, target_states: set[str], timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = runner.get(job_id)
        if snap is not None and snap.get("state") in target_states:
            return snap
        time.sleep(0.1)
    snap = runner.get(job_id) or {}
    raise AssertionError(f"job did not reach {target_states} in {timeout}s; last={snap.get('state')}")


def test_subprocess_worker_completes_through_blackboard(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER",
        "ms4_double_agent._fake_chat_runners:quick_factory",
    )
    board = Blackboard(tmp_path / "double_agent.sqlite3")
    # No chat_runner_factory -> subprocess backend.
    runner = JobRunner(blackboard=board)
    try:
        env = _envelope()
        snap = runner.submit(env)
        assert snap["state"] in {"queued", "running", "completed"}
        final = _await_state(runner, env.job_id, {"completed"})
        assert final["state"] == "completed"
        events = board.list_events(env.job_id, limit=100)
        types = [e["type"] for e in events]
        assert "job.started" in types
        assert "job.completed" in types
        for t in types:
            assert t in safety.EVENT_TYPES, t
    finally:
        runner.shutdown(wait=False)


def test_subprocess_worker_dies_on_cancel_within_grace(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER",
        "ms4_double_agent._fake_chat_runners:sleeping_factory",
    )
    board = Blackboard(tmp_path / "double_agent.sqlite3")
    runner = JobRunner(blackboard=board, cancel_grace_seconds=2.0)
    try:
        env = _envelope()
        runner.submit(env)
        # Give the subprocess a moment to actually start the sleeping chat.
        _await_state(runner, env.job_id, {"running"}, timeout=15.0)
        t0 = time.time()
        runner.cancel(env.job_id)
        final = _await_state(runner, env.job_id, {"canceled", "failed"}, timeout=10.0)
        elapsed = time.time() - t0
        assert elapsed < 8.0, f"cancel took too long: {elapsed:.2f}s"
        assert final["state"] in {"canceled", "failed"}
        # The blackboard owns truth: regardless of canceled-vs-failed reporting,
        # there must be no lingering subprocess for this job. Watcher cleanup
        # happens in a separate thread, so allow a brief settle.
        settle_deadline = time.time() + 3.0
        while time.time() < settle_deadline and env.job_id in runner._procs:  # type: ignore[attr-defined]
            time.sleep(0.05)
        assert env.job_id not in runner._procs  # type: ignore[attr-defined]
    finally:
        runner.shutdown(wait=False)


def test_hydrate_recover_picks_up_after_parent_crash(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER",
        "ms4_double_agent._fake_chat_runners:sleeping_factory",
    )
    db = tmp_path / "double_agent.sqlite3"
    board = Blackboard(db)
    # Submit a job, then simulate a parent crash by shutting down the runner
    # while the worker subprocess is still running (we cancel-kill on shutdown,
    # so to simulate a crash we mark the row running and then forget it).
    runner = JobRunner(blackboard=board, cancel_grace_seconds=2.0)
    env = _envelope()
    try:
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"}, timeout=15.0)
    finally:
        runner.shutdown(wait=False)
    # Manually leave the blackboard row in running to simulate a crash.
    board.update_job_state(env.job_id, state="running")

    fresh_runner = JobRunner(blackboard=Blackboard(db))
    try:
        affected = fresh_runner.recover_on_startup()
        assert env.job_id in affected
        snap = fresh_runner.get(env.job_id)
        assert snap is not None
        assert snap["state"] == "failed"
        assert "did not survive restart" in (snap.get("last_safe_user_status") or "")
    finally:
        fresh_runner.shutdown(wait=False)


def test_subprocess_worker_tool_events_land_in_blackboard(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER",
        "ms4_double_agent._fake_chat_runners:tool_calling_factory",
    )
    board = Blackboard(tmp_path / "double_agent.sqlite3")
    runner = JobRunner(blackboard=board)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
        types = [e["type"] for e in board.list_events(env.job_id, limit=100)]
        assert "job.tool.call.started" in types
        assert "job.tool.call.completed" in types
    finally:
        runner.shutdown(wait=False)
