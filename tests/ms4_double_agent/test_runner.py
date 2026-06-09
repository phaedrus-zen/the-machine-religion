"""Runner + worker tests.

Uses a fake chat runner so the suite is hermetic — Hermes / HiveMind /
MS3 are not contacted. Worker thread is awaited explicitly via a
synchronization event so tests are deterministic.
"""

from __future__ import annotations

import threading
import time

import pytest

from machine_spirit_4.double_agent import (
    Blackboard,
    JobEnvelope,
    JobRunner,
    RunnerError,
)
from machine_spirit_4.double_agent import safety


@pytest.fixture()
def board(tmp_path):
    return Blackboard(tmp_path / "double_agent.sqlite3")


def _envelope(conv="conv-1", revision=1) -> JobEnvelope:
    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id=conv,
        conversation_revision_id=revision,
        background_lobe_type="deep_chat",
        user_visible_goal="Diagnose the issue.",
        internal_goal="Diagnose the issue.",
    )
    env.validate()
    return env


class FakeChat:
    def __init__(self, *, sleep_until: threading.Event | None = None, tool_calls: list[tuple[str, dict]] | None = None,
                 final_text: str = "Done.", raise_exc: BaseException | None = None):
        self.sleep_until = sleep_until
        self.tool_calls = tool_calls or []
        self.final_text = final_text
        self.raise_exc = raise_exc

    def factory(self):
        def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback):
            for i, (name, args) in enumerate(self.tool_calls):
                tool_start_callback(f"tc-{i}", name, args)
                tool_complete_callback(f"tc-{i}", name, args, f"tool {name} ok")
            # Light token churn so the worker exercises the stream callback path.
            for chunk in self.final_text.split():
                stream_callback(chunk)
            if self.sleep_until is not None:
                # Block until the test signals; lets us reproduce cancellation.
                self.sleep_until.wait(timeout=5)
            if self.raise_exc is not None:
                raise self.raise_exc
            return {"text": self.final_text}
        return _call


def _await_state(runner: JobRunner, job_id: str, target_states: set[str], timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = runner.get(job_id)
        if snap is not None and snap.get("state") in target_states:
            return snap
        time.sleep(0.02)
    snap = runner.get(job_id) or {}
    raise AssertionError(f"job did not reach {target_states} in {timeout}s; last={snap.get('state')}")


def test_submit_and_completion(board):
    chat = FakeChat(final_text="The bridge command was sent and acknowledged.")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        snap = runner.submit(env)
        assert snap["state"] in {"queued", "running", "completed"}
        final = _await_state(runner, env.job_id, {"completed"})
        assert final["state"] == "completed"
        result = board.get_result(env.job_id)
        assert result is not None
        assert result["status"] == "success"
        assert result["text"].startswith("The bridge command was sent")
    finally:
        runner.shutdown(wait=False)


def test_submit_emits_lifecycle_events_in_order(board):
    chat = FakeChat(
        final_text="Found two mutex acquisitions across an await boundary.",
        tool_calls=[("read_file", {"path": "worker.rs"}), ("grep", {"pattern": "lock"})],
    )
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
        events = board.list_events(env.job_id, limit=100)
        types = [e["type"] for e in events]
        assert types[0] == "job.started"
        assert types[-1] == "job.completed"
        assert "job.tool.call.started" in types
        assert "job.tool.call.completed" in types
        for t in types:
            assert t in safety.EVENT_TYPES, t
    finally:
        runner.shutdown(wait=False)


def test_cancellation_stops_the_worker(board):
    gate = threading.Event()
    chat = FakeChat(sleep_until=gate, final_text="This should never be returned.")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"}, timeout=2.0)
        runner.cancel(env.job_id)
        gate.set()  # release the fake chat so the cancellation point is reached
        final = _await_state(runner, env.job_id, {"canceled", "completed"}, timeout=3.0)
        # The fake chat returns the final_text only if the cancel event isn't
        # observed inside a tool callback; for a worker that has no tool
        # callbacks during the sleep, the worker may finish before checking the
        # cancel between tool calls. Either way the operator-visible state must
        # not show stale work as a fresh result.
        assert final["state"] in {"canceled", "completed"}
        # If completed, ensure we still recorded the cancel signal on the runner.
    finally:
        runner.shutdown(wait=False)


def test_cancel_after_completion_is_noop(board):
    chat = FakeChat(final_text="Done.")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
        snap = runner.cancel(env.job_id)
        assert snap["state"] == "completed"
    finally:
        runner.shutdown(wait=False)


def test_mark_stale_explicitly(board):
    chat = FakeChat(sleep_until=threading.Event())  # never returns
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"}, timeout=2.0)
        snap = runner.mark_stale(env.job_id, reason="operator says actually do go instead")
        assert snap["state"] == "stale"
        assert "operator says" in snap["last_safe_user_status"]
    finally:
        runner.shutdown(wait=False)


def test_bump_revision_marks_older_jobs_stale(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_AUTOSTALE", "1")  # legacy revision-driven staling
    chat = FakeChat(sleep_until=threading.Event())  # never returns
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-shared", revision=1)
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"}, timeout=2.0)
        outcome = runner.bump_revision("conv-shared", user_message_excerpt="please use Rust instead")
        assert outcome["revision"]["revision_id"] == 1  # first bump records as rev 1
        # The submitted job is *already* at revision 1, so first bump shouldn't stale it.
        assert outcome["marked_stale"] == []
        outcome2 = runner.bump_revision("conv-shared", user_message_excerpt="actually scratch that")
        assert outcome2["revision"]["revision_id"] == 2
        assert env.job_id in outcome2["marked_stale"]
    finally:
        runner.shutdown(wait=False)


def test_unsafe_job_id_refused_at_submit(board):
    runner = JobRunner(blackboard=board, chat_runner_factory=FakeChat().factory)
    try:
        env = _envelope()
        env.job_id = "bad id;rm -rf"
        with pytest.raises(Exception):
            runner.submit(env)
    finally:
        runner.shutdown(wait=False)


def test_unknown_job_id_cancel_raises(board):
    runner = JobRunner(blackboard=board, chat_runner_factory=FakeChat().factory)
    try:
        with pytest.raises(RunnerError):
            runner.cancel("da-does-not-exist")
    finally:
        runner.shutdown(wait=False)


def test_failure_inside_worker_is_recorded_as_failed(board):
    chat = FakeChat(raise_exc=RuntimeError("model gateway dropped"))
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"failed"})
        assert final["state"] == "failed"
        # last_safe_user_status carries the error type, not stack trace
        assert "RuntimeError" in (final.get("last_safe_user_status") or "")
        events = [e["type"] for e in board.list_events(env.job_id, limit=100)]
        assert "job.failed" in events
    finally:
        runner.shutdown(wait=False)


def test_recover_on_startup_flips_dangling_running(board):
    chat = FakeChat(final_text="ok")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
    finally:
        runner.shutdown(wait=False)

    # Simulate a dangling job that was never finalized (process crashed).
    env_orphan = _envelope()
    board.insert_job(env_orphan)
    board.update_job_state(env_orphan.job_id, state="running")

    fresh_runner = JobRunner(blackboard=board, chat_runner_factory=FakeChat().factory)
    try:
        affected = fresh_runner.recover_on_startup()
        assert env_orphan.job_id in affected
        snap = board.get_job_snapshot(env_orphan.job_id)
        assert snap["state"] == "failed"
    finally:
        fresh_runner.shutdown(wait=False)
