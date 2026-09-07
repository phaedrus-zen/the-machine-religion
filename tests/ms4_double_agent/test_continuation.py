"""Continuation detection + runner config tests.

Covers the May 28 2026 hardening of Double Agent:

* phrase fast-path (moved to ``double_agent.continuation``)
* injectable LLM continuation classifier — consulted only when there
  are staleable jobs, the phrase path missed, and the message is short
  enough; fail-safe on unsure/exception
* env-driven concurrency + cancel grace
* the ``_parse_verdict`` / ``_extract_text`` helpers

Hermetic: no Hermes / HiveMind / MS3 contact. The classifier is a
plain callable injected by the test.
"""

from __future__ import annotations

import threading
import time

import pytest

from machine_spirit_4.double_agent import Blackboard, JobEnvelope, JobRunner, safety
from machine_spirit_4.double_agent.continuation import (
    CLASSIFIER_CONSULT_MAX_LEN,
    _extract_text,
    _parse_verdict,
    make_llm_continuation_classifier,
    phrase_is_continuation,
)


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


class _BlockingChat:
    """Chat runner that blocks until released — keeps a job 'running'."""

    def __init__(self):
        self.release = threading.Event()

    def factory(self):
        def _call(*, message, session_id, model, stream_callback,
                  tool_start_callback, tool_complete_callback):
            self.release.wait(timeout=5)
            return {"text": "done"}
        return _call


def _await_state(runner, job_id, target, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = runner.get(job_id)
        if snap and snap.get("state") in target:
            return snap
        time.sleep(0.02)
    raise AssertionError(f"job never reached {target}")


# ---------------------------------------------------------------------------
# Phrase fast-path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "ok", "yes", "do that", "go ahead", "continue", "any update?",
    "yes please", "ok do it now", "what's next", "keep going",
])
def test_phrase_continuation_positive(msg):
    assert phrase_is_continuation(msg) is True


@pytest.mark.parametrize("msg", [
    "actually use rust instead", "completely different topic now",
    "no, scratch that and start over", "", "   ",
    "yes but first rewrite the whole module in go please",  # too long / new dir
])
def test_phrase_continuation_negative(msg):
    assert phrase_is_continuation(msg) is False


# ---------------------------------------------------------------------------
# Classifier decision logic (_detect_continuation) — no threads
# ---------------------------------------------------------------------------


def test_detect_phrase_always_wins(board):
    runner = JobRunner(blackboard=board)
    # Even with no jobs and no classifier, the phrase path fires.
    assert runner._detect_continuation("ok do that", has_jobs=False) == (True, "phrase")


def test_detect_classifier_only_when_jobs_and_phrase_missed(board):
    runner = JobRunner(blackboard=board)
    runner.set_continuation_classifier(lambda m: True)
    # No staleable jobs -> classifier NOT consulted, even though it would
    # say True. Nothing to protect, so it's "none".
    assert runner._detect_continuation("go for it", has_jobs=False) == (False, "none")
    # Staleable jobs + phrase miss -> classifier consulted, says True.
    assert runner._detect_continuation("go for it", has_jobs=True) == (True, "classifier")


def test_detect_classifier_negative_stales(board):
    runner = JobRunner(blackboard=board)
    runner.set_continuation_classifier(lambda m: False)
    assert runner._detect_continuation("switch to a new plan", has_jobs=True) == (False, "none")


def test_detect_classifier_unsure_is_fail_safe(board):
    runner = JobRunner(blackboard=board)
    runner.set_continuation_classifier(lambda m: None)
    assert runner._detect_continuation("hmm interesting", has_jobs=True) == (False, "none")


def test_detect_classifier_exception_is_fail_safe(board):
    runner = JobRunner(blackboard=board)

    def boom(_m):
        raise RuntimeError("classifier down")

    runner.set_continuation_classifier(boom)
    # Must NOT propagate — a broken classifier can never break a turn.
    assert runner._detect_continuation("go for it", has_jobs=True) == (False, "none")


def test_detect_long_message_skips_classifier(board):
    runner = JobRunner(blackboard=board)
    called = {"n": 0}

    def cls(_m):
        called["n"] += 1
        return True

    runner.set_continuation_classifier(cls)
    long_msg = "x" * (CLASSIFIER_CONSULT_MAX_LEN + 5)
    assert runner._detect_continuation(long_msg, has_jobs=True) == (False, "none")
    assert called["n"] == 0  # never consulted for a long (clearly new-direction) message


# ---------------------------------------------------------------------------
# bump_revision integration — classifier rescues / stales a real job
# ---------------------------------------------------------------------------


def test_bump_revision_classifier_rescues_paraphrase(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_AUTOSTALE", "1")  # legacy continuation-aware staling
    chat = _BlockingChat()
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-rescue")
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"})
        runner.bump_revision("conv-rescue", user_message_excerpt="first message")
        runner.set_continuation_classifier(lambda m: True)
        out = runner.bump_revision(
            "conv-rescue", user_message_excerpt="go for it, run with that"
        )
        assert out["continuation_detected"] is True
        assert out["continuation_method"] == "classifier"
        assert out["marked_stale"] == []
        # The running job survives the bump.
        assert runner.get(env.job_id)["state"] == "running"
    finally:
        chat.release.set()
        runner.shutdown(wait=False)


def test_bump_revision_unsure_classifier_stales(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_AUTOSTALE", "1")  # legacy continuation-aware staling
    chat = _BlockingChat()
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-stale")
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"})
        runner.bump_revision("conv-stale", user_message_excerpt="first message")
        runner.set_continuation_classifier(lambda m: None)  # unsure -> fail-safe stale
        out = runner.bump_revision(
            "conv-stale", user_message_excerpt="actually a totally different request"
        )
        assert out["continuation_detected"] is False
        assert env.job_id in out["marked_stale"]
    finally:
        chat.release.set()
        runner.shutdown(wait=False)


def test_bump_revision_phrase_path_needs_no_classifier(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_AUTOSTALE", "1")  # legacy continuation-aware staling
    chat = _BlockingChat()
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    consulted = {"n": 0}

    def cls(_m):
        consulted["n"] += 1
        return False

    try:
        env = _envelope(conv="conv-phrase")
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"})
        runner.bump_revision("conv-phrase", user_message_excerpt="first message")
        runner.set_continuation_classifier(cls)
        out = runner.bump_revision("conv-phrase", user_message_excerpt="ok do that")
        assert out["continuation_detected"] is True
        assert out["continuation_method"] == "phrase"
        assert consulted["n"] == 0  # phrase short-circuits before the classifier
    finally:
        chat.release.set()
        runner.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Default (May 30 2026): auto-stale OFF — deep jobs run to completion
# ---------------------------------------------------------------------------


def test_bump_revision_does_not_stale_by_default(board, monkeypatch):
    """The new default: a new-direction user turn does NOT stale an
    in-flight job. The job runs to completion; the system delivers it."""
    monkeypatch.delenv("MS4_DA_AUTOSTALE", raising=False)
    chat = _BlockingChat()
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-nostale")
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"})
        runner.bump_revision("conv-nostale", user_message_excerpt="first message")
        # A genuinely different request would have staled under the legacy
        # path; by default it must NOT.
        out = runner.bump_revision(
            "conv-nostale", user_message_excerpt="actually a totally different request"
        )
        assert out["marked_stale"] == []
        assert out.get("autostale") is False
        assert runner.get(env.job_id)["state"] == "running"
    finally:
        chat.release.set()
        runner.shutdown(wait=False)


def test_default_no_stale_even_without_classifier(board, monkeypatch):
    """No classifier wired + new-direction message + auto-stale off ->
    job survives (previously this fail-safe-staled)."""
    monkeypatch.delenv("MS4_DA_AUTOSTALE", raising=False)
    chat = _BlockingChat()
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-nostale2")
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"})
        out = runner.bump_revision(
            "conv-nostale2", user_message_excerpt="something completely unrelated and long enough"
        )
        assert out["marked_stale"] == []
        assert runner.get(env.job_id)["state"] == "running"
    finally:
        chat.release.set()
        runner.shutdown(wait=False)


def test_explicit_cancel_still_stops_job_when_autostale_off(board, monkeypatch):
    """Auto-stale off must not remove the explicit-cancel escape hatch:
    the job survives a new-direction bump, then cancel takes it terminal.
    (Mirrors test_runner's cancel pattern: a worker with no tool
    callbacks may finish before observing the cancel event, so accept
    canceled OR completed once released.)"""
    monkeypatch.delenv("MS4_DA_AUTOSTALE", raising=False)
    chat = _BlockingChat()
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-cancel")
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"})
        runner.bump_revision("conv-cancel", user_message_excerpt="new direction entirely")
        assert runner.get(env.job_id)["state"] == "running"  # survived the bump (no auto-stale)
        runner.cancel(env.job_id)
        chat.release.set()  # let the worker reach its cancellation point / finish
        final = _await_state(runner, env.job_id, {"canceled", "completed"})
        assert final["state"] in {"canceled", "completed"}
    finally:
        chat.release.set()
        runner.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Env-driven config
# ---------------------------------------------------------------------------


def test_env_overrides_concurrency_and_grace(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_MAX_CONCURRENT_JOBS", "5")
    monkeypatch.setenv("MS4_DA_CANCEL_GRACE_SECONDS", "1.5")
    r = JobRunner(blackboard=board)
    try:
        assert r._max_concurrent == 5
        assert r._cancel_grace_seconds == 1.5
    finally:
        r.shutdown(wait=False)


def test_env_garbage_falls_back_to_defaults(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_MAX_CONCURRENT_JOBS", "notanint")
    monkeypatch.setenv("MS4_DA_CANCEL_GRACE_SECONDS", "")
    r = JobRunner(blackboard=board)
    try:
        assert r._max_concurrent == safety.DEFAULT_MAX_CONCURRENT_JOBS
        assert r._cancel_grace_seconds == 5.0
    finally:
        r.shutdown(wait=False)


def test_explicit_args_beat_env(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_MAX_CONCURRENT_JOBS", "9")
    r = JobRunner(blackboard=board, max_concurrent_jobs=3, cancel_grace_seconds=2.0)
    try:
        assert r._max_concurrent == 3
        assert r._cancel_grace_seconds == 2.0
    finally:
        r.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Verdict + text-extraction helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("CONTINUE", True),
    ("continue", True),
    ("Continue.", True),
    ("NEW", False),
    ("new", False),
    ("New direction", False),
    ("", None),
    ("maybe?", None),
    ("I think continue", None),  # 'continue' not in first 12 chars -> unsure
])
def test_parse_verdict(text, expected):
    assert _parse_verdict(text) is expected


def test_extract_text_openai_shape():
    resp = {"choices": [{"message": {"content": "CONTINUE"}}]}
    assert _extract_text(resp) == "CONTINUE"


def test_extract_text_text_field():
    assert _extract_text({"choices": [{"text": "NEW"}]}) == "NEW"
    assert _extract_text({"text": "CONTINUE"}) == "CONTINUE"
    assert _extract_text("CONTINUE") == "CONTINUE"
    assert _extract_text({"weird": 1}) == ""
    assert _extract_text(None) == ""


def test_hivemind_classifier_uses_current_recommendation_contract(monkeypatch):
    from machine_spirit_4.gateway import hivemind_tools

    observed = {}

    def recommend(url, **kwargs):
        observed["recommend"] = {"url": url, "kwargs": kwargs}
        return {
            "capability": "chat",
            "quality": "balanced",
            "recommended_model": "opaque-continuation-chat",
            "backend": "ollama",
        }

    def inference(url, **kwargs):
        observed["inference"] = {"url": url, "kwargs": kwargs}
        return {"choices": [{"message": {"content": "CONTINUE"}}]}

    monkeypatch.setattr(hivemind_tools, "models_recommend", recommend)
    monkeypatch.setattr(hivemind_tools, "inference_chat", inference)
    classifier = make_llm_continuation_classifier("http://hive")

    assert classifier("go for it") is True
    assert observed["recommend"] == {
        "url": "http://hive",
        "kwargs": {"capability": "chat"},
    }
    assert observed["inference"]["kwargs"]["model"] == "opaque-continuation-chat"


def test_hivemind_classifier_rejects_non_chat_recommendation(monkeypatch):
    from machine_spirit_4.gateway import hivemind_tools

    monkeypatch.setattr(
        hivemind_tools,
        "models_recommend",
        lambda *_args, **_kwargs: {
            "capability": "tts",
            "recommended_model": "opaque-non-chat",
            "backend": "tts_gim",
        },
    )
    monkeypatch.setattr(
        hivemind_tools,
        "inference_chat",
        lambda *_args, **_kwargs: pytest.fail("non-chat recommendation must be rejected"),
    )
    classifier = make_llm_continuation_classifier("http://hive")

    assert classifier("go for it") is None
