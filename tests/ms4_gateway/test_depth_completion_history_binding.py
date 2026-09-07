"""Regression contract for exact, durable Depth-result delivery.

A completed Depth answer must be attached to the same Face conversation that
created the job.  Delivery is idempotent because browser polling, reconnects,
and server reconciliation may all observe the same terminal job.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

import machine_spirit_4.gateway.hermes_runner as hermes_module
from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat
from machine_spirit_4.gateway.hermes_runner import (
    Ms4HermesRunner,
    _face_lobe_prior_context,
)


def _job_id() -> str:
    return f"da-{uuid.uuid4()}"


def _snapshot(
    job_id: str,
    *,
    conversation_id: str = "conv-a",
    state: str = "completed",
    goal: str = "Explain the recovery plan",
) -> dict[str, Any]:
    return {
        "schema": "DoubleAgentJobEnvelope.v1",
        "job_id": job_id,
        "parent_conversation_id": conversation_id,
        "conversation_revision_id": 1,
        "background_lobe_type": "deep_chat",
        "user_visible_goal": goal,
        "internal_goal": goal,
        "state": state,
        "is_stale": False,
    }


def _result(
    job_id: str,
    *,
    status: str = "success",
    text: str = "Complete verified Depth answer.",
) -> dict[str, Any]:
    return {
        "schema": "DoubleAgentJobResult.v1",
        "job_id": job_id,
        "status": status,
        "summary": "Depth completed.",
        "text": text,
        "confidence": "high",
        "conversation_revision_id": 1,
    }


class _FakeDepthRunner:
    def __init__(
        self,
        snapshots: dict[str, dict[str, Any]],
        results: dict[str, dict[str, Any] | None],
    ) -> None:
        self.snapshots = snapshots
        self.results = results
        self.get_calls: list[str] = []
        self.result_calls: list[str] = []
        self.list_calls: list[dict[str, Any]] = []
        self.submit_calls = 0

    def get(self, job_id: str) -> dict[str, Any] | None:
        self.get_calls.append(job_id)
        return self.snapshots.get(job_id)

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        self.result_calls.append(job_id)
        return self.results.get(job_id)

    def list(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.list_calls.append(dict(kwargs))
        conversation_id = kwargs.get("conversation_id")
        return [
            snapshot
            for snapshot in self.snapshots.values()
            if snapshot.get("parent_conversation_id") == conversation_id
        ]

    def submit(self, *_args: Any, **_kwargs: Any) -> None:
        self.submit_calls += 1
        raise AssertionError("completion reconciliation must never dispatch a new job")


def _runner(tmp_path: Path, face: FaceLobeChat) -> Ms4HermesRunner:
    return Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url="http://hive.test:6089",
        face_lobe_chat=face,
    )


def test_bind_depth_result_is_exact_session_idempotent_and_bounded() -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    state_a = face.get_or_create_session("conv-a", "face-test")
    state_b = face.get_or_create_session("conv-b", "face-test")
    state_a.messages.extend(
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "Depth is working on it."},
        ]
    )
    job_id = _job_id()
    long_text = "BEGIN-DEPTH\n" + ("substantive result paragraph. " * 1200) + "\nEND-DEPTH"

    first = face.bind_depth_result(
        conversation_id="conv-a",
        job_id=job_id,
        result_text=long_text,
        goal="original request",
    )
    messages_after_first = list(state_a.messages)
    second = face.bind_depth_result(
        conversation_id="conv-a",
        job_id=job_id,
        result_text=long_text,
        goal="original request",
    )

    assert first["history_bound"] is True
    assert first["already_bound"] is False
    assert second["history_bound"] is True
    assert second["already_bound"] is True
    assert state_a.messages == messages_after_first
    assert state_b.messages == []
    assert len(state_a.messages) == 3
    assert state_a.messages[-1]["role"] == "assistant"
    assert "BEGIN-DEPTH" in state_a.messages[-1]["content"]
    assert len(state_a.messages[-1]["content"]) <= 16_000
    assert job_id in state_a.delivered_depth_job_ids
    assert state_a.last_depth_delivery["job_id"] == job_id
    # A Depth completion is an assistant continuation of the existing user
    # turn, not a second synthetic user turn.
    sessions = {item["session_id"]: item for item in face.sessions()}
    assert sessions["conv-a"]["turns"] == 1


def test_depth_history_is_sorted_by_completion_not_delivery_arrival() -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    earlier_job = _job_id()
    later_job = _job_id()
    long_earlier = "EARLIER\n" + ("substantive detail " * 1400)

    later = face.bind_depth_result(
        conversation_id="conv-order",
        job_id=later_job,
        result_text="LATER",
        sort_key="2026-08-07T12:00:02Z",
    )
    earlier = face.bind_depth_result(
        conversation_id="conv-order",
        job_id=earlier_job,
        result_text=long_earlier,
        sort_key="2026-08-07T12:00:01Z",
    )
    later_again = face.bind_depth_result(
        conversation_id="conv-order",
        job_id=later_job,
        result_text="LATER",
        sort_key="2026-08-07T12:00:02Z",
    )

    messages = face._sessions["conv-order"].messages
    assert "EARLIER" in messages[0]["content"]
    assert "LATER" in messages[1]["content"]
    assert earlier["history_truncated"] is True
    assert later["history_truncated"] is False
    assert later_again["already_bound"] is True
    assert later_again["history_truncated"] is False


def test_new_depth_followup_keeps_later_parts_of_verified_answer() -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    state = face.get_or_create_session("conv-depth-followup", "face-test")
    state.messages.extend(
        [
            {"role": "user", "content": "Remember ALPHA means stale routing and the window is 12 minutes."},
            {"role": "assistant", "content": "READY"},
        ]
    )
    job_id = _job_id()
    result = (
        "DIRECT_CONCLUSION\n"
        + ("opening mechanism " * 250)
        + "SECOND_TRADEOFF_CRITICAL\n"
        + ("supporting evidence " * 700)
        + "CONCLUSION_TOKEN"
    )
    face.bind_depth_result(
        conversation_id="conv-depth-followup",
        job_id=job_id,
        result_text=result,
        sort_key="2026-08-07T12:00:01Z",
    )

    prior = _face_lobe_prior_context(face, "conv-depth-followup")

    combined = "\n".join(item["content"] for item in prior)
    assert len(prior) == 3
    assert any(len(item["content"]) > 1200 for item in prior)
    assert "ALPHA means stale routing" in combined
    assert "12 minutes" in combined
    assert "SECOND_TRADEOFF_CRITICAL" in combined
    assert "CONCLUSION_TOKEN" in combined


def test_runner_delivers_verified_success_once_to_exact_conversation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    face.get_or_create_session("conv-a", "face-test")
    job_id = _job_id()
    depth = _FakeDepthRunner(
        {job_id: _snapshot(job_id)},
        {job_id: _result(job_id, text="Full answer for the operator.")},
    )
    monkeypatch.setattr(hermes_module, "default_runner", lambda: depth)
    runner = _runner(tmp_path, face)

    first = runner.deliver_depth_result(job_id, "conv-a")
    count_after_first = len(face._sessions["conv-a"].messages)
    second = runner.deliver_depth_result(job_id, "conv-a")

    assert first["schema"] == "Ms4DepthCompletionDelivery.v1"
    assert first["delivery_kind"] == "answer"
    assert first["history_bound"] is True
    assert first["already_bound"] is False
    assert first["job_id"] == job_id
    assert first["conversation_id"] == "conv-a"
    assert first["result"]["text"] == "Full answer for the operator."
    assert second["delivery_kind"] == "answer"
    assert second["history_bound"] is True
    assert second["already_bound"] is True
    assert len(face._sessions["conv-a"].messages) == count_after_first


def test_runner_rejects_wrong_conversation_without_creating_or_mutating_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    job_id = _job_id()
    depth = _FakeDepthRunner(
        {job_id: _snapshot(job_id, conversation_id="conv-a")},
        {job_id: _result(job_id)},
    )
    monkeypatch.setattr(hermes_module, "default_runner", lambda: depth)
    runner = _runner(tmp_path, face)

    with pytest.raises(ValueError, match="conversation"):
        runner.deliver_depth_result(job_id, "conv-b")

    assert face._sessions == {}


@pytest.mark.parametrize(
    ("state", "result_status", "text", "expected_kind"),
    [
        ("failed", "success", "failed output", "failure"),
        ("canceled", "success", "canceled output", "failure"),
        ("completed", "partial", "partial output", "incomplete"),
        ("completed", "failed", "failed output", "incomplete"),
        ("completed", "success", "", "incomplete"),
    ],
)
def test_runner_rejects_failed_or_incomplete_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: str,
    result_status: str,
    text: str,
    expected_kind: str,
) -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    job_id = _job_id()
    depth = _FakeDepthRunner(
        {job_id: _snapshot(job_id, state=state)},
        {job_id: _result(job_id, status=result_status, text=text)},
    )
    monkeypatch.setattr(hermes_module, "default_runner", lambda: depth)
    runner = _runner(tmp_path, face)

    delivery = runner.deliver_depth_result(job_id, "conv-a")

    assert delivery["schema"] == "Ms4DepthCompletionDelivery.v1"
    assert delivery["delivery_kind"] == expected_kind
    assert delivery["history_bound"] is False
    assert delivery["already_bound"] is False
    assert face._sessions == {}


def test_runner_rejects_nonterminal_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    job_id = _job_id()
    depth = _FakeDepthRunner(
        {job_id: _snapshot(job_id, state="running")},
        {job_id: _result(job_id, text="premature")},
    )
    monkeypatch.setattr(hermes_module, "default_runner", lambda: depth)
    runner = _runner(tmp_path, face)

    with pytest.raises(RuntimeError, match="not terminal"):
        runner.deliver_depth_result(job_id, "conv-a")

    assert face._sessions == {}


def test_runner_rejects_unknown_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    job_id = _job_id()
    depth = _FakeDepthRunner({}, {})
    monkeypatch.setattr(hermes_module, "default_runner", lambda: depth)
    runner = _runner(tmp_path, face)

    with pytest.raises(KeyError):
        runner.deliver_depth_result(job_id, "conv-a")

    assert face._sessions == {}


def test_reconciliation_is_idempotent_and_never_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    face.get_or_create_session("conv-a", "face-test")
    job_id = _job_id()
    depth = _FakeDepthRunner(
        {job_id: _snapshot(job_id)},
        {job_id: _result(job_id, text="Recovered completion.")},
    )
    monkeypatch.setattr(hermes_module, "default_runner", lambda: depth)
    runner = _runner(tmp_path, face)

    runner._reconcile_depth_results("conv-a")
    messages_after_first = list(face._sessions["conv-a"].messages)
    runner._reconcile_depth_results("conv-a")

    assert face._sessions["conv-a"].messages == messages_after_first
    assert len(messages_after_first) == 1
    assert messages_after_first[0]["role"] == "assistant"
    assert "Recovered completion" in messages_after_first[0]["content"]
    assert len(depth.list_calls) == 2
    assert depth.submit_calls == 0
