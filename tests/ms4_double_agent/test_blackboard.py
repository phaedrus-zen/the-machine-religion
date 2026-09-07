"""Blackboard CRUD + recovery tests."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from machine_spirit_4.double_agent import (
    Blackboard,
    JobEnvelope,
    JobEvent,
    JobResult,
    SchemaError,
)
from machine_spirit_4.double_agent import safety


@pytest.fixture()
def board(tmp_path):
    return Blackboard(tmp_path / "double_agent.sqlite3")


def _make_envelope(conv="conv-1", revision=1, job_id=None) -> JobEnvelope:
    env = JobEnvelope(
        job_id=job_id or safety.new_job_id(),
        parent_conversation_id=conv,
        conversation_revision_id=revision,
        background_lobe_type="deep_chat",
        user_visible_goal="Diagnose the issue.",
        internal_goal="Diagnose the issue and check evidence.",
    )
    env.validate()
    return env


def test_insert_and_get_roundtrip(board):
    env = _make_envelope()
    board.insert_job(env)
    snap = board.get_job_snapshot(env.job_id)
    assert snap is not None
    assert snap["state"] == "queued"
    assert snap["user_visible_goal"] == env.user_visible_goal
    assert snap["last_safe_user_status"] == env.user_visible_goal
    assert snap["is_stale"] is False


def test_insert_event_updates_last_safe_user_status(board):
    env = _make_envelope()
    board.insert_job(env)
    board.insert_event(
        JobEvent.make(
            job_id=env.job_id,
            type="job.started",
            safe_user_status="Background work started.",
        )
    )
    snap = board.get_job_snapshot(env.job_id)
    assert snap["last_safe_user_status"] == "Background work started."
    assert snap["last_event_type"] == "job.started"


def test_operator_only_event_does_not_update_user_status(board):
    env = _make_envelope()
    board.insert_job(env)
    board.insert_event(
        JobEvent.make(
            job_id=env.job_id,
            type="job.checkpoint",
            safe_user_status="user-safe text",
        )
    )
    board.insert_event(
        JobEvent.make(
            job_id=env.job_id,
            type="job.failed",
            safe_user_status="raw traceback",
            visibility="operator_only",
        )
    )
    snap = board.get_job_snapshot(env.job_id)
    assert snap["last_safe_user_status"] == "user-safe text"


def test_list_events_pagination(board):
    env = _make_envelope()
    board.insert_job(env)
    first_id = None
    for i in range(5):
        event = JobEvent.make(
            job_id=env.job_id,
            type="job.checkpoint",
            safe_user_status=f"checkpoint {i}",
        )
        board.insert_event(event)
        if first_id is None:
            first_id = event.event_id
        time.sleep(0.005)  # ensure distinct timestamps for ordering
    all_events = board.list_events(env.job_id)
    assert len(all_events) == 5
    later = board.list_events(env.job_id, after_event_id=first_id)
    assert len(later) == 4


def test_list_jobs_filters_by_conversation_and_state(board):
    env_a = _make_envelope(conv="conv-A")
    env_b = _make_envelope(conv="conv-B")
    env_c = _make_envelope(conv="conv-A")
    for env in (env_a, env_b, env_c):
        board.insert_job(env)
    board.update_job_state(env_c.job_id, state="completed")
    queued_for_a = board.list_jobs(conversation_id="conv-A", states=("queued",))
    assert {j["job_id"] for j in queued_for_a} == {env_a.job_id}
    all_for_a = board.list_jobs(conversation_id="conv-A")
    assert {j["job_id"] for j in all_for_a} == {env_a.job_id, env_c.job_id}


def test_result_round_trip(board):
    env = _make_envelope()
    board.insert_job(env)
    result = JobResult(
        job_id=env.job_id,
        status="success",
        summary="found bridge issue",
        text="long final response",
        confidence="medium",
        conversation_revision_id=1,
    )
    board.insert_result(result)
    restored = board.get_result(env.job_id)
    assert restored is not None
    assert restored["status"] == "success"
    assert restored["summary"] == "found bridge issue"


def test_voice_turn_identity_survives_revision_job_and_result_reload(board):
    turn_id = "ms4-turn-0123456789abcdef"
    revision = board.bump_revision(
        "conv-voice-reload",
        user_message_excerpt="run the depth check",
        turn_id=turn_id,
    )
    assert revision.to_dict()["turn_id"] == turn_id

    envelope = _make_envelope(conv="conv-voice-reload", revision=revision.revision_id)
    envelope.turn_id = turn_id
    board.insert_job(envelope)
    board.insert_result(
        JobResult(
            job_id=envelope.job_id,
            status="success",
            summary="depth check complete",
            conversation_revision_id=revision.revision_id,
            turn_id=turn_id,
        )
    )

    reloaded = Blackboard(board.path)
    restored_envelope = reloaded.get_job(envelope.job_id)
    restored_result = reloaded.get_result(envelope.job_id)
    assert restored_envelope is not None
    assert restored_envelope.turn_id == turn_id
    assert restored_result is not None
    assert restored_result["turn_id"] == turn_id
    with reloaded._connect() as conn:
        restored_revision = conn.execute(
            "SELECT turn_id FROM conversation_revisions "
            "WHERE conversation_id = ? AND revision_id = ?",
            (revision.conversation_id, revision.revision_id),
        ).fetchone()
    assert restored_revision is not None
    assert restored_revision["turn_id"] == turn_id


def test_existing_revision_table_adds_turn_identity_column(tmp_path):
    db_path = tmp_path / "legacy-double-agent.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE conversation_revisions ("
            "conversation_id TEXT NOT NULL, revision_id INTEGER NOT NULL, "
            "created_at TEXT NOT NULL, user_message_excerpt TEXT NOT NULL, "
            "PRIMARY KEY (conversation_id, revision_id))"
        )
        conn.execute(
            "INSERT INTO conversation_revisions "
            "(conversation_id, revision_id, created_at, user_message_excerpt) "
            "VALUES (?,?,?,?)",
            (
                "conv-legacy-voice",
                1,
                "2026-09-06T00:00:00+00:00",
                "legacy revision",
            ),
        )

    board = Blackboard(db_path)
    revision = board.bump_revision(
        "conv-legacy-voice",
        user_message_excerpt="preserve this identity",
        turn_id="ms4-turn-fedcba9876543210",
    )
    assert revision.revision_id == 2
    assert revision.turn_id == "ms4-turn-fedcba9876543210"
    with board._connect() as conn:
        rows = conn.execute(
            "SELECT revision_id, turn_id FROM conversation_revisions "
            "WHERE conversation_id = ? ORDER BY revision_id",
            ("conv-legacy-voice",),
        ).fetchall()
    assert [(row["revision_id"], row["turn_id"]) for row in rows] == [
        (1, None),
        (2, "ms4-turn-fedcba9876543210"),
    ]


def test_bump_revision_and_mark_stale(board):
    env = _make_envelope(revision=1)
    board.insert_job(env)
    board.update_job_state(env.job_id, state="running")
    assert board.current_revision(env.parent_conversation_id) is None
    rev = board.bump_revision(env.parent_conversation_id, user_message_excerpt="actually make it rust")
    assert rev.revision_id == 1
    # job is at revision 1 still; no stale yet.
    stale = board.mark_stale_jobs_for_revision(env.parent_conversation_id, rev.revision_id)
    assert stale == []
    rev2 = board.bump_revision(env.parent_conversation_id, user_message_excerpt="actually rewrite in go")
    assert rev2.revision_id == 2
    stale = board.mark_stale_jobs_for_revision(env.parent_conversation_id, rev2.revision_id)
    assert stale == [env.job_id]
    snap = board.get_job_snapshot(env.job_id)
    assert snap["state"] == "stale"


def test_hydrate_recover_flips_queued_and_running_jobs_to_failed(board):
    env_q = _make_envelope(job_id="da-queued")
    env_r = _make_envelope(job_id="da-running")
    env_done = _make_envelope(job_id="da-done")
    for env in (env_q, env_r, env_done):
        board.insert_job(env)
    board.update_job_state(env_r.job_id, state="running")
    board.update_job_state(env_done.job_id, state="completed", finished_at="2026-05-21T00:00:00Z")

    affected = board.hydrate_recover()
    assert set(affected) == {env_q.job_id, env_r.job_id}
    snap_done = board.get_job_snapshot(env_done.job_id)
    assert snap_done["state"] == "completed"
    snap_q = board.get_job_snapshot(env_q.job_id)
    snap_r = board.get_job_snapshot(env_r.job_id)
    assert snap_q["state"] == "failed"
    assert snap_r["state"] == "failed"
    assert "did not survive restart" in snap_q["last_safe_user_status"]


def test_blackboard_is_thread_safe(board):
    """Smoke test: many threads can append events without losing rows."""
    env = _make_envelope()
    board.insert_job(env)
    errors: list[Exception] = []

    def worker(n: int):
        try:
            for i in range(20):
                board.insert_event(
                    JobEvent.make(
                        job_id=env.job_id,
                        type="job.checkpoint",
                        safe_user_status=f"worker {n} ev {i}",
                    )
                )
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    events = board.list_events(env.job_id, limit=1000)
    assert len(events) == 4 * 20


def test_unsafe_ids_return_none_or_empty(board):
    assert board.get_job_snapshot("bad id;") is None
    assert board.list_events("bad id;") == []
    assert board.list_jobs(conversation_id="bad id;") == []
    with pytest.raises(SchemaError):
        board.bump_revision("bad id;")
