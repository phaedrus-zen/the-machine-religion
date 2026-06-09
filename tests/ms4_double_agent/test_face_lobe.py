"""Face Lobe context block + revision bump tests."""

from __future__ import annotations

import pytest

from machine_spirit_4.double_agent import (
    Blackboard,
    JobEnvelope,
    JobRunner,
    build_face_lobe_context_block,
    face_lobe_turn_start,
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


def test_face_lobe_turn_start_bumps_revision_and_stales_old_jobs(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_AUTOSTALE", "1")  # legacy revision-driven staling
    runner = JobRunner(blackboard=board, chat_runner_factory=lambda: (lambda **_: {"text": "n/a"}))
    try:
        env = _envelope()
        board.insert_job(env)
        # The job is at revision 1; first turn = bump to 1 (no stale).
        outcome = face_lobe_turn_start(
            conversation_id="conv-1",
            user_message="hello first turn",
            runner=runner,
        )
        assert outcome["revision"]["revision_id"] == 1
        assert outcome["marked_stale"] == []
        # Second turn bumps to 2 and stales the running job.
        board.update_job_state(env.job_id, state="running")
        outcome2 = face_lobe_turn_start(
            conversation_id="conv-1",
            user_message="actually do it differently",
            runner=runner,
        )
        assert outcome2["revision"]["revision_id"] == 2
        assert env.job_id in outcome2["marked_stale"]
    finally:
        runner.shutdown(wait=False)


def test_context_block_is_none_when_conversation_is_brand_new(board):
    # No revisions, no jobs -> no block (avoid noise on every turn).
    assert build_face_lobe_context_block(conversation_id="fresh-conv", blackboard=board) is None


def test_context_block_includes_revision_and_active_jobs(board):
    env = _envelope(conv="conv-2", revision=1)
    board.insert_job(env)
    board.bump_revision("conv-2", user_message_excerpt="first user message")
    block = build_face_lobe_context_block(conversation_id="conv-2", blackboard=board)
    assert block is not None
    assert "MS4 Face Lobe context" in block
    assert "conversation_revision_id: 1" in block
    assert env.job_id in block
    assert "active or stale Double Agent jobs" in block
    # Anti-hallucination contract must always be present.
    assert "Face Lobe rules" in block
    assert "may NOT invent" in block


def test_context_block_marks_stale_jobs_visibly(board):
    env = _envelope(conv="conv-3", revision=1)
    board.insert_job(env)
    board.update_job_state(env.job_id, state="running")
    # Bump twice — the second bump should stale the running job.
    board.bump_revision("conv-3", user_message_excerpt="first")
    board.bump_revision("conv-3", user_message_excerpt="actually rust")
    board.mark_stale_jobs_for_revision("conv-3", 2)
    block = build_face_lobe_context_block(conversation_id="conv-3", blackboard=board)
    assert block is not None
    assert "(STALE)" in block


def test_context_block_dispatch_flag_says_yes_when_dispatched(board):
    env = _envelope(conv="conv-dispatch-yes", revision=1)
    board.insert_job(env)
    board.bump_revision("conv-dispatch-yes", user_message_excerpt="first")
    block = build_face_lobe_context_block(
        conversation_id="conv-dispatch-yes",
        blackboard=board,
        dispatched_this_turn_job_id=env.job_id,
    )
    assert block is not None
    assert f"THIS TURN DISPATCHED job {env.job_id}" in block


def test_context_block_dispatch_flag_says_no_when_not_dispatched(board):
    """Critical anti-hallucination signal: when the router routes
    'direct' (no dispatch), the context block MUST plainly say
    'THIS TURN DID NOT DISPATCH any background work' so the strict
    system prompt rule can refuse to fabricate a dispatch claim."""
    board.bump_revision("conv-direct", user_message_excerpt="just chat")
    block = build_face_lobe_context_block(
        conversation_id="conv-direct",
        blackboard=board,
        dispatched_this_turn_job_id=None,
    )
    assert block is not None
    assert "THIS TURN DID NOT DISPATCH any background work" in block


def test_context_block_surfaces_completed_job_result_text(board):
    """The user-reported bug: Depth Lobe completed 'Today is May 21,
    2026.' was visible in the panel, but the Face Lobe still said
    'I don't have access to clock data' on follow-up turns. The
    context block now includes completed jobs WITH result text so the
    Face Lobe can quote them as the answer."""
    from machine_spirit_4.double_agent.schemas import JobResult

    env = _envelope(conv="conv-completed", revision=1)
    board.insert_job(env)
    board.update_job_state(env.job_id, state="completed",
                            last_safe_user_status="Date question answered.")
    board.insert_result(JobResult(
        job_id=env.job_id,
        status="success",
        summary="Date question answered.",
        text="Today is Thursday, May 21, 2026.",
        evidence=[],
        actions_taken=[],
        next_steps=[],
        confidence="high",
        conversation_revision_id=1,
    ))
    board.bump_revision("conv-completed", user_message_excerpt="follow up")
    block = build_face_lobe_context_block(
        conversation_id="conv-completed",
        blackboard=board,
    )
    assert block is not None
    assert "recently completed Depth Lobe jobs" in block
    assert "Today is Thursday, May 21, 2026" in block, \
        "completed job result text must be quoted in the context block so the Face Lobe can use it"


def test_context_block_states_auto_delivery_and_no_speculation(board):
    """Phase 4: the block must tell the Face Lobe that the system
    auto-delivers completed jobs and that it must not speculate about
    job state (fixes the observed 'already queued / might be cancelled'
    hallucinations)."""
    env = _envelope(conv="conv-rules", revision=1)
    board.insert_job(env)
    board.update_job_state(env.job_id, state="running")
    board.bump_revision("conv-rules", user_message_excerpt="is it done?")
    block = build_face_lobe_context_block(conversation_id="conv-rules", blackboard=board)
    assert block is not None
    assert "delivers each result to the user automatically" in block
    assert "Do NOT speculate" in block
    assert "delivered automatically when ready" in block
    # Status-query handling: 'is it ready / what finished' must be answered
    # from the completed list, not deflected to /deep.
    assert "STATUS question" in block
    assert "NEVER tell the user to re-dispatch" in block


def test_completed_jobs_labeled_most_recent_first(board):
    """The completed list is updated_at DESC; the block says so + the
    status rule keys off 'most recent' to deliver the right result."""
    from machine_spirit_4.double_agent import JobResult

    for i in range(2):
        env = _envelope(conv="conv-order", revision=1)
        board.insert_job(env)
        board.update_job_state(env.job_id, state="completed",
                                last_safe_user_status=f"job {i} status")
        board.insert_result(JobResult(
            job_id=env.job_id, status="success", summary=f"job {i}",
            text=f"result number {i}", confidence="high",
        ))
    block = build_face_lobe_context_block(conversation_id="conv-order", blackboard=board)
    assert block is not None
    assert "MOST RECENT FIRST" in block


def test_context_block_extra_authoritative_lines_appear_first(board):
    """The runner injects 'current local date/time: ...' via
    extra_authoritative_lines so the model can answer date/time
    questions without any dispatch at all."""
    board.bump_revision("conv-grounding", user_message_excerpt="what time")
    block = build_face_lobe_context_block(
        conversation_id="conv-grounding",
        blackboard=board,
        extra_authoritative_lines=["current local date/time: 2026-05-21T23:00:00-04:00"],
    )
    assert block is not None
    assert "current local date/time: 2026-05-21T23:00:00-04:00" in block
