"""Face Lobe behavior: revision bumping + anti-hallucination context block.

The Face Lobe is the gateway's existing chat path. The two
adjustments this module owns are:

1. **Revision bump per user message** — every call into
   :func:`face_lobe_turn_start` increments the conversation revision.
   As of May 30 2026 this no longer auto-stales older in-flight jobs by
   default (``MS4_DA_AUTOSTALE=0``): deep jobs run to completion and the
   system delivers their results automatically; only an explicit cancel
   stops a job. Set ``MS4_DA_AUTOSTALE=1`` to restore the legacy
   revision-driven staling (artifact §12).

2. **Anti-hallucination context block** — :func:`build_face_lobe_context_block`
   produces a structured block to prepend to the grounded user
   message. It lists active Double Agent jobs with their
   `last_safe_user_status`, the current revision id, and the
   May-Report / May-Not-Invent rules from artifact §13. The block is
   intentionally short and predictable; it is *not* model-generated
   to keep it tamper-proof.

The context block is appended in addition to (not in place of)
existing grounding sources like ``TMR_GROUNDING`` or the local-image
vision guidance, because those are unrelated to Double Agent.
"""

from __future__ import annotations

from typing import Any

from .blackboard import Blackboard, default_blackboard
from .runner import JobRunner, default_runner
from .schemas import result_matches_job_identity


_ANTI_HALLUCINATION_RULES = (
    "Face Lobe rules (Double Agent §13):\n"
    "- You may report: your own action, confirmed job state, confirmed tool events, "
    "completed background results, explicitly labeled provisional guesses, explicit uncertainty.\n"
    "- You may NOT invent: tool results, file reads, device state, background progress, "
    "completion status, worker actions, or evidence that does not exist.\n"
    "- The line above starting with 'THIS TURN' is ground truth for dispatch state. "
    "If it says NO dispatch, do NOT claim to have dispatched anything.\n"
    "- For active jobs, refer only to the listed safe_user_status. Do not summarize beyond it.\n"
    "- Only jobs under 'verified completed Depth Lobe jobs' with a 'result:' line are verified "
    "answers. Quote those when relevant rather than re-running or re-asking.\n"
    "- For referential follow-ups such as 'that', 'it', or 'the first point', use the latest "
    "relevant verified completed result and answer every requested part from its structured text.\n"
    "- Jobs under 'terminal Depth Lobe jobs without verified results' are failures, cancellations, "
    "or incomplete terminal records. Do NOT present them as successful results.\n"
    "- Deep jobs run to completion and the SYSTEM delivers each result to the user "
    "automatically the moment it finishes. You do NOT poll, re-dispatch, or chase them.\n"
    "- Do NOT speculate about or invent a job's state. Never say a job is 'queued', "
    "'cancelled', 'might still be running', or 'already running' unless that exact state is "
    "listed above. State only what the lines above say.\n"
    "- If the user asks whether an in-flight job is done and it is still listed active, say "
    "it's still working and that its result will be delivered automatically when ready — "
    "do not guess a status or a result.\n"
    "- If the user asks a STATUS question ('is it ready?', 'what just finished?', 'any "
    "update?', 'did anything finish?', 'is it done?', 'status?'): answer from the lists "
    "above. If there are verified completed jobs, report the MOST RECENT one(s) and their "
    "result - that completed work IS the answer. Only say nothing finished if the verified "
    "completed list is empty. NEVER tell the user to re-dispatch or use /deep for work that already "
    "appears completed above."
)


def face_lobe_turn_start(
    *,
    conversation_id: str,
    user_message: str,
    runner: JobRunner | None = None,
    turn_id: str | None = None,
) -> dict[str, Any]:
    """Mark the start of a Face Lobe turn.

    Bumps the conversation revision and stales any older-revision
    in-flight jobs in the same conversation. Returns a dict with
    ``revision_id`` and ``marked_stale``.
    """
    active_runner = runner or default_runner()
    excerpt = (user_message or "").strip().replace("\n", " ")
    if len(excerpt) > 200:
        excerpt = excerpt[:197] + "..."
    outcome = active_runner.bump_revision(
        conversation_id,
        user_message_excerpt=excerpt,
        turn_id=turn_id,
    )
    return outcome


_COMPLETED_RESULTS_TOTAL_CHAR_LIMIT = 4800
_LATEST_COMPLETED_RESULT_CHAR_LIMIT = 3300
_OLDER_COMPLETED_RESULT_CHAR_LIMIT = 500


def _single_line(text: Any, *, limit: int = 400) -> str:
    value = str(text or "").strip().replace("\n", " ")
    if len(value) > limit:
        value = value[: limit - 3].rstrip() + "..."
    return value


def _bounded_structured_text(text: Any, *, limit: int) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    value = "\n".join(line.rstrip() for line in value.split("\n"))
    if len(value) > limit:
        value = value[: limit - 3].rstrip() + "..."
    return value


def _verified_success_text(
    job: dict[str, Any],
    result: dict[str, Any] | None,
) -> str:
    if not result_matches_job_identity(job, result):
        return ""
    assert isinstance(result, dict)
    if str(result.get("status") or "").lower() != "success":
        return ""
    return str(result.get("text") or result.get("summary") or "").strip()


def _terminal_status_text(job: dict[str, Any], result: dict[str, Any] | None) -> str:
    if result_matches_job_identity(job, result):
        assert isinstance(result, dict)
        summary = _single_line(result.get("summary") or "", limit=400)
        status = str(result.get("status") or "unknown").strip() or "unknown"
        if summary:
            return f"result_status={status}; status: {summary}"
        return f"result_status={status}; no verified result text recorded"
    status = _single_line(job.get("last_safe_user_status") or "", limit=400)
    return status or "no verified result recorded"


def build_face_lobe_context_block(
    *,
    conversation_id: str,
    blackboard: Blackboard | None = None,
    include_completed: bool = True,
    max_active_jobs: int = 6,
    max_completed_jobs: int = 4,
    dispatched_this_turn_job_id: str | None = None,
    extra_authoritative_lines: list[str] | None = None,
) -> str | None:
    """Produce the context block to prepend to the grounded message.

    Returns ``None`` when there are no active Double Agent jobs, no
    completed jobs visible to this conversation, no revision history,
    and no extra authoritative lines.

    Now always lists recently *completed* Depth Lobe jobs WITH their
    result text (clamped) so the Face Lobe can answer follow-up
    questions from prior work instead of pretending it doesn't know.

    ``dispatched_this_turn_job_id`` lets the runner pin a single
    ground-truth signal into the prompt — either ``THIS TURN
    DISPATCHED job <id>`` or ``THIS TURN DID NOT DISPATCH any
    background work``. The strengthened Face Lobe system prompt keys
    off this line to refuse to fabricate dispatch claims.
    """
    bb = blackboard or default_blackboard()
    revision = bb.current_revision(conversation_id)
    # Active = queued/running/stale (work in flight or just stranded).
    active = bb.list_jobs(
        conversation_id=conversation_id,
        states=("queued", "running", "stale"),
        limit=max_active_jobs,
    )
    completed: list[dict[str, Any]] = []
    if include_completed:
        completed = bb.list_jobs(
            conversation_id=conversation_id,
            states=("completed", "failed", "canceled"),
            limit=max_completed_jobs,
        )

    if (
        revision is None
        and not active
        and not completed
        and not extra_authoritative_lines
        and dispatched_this_turn_job_id is None
    ):
        return None

    lines = ["MS4 Face Lobe context (authoritative; do not contradict):"]
    if revision is not None:
        lines.append(f"- conversation_revision_id: {revision}")
    if dispatched_this_turn_job_id:
        lines.append(
            f"- THIS TURN DISPATCHED job {dispatched_this_turn_job_id} to the Depth Lobe."
        )
    else:
        lines.append("- THIS TURN DID NOT DISPATCH any background work.")
    if extra_authoritative_lines:
        for line in extra_authoritative_lines:
            lines.append(f"- {line}")
    if active:
        lines.append("- active or stale Double Agent jobs:")
        for job in active:
            jid = job.get("job_id", "?")
            state = job.get("state", "?")
            lobe = job.get("background_lobe_type", "?")
            status = job.get("last_safe_user_status") or "no status yet"
            is_stale = " (STALE)" if job.get("is_stale") else ""
            lines.append(f"    - {jid} [{state} · {lobe}]{is_stale}: {status}")
    else:
        lines.append("- no active background jobs")
    verified_completed: list[tuple[dict[str, Any], str]] = []
    unverified_terminal: list[tuple[dict[str, Any], str]] = []
    for job in completed:
        jid = job.get("job_id", "?")
        state = str(job.get("state", "?"))
        result = bb.get_result(jid) if hasattr(bb, "get_result") else None
        verified_text = _verified_success_text(job, result) if state == "completed" else ""
        if verified_text:
            verified_completed.append((job, verified_text))
        else:
            unverified_terminal.append((job, _terminal_status_text(job, result)))
    if verified_completed:
        lines.append("- verified completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):")
        remaining_result_chars = _COMPLETED_RESULTS_TOTAL_CHAR_LIMIT
        for index, (job, text) in enumerate(verified_completed):
            jid = job.get("job_id", "?")
            state = job.get("state", "?")
            goal = (job.get("user_visible_goal") or "")[:120]
            preferred_limit = (
                _LATEST_COMPLETED_RESULT_CHAR_LIMIT
                if index == 0
                else _OLDER_COMPLETED_RESULT_CHAR_LIMIT
            )
            text = _bounded_structured_text(
                text,
                limit=min(preferred_limit, remaining_result_chars),
            )
            remaining_result_chars -= len(text)
            lines.append(f"    - {jid} [{state}] goal={goal!r}")
            first_line, *remaining_lines = text.split("\n")
            lines.append(f"      result: {first_line}")
            lines.extend(f"        {line}" for line in remaining_lines)
    if unverified_terminal:
        lines.append("- terminal Depth Lobe jobs without verified results (MOST RECENT FIRST; do NOT present as successful):")
        for job, status_text in unverified_terminal:
            jid = job.get("job_id", "?")
            state = job.get("state", "?")
            goal = (job.get("user_visible_goal") or "")[:120]
            lines.append(f"    - {jid} [{state}] goal={goal!r}")
            lines.append(f"      status: {status_text}")
    lines.append(_ANTI_HALLUCINATION_RULES)
    return "\n".join(lines)
