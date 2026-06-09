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


_ANTI_HALLUCINATION_RULES = (
    "Face Lobe rules (Double Agent §13):\n"
    "- You may report: your own action, confirmed job state, confirmed tool events, "
    "completed background results, explicitly labeled provisional guesses, explicit uncertainty.\n"
    "- You may NOT invent: tool results, file reads, device state, background progress, "
    "completion status, worker actions, or evidence that does not exist.\n"
    "- The line above starting with 'THIS TURN' is ground truth for dispatch state. "
    "If it says NO dispatch, do NOT claim to have dispatched anything.\n"
    "- For active jobs, refer only to the listed safe_user_status. Do not summarize beyond it.\n"
    "- For completed jobs listed above with a 'result:' line, the result IS the answer when "
    "relevant — quote it back rather than re-running or re-asking.\n"
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
    "above. If there are recently completed jobs, report the MOST RECENT one(s) and their "
    "result — that completed work IS the answer. Only say nothing finished if the completed "
    "list is empty. NEVER tell the user to re-dispatch or use /deep for work that already "
    "appears completed above."
)


def face_lobe_turn_start(
    *,
    conversation_id: str,
    user_message: str,
    runner: JobRunner | None = None,
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
    outcome = active_runner.bump_revision(conversation_id, user_message_excerpt=excerpt)
    return outcome


_COMPLETED_RESULT_CHAR_LIMIT = 1200


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
    if completed:
        lines.append("- recently completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):")
        for job in completed:
            jid = job.get("job_id", "?")
            state = job.get("state", "?")
            goal = (job.get("user_visible_goal") or "")[:120]
            result = bb.get_result(jid) if hasattr(bb, "get_result") else None
            text = ""
            if isinstance(result, dict):
                text = str(result.get("text") or result.get("summary") or "")
            if not text:
                text = str(job.get("last_safe_user_status") or "")
            text = text.strip().replace("\n", " ")
            if len(text) > _COMPLETED_RESULT_CHAR_LIMIT:
                text = text[: _COMPLETED_RESULT_CHAR_LIMIT - 3].rstrip() + "..."
            lines.append(f"    - {jid} [{state}] goal={goal!r}")
            if text:
                lines.append(f"      result: {text}")
    lines.append(_ANTI_HALLUCINATION_RULES)
    return "\n".join(lines)
