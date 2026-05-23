"""Safety guards for the Double Agent runtime.

Every string the operator can supply that ends up in a SQL identifier,
shell command, file path, or rendered to the user passes through one of
the guards in this module. Mirrors the ``is_safe_target_version`` /
``is_safe_*`` allowlist pattern that ``machine_spirit_4.hermes_admin``
already uses for Hermes upgrades.
"""

from __future__ import annotations

import re
import uuid


# ----- allowlists -----------------------------------------------------------

JOB_STATES = ("queued", "running", "completed", "failed", "canceled", "stale")
PRIORITIES = ("interactive", "batch")
LATENCY_CLASSES = ("foreground", "interactive_background", "batch_background")
BACKGROUND_LOBE_TYPES = (
    "deep_chat",
    "deep_coder",
    "diagnostic",
    "research",
    "verifier",
)
JOB_RESULT_STATUSES = ("success", "partial", "failed", "stale", "needs_user")
CONFIDENCE_LEVELS = ("low", "medium", "high")
VISIBILITY = ("user_safe", "operator_only")

EVENT_TYPES = (
    "job.queued",
    "job.started",
    "job.tool.call.started",
    "job.tool.call.completed",
    "job.checkpoint",
    "job.partial_finding",
    "job.needs_input",
    "job.ethics_block",
    "job.completed",
    "job.failed",
    "job.canceled",
    "job.stale",
)

# Status policy hard ceiling. Mirrors the artifact "stream safe operational
# events, not hidden reasoning" rule. Workers cannot bypass it.
SAFE_USER_STATUS_MAX_CHARS = 280
USER_VISIBLE_GOAL_MAX_CHARS = 280
INTERNAL_GOAL_MAX_CHARS = 1024
USER_MESSAGE_EXCERPT_MAX_CHARS = 200

# Job concurrency cap (foreground-starvation guard, §17).
DEFAULT_MAX_CONCURRENT_JOBS = 2


# ----- regex guards --------------------------------------------------------

# Safe ids: uuid, conversation ids, free string ids the operator supplies.
# Allow letters, digits, hyphens, underscores, dots, max 128 chars.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")

# Conversation ids may be a touch longer (e.g. carry path-style namespaces).
_SAFE_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9._\-/:]{1,256}$")

# Free text fields that get persisted but never executed. Forbid control
# characters except common whitespace; everything else is allowed.
_FORBIDDEN_TEXT_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def is_safe_job_id(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    if not _SAFE_ID_RE.match(value):
        return False
    return True


def is_safe_conversation_id(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_SAFE_CONVERSATION_ID_RE.match(value))


def is_safe_event_type(value: str | None) -> bool:
    return value in EVENT_TYPES


def is_safe_state(value: str | None) -> bool:
    return value in JOB_STATES


def is_safe_result_status(value: str | None) -> bool:
    return value in JOB_RESULT_STATUSES


def is_safe_priority(value: str | None) -> bool:
    return value in PRIORITIES


def is_safe_latency_class(value: str | None) -> bool:
    return value in LATENCY_CLASSES


def is_safe_background_lobe_type(value: str | None) -> bool:
    return value in BACKGROUND_LOBE_TYPES


def is_safe_visibility(value: str | None) -> bool:
    return value in VISIBILITY


def is_safe_confidence(value: str | None) -> bool:
    return value in CONFIDENCE_LEVELS


def coerce_safe_text(value: str | None, *, max_chars: int) -> str:
    """Sanitize a free-text field for persistence and rendering.

    Strips disallowed control characters and clamps length. Used for
    ``safe_user_status``, ``user_visible_goal``, ``summary``, etc.
    """
    if value is None:
        return ""
    text = str(value)
    text = _FORBIDDEN_TEXT_CHARS.sub("", text)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def coerce_safe_user_status(value: str | None) -> str:
    return coerce_safe_text(value, max_chars=SAFE_USER_STATUS_MAX_CHARS)


def coerce_user_visible_goal(value: str | None) -> str:
    return coerce_safe_text(value, max_chars=USER_VISIBLE_GOAL_MAX_CHARS)


def coerce_internal_goal(value: str | None) -> str:
    return coerce_safe_text(value, max_chars=INTERNAL_GOAL_MAX_CHARS)


def coerce_user_message_excerpt(value: str | None) -> str:
    return coerce_safe_text(value, max_chars=USER_MESSAGE_EXCERPT_MAX_CHARS)


def new_job_id() -> str:
    return f"da-{uuid.uuid4()}"


def new_event_id() -> str:
    return f"evt-{uuid.uuid4()}"
