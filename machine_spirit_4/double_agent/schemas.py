"""Schemas for the Double Agent runtime.

These dataclasses model the artifact-defined types
(``DoubleAgentJobEnvelope.v1``, ``DoubleAgentJobEvent.v1``,
``DoubleAgentJobResult.v1``, ``ConversationRevision.v1``) plus their
JSON (de)serialization. Validation is done at construction via the
allowlists in :mod:`machine_spirit_4.double_agent.safety` so that
nothing downstream — the SQLite blackboard, the SSE stream, the MCP
tool surface — has to re-validate.

The schemas are intentionally minimal for the phase 1 MVP. Fields that
are deferred to later phases (fanout, reducer/reconciler, distributed
leases) are omitted entirely rather than carried as "optional but
unused" — the fit/gap doc spells out the deferral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import safety


SCHEMA_JOB = "DoubleAgentJobEnvelope.v1"
SCHEMA_EVENT = "DoubleAgentJobEvent.v1"
SCHEMA_RESULT = "DoubleAgentJobResult.v1"
SCHEMA_REVISION = "ConversationRevision.v1"
SCHEMA_AUTHORITY = "AuthorityEnvelope.v1"
SCHEMA_RESOURCE_REQUEST = "ResourceRequest.v1"
SCHEMA_STATUS_POLICY = "StatusPolicy.v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SchemaError(ValueError):
    """Raised when an envelope or event fails the allowlist guards."""


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


@dataclass
class AuthorityEnvelope:
    can_read_state: bool = True
    can_call_tools: bool = True
    # Phase 1: must be false. The artifact's authority envelope is the
    # primary mechanism preventing background workers from sneaking
    # effectful behavior past the foreground governor.
    can_mutate_world: bool = False
    allowed_toolsets: list[str] = field(default_factory=list)
    requires_approval_for: list[str] = field(default_factory=lambda: [
        "purchase",
        "delete",
        "deploy",
        "unlock",
        "electrical_control",
        "security_state_change",
    ])

    def validate_phase1(self) -> None:
        if self.can_mutate_world:
            raise SchemaError(
                "AuthorityEnvelope.can_mutate_world must be false in phase 1; "
                "mutating actions still flow through MS3 ethics per tool call."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_AUTHORITY,
            "can_read_state": bool(self.can_read_state),
            "can_call_tools": bool(self.can_call_tools),
            "can_mutate_world": bool(self.can_mutate_world),
            "allowed_toolsets": list(self.allowed_toolsets),
            "requires_approval_for": list(self.requires_approval_for),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AuthorityEnvelope":
        raw = raw or {}
        return cls(
            can_read_state=bool(raw.get("can_read_state", True)),
            can_call_tools=bool(raw.get("can_call_tools", True)),
            can_mutate_world=bool(raw.get("can_mutate_world", False)),
            allowed_toolsets=list(raw.get("allowed_toolsets") or []),
            requires_approval_for=list(raw.get("requires_approval_for") or []),
        )


@dataclass
class ResourceRequest:
    model_class: str = "deep_reasoning"
    preferred_runtime: str = "local"
    preferred_hardware: str = "best_available"
    fallback_allowed: bool = False
    model_override: str | None = None
    # Optional Hermes toolset filter for the Depth Lobe worker (Phase E).
    # ``None`` = full Hermes tool catalog (the escape hatch). When set,
    # the worker constructs its agent with only these Hermes toolsets,
    # trimming the per-job tool context. Driven by the Quartermaster's
    # read of the job intent; conservative (None) by default.
    enabled_toolsets: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_RESOURCE_REQUEST,
            "model_class": self.model_class,
            "preferred_runtime": self.preferred_runtime,
            "preferred_hardware": self.preferred_hardware,
            "fallback_allowed": bool(self.fallback_allowed),
            "model_override": self.model_override,
            "enabled_toolsets": list(self.enabled_toolsets) if self.enabled_toolsets else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ResourceRequest":
        raw = raw or {}
        ets_raw = raw.get("enabled_toolsets")
        enabled_toolsets = None
        if isinstance(ets_raw, list):
            # Keep only safe, short identifier-ish toolset names.
            enabled_toolsets = [
                str(t) for t in ets_raw
                if isinstance(t, str) and t and len(t) <= 64
            ] or None
        return cls(
            model_class=str(raw.get("model_class") or "deep_reasoning"),
            preferred_runtime=str(raw.get("preferred_runtime") or "local"),
            preferred_hardware=str(raw.get("preferred_hardware") or "best_available"),
            fallback_allowed=bool(raw.get("fallback_allowed", False)),
            model_override=(str(raw.get("model_override")) if raw.get("model_override") else None),
            enabled_toolsets=enabled_toolsets,
        )


@dataclass
class StatusPolicy:
    emit_progress_events: bool = True
    # Hard-pinned to False in phase 1; the worker never emits raw model
    # tokens. The flag stays in the schema so future phases can flip it
    # behind explicit operator opt-in plus a separate safety review.
    emit_raw_tokens: bool = False
    emit_tool_events: bool = True
    summarize_every_seconds: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_STATUS_POLICY,
            "emit_progress_events": bool(self.emit_progress_events),
            "emit_raw_tokens": False,  # locked
            "emit_tool_events": bool(self.emit_tool_events),
            "summarize_every_seconds": int(self.summarize_every_seconds),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "StatusPolicy":
        raw = raw or {}
        return cls(
            emit_progress_events=bool(raw.get("emit_progress_events", True)),
            emit_raw_tokens=False,
            emit_tool_events=bool(raw.get("emit_tool_events", True)),
            summarize_every_seconds=max(1, int(raw.get("summarize_every_seconds") or 10)),
        )


# ---------------------------------------------------------------------------
# Top-level schemas
# ---------------------------------------------------------------------------


@dataclass
class JobEnvelope:
    job_id: str
    parent_conversation_id: str
    conversation_revision_id: int
    background_lobe_type: str
    user_visible_goal: str
    internal_goal: str
    authority: AuthorityEnvelope = field(default_factory=AuthorityEnvelope)
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    status_policy: StatusPolicy = field(default_factory=StatusPolicy)
    owner_identity: str = "MachineSpirit4"
    foreground_lobe: str = "ms4_face"
    priority: str = "interactive"
    latency_class: str = "interactive_background"
    state: str = "queued"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    request_user: str | None = None

    def validate(self) -> None:
        if not safety.is_safe_job_id(self.job_id):
            raise SchemaError(f"unsafe job_id: {self.job_id!r}")
        if not safety.is_safe_conversation_id(self.parent_conversation_id):
            raise SchemaError(f"unsafe parent_conversation_id: {self.parent_conversation_id!r}")
        if not isinstance(self.conversation_revision_id, int) or self.conversation_revision_id < 1:
            raise SchemaError("conversation_revision_id must be a positive integer")
        if not safety.is_safe_background_lobe_type(self.background_lobe_type):
            raise SchemaError(f"unsafe background_lobe_type: {self.background_lobe_type!r}")
        if not safety.is_safe_priority(self.priority):
            raise SchemaError(f"unsafe priority: {self.priority!r}")
        if not safety.is_safe_latency_class(self.latency_class):
            raise SchemaError(f"unsafe latency_class: {self.latency_class!r}")
        if not safety.is_safe_state(self.state):
            raise SchemaError(f"unsafe state: {self.state!r}")
        self.authority.validate_phase1()
        # Sanitize free text in place (idempotent).
        self.user_visible_goal = safety.coerce_user_visible_goal(self.user_visible_goal)
        self.internal_goal = safety.coerce_internal_goal(self.internal_goal)
        if not self.user_visible_goal:
            raise SchemaError("user_visible_goal must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_JOB,
            "job_id": self.job_id,
            "parent_conversation_id": self.parent_conversation_id,
            "conversation_revision_id": int(self.conversation_revision_id),
            "owner_identity": self.owner_identity,
            "foreground_lobe": self.foreground_lobe,
            "background_lobe_type": self.background_lobe_type,
            "user_visible_goal": self.user_visible_goal,
            "internal_goal": self.internal_goal,
            "priority": self.priority,
            "latency_class": self.latency_class,
            "state": self.state,
            "authority": self.authority.to_dict(),
            "resource_request": self.resource_request.to_dict(),
            "status_policy": self.status_policy.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "request_user": self.request_user,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JobEnvelope":
        if not isinstance(raw, dict):
            raise SchemaError("envelope must be a JSON object")
        env = cls(
            job_id=str(raw.get("job_id") or safety.new_job_id()),
            parent_conversation_id=str(raw.get("parent_conversation_id") or ""),
            conversation_revision_id=int(raw.get("conversation_revision_id") or 1),
            background_lobe_type=str(raw.get("background_lobe_type") or "deep_chat"),
            user_visible_goal=str(raw.get("user_visible_goal") or ""),
            internal_goal=str(raw.get("internal_goal") or raw.get("user_visible_goal") or ""),
            authority=AuthorityEnvelope.from_dict(raw.get("authority")),
            resource_request=ResourceRequest.from_dict(raw.get("resource_request")),
            status_policy=StatusPolicy.from_dict(raw.get("status_policy")),
            owner_identity=str(raw.get("owner_identity") or "MachineSpirit4"),
            foreground_lobe=str(raw.get("foreground_lobe") or "ms4_face"),
            priority=str(raw.get("priority") or "interactive"),
            latency_class=str(raw.get("latency_class") or "interactive_background"),
            state=str(raw.get("state") or "queued"),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
            finished_at=raw.get("finished_at"),
            request_user=(str(raw.get("request_user")) if raw.get("request_user") else None),
        )
        env.validate()
        return env


@dataclass
class JobEvent:
    event_id: str
    job_id: str
    type: str
    timestamp: str
    safe_user_status: str
    visibility: str = "user_safe"
    evidence_refs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not safety.is_safe_job_id(self.job_id):
            raise SchemaError(f"unsafe job_id: {self.job_id!r}")
        if not safety.is_safe_event_type(self.type):
            raise SchemaError(
                f"event type not in allowlist: {self.type!r}. "
                "Workers may not invent new event types; see safety.EVENT_TYPES."
            )
        if not safety.is_safe_visibility(self.visibility):
            raise SchemaError(f"unsafe visibility: {self.visibility!r}")
        # Always sanitize. We do this even for operator_only because we may
        # still render the string in the operator console.
        self.safe_user_status = safety.coerce_safe_user_status(self.safe_user_status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_EVENT,
            "event_id": self.event_id,
            "job_id": self.job_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "safe_user_status": self.safe_user_status,
            "visibility": self.visibility,
            "evidence_refs": list(self.evidence_refs),
            "payload": dict(self.payload),
        }

    @classmethod
    def make(
        cls,
        *,
        job_id: str,
        type: str,
        safe_user_status: str,
        visibility: str = "user_safe",
        evidence_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> "JobEvent":
        event = cls(
            event_id=safety.new_event_id(),
            job_id=job_id,
            type=type,
            timestamp=timestamp or _now_iso(),
            safe_user_status=safe_user_status,
            visibility=visibility,
            evidence_refs=list(evidence_refs or []),
            payload=dict(payload or {}),
        )
        event.validate()
        return event

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JobEvent":
        if not isinstance(raw, dict):
            raise SchemaError("event must be a JSON object")
        event = cls(
            event_id=str(raw.get("event_id") or safety.new_event_id()),
            job_id=str(raw.get("job_id") or ""),
            type=str(raw.get("type") or ""),
            timestamp=str(raw.get("timestamp") or _now_iso()),
            safe_user_status=str(raw.get("safe_user_status") or ""),
            visibility=str(raw.get("visibility") or "user_safe"),
            evidence_refs=list(raw.get("evidence_refs") or []),
            payload=dict(raw.get("payload") or {}),
        )
        event.validate()
        return event


@dataclass
class JobResult:
    job_id: str
    status: str
    summary: str
    text: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    confidence: str = "low"
    finished_at: str = field(default_factory=_now_iso)
    conversation_revision_id: int = 1

    def validate(self) -> None:
        if not safety.is_safe_job_id(self.job_id):
            raise SchemaError(f"unsafe job_id: {self.job_id!r}")
        if not safety.is_safe_result_status(self.status):
            raise SchemaError(f"unsafe result status: {self.status!r}")
        if not safety.is_safe_confidence(self.confidence):
            raise SchemaError(f"unsafe confidence: {self.confidence!r}")
        self.summary = safety.coerce_safe_text(self.summary, max_chars=1024)
        # text is full Hermes final response — allowed length, but still scrub control chars
        self.text = safety.coerce_safe_text(self.text, max_chars=200_000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_RESULT,
            "job_id": self.job_id,
            "status": self.status,
            "summary": self.summary,
            "text": self.text,
            "evidence": list(self.evidence),
            "actions_taken": list(self.actions_taken),
            "next_steps": list(self.next_steps),
            "confidence": self.confidence,
            "finished_at": self.finished_at,
            "conversation_revision_id": int(self.conversation_revision_id),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JobResult":
        if not isinstance(raw, dict):
            raise SchemaError("result must be a JSON object")
        result = cls(
            job_id=str(raw.get("job_id") or ""),
            status=str(raw.get("status") or "success"),
            summary=str(raw.get("summary") or ""),
            text=str(raw.get("text") or ""),
            evidence=list(raw.get("evidence") or []),
            actions_taken=list(raw.get("actions_taken") or []),
            next_steps=list(raw.get("next_steps") or []),
            confidence=str(raw.get("confidence") or "low"),
            finished_at=str(raw.get("finished_at") or _now_iso()),
            conversation_revision_id=int(raw.get("conversation_revision_id") or 1),
        )
        result.validate()
        return result


@dataclass
class ConversationRevision:
    conversation_id: str
    revision_id: int
    user_message_excerpt: str
    created_at: str = field(default_factory=_now_iso)

    def validate(self) -> None:
        if not safety.is_safe_conversation_id(self.conversation_id):
            raise SchemaError(f"unsafe conversation_id: {self.conversation_id!r}")
        if not isinstance(self.revision_id, int) or self.revision_id < 1:
            raise SchemaError("revision_id must be a positive integer")
        self.user_message_excerpt = safety.coerce_user_message_excerpt(self.user_message_excerpt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_REVISION,
            "conversation_id": self.conversation_id,
            "revision_id": int(self.revision_id),
            "user_message_excerpt": self.user_message_excerpt,
            "created_at": self.created_at,
        }
