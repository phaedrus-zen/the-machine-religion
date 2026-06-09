"""MS4 Double Agent runtime — public API.

The artifact's "one Machine Spirit identity with multiple execution
lobes" model rendered as concrete code. See
``machine_spirit_4/docs/double_agent/`` for the architecture
discussion and fit/gap. The module split here matches the fit/gap's
new-module layout exactly:

* :mod:`schemas` — :class:`JobEnvelope`, :class:`JobEvent`,
  :class:`JobResult`, :class:`ConversationRevision`, plus the embedded
  authority / resource / status sub-schemas. Validation lives here.
* :mod:`safety` — allowlists and ``is_safe_*`` guards. Everything that
  ends up in a SQL identifier or rendered to the user passes through
  this module.
* :mod:`blackboard` — SQLite-backed persistent store for jobs, events,
  results, and conversation revisions. Provides
  :func:`hydrate_recover` so a gateway restart can't leave dangling
  ``running`` jobs.
* :mod:`worker` — background worker that wraps Hermes and translates
  its lifecycle callbacks into allowlisted blackboard events. No raw
  model tokens ever leak into events (anti-hallucination contract).
* :mod:`runner` — orchestrates ``submit/list/get/cancel/mark_stale``
  and ``bump_revision``. Caps concurrent jobs so background work
  cannot starve the foreground.
"""

from __future__ import annotations

from .blackboard import (
    Blackboard,
    _reset_default_blackboard_for_tests,
    default_blackboard,
)
from .depth_picker import (
    DEPTH_PRIORITY_PATTERNS,
    ENV_OVERRIDE as MS4_DEPTH_MODEL_ENV,
    DepthChoice,
    choose_depth_model,
)
from .continuation import (
    ContinuationClassifier,
    make_llm_continuation_classifier,
    phrase_is_continuation,
)
from .face_lobe import build_face_lobe_context_block, face_lobe_turn_start
from .model_picker import (
    DEFAULT_FOREGROUND_MODEL,
    ENV_OVERRIDE as MS4_FOREGROUND_MODEL_ENV,
    FOREGROUND_PRIORITY_PATTERNS,
    ForegroundChoice,
    choose_foreground_model,
)
from .router import RouteDecision, route as router_route
from .runner import (
    JobRunner,
    RunnerError,
    _reset_default_runner_for_tests,
    default_runner,
)
from .schemas import (
    AuthorityEnvelope,
    ConversationRevision,
    JobEnvelope,
    JobEvent,
    JobResult,
    ResourceRequest,
    SchemaError,
    StatusPolicy,
)
from .worker import DoubleAgentWorker, WorkerCanceled, build_real_chat_runner

__all__ = [
    "AuthorityEnvelope",
    "Blackboard",
    "ContinuationClassifier",
    "ConversationRevision",
    "DEFAULT_FOREGROUND_MODEL",
    "DEPTH_PRIORITY_PATTERNS",
    "DepthChoice",
    "DoubleAgentWorker",
    "FOREGROUND_PRIORITY_PATTERNS",
    "ForegroundChoice",
    "JobEnvelope",
    "JobEvent",
    "JobResult",
    "JobRunner",
    "MS4_DEPTH_MODEL_ENV",
    "MS4_FOREGROUND_MODEL_ENV",
    "ResourceRequest",
    "RouteDecision",
    "RunnerError",
    "SchemaError",
    "StatusPolicy",
    "WorkerCanceled",
    "_reset_default_blackboard_for_tests",
    "_reset_default_runner_for_tests",
    "build_face_lobe_context_block",
    "build_real_chat_runner",
    "choose_depth_model",
    "choose_foreground_model",
    "default_blackboard",
    "default_runner",
    "face_lobe_turn_start",
    "make_llm_continuation_classifier",
    "phrase_is_continuation",
    "router_route",
]
