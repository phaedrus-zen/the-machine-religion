"""HiveMind training admin (LoRA / PEFT / forge fine-tuning).

Wraps ``hivemind.training.{backends,start,status}@v1`` for MS4. Lets
the operator (or another agent driving MS4's MCP) start a fine-tune
job and poll its status without learning HiveMind's MCP shape.

Trainings live in HiveMind's `menta_forge` (Python GIM on port 6115).
The MCP layer routes there. MS4's only role is the thin admin surface
+ UI snapshot.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.training_admin")


class TrainingAdminError(RuntimeError):
    """Raised on non-recoverable training admin failure."""


def list_backends(hivemind_url: str) -> list[dict[str, Any]]:
    """Available training backends. Each entry is typically
    ``{name, version, supports: [lora|qlora|full|peft|...]}``."""
    try:
        raw = tools.training_backends(hivemind_url)
    except HivemindToolError as exc:
        raise TrainingAdminError(f"training.backends failed: {exc}") from exc
    if isinstance(raw, dict):
        cand = raw.get("backends") or raw.get("data") or []
        return [b for b in cand if isinstance(b, dict)]
    if isinstance(raw, list):
        return [b for b in raw if isinstance(b, dict)]
    return []


def start_job(hivemind_url: str, recipe: dict[str, Any]) -> dict[str, Any]:
    """Start a training job. ``recipe`` is backend-specific
    (typically ``{backend, model, dataset, hyperparams}``)."""
    if not isinstance(recipe, dict) or not recipe:
        raise ValueError("recipe must be a non-empty object")
    try:
        raw = tools.training_start(hivemind_url, recipe=recipe)
    except HivemindToolError as exc:
        raise TrainingAdminError(f"training.start failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def status(hivemind_url: str, job_id: str | None = None) -> dict[str, Any]:
    """Current training jobs (or one specific job's status)."""
    try:
        raw = tools.training_status(hivemind_url, job_id=job_id)
    except HivemindToolError as exc:
        raise TrainingAdminError(f"training.status failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def combined_snapshot(hivemind_url: str) -> dict[str, Any]:
    """UI-friendly snapshot: list of backends + active job summary.
    Schema ``Ms4TrainingSnapshot.v1``. Fail-soft per-call."""
    snap: dict[str, Any] = {
        "schema": "Ms4TrainingSnapshot.v1",
        "backends": [],
        "status": None,
        "errors": [],
    }
    try:
        snap["backends"] = list_backends(hivemind_url)
    except TrainingAdminError as exc:
        snap["errors"].append(f"backends: {exc}")
    try:
        snap["status"] = status(hivemind_url)
    except TrainingAdminError as exc:
        snap["errors"].append(f"status: {exc}")
    return snap
