"""HiveMind Oracle planner admin.

What this module owns
---------------------

Wraps the ``hivemind.oracle.*`` tool family (planner / coordinator
running at port 6089's Oracle route + the dedicated MCP tools). The
Oracle is HiveMind's reasoning surface: given a goal, it plans steps,
checks cluster capacity, and chooses tools. MS4 exposes it so the
operator can:

  * see Oracle's current state (`status`)
  * push runtime config (`configure`)
  * ask Oracle to plan a task (`chat`)

This is intentionally a thin pass-through. MS4 doesn't substitute its
own planner; if Oracle is up, MS4 surfaces it. If Oracle is down or
disabled in the cluster config, the per-method calls raise
:class:`OracleAdminError` and the UI/REST routes degrade gracefully
to "Oracle unavailable" rather than blocking other functionality.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.oracle_admin")


class OracleAdminError(RuntimeError):
    """Raised on non-recoverable Oracle admin failure."""


def status(hivemind_url: str) -> dict[str, Any]:
    """Oracle planner state. Schema ``Ms4OracleSnapshot.v1``."""
    try:
        raw = tools.oracle_status(hivemind_url)
    except HivemindToolError as exc:
        raise OracleAdminError(f"oracle.status failed: {exc}") from exc
    body = raw if isinstance(raw, dict) else {"raw": raw}
    return {"schema": "Ms4OracleSnapshot.v1", **body}


def configure(hivemind_url: str, config: dict[str, Any]) -> dict[str, Any]:
    """Push runtime config to the Oracle planner."""
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    try:
        raw = tools.oracle_configure(hivemind_url, config=config)
    except HivemindToolError as exc:
        raise OracleAdminError(f"oracle.configure failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def chat(hivemind_url: str, message: str, **opts: Any) -> dict[str, Any]:
    """Ask Oracle to reason about / plan for ``message``.

    Used by MS4 operator workflows like "what should I run next on the
    cluster?". The Face Lobe can also dispatch to Oracle for planning
    when its router decides a question is best handled there rather
    than by Hermes."""
    if not message or not isinstance(message, str):
        raise ValueError("message is required")
    try:
        raw = tools.oracle_chat(hivemind_url, message=message, **opts)
    except HivemindToolError as exc:
        raise OracleAdminError(f"oracle.chat failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}
