"""Convert an upstream MCP ``tools/list`` entry into the MS4 /
Quartermaster catalog schema.

Two safety-relevant rules live here:

* **Namespacing** — every imported tool id is ``ext.<server>.<tool>@v1``
  so it can never collide with a native ``hivemind.*`` / ``ms4.*`` id,
  and the toolbox derives to ``<server>`` (see ``quartermaster.taxonomy``).
* **Fail-closed kind** — an imported tool is classified ``read_only``
  (and therefore eligible for the Face Lobe inline fast-path) ONLY when
  the upstream explicitly declares ``readOnlyHint`` AND is NOT
  ``destructiveHint`` AND the operator opted the server in
  (``allow_inline``). Everything else is ``runtime_action`` — routed to
  the Depth Lobe under MS3 ethics. Unknown tools are treated as unsafe.

These ``kind`` strings intentionally match
``quartermaster.catalog.KIND_*`` by value; we keep them local so this
package has no dependency on the quartermaster package (the dependency
runs the other way — the catalog reads imported tools).
"""

from __future__ import annotations

import re
from typing import Any


KIND_READ_ONLY = "read_only"
KIND_RUNTIME_ACTION = "runtime_action"

IMPORTED_TOOL_SCHEMA = "Ms4ImportedTool.v1"

_SEGMENT_RE = re.compile(r"[^a-z0-9_]+")


def sanitize_segment(value: str) -> str:
    """Lower-case, collapse non ``[a-z0-9_]`` to ``_``. Used for both
    the server id and the upstream tool name so the composed id is a
    safe, predictable identifier."""
    s = (value or "").strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_")
    s = _SEGMENT_RE.sub("_", s).strip("_")
    return s or "unknown"


def imported_tool_id(server_id: str, upstream_name: str) -> str:
    return f"ext.{sanitize_segment(server_id)}.{sanitize_segment(upstream_name)}@v1"


def kind_for_tool(tool: dict[str, Any], *, allow_inline: bool) -> str:
    """Fail-closed classification. ``read_only`` only when the upstream
    affirmatively says so and the operator opted in."""
    annotations = tool.get("annotations") if isinstance(tool.get("annotations"), dict) else {}
    read_only = bool(annotations.get("readOnlyHint"))
    destructive = bool(annotations.get("destructiveHint"))
    if read_only and not destructive and allow_inline:
        return KIND_READ_ONLY
    return KIND_RUNTIME_ACTION


def upstream_tool_to_record(
    server_id: str,
    tool: dict[str, Any],
    *,
    allow_inline: bool,
) -> dict[str, Any] | None:
    """Return the persisted/imported-tool record, or ``None`` if the
    upstream entry is malformed."""
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    input_schema = tool.get("inputSchema")
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}
    annotations = tool.get("annotations") if isinstance(tool.get("annotations"), dict) else {}
    description = str(tool.get("description") or "").strip()
    return {
        "schema": IMPORTED_TOOL_SCHEMA,
        "name": imported_tool_id(server_id, name),
        "server_id": server_id,
        "upstream_name": name,
        "description": description,
        "kind": kind_for_tool(tool, allow_inline=allow_inline),
        "input_schema": input_schema,
        "annotations": annotations,
    }


def convert_tools(
    server_id: str,
    tools: list[dict[str, Any]],
    *,
    allow_inline: bool,
) -> list[dict[str, Any]]:
    """Convert an upstream ``tools/list`` payload. Drops malformed
    entries and de-duplicates by composed id (last wins)."""
    by_id: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        record = upstream_tool_to_record(server_id, tool, allow_inline=allow_inline)
        if record is not None:
            by_id[record["name"]] = record
    return list(by_id.values())
