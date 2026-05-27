"""HiveMind VM lifecycle admin.

What this module owns
---------------------

Surfaces ``hivemind.vm.*@v1`` for MS4. The HiveMind cluster gained a
full VM lifecycle API in the May-2026 catalog refresh (the operator
literally asked "Can you create a VM in HiveMind?" in a prior
session — now there's a tool for it). This module is the thin,
typed, fail-soft Python layer the gateway routes + Settings UI panel
sit on top of.

Why a separate module from :mod:`hivemind_tools`
------------------------------------------------

:mod:`hivemind_tools` is the catalog — one function per HiveMind tool.
This module is the *feature surface*: it composes those primitives
into UI-friendly shapes (e.g. ``list_with_gpu_assignments`` joins
``vm.list`` and ``vm.gpus`` so the Settings panel renders both in
one call), adds intent-level guardrails (force-stop requires explicit
``confirm=True``), and exposes a stable schema so the UI doesn't
break when HiveMind's raw shapes evolve.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.vm_admin")


class VmAdminError(RuntimeError):
    """Raised on a non-recoverable VM admin failure."""


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


def list_vms(hivemind_url: str) -> list[dict[str, Any]]:
    """Return ``vm.list`` normalized to a list of dicts. HiveMind's raw
    response is a list of vm records; if the cluster returns a dict
    with a ``vms`` key we unwrap that too."""
    try:
        raw = tools.vm_list(hivemind_url)
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.list failed: {exc}") from exc
    if isinstance(raw, dict):
        candidate = raw.get("vms") or raw.get("data")
        if isinstance(candidate, list):
            return [v for v in candidate if isinstance(v, dict)]
        return [raw]
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, dict)]
    return []


def list_with_gpu_assignments(hivemind_url: str) -> dict[str, Any]:
    """Combine ``vm.list`` + ``vm.gpus`` into one snapshot the UI can
    render in a single call.

    Returns ``{schema: 'Ms4VmSnapshot.v1', vms: [...], gpus: [...],
    errors: [...]}``. Per-call failures land in ``errors`` rather
    than throwing.
    """
    snapshot: dict[str, Any] = {
        "schema": "Ms4VmSnapshot.v1",
        "vms": [],
        "gpus": None,
        "errors": [],
    }
    try:
        snapshot["vms"] = list_vms(hivemind_url)
    except VmAdminError as exc:
        snapshot["errors"].append(f"vm.list: {exc}")
    try:
        snapshot["gpus"] = tools.vm_gpus(hivemind_url)
    except HivemindToolError as exc:
        snapshot["errors"].append(f"vm.gpus: {exc}")
    return snapshot


def get_screenshot(hivemind_url: str, vm_id: str) -> dict[str, Any]:
    """Capture a VM's display.

    The HiveMind MCP gateway (per CHANGELOG entry ``ace3d899``) now
    emits BOTH a text JSON metadata block AND an MCP-native
    ``image`` content block alongside it. We surface BOTH so:

    * MS4's REST consumers can render the screenshot inline via a
      data: URL with no double base64 round-trip
    * MS4's MCP proxy can pass the image content block through to
      MCP-aware clients (Cursor, Claude Desktop) that render images
      natively in the tool reply.

    Returns ``{schema: 'Ms4VmScreenshot.v1', image_base64, mime_type,
    metadata: {...original JSON...}}``. Falls back to the legacy text-
    only shape on clusters that don't emit image content yet.
    """
    try:
        raw = tools.call_tool_with_image(
            hivemind_url, "hivemind.vm.screenshot@v1", {"vm_id": vm_id}, timeout=30
        )
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.screenshot {vm_id!r} failed: {exc}") from exc
    metadata = raw.get("json") if isinstance(raw, dict) else None
    image_b64 = raw.get("image_base64") if isinstance(raw, dict) else None
    mime = raw.get("image_mime_type") if isinstance(raw, dict) else None
    # Backwards-compatible fallback: if the cluster only returned a
    # text block with ``data_base64`` (the pre-ace3d899 shape), copy
    # that into image_base64 so callers don't need to special-case.
    if not image_b64 and isinstance(metadata, dict):
        image_b64 = metadata.get("data_base64") or metadata.get("image_base64") or ""
        if not mime:
            fmt = metadata.get("format") or "png"
            mime = f"image/{fmt}" if fmt and "/" not in fmt else (fmt or "image/png")
    return {
        "schema": "Ms4VmScreenshot.v1",
        "vm_id": vm_id,
        "image_base64": image_b64 or "",
        "mime_type": mime or "image/png",
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Write paths — every mutation goes through here so we can add audit
# logging + MS3 ethics + human approval in one place later.
# ---------------------------------------------------------------------------


def start_vm(hivemind_url: str, vm_id: str) -> dict[str, Any]:
    try:
        return tools.vm_start(hivemind_url, vm_id)
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.start {vm_id!r} failed: {exc}") from exc


def stop_vm(hivemind_url: str, vm_id: str) -> dict[str, Any]:
    """Graceful stop (ACPI shutdown). For unresponsive VMs use
    :func:`force_stop_vm`."""
    try:
        return tools.vm_stop(hivemind_url, vm_id)
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.stop {vm_id!r} failed: {exc}") from exc


def force_stop_vm(hivemind_url: str, vm_id: str, *, confirm: bool) -> dict[str, Any]:
    """Hard power-off. ``confirm=True`` is required; the call raises a
    plain :class:`ValueError` otherwise so an accidental UI click
    can't power-cycle a busy VM."""
    if not confirm:
        raise ValueError("force_stop requires confirm=True (power-off is destructive to running VM state)")
    try:
        return tools.vm_force_stop(hivemind_url, vm_id)
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.force_stop {vm_id!r} failed: {exc}") from exc


def delete_vm(hivemind_url: str, vm_id: str, *, confirm: bool) -> dict[str, Any]:
    """Destroy a VM. Irreversible — requires ``confirm=True``."""
    if not confirm:
        raise ValueError("delete_vm requires confirm=True (irreversible)")
    try:
        return tools.vm_delete(hivemind_url, vm_id)
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.delete {vm_id!r} failed: {exc}") from exc


def create_prebuilt(hivemind_url: str, template: str, **opts: Any) -> dict[str, Any]:
    try:
        return tools.vm_create_prebuilt(hivemind_url, template, **opts)
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.create_prebuilt {template!r} failed: {exc}") from exc


def deploy_vm(hivemind_url: str, vm_id: str, target_node: str | None = None) -> dict[str, Any]:
    try:
        return tools.vm_deploy(hivemind_url, vm_id, target_node)
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.deploy {vm_id!r} failed: {exc}") from exc


def undeploy_vm(hivemind_url: str, vm_id: str) -> dict[str, Any]:
    try:
        return tools.vm_undeploy(hivemind_url, vm_id)
    except HivemindToolError as exc:
        raise VmAdminError(f"vm.undeploy {vm_id!r} failed: {exc}") from exc
