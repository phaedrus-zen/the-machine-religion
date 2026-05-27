"""HiveMind GPU mode / vGPU admin (``hivemind.gpu_mode.*@v1`` and
``hivemind.gpu.availability@v1``).

The May-2026 HiveMind release added per-GPU mode control (passthrough
vs vGPU vs shared) and on-the-fly vGPU creation from profiles. MS4
exposes these so Settings can show "this card supports 4x vGPU
profiles, current mode = passthrough" and so the model picker /
Depth Lobe scheduler can take available compute capacity into account
(via ``gpu.availability``).
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.gpu_mode_admin")


class GpuModeAdminError(RuntimeError):
    pass


def capabilities(hivemind_url: str) -> Any:
    try:
        return tools.gpu_mode_capabilities(hivemind_url)
    except HivemindToolError as exc:
        raise GpuModeAdminError(f"gpu_mode.capabilities failed: {exc}") from exc


def set_mode(hivemind_url: str, *, node_id: str, gpu_id: str, mode: str) -> dict[str, Any]:
    """Set a GPU's mode. Mode is typically one of ``passthrough``,
    ``vgpu``, or ``shared`` (HiveMind defines the exact set per
    vendor — query ``capabilities()`` first)."""
    try:
        return tools.gpu_mode_set(hivemind_url, node_id=node_id, gpu_id=gpu_id, mode=mode)
    except HivemindToolError as exc:
        raise GpuModeAdminError(
            f"gpu_mode.set {node_id!r}/{gpu_id!r}->{mode!r} failed: {exc}"
        ) from exc


def create_vgpu(hivemind_url: str, *, node_id: str, gpu_id: str, profile: str) -> dict[str, Any]:
    try:
        return tools.gpu_mode_vgpu_create(
            hivemind_url, node_id=node_id, gpu_id=gpu_id, profile=profile
        )
    except HivemindToolError as exc:
        raise GpuModeAdminError(
            f"gpu_mode.vgpu_create {node_id!r}/{gpu_id!r}/{profile!r} failed: {exc}"
        ) from exc


def vgpu_status(hivemind_url: str, vgpu_id: str | None = None) -> Any:
    try:
        return tools.gpu_mode_vgpu_status(hivemind_url, vgpu_id)
    except HivemindToolError as exc:
        raise GpuModeAdminError(f"gpu_mode.vgpu_status failed: {exc}") from exc


def availability(hivemind_url: str) -> Any:
    """``hivemind.gpu.availability@v1`` — which GPUs are free to schedule
    work on right now."""
    try:
        return tools.gpu_availability(hivemind_url)
    except HivemindToolError as exc:
        raise GpuModeAdminError(f"gpu.availability failed: {exc}") from exc


def combined_snapshot(hivemind_url: str) -> dict[str, Any]:
    """Capabilities + vGPU status + live availability in one snapshot."""
    snap: dict[str, Any] = {
        "schema": "Ms4GpuSnapshot.v1",
        "capabilities": None,
        "vgpu_status": None,
        "availability": None,
        "errors": [],
    }
    for key, fn in (
        ("capabilities", capabilities),
        ("vgpu_status", lambda h: vgpu_status(h)),
        ("availability", availability),
    ):
        try:
            snap[key] = fn(hivemind_url)
        except GpuModeAdminError as exc:
            snap["errors"].append(f"{key}: {exc}")
    return snap
