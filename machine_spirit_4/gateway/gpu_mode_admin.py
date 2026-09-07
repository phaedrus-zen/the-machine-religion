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
    """List NVIDIA GPUs detected on this host (one ``lspci`` line per
    GPU). Read-only inventory; call before mode switching or vGPU
    provisioning."""
    try:
        return tools.gpu_mode_capabilities(hivemind_url)
    except HivemindToolError as exc:
        raise GpuModeAdminError(f"gpu_mode.capabilities failed: {exc}") from exc


def set_mode(
    hivemind_url: str,
    *,
    gpu_pci_id: str,
    desired_mode: str,
    vm_uuid: str | None = None,
) -> dict[str, Any]:
    """Change the driver mode of a GPU.

    ``desired_mode`` must be ``'vgpu'`` or ``'passthrough'``:

    * ``vgpu`` — enable the NVIDIA vGPU stack (requires NVIDIA vGPU
      license to provision usable virtual GPUs).
    * ``passthrough`` — unbind the host driver and bind the GPU to
      ``vfio-pci``. On Windows this is the **Hyper-V DDA (Discrete
      Device Assignment)** path; the rebind works on consumer SKUs
      but the actual VM attach requires Windows Server / Hyper-V
      Server license.

    NB: Hyper-V **GPU-P (GPU Partitioning)** — the consumer-licensed
    GPU passthrough for game streaming — is NOT exposed as a
    ``gpu_mode``. It's reached via
    ``vm.create_prebuilt(vm_type='windows_game_stream_prebuilt')``.
    See :mod:`gpu_passthrough` for the high-level workflow."""
    try:
        return tools.gpu_mode_set(
            hivemind_url,
            gpu_pci_id=gpu_pci_id,
            desired_mode=desired_mode,
            vm_uuid=vm_uuid,
        )
    except HivemindToolError as exc:
        raise GpuModeAdminError(
            f"gpu_mode.set {gpu_pci_id!r}->{desired_mode!r} failed: {exc}"
        ) from exc


def create_vgpu(
    hivemind_url: str,
    *,
    gpu_pci_id: str,
    profile: str,
    count: int = 1,
) -> dict[str, Any]:
    """Create ``count`` vGPU mediated devices on the GPU at
    ``gpu_pci_id`` using ``profile`` (e.g. ``nvidia-256`` for an
    A100 1g.5gb slice). Host must already be in ``vgpu`` mode."""
    try:
        return tools.gpu_mode_vgpu_create(
            hivemind_url, gpu_pci_id=gpu_pci_id, profile=profile, count=count
        )
    except HivemindToolError as exc:
        raise GpuModeAdminError(
            f"gpu_mode.vgpu_create {gpu_pci_id!r}/{profile!r}x{count} failed: {exc}"
        ) from exc


def vgpu_status(hivemind_url: str) -> Any:
    """Status of the NVIDIA vGPU stack (``nvidia-vgpu-mgr`` systemd
    unit + presence of the ``nvidia`` kernel module)."""
    try:
        return tools.gpu_mode_vgpu_status(hivemind_url)
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
