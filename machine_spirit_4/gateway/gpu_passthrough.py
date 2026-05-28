"""HiveMind GPU passthrough / partitioning admin (feature layer).

What this module owns
---------------------

This module is the **operator-facing workflow layer** for the three
ways a HiveMind-managed VM can get GPU acceleration:

1. **GPU-P (Hyper-V GPU Partitioning)** — the consumer-licensed
   path. Windows 11 host shares a physical NVIDIA GPU across one or
   more VMs by carving it into partitions. No special license
   required, works on any modern NVIDIA card. Provisioned via
   ``hivemind.vm.create_prebuilt@v1`` with
   ``vm_type='windows_game_stream_prebuilt'``; the template wires
   up GPU-P internally so MS4 doesn't have to call a separate "set
   GPU mode" tool.

2. **DDA (Hyper-V Discrete Device Assignment)** — the enterprise
   path. The GPU is dismounted from the host and assigned wholesale
   to one VM via vfio-pci binding. On Windows this requires
   **Windows Server / Hyper-V Server** (not Windows 11 Home/Pro).
   The driver rebind itself works on consumer SKUs but the actual
   ``Add-VMAssignableDevice`` PowerShell step fails without the
   license. Provisioned via ``hivemind.gpu_mode.set@v1`` with
   ``desired_mode='passthrough'`` followed by
   ``hivemind.vm.create_prebuilt`` + manual GPU attach.

3. **NVIDIA vGPU** — the datacenter path. Requires an active NVIDIA
   vGPU license server. Provisioned via
   ``hivemind.gpu_mode.set(desired_mode='vgpu')`` + per-profile
   ``gpu_mode.vgpu_create``.

The user has confirmed (May 27 2026):
  * **GPU-P** is the active path for consumer Windows 11 +
    Cyberpunk 2077 / PsyKyo_Engine game streaming.
  * **DDA** wiring must be complete so the same buttons work when
    the WS license is acquired later.
  * **vGPU** is wired but not the priority path on this hardware.

This module exposes a single combined snapshot + three high-level
"prepare and reserve" helpers that the Settings UI can drive
directly. Each helper is intentionally idempotent and records what
it would do via the existing audit-log infrastructure so a future
Phase-2 game-session orchestrator inherits the same trail.
"""

from __future__ import annotations

import logging
from typing import Any

from . import gpu_mode_admin, hivemind_tools as tools, vm_admin
from .gpu_mode_admin import GpuModeAdminError
from .hivemind_tools import HivemindToolError
from .vm_admin import VmAdminError


log = logging.getLogger("ms4.gateway.gpu_passthrough")


class GpuPassthroughError(RuntimeError):
    """Raised when a GPU passthrough workflow cannot complete."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAME_STREAM_VM_TYPE = "windows_game_stream_prebuilt"
"""HiveMind ``vm.create_prebuilt`` template id that provisions a
Windows 11 VM with GPU-P partitioning pre-wired. This is the
recommended path for consumer hardware."""

# vfio-pci is the Linux kernel name; on Windows the effective unbind
# happens via Hyper-V's Dismount-VMHostAssignableDevice followed by
# Add-VMAssignableDevice. HiveMind abstracts the host OS away.
PASSTHROUGH_MODE = "passthrough"
VGPU_MODE = "vgpu"
ALLOWED_MODES = (PASSTHROUGH_MODE, VGPU_MODE)


# ---------------------------------------------------------------------------
# Combined snapshot
# ---------------------------------------------------------------------------


def snapshot(hivemind_url: str) -> dict[str, Any]:
    """Build the ``Ms4GpuPassthroughSnapshot.v1`` shape the Settings
    UI renders in one round-trip.

    Fans out four read-only calls in parallel-ish (sequentially today
    because we have no shared event loop here, but each call is
    bounded so total wall-time stays under a few seconds):

    * ``hivemind.gpu_mode.capabilities@v1`` — host inventory of NVIDIA
      GPUs (raw lspci lines + which modes each card supports).
    * ``hivemind.vm.gpus@v1`` — GPUs visible to ``menta_vm_manager``
      with their current driver mode and assignment state.
    * ``hivemind.gpu_mode.vgpu_status@v1`` — health of the vGPU stack.
    * ``hivemind.gpu.availability@v1`` — live scheduler view of which
      GPUs are free right now.

    Returns ``Ms4GpuPassthroughSnapshot.v1``:

    .. code-block:: python

        {
          "schema": "Ms4GpuPassthroughSnapshot.v1",
          "capabilities": [...] | None,    # gpu_mode.capabilities raw
          "vm_visible_gpus": [...] | None, # vm.gpus
          "vgpu_stack_status": {...} | None,
          "availability": {...} | None,
          "modes": {
            "gpu_p":       {"available": True,  "needs": "Windows 11 host",
                            "via": "vm.create_prebuilt(windows_game_stream_prebuilt)"},
            "dda":         {"available": True,  "needs": "Windows Server license to attach to VM",
                            "via": "gpu_mode.set(passthrough) + vm.create_prebuilt"},
            "vgpu":        {"available": True,  "needs": "NVIDIA vGPU license server",
                            "via": "gpu_mode.set(vgpu) + gpu_mode.vgpu_create"},
          },
          "errors": [...],
        }

    Each per-call failure lands in ``errors`` (string list) rather
    than throwing — the UI surfaces partial state.
    """
    snap: dict[str, Any] = {
        "schema": "Ms4GpuPassthroughSnapshot.v1",
        "capabilities": None,
        "vm_visible_gpus": None,
        "vgpu_stack_status": None,
        "availability": None,
        "modes": {
            "gpu_p": {
                "label": "GPU-P (Hyper-V GPU Partitioning)",
                "available": True,
                "needs": "Windows 11 host",
                "license": "consumer (free)",
                "via": "vm.create_prebuilt(windows_game_stream_prebuilt)",
            },
            "dda": {
                "label": "DDA (Hyper-V Discrete Device Assignment)",
                "available": True,
                "needs": "Windows Server / Hyper-V Server SKU to actually attach a dismounted GPU to a VM. The driver rebind itself works on consumer SKUs.",
                "license": "Windows Server",
                "via": "gpu_mode.set(passthrough) + vm.create_prebuilt",
            },
            "vgpu": {
                "label": "NVIDIA vGPU (datacenter)",
                "available": True,
                "needs": "Active NVIDIA vGPU license server",
                "license": "NVIDIA vGPU",
                "via": "gpu_mode.set(vgpu) + gpu_mode.vgpu_create",
            },
        },
        "errors": [],
    }

    try:
        snap["capabilities"] = gpu_mode_admin.capabilities(hivemind_url)
    except GpuModeAdminError as exc:
        snap["errors"].append(f"gpu_mode.capabilities: {exc}")
    try:
        snap["vm_visible_gpus"] = tools.vm_gpus(hivemind_url)
    except HivemindToolError as exc:
        snap["errors"].append(f"vm.gpus: {exc}")
    try:
        snap["vgpu_stack_status"] = gpu_mode_admin.vgpu_status(hivemind_url)
    except GpuModeAdminError as exc:
        snap["errors"].append(f"gpu_mode.vgpu_status: {exc}")
    try:
        snap["availability"] = gpu_mode_admin.availability(hivemind_url)
    except GpuModeAdminError as exc:
        snap["errors"].append(f"gpu.availability: {exc}")

    return snap


# ---------------------------------------------------------------------------
# DDA / vGPU prepare (idempotent mode change)
# ---------------------------------------------------------------------------


def prepare_mode(
    hivemind_url: str,
    *,
    gpu_pci_id: str,
    desired_mode: str,
    vm_uuid: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Switch a GPU to ``vgpu`` or ``passthrough`` mode.

    This is a **destructive host operation** — it unbinds the host's
    NVIDIA driver from the device. Any process currently using the
    GPU on the host loses access. We require ``confirm=True`` to
    prevent UI-triggered accidents.

    Mode semantics:

    * ``passthrough`` is the **DDA path**. Driver is unbound and
      rebound to ``vfio-pci``. The actual ``Add-VMAssignableDevice``
      step still needs Windows Server license — but the rebind is
      what makes the GPU eligible. So calling this on a consumer SKU
      is the right move when preparing for a future license arrival.
    * ``vgpu`` enables the NVIDIA vGPU stack; subsequent
      :func:`create_vgpu` calls partition the GPU into mediated
      devices.

    Returns the raw HiveMind response augmented with
    ``{schema: 'Ms4GpuPassthroughAction.v1', intent, gpu_pci_id,
    desired_mode}`` so the UI knows what was attempted.
    """
    if desired_mode not in ALLOWED_MODES:
        raise ValueError(
            f"desired_mode must be 'passthrough' or 'vgpu', got {desired_mode!r}"
        )
    if not confirm:
        raise ValueError(
            "prepare_mode requires confirm=True (driver rebind dismounts the GPU "
            "from the host, evicting any current users)"
        )
    if not gpu_pci_id:
        raise ValueError("gpu_pci_id is required")

    intent = "dda_prepare" if desired_mode == PASSTHROUGH_MODE else "vgpu_prepare"
    try:
        raw = gpu_mode_admin.set_mode(
            hivemind_url,
            gpu_pci_id=gpu_pci_id,
            desired_mode=desired_mode,
            vm_uuid=vm_uuid,
        )
    except GpuModeAdminError as exc:
        raise GpuPassthroughError(
            f"{intent} {gpu_pci_id!r} failed: {exc}"
        ) from exc
    body = raw if isinstance(raw, dict) else {"raw": raw}
    return {
        "schema": "Ms4GpuPassthroughAction.v1",
        "intent": intent,
        "gpu_pci_id": gpu_pci_id,
        "desired_mode": desired_mode,
        "vm_uuid": vm_uuid,
        **body,
    }


def create_vgpu(
    hivemind_url: str,
    *,
    gpu_pci_id: str,
    profile: str,
    count: int = 1,
    confirm: bool = False,
) -> dict[str, Any]:
    """Create ``count`` vGPU mediated devices on the GPU at
    ``gpu_pci_id`` using ``profile``. The GPU must already be in
    ``vgpu`` mode — call :func:`prepare_mode` with
    ``desired_mode='vgpu'`` first.

    Requires ``confirm=True`` because mdev creation persists into
    sysfs and consumes the vGPU license pool.
    """
    if not confirm:
        raise ValueError("create_vgpu requires confirm=True")
    if not gpu_pci_id or not profile:
        raise ValueError("gpu_pci_id and profile are required")
    try:
        raw = gpu_mode_admin.create_vgpu(
            hivemind_url,
            gpu_pci_id=gpu_pci_id,
            profile=profile,
            count=count,
        )
    except GpuModeAdminError as exc:
        raise GpuPassthroughError(
            f"vgpu_create {gpu_pci_id!r}/{profile!r}x{count} failed: {exc}"
        ) from exc
    body = raw if isinstance(raw, dict) else {"raw": raw}
    return {
        "schema": "Ms4GpuPassthroughAction.v1",
        "intent": "vgpu_create",
        "gpu_pci_id": gpu_pci_id,
        "profile": profile,
        "count": int(count),
        **body,
    }


# ---------------------------------------------------------------------------
# GPU-P fast path: create a Windows game-stream VM
# ---------------------------------------------------------------------------


def create_game_stream_vm(
    hivemind_url: str,
    *,
    name: str,
    confirm: bool = False,
    **opts: Any,
) -> dict[str, Any]:
    """Provision a Windows 11 game-streaming VM via the
    ``windows_game_stream_prebuilt`` template. HiveMind wires up
    GPU-P partitioning inside the template; MS4 just calls
    create + deploy.

    This is the **GPU-P fast path** — works on consumer Windows 11
    hardware with no extra licensing, suitable for streaming
    Cyberpunk 2077 / PsyKyo_Engine via Sunshine/Moonlight or HiveMind's
    ScreenStream bridge.

    Returns ``Ms4GpuPassthroughAction.v1`` shape with the create +
    deploy results merged in:

    .. code-block:: python

        {
          "schema": "Ms4GpuPassthroughAction.v1",
          "intent": "gpu_p_game_stream_vm",
          "name": "...",
          "create": {...vm.create_prebuilt response...},
          "deploy": {...vm.deploy response...} | None,
          "errors": [],
        }

    Deploy is best-effort: if create succeeds but deploy fails, the
    create result is preserved so the operator can retry deploy
    manually. Requires ``confirm=True`` because this provisions a
    new VM with non-trivial host resource cost.
    """
    if not confirm:
        raise ValueError("create_game_stream_vm requires confirm=True")
    if not name:
        raise ValueError("name is required (must match ^[A-Za-z0-9._-]+$)")

    result: dict[str, Any] = {
        "schema": "Ms4GpuPassthroughAction.v1",
        "intent": "gpu_p_game_stream_vm",
        "name": name,
        "vm_type": GAME_STREAM_VM_TYPE,
        "create": None,
        "deploy": None,
        "errors": [],
    }
    try:
        result["create"] = vm_admin.create_prebuilt(
            hivemind_url, GAME_STREAM_VM_TYPE, name, **opts
        )
    except VmAdminError as exc:
        raise GpuPassthroughError(
            f"create_game_stream_vm {name!r}: vm.create_prebuilt failed: {exc}"
        ) from exc

    # Best-effort deploy. If create succeeded but the host's
    # hypervisor can't register the VM right now (e.g. Hyper-V not
    # running, F33/F35 hardening rejecting the name late), keep the
    # create result so the operator can retry manually.
    try:
        result["deploy"] = vm_admin.deploy_vm(hivemind_url, name)
    except VmAdminError as exc:
        result["errors"].append(f"vm.deploy: {exc}")

    return result
