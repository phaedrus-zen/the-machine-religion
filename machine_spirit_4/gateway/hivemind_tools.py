"""Typed Python wrappers around every HiveMind MCP tool MS4 cares about.

Why this module exists
----------------------

HiveMind's May-2026 catalog ships 143 tools across 30+ domains. MS4
historically reached for them ad-hoc — :mod:`voice_admin` for ASR/TTS
provisioning, :mod:`hivemind_state` for jobs/load/health, inline
:func:`mcp_call` invocations in :mod:`context` for grounding. Each
new feature wanting to call a tool either duplicated the
JSON-RPC + content-text unwrap dance or grew a one-off helper.

This module is the single, typed surface every MS4 module imports.
It:

  * uses :func:`hivemind_state.post_mcp_envelope` so direct MCP
    (port 6105) + HLI proxy fallback + bearer auth (via
    ``MS4_HIVEMIND_API_KEY``) are inherited automatically;
  * normalizes errors: every tool wrapper raises
    :class:`HivemindToolError` on a transport or ``isError: true``
    failure so callers never see raw urllib exceptions;
  * unwraps the standard MCP ``result.content[0].text`` JSON envelope
    so callers get a plain Python ``dict`` / ``list`` instead of
    poking at the JSON-RPC shape;
  * documents the live tool id (e.g. ``hivemind.vm.list@v1``) in
    each function's docstring so a future caller can trace the
    Python wrapper back to the HiveMind tool reference.

Coverage maps to the HiveMind ``tools/list`` reply observed live on
the operator's cluster (143 tools as of May 25 2026). Wrappers exist
for every tool we use today plus every tool we plan to surface in
MS4 over the next few iterations (VM lifecycle, voice identities,
human approval, storage, network, vGPU, maintenance windows,
multi-turn VLM, cluster time, model recommendation).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .hivemind_state import HivemindStateError, post_mcp_envelope

# Identity signing for PsyKyo caller-trust gate. Optional: when the
# shared secret env var is unset we silently omit the envelope and let
# the HiveMind MCP gateway fall back to legacy auth (e.g. the
# PSYKYO_EXECUTE_SAFE_AUTO actuation token). Import guarded so a
# partially-deployed MS4 (mcp/ module missing identity_signing.py)
# never breaks tool calls.
try:
    from machine_spirit_4.mcp import identity_signing as _identity_signing
    _IDENTITY_SIGNING_AVAILABLE = True
except Exception:
    _IDENTITY_SIGNING_AVAILABLE = False


log = logging.getLogger("ms4.gateway.hivemind_tools")


def identity_envelope_status() -> dict:
    """Return a small status dict for MS4's deps_status / health
    surfaces. ``available`` reflects whether the signing module
    imported successfully; ``secret_configured`` reflects whether
    the operator has set the shared secret env var right now (not
    cached). Safe to call without raising even if the module is
    missing or the env var is unset."""
    if not _IDENTITY_SIGNING_AVAILABLE:
        return {
            "available": False,
            "secret_configured": False,
            "agent_id": None,
        }
    return {
        "available": True,
        "secret_configured": _identity_signing.is_secret_configured(),
        "agent_id": _identity_signing.AGENT_ID,
    }


class HivemindToolError(RuntimeError):
    """Raised when a HiveMind MCP tool call fails. Wraps both protocol
    errors and tool-execution errors (``result.isError: true``) so
    callers only need one exception handler."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_tool(
    hivemind_url: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout: int = 15,
) -> Any:
    """Call a HiveMind MCP tool by name and return the unwrapped JSON
    payload.

    Returns:
      * a ``dict`` when the tool returns a structured object;
      * a ``list`` when the tool returns a JSON array (unwrapped from
        a single text content);
      * a ``str`` (raw text) when the tool returns plain text that
        isn't valid JSON.

    Raises :class:`HivemindToolError` on any failure mode.
    """
    # PsyKyo caller-trust gate: attach a signed agent_identity envelope
    # when the shared secret is configured. Honest fallback on any
    # signing error -- log + omit, never block the call. HiveMind MCP
    # gateway / PsyKyo can fall back to the legacy actuation token.
    agent_identity = None
    if _IDENTITY_SIGNING_AVAILABLE and _identity_signing.is_secret_configured():
        try:
            agent_identity = _identity_signing.sign_envelope(
                parent_action_authorization=f"ms4_call_tool:{tool_name}"
            )
        except Exception as e:
            log.warning(
                "identity_envelope_sign_failed tool=%s err=%s",
                tool_name,
                e,
            )
    payload = {
        "jsonrpc": "2.0",
        "id": f"ms4-tool-{uuid.uuid4().hex[:8]}",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments or {}},
    }
    if agent_identity is not None:
        # Attach at params top level (NOT inside arguments) so the
        # HiveMind MCP gateway can read it as call metadata without
        # polluting the tool's typed argument schema.
        payload["params"]["agent_identity"] = agent_identity
    try:
        envelope = post_mcp_envelope(hivemind_url, payload, timeout=timeout)
    except HivemindStateError as exc:
        raise HivemindToolError(f"{tool_name!r}: transport failure: {exc}") from exc
    if not isinstance(envelope, dict):
        raise HivemindToolError(f"{tool_name!r}: non-object JSON-RPC envelope")
    if envelope.get("error"):
        raise HivemindToolError(f"{tool_name!r}: protocol error: {envelope['error']}")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise HivemindToolError(f"{tool_name!r}: missing result block")
    if result.get("isError"):
        text_chunks = [
            chunk.get("text", "")
            for chunk in (result.get("content") or [])
            if isinstance(chunk, dict)
        ]
        raise HivemindToolError(f"{tool_name!r}: tool error: {' '.join(text_chunks)[:240]}")
    content = result.get("content")
    if isinstance(content, list):
        for chunk in content:
            if not isinstance(chunk, dict) or chunk.get("type") != "text":
                continue
            text = chunk.get("text") or ""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return result


# ===========================================================================
# Apps & app registry
# ===========================================================================


def app_list(hivemind_url: str) -> Any:
    """``hivemind.app.list@v1`` — registered app records (id, name, version,
    state, owner)."""
    return _call_tool(hivemind_url, "hivemind.app.list@v1")


# NOTE: HiveMind's app.* tools key the app by ``id`` on the wire (live
# May-27 2026 schema: app.{get,status,metrics,start,stop} all req=['id']).
# MS4 keeps the historical ``app_id`` Python param for caller stability
# and translates to ``id`` when building the envelope.


def app_get(hivemind_url: str, app_id: str) -> Any:
    """``hivemind.app.get@v1`` — full record for a single app id."""
    return _call_tool(hivemind_url, "hivemind.app.get@v1", {"id": app_id})


def app_status(hivemind_url: str, app_id: str) -> Any:
    """``hivemind.app.status@v1`` — runtime status (running/stopped/failed)."""
    return _call_tool(hivemind_url, "hivemind.app.status@v1", {"id": app_id})


def app_metrics(hivemind_url: str, app_id: str) -> Any:
    """``hivemind.app.metrics@v1`` — resource usage snapshot."""
    return _call_tool(hivemind_url, "hivemind.app.metrics@v1", {"id": app_id})


def app_start(hivemind_url: str, app_id: str) -> Any:
    """``hivemind.app.start@v1`` — request app start. Returns dispatch ack."""
    return _call_tool(hivemind_url, "hivemind.app.start@v1", {"id": app_id})


def app_stop(hivemind_url: str, app_id: str) -> Any:
    """``hivemind.app.stop@v1`` — request app stop. Returns dispatch ack."""
    return _call_tool(hivemind_url, "hivemind.app.stop@v1", {"id": app_id})


def app_resolve(hivemind_url: str, name: str) -> Any:
    """``hivemind.app.resolve@v1`` — look up an app id from a name."""
    return _call_tool(hivemind_url, "hivemind.app.resolve@v1", {"name": name})


def app_register(hivemind_url: str, manifest: dict[str, Any]) -> Any:
    """``hivemind.app.register@v1`` — register a new app from a manifest."""
    return _call_tool(hivemind_url, "hivemind.app.register@v1", {"manifest": manifest})


def apps_discover(hivemind_url: str) -> Any:
    """``hivemind.apps.discover@v1`` — discoverable apps across the cluster
    (broader than ``app.list`` which is local registry only)."""
    return _call_tool(hivemind_url, "hivemind.apps.discover@v1")


# ===========================================================================
# VMs
# ===========================================================================


def vm_list(hivemind_url: str) -> Any:
    """``hivemind.vm.list@v1`` — VM inventory across the cluster."""
    return _call_tool(hivemind_url, "hivemind.vm.list@v1")


def vm_gpus(hivemind_url: str) -> Any:
    """``hivemind.vm.gpus@v1`` — list GPUs visible to menta_vm_manager
    and assignable to VMs (Hyper-V DDA / VFIO passthrough). Returns
    PCI addresses, vendor/model strings, current driver mode, and
    assignment state. Takes NO arguments per the May-26 2026 cluster
    contract — earlier MS4 wrappers passed an optional ``vm_id``
    filter that the cluster silently ignored."""
    return _call_tool(hivemind_url, "hivemind.vm.gpus@v1")


# NOTE: HiveMind's vm.* mutation tools use ``name`` as the VM
# identifier on the wire (see live tool schemas). MS4's public Python
# API keeps the historical ``vm_id`` keyword for backwards
# compatibility with everything that already imports these helpers;
# we just translate to ``name`` when building the MCP envelope.


def vm_start(hivemind_url: str, vm_id: str) -> Any:
    """``hivemind.vm.start@v1`` — start a VM by name."""
    return _call_tool(hivemind_url, "hivemind.vm.start@v1", {"name": vm_id})


def vm_stop(hivemind_url: str, vm_id: str) -> Any:
    """``hivemind.vm.stop@v1`` — graceful stop (sends ACPI shutdown)."""
    return _call_tool(hivemind_url, "hivemind.vm.stop@v1", {"name": vm_id})


def vm_force_stop(hivemind_url: str, vm_id: str) -> Any:
    """``hivemind.vm.force_stop@v1`` — hard power-off. Use as last resort."""
    return _call_tool(hivemind_url, "hivemind.vm.force_stop@v1", {"name": vm_id})


def vm_delete(hivemind_url: str, vm_id: str) -> Any:
    """``hivemind.vm.delete@v1`` — destroy a VM (irreversible)."""
    return _call_tool(hivemind_url, "hivemind.vm.delete@v1", {"name": vm_id})


def vm_screenshot(
    hivemind_url: str,
    vm_id: str,
    *,
    width: int = 1280,
    height: int = 720,
) -> Any:
    """``hivemind.vm.screenshot@v1`` — capture VM display. Live cluster
    contract requires explicit ``width`` + ``height`` (Hyper-V's
    screenshot API needs target dimensions; we default to 1280x720
    which matches the prior implicit shape). Returns ``{format,
    data_base64, ...}`` plus an MCP-native ``image`` content block
    (handled via :func:`call_tool_with_image`)."""
    return _call_tool(
        hivemind_url,
        "hivemind.vm.screenshot@v1",
        {"name": vm_id, "width": int(width), "height": int(height)},
        timeout=30,
    )


def vm_create_prebuilt(
    hivemind_url: str,
    vm_type: str,
    name: str,
    **opts: Any,
) -> Any:
    """``hivemind.vm.create_prebuilt@v1`` — instantiate a VM from a
    template name.

    ``vm_type`` is the template id; live cluster enum is
    ``windows_game_stream_prebuilt`` | ``linux_inference_prebuilt`` |
    ``home_assistant_prebuilt`` | ``windows_generic_install`` |
    ``windows_11_install`` | ``ubuntu_server_install`` |
    ``home_assistant_install``. ``name`` is the VM name and must
    match ``^[A-Za-z0-9._-]+$`` per HiveMind F33/F35 hardening. Extra
    ``opts`` forward through to the cluster for forward-compat with
    future template parameters."""
    args = {"vm_type": vm_type, "name": name, **opts}
    return _call_tool(hivemind_url, "hivemind.vm.create_prebuilt@v1", args, timeout=60)


def vm_deploy(hivemind_url: str, vm_id: str, **opts: Any) -> Any:
    """``hivemind.vm.deploy@v1`` — register an already-defined VM with
    the hypervisor (Hyper-V Import-VM on Windows, virsh define on
    Linux). Required before :func:`vm_start`. Idempotent.

    The live contract only requires ``name``; ``**opts`` is accepted
    for forward-compat if HiveMind grows the schema (e.g. target
    node hints) — the cluster ignores unknown fields today."""
    args = {"name": vm_id, **opts}
    return _call_tool(hivemind_url, "hivemind.vm.deploy@v1", args, timeout=60)


def vm_undeploy(hivemind_url: str, vm_id: str) -> Any:
    """``hivemind.vm.undeploy@v1`` — unregister a VM from its
    hypervisor (Hyper-V Remove-VM on Windows, virsh undefine on
    Linux)."""
    return _call_tool(hivemind_url, "hivemind.vm.undeploy@v1", {"name": vm_id})


# ===========================================================================
# Storage
# ===========================================================================


def storage_status(hivemind_url: str) -> Any:
    """``hivemind.storage.status@v1`` — top-level storage health/capacity."""
    return _call_tool(hivemind_url, "hivemind.storage.status@v1")


def storage_pools(hivemind_url: str) -> Any:
    """``hivemind.storage.pools@v1`` — list storage pools."""
    return _call_tool(hivemind_url, "hivemind.storage.pools@v1")


def storage_pool_create(hivemind_url: str, **opts: Any) -> Any:
    """``hivemind.storage.pool_create@v1`` — create a new storage pool."""
    return _call_tool(hivemind_url, "hivemind.storage.pool_create@v1", opts, timeout=30)


# NOTE: HiveMind's storage mutation tools key the target by ``id`` on the
# wire (live May-27 2026: pool_delete/volume_delete/volume_detach/
# volume_resize/snapshot_* all req=['id']; volume_attach req=['id','vm_id'];
# volume_resize uses ``size_gb``; snapshot_create uses ``name``). MS4 keeps
# its descriptive Python params and translates to the live keys here.


def storage_pool_delete(hivemind_url: str, pool_id: str) -> Any:
    """``hivemind.storage.pool_delete@v1`` — delete a storage pool."""
    return _call_tool(hivemind_url, "hivemind.storage.pool_delete@v1", {"id": pool_id})


def storage_volumes(hivemind_url: str, pool_id: str | None = None) -> Any:
    """``hivemind.storage.volumes@v1`` — list volumes (optional pool filter)."""
    args = {"pool_id": pool_id} if pool_id else {}
    return _call_tool(hivemind_url, "hivemind.storage.volumes@v1", args)


def storage_volume_create(hivemind_url: str, **opts: Any) -> Any:
    """``hivemind.storage.volume_create@v1`` — create a new volume."""
    return _call_tool(hivemind_url, "hivemind.storage.volume_create@v1", opts, timeout=30)


def storage_volume_delete(hivemind_url: str, volume_id: str) -> Any:
    """``hivemind.storage.volume_delete@v1`` — delete a volume."""
    return _call_tool(hivemind_url, "hivemind.storage.volume_delete@v1", {"id": volume_id})


def storage_volume_attach(hivemind_url: str, volume_id: str, target: str) -> Any:
    """``hivemind.storage.volume_attach@v1`` — attach a volume to a VM.
    Live contract: ``{id, vm_id}`` (``target`` is the VM identifier)."""
    return _call_tool(
        hivemind_url,
        "hivemind.storage.volume_attach@v1",
        {"id": volume_id, "vm_id": target},
    )


def storage_volume_detach(hivemind_url: str, volume_id: str) -> Any:
    """``hivemind.storage.volume_detach@v1`` — detach a volume from its
    current target."""
    return _call_tool(hivemind_url, "hivemind.storage.volume_detach@v1", {"id": volume_id})


def storage_volume_resize(hivemind_url: str, volume_id: str, new_size_bytes: int) -> Any:
    """``hivemind.storage.volume_resize@v1`` — resize a volume. MS4's
    public API takes ``new_size_bytes``; the live cluster contract uses
    ``size_gb``, so we convert (rounding up to at least 1 GiB)."""
    size_gb = max(1, round(int(new_size_bytes) / (1024 ** 3)))
    return _call_tool(
        hivemind_url,
        "hivemind.storage.volume_resize@v1",
        {"id": volume_id, "size_gb": size_gb},
    )


def storage_snapshots(hivemind_url: str, volume_id: str | None = None) -> Any:
    """``hivemind.storage.snapshots@v1`` — list snapshots (optional volume
    filter)."""
    args = {"volume_id": volume_id} if volume_id else {}
    return _call_tool(hivemind_url, "hivemind.storage.snapshots@v1", args)


def storage_snapshot_create(hivemind_url: str, volume_id: str, label: str | None = None) -> Any:
    """``hivemind.storage.snapshot_create@v1`` — snapshot a volume. Live
    contract: ``{id}`` (the volume id) + optional ``name``/``description``."""
    args: dict[str, Any] = {"id": volume_id}
    if label:
        args["name"] = label
    return _call_tool(hivemind_url, "hivemind.storage.snapshot_create@v1", args, timeout=30)


def storage_snapshot_delete(hivemind_url: str, snapshot_id: str) -> Any:
    """``hivemind.storage.snapshot_delete@v1`` — delete a snapshot."""
    return _call_tool(
        hivemind_url, "hivemind.storage.snapshot_delete@v1", {"id": snapshot_id}
    )


def storage_snapshot_restore(hivemind_url: str, snapshot_id: str) -> Any:
    """``hivemind.storage.snapshot_restore@v1`` — roll a volume back to a
    snapshot."""
    return _call_tool(
        hivemind_url,
        "hivemind.storage.snapshot_restore@v1",
        {"id": snapshot_id},
        timeout=60,
    )


# ===========================================================================
# Network
# ===========================================================================


def network_list(hivemind_url: str) -> Any:
    """``hivemind.network.list@v1`` — list cluster networks."""
    return _call_tool(hivemind_url, "hivemind.network.list@v1")


# NOTE: HiveMind's network mutation tools key the network by ``id`` on the
# wire (live May-27 2026: delete/attachments/isolate req=['id'];
# attach/detach req=['id','vm_id']; status takes no args). MS4 keeps its
# ``network_id``/``target`` Python params and translates here.


def network_status(hivemind_url: str, network_id: str | None = None) -> Any:
    """``hivemind.network.status@v1`` — overall network status. The live
    contract takes no arguments; ``network_id`` is accepted for API
    stability but not sent (filter client-side)."""
    return _call_tool(hivemind_url, "hivemind.network.status@v1")


def network_create(hivemind_url: str, **opts: Any) -> Any:
    """``hivemind.network.create@v1`` — create a network. Live required
    fields: ``name``, ``network_type``, ``isolated``, ``enable_dhcp``."""
    return _call_tool(hivemind_url, "hivemind.network.create@v1", opts, timeout=30)


def network_delete(hivemind_url: str, network_id: str) -> Any:
    """``hivemind.network.delete@v1`` — delete a network."""
    return _call_tool(hivemind_url, "hivemind.network.delete@v1", {"id": network_id})


def network_attach(hivemind_url: str, network_id: str, target: str) -> Any:
    """``hivemind.network.attach@v1`` — attach a VM to a network. Live
    contract: ``{id, vm_id}`` (``target`` is the VM identifier)."""
    return _call_tool(
        hivemind_url,
        "hivemind.network.attach@v1",
        {"id": network_id, "vm_id": target},
    )


def network_detach(hivemind_url: str, network_id: str, target: str) -> Any:
    """``hivemind.network.detach@v1`` — detach a VM from a network. Live
    contract: ``{id, vm_id}``."""
    return _call_tool(
        hivemind_url,
        "hivemind.network.detach@v1",
        {"id": network_id, "vm_id": target},
    )


def network_bridges(hivemind_url: str) -> Any:
    """``hivemind.network.bridges@v1`` — list cluster bridges."""
    return _call_tool(hivemind_url, "hivemind.network.bridges@v1")


def network_interfaces(hivemind_url: str) -> Any:
    """``hivemind.network.interfaces@v1`` — list physical interfaces."""
    return _call_tool(hivemind_url, "hivemind.network.interfaces@v1")


def network_isolate(hivemind_url: str, network_id: str, isolated: bool = True) -> Any:
    """``hivemind.network.isolate@v1`` — isolate a network. Live contract:
    ``{id}`` (``isolated`` kept for API stability; not in the live schema
    but forwarded harmlessly for forward-compat)."""
    return _call_tool(
        hivemind_url,
        "hivemind.network.isolate@v1",
        {"id": network_id, "isolated": isolated},
    )


def network_attachments(hivemind_url: str, network_id: str) -> Any:
    """``hivemind.network.attachments@v1`` — list attachments for a
    network. Live contract requires ``{id}``."""
    return _call_tool(hivemind_url, "hivemind.network.attachments@v1", {"id": network_id})


# ===========================================================================
# GPU mode / vGPU
# ===========================================================================


def gpu_mode_capabilities(hivemind_url: str) -> Any:
    """``hivemind.gpu_mode.capabilities@v1`` — list NVIDIA GPUs detected
    on this host (one raw ``lspci`` line per GPU). Read-only inventory;
    call this BEFORE mode switching or vGPU provisioning."""
    return _call_tool(hivemind_url, "hivemind.gpu_mode.capabilities@v1")


def gpu_mode_set(
    hivemind_url: str,
    *,
    gpu_pci_id: str,
    desired_mode: str,
    vm_uuid: str | None = None,
) -> Any:
    """``hivemind.gpu_mode.set@v1`` — change the driver mode of a GPU.

    ``desired_mode`` MUST be one of:

    * ``"vgpu"`` — enable the NVIDIA vGPU stack on the device.
    * ``"passthrough"`` — unbind the NVIDIA driver and bind the
      device to ``vfio-pci`` so a VM can claim it via DDA/VFIO. On
      Windows this is Hyper-V Discrete Device Assignment (requires a
      Windows Server license to actually attach; the rebind itself
      works on consumer SKUs).

    HiveMind does NOT currently model Hyper-V GPU Partitioning
    (GPU-P) as a ``gpu_mode``. GPU-P is exposed indirectly via
    :func:`vm_create_prebuilt` (``vm_type='windows_game_stream_prebuilt'``)
    which provisions a VM with a partitioned GPU surface.

    ``gpu_pci_id`` is the PCI address (e.g. ``'0000:01:00.0'``).
    ``vm_uuid`` is an optional informational hint identifying the VM
    that will receive the GPU after rebind."""
    if desired_mode not in ("vgpu", "passthrough"):
        raise ValueError(
            f"desired_mode must be 'vgpu' or 'passthrough', got {desired_mode!r}"
        )
    args: dict[str, Any] = {
        "gpu_pci_id": gpu_pci_id,
        "desired_mode": desired_mode,
    }
    if vm_uuid:
        args["vm_uuid"] = vm_uuid
    return _call_tool(hivemind_url, "hivemind.gpu_mode.set@v1", args, timeout=30)


def gpu_mode_vgpu_create(
    hivemind_url: str,
    *,
    gpu_pci_id: str,
    profile: str,
    count: int = 1,
) -> Any:
    """``hivemind.gpu_mode.vgpu_create@v1`` — create one or more vGPU
    mediated device (mdev) instances on a GPU using the given
    profile.

    Writes UUIDs to
    ``/sys/bus/pci/devices/{gpu_pci_id}/mdev_supported_types/{profile}/create``.
    ``count`` is server-validated (1..64). The host must already be
    in ``vgpu`` mode — call :func:`gpu_mode_set` first."""
    if count < 1 or count > 64:
        raise ValueError(f"count must be between 1 and 64 (got {count})")
    return _call_tool(
        hivemind_url,
        "hivemind.gpu_mode.vgpu_create@v1",
        {"gpu_pci_id": gpu_pci_id, "profile": profile, "count": int(count)},
        timeout=30,
    )


def gpu_mode_vgpu_status(hivemind_url: str) -> Any:
    """``hivemind.gpu_mode.vgpu_status@v1`` — status of the NVIDIA vGPU
    stack (``nvidia-vgpu-mgr`` systemd unit + presence of the
    ``nvidia`` kernel module). Takes NO arguments per the May-26
    2026 cluster contract."""
    return _call_tool(hivemind_url, "hivemind.gpu_mode.vgpu_status@v1")


def gpu_availability(hivemind_url: str) -> Any:
    """``hivemind.gpu.availability@v1`` — which GPUs are free for scheduling
    right now (not the same as the static inventory)."""
    return _call_tool(hivemind_url, "hivemind.gpu.availability@v1")


# ===========================================================================
# Voice identities (speaker enrollment / identification)
# ===========================================================================


def voice_identities_list(hivemind_url: str) -> Any:
    """``hivemind.voice_identities.list@v1`` — enrolled voice identities."""
    return _call_tool(hivemind_url, "hivemind.voice_identities.list@v1")


def voice_identities_enroll(
    hivemind_url: str,
    *,
    audio_base64: str,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """``hivemind.voice_identities.enroll@v1`` — register a new voice from
    a sample. ``audio_base64`` is a base64 WAV (recommended ≥5s)."""
    args: dict[str, Any] = {"audio_base64": audio_base64, "name": name}
    if metadata:
        args["metadata"] = metadata
    return _call_tool(
        hivemind_url, "hivemind.voice_identities.enroll@v1", args, timeout=30
    )


def voice_identities_identify(
    hivemind_url: str,
    *,
    audio_base64: str,
    top_k: int = 1,
    threshold: float | None = None,
) -> Any:
    """``hivemind.voice_identities.identify@v1`` — match an audio sample to
    enrolled identities. Returns ``{matches: [{identity_id, name, score}], ...}``.

    Live contract props: ``audio_base64``/``audio_data``,
    ``return_embeddings``, ``sample_rate``, ``threshold``. ``top_k`` is
    NOT a live parameter — kept in the Python signature for caller
    stability but not sent over the wire (the cluster returns ranked
    matches; MS4 picks the top one client-side)."""
    args: dict[str, Any] = {"audio_base64": audio_base64}
    if threshold is not None:
        args["threshold"] = threshold
    return _call_tool(
        hivemind_url,
        "hivemind.voice_identities.identify@v1",
        args,
        timeout=15,
    )


def voice_identities_refine(
    hivemind_url: str,
    *,
    identity_id: str,
    audio_base64: str | None = None,
    embedding: list[float] | None = None,
) -> Any:
    """``hivemind.voice_identities.refine@v1`` — refine an existing
    identity. Live contract: ``{name, embedding}`` (req=['name','embedding']).

    NOTE / KNOWN GAP: the live cluster refines by *embedding*, not by
    raw audio — ``audio_base64`` is not a live ``refine`` property
    (enroll accepts audio, refine does not). MS4 has no local embedding
    pipeline yet, so refine-by-audio cannot complete against this
    cluster contract. We translate ``identity_id`` → the live ``name``
    key and forward an ``embedding`` when one is supplied; passing only
    audio will surface the cluster's validation error (honest failure)."""
    args: dict[str, Any] = {"name": identity_id}
    if embedding is not None:
        args["embedding"] = embedding
    if audio_base64 is not None:
        # Forwarded for forward-compat; current live refine ignores/
        # rejects audio (see docstring gap note).
        args["audio_base64"] = audio_base64
    return _call_tool(
        hivemind_url,
        "hivemind.voice_identities.refine@v1",
        args,
        timeout=30,
    )


def voice_identities_delete(hivemind_url: str, identity_id: str) -> Any:
    """``hivemind.voice_identities.delete@v1`` — remove an enrolled
    identity. Live contract keys by ``name``."""
    return _call_tool(
        hivemind_url, "hivemind.voice_identities.delete@v1", {"name": identity_id}
    )


# ===========================================================================
# Human-in-the-loop
# ===========================================================================


def human_approval_request(
    hivemind_url: str,
    *,
    action_id: str,
    summary: str,
    details: dict[str, Any] | None = None,
    risk_level: str = "medium",
    timeout_secs: int = 300,
) -> Any:
    """``hivemind.human.approval.request@v1`` — request human approval for
    an action. Returns a request id immediately; poll
    :func:`human_approval_status` until decided or timed out."""
    args: dict[str, Any] = {
        "action_id": action_id,
        "summary": summary,
        "risk_level": risk_level,
        "timeout_secs": timeout_secs,
    }
    if details:
        args["details"] = details
    return _call_tool(hivemind_url, "hivemind.human.approval.request@v1", args, timeout=15)


def human_approval_status(hivemind_url: str, request_id: str) -> Any:
    """``hivemind.human.approval.status@v1`` — poll for a decision.
    Returns ``{decision: 'pending'|'approved'|'rejected'|'timeout', ...}``."""
    return _call_tool(
        hivemind_url, "hivemind.human.approval.status@v1", {"request_id": request_id}
    )


def human_notify(
    hivemind_url: str,
    *,
    channel: str,
    message: str,
    severity: str = "info",
) -> Any:
    """``hivemind.human.notify@v1`` — fire-and-forget operator notification.
    Channel typically ``telegram`` or ``log``."""
    return _call_tool(
        hivemind_url,
        "hivemind.human.notify@v1",
        {"channel": channel, "message": message, "severity": severity},
    )


def human_telegram_poll(hivemind_url: str, since_ts: int | None = None) -> Any:
    """``hivemind.human.telegram.poll@v1`` — pull recent Telegram operator
    messages (when configured). Optional ``since_ts`` UNIX timestamp."""
    args = {"since_ts": since_ts} if since_ts else {}
    return _call_tool(hivemind_url, "hivemind.human.telegram.poll@v1", args)


# ===========================================================================
# Services & maintenance windows
# ===========================================================================


def services_maintenance_enter(
    hivemind_url: str,
    *,
    service_name: str,
    reason: str | None = None,
    duration_secs: int | None = None,
) -> Any:
    """``hivemind.services.maintenance.enter@v1`` — declare a service to be
    in a maintenance window (suppresses health alerts, signals to
    callers that the service is intentionally degraded)."""
    args: dict[str, Any] = {"service_name": service_name}
    if reason:
        args["reason"] = reason
    if duration_secs:
        args["duration_secs"] = duration_secs
    return _call_tool(
        hivemind_url, "hivemind.services.maintenance.enter@v1", args
    )


def services_maintenance_clear(hivemind_url: str, service_name: str) -> Any:
    """``hivemind.services.maintenance.clear@v1`` — leave maintenance mode."""
    return _call_tool(
        hivemind_url,
        "hivemind.services.maintenance.clear@v1",
        {"service_name": service_name},
    )


def is_service_in_maintenance(hivemind_url: str, service_name: str) -> bool:
    """Convenience helper for callers that just want a boolean. Resolves
    the maintenance state by checking ``hivemind.service_health@v1``
    for the service's metadata; returns ``False`` on any error so we
    never block work on a stale maintenance-flag lookup."""
    try:
        from .hivemind_state import get_service_health

        health = get_service_health(hivemind_url)
        entry = (health or {}).get(service_name)
        if isinstance(entry, dict):
            return bool(entry.get("maintenance"))
    except Exception:
        pass
    return False


# ===========================================================================
# Cluster time
# ===========================================================================


def time_now(hivemind_url: str) -> Any:
    """``hivemind.time.now@v1`` — authoritative cluster time. Useful for
    Face Lobe context-block grounding so the model isn't relying on
    the operator's local clock if MS4 is running on a different
    timezone than the cluster."""
    return _call_tool(hivemind_url, "hivemind.time.now@v1", {}, timeout=5)


def time_convert(
    hivemind_url: str, *, value: str, from_format: str, to_format: str
) -> Any:
    """``hivemind.time.convert@v1`` — convert between time formats / timezones."""
    return _call_tool(
        hivemind_url,
        "hivemind.time.convert@v1",
        {"value": value, "from_format": from_format, "to_format": to_format},
    )


# ===========================================================================
# Models
# ===========================================================================


def models_list(hivemind_url: str) -> Any:
    """``hivemind.models.list@v1`` — native resilient catalog with HLI
    fallback."""
    return _call_tool(hivemind_url, "hivemind.models.list@v1", {}, timeout=30)


def models_catalog(hivemind_url: str) -> Any:
    """``hivemind.models.catalog@v1`` — full catalog with metadata."""
    return _call_tool(hivemind_url, "hivemind.models.catalog@v1", {}, timeout=30)


def models_recommend(
    hivemind_url: str,
    *,
    workload: str,
    constraints: dict[str, Any] | None = None,
) -> Any:
    """``hivemind.models.recommend@v1`` — let HiveMind pick a model for a
    workload description (e.g. ``"foreground_chat_small"``,
    ``"depth_reasoning_large"``, ``"vision_describe"``).

    Returns ``{recommended: [{model_id, score, reason}], ...}``. MS4's
    model_picker tries this first and falls back to its hand-rolled
    priority list when the recommendation is empty or fails.
    """
    args: dict[str, Any] = {"workload": workload}
    if constraints:
        args["constraints"] = constraints
    return _call_tool(hivemind_url, "hivemind.models.recommend@v1", args, timeout=10)


# ===========================================================================
# Capability matrix
# ===========================================================================


def capability_matrix(hivemind_url: str) -> Any:
    """``hivemind.capability.matrix@v1`` — what each cluster node can do
    (which GIMs available, which models loaded, vGPU support, etc.).
    Useful for scheduling decisions and for the UI's cluster status."""
    return _call_tool(hivemind_url, "hivemind.capability.matrix@v1")


# ===========================================================================
# Files / search / web / http
# ===========================================================================


def files_read(hivemind_url: str, path: str) -> Any:
    """``hivemind.files.read@v1`` — read a file from a cluster node."""
    return _call_tool(hivemind_url, "hivemind.files.read@v1", {"path": path})


def search_local(hivemind_url: str, query: str, *, max_results: int = 20) -> Any:
    """``hivemind.search.local@v1`` — search across cluster-local indexed
    content."""
    return _call_tool(
        hivemind_url,
        "hivemind.search.local@v1",
        {"query": query, "max_results": max_results},
    )


def web_search(hivemind_url: str, query: str, *, max_results: int = 10) -> Any:
    """``hivemind.web.search@v1`` — public web search."""
    return _call_tool(
        hivemind_url, "hivemind.web.search@v1", {"query": query, "max_results": max_results}
    )


def web_fetch(hivemind_url: str, url: str) -> Any:
    """``hivemind.web.fetch@v1`` — fetch a URL through HiveMind's outbound
    proxy."""
    return _call_tool(hivemind_url, "hivemind.web.fetch@v1", {"url": url}, timeout=30)


def http_fetch(hivemind_url: str, url: str, **opts: Any) -> Any:
    """``hivemind.http.fetch@v1`` — lower-level HTTP fetch with method/body
    control."""
    args = {"url": url, **opts}
    return _call_tool(hivemind_url, "hivemind.http.fetch@v1", args, timeout=30)


# ===========================================================================
# Embeddings & VLM
# ===========================================================================


def embeddings_create(hivemind_url: str, *, input_text: str | list[str], model: str | None = None) -> Any:
    """``hivemind.embeddings.create@v1`` — produce embedding vectors."""
    args: dict[str, Any] = {"input": input_text}
    if model:
        args["model"] = model
    return _call_tool(hivemind_url, "hivemind.embeddings.create@v1", args, timeout=30)


def vlm_chat(
    hivemind_url: str,
    *,
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> Any:
    """``hivemind.vlm.chat@v1`` — multi-turn VLM conversation. ``messages``
    follow the OpenAI chat shape with image-content support."""
    args: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        args["model"] = model
    return _call_tool(hivemind_url, "hivemind.vlm.chat@v1", args, timeout=60)


def vlm_describe_image(
    hivemind_url: str,
    *,
    image_base64: str,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> Any:
    """``hivemind.vlm.describe_image@v1`` — single-shot image description.
    Used as the fallback when ``vlm_chat`` isn't reachable."""
    args: dict[str, Any] = {
        "image_base64": image_base64,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        args["model"] = model
    return _call_tool(
        hivemind_url, "hivemind.vlm.describe_image@v1", args, timeout=60
    )


# ===========================================================================
# Images (vision-side generation / detection / OCR)
# ===========================================================================


def images_generate(hivemind_url: str, *, prompt: str, **opts: Any) -> Any:
    """``hivemind.images.generate@v1`` — text-to-image."""
    args = {"prompt": prompt, **opts}
    return _call_tool(hivemind_url, "hivemind.images.generate@v1", args, timeout=120)


def images_detect(hivemind_url: str, *, image_base64: str, **opts: Any) -> Any:
    """``hivemind.images.detect@v1`` — object detection."""
    args = {"image_base64": image_base64, **opts}
    return _call_tool(hivemind_url, "hivemind.images.detect@v1", args, timeout=60)


def images_ocr(hivemind_url: str, *, image_base64: str) -> Any:
    """``hivemind.images.ocr@v1`` — extract text from an image."""
    return _call_tool(
        hivemind_url, "hivemind.images.ocr@v1", {"image_base64": image_base64}, timeout=60
    )


# ===========================================================================
# API keys / auth introspection
# ===========================================================================


def api_keys_status(hivemind_url: str) -> Any:
    """``hivemind.api_keys.status@v1`` — redacted view of configured API
    keys (does NOT return secrets). Useful for the Settings panel to
    show which providers HiveMind is wired up to."""
    return _call_tool(hivemind_url, "hivemind.api_keys.status@v1")


# ===========================================================================
# Ollama facade
# ===========================================================================


def ollama_tags(hivemind_url: str) -> Any:
    """``hivemind.ollama.tags@v1`` — list locally available Ollama models."""
    return _call_tool(hivemind_url, "hivemind.ollama.tags@v1")


def ollama_service_control(hivemind_url: str, action: str) -> Any:
    """``hivemind.ollama.service_control@v1`` — start/stop/restart the
    Ollama backend."""
    return _call_tool(
        hivemind_url, "hivemind.ollama.service_control@v1", {"action": action}
    )


# ===========================================================================
# Crown / Neuro (thin pass-through; MS4 doesn't drive these but
# exposes them so other agents using the MS4 MCP can.)
# ===========================================================================


def crown_status(hivemind_url: str) -> Any:
    """``hivemind.crown.status@v1`` — Crown SDK headset/session status."""
    return _call_tool(hivemind_url, "hivemind.crown.status@v1")


def crown_latest(hivemind_url: str) -> Any:
    """``hivemind.crown.latest@v1`` — most recent Crown event snapshot."""
    return _call_tool(hivemind_url, "hivemind.crown.latest@v1")


def crown_signal_quality(hivemind_url: str) -> Any:
    """``hivemind.crown.signal_quality@v1`` — current EEG signal quality."""
    return _call_tool(hivemind_url, "hivemind.crown.signal_quality@v1")


# Crown — operationally useful subset of the 25-tool surface.
def crown_session_current(hivemind_url: str) -> Any:
    """``hivemind.crown.session.current@v1`` — currently active Crown session."""
    return _call_tool(hivemind_url, "hivemind.crown.session.current@v1")


def crown_session_start(hivemind_url: str, **opts: Any) -> Any:
    """``hivemind.crown.session.start@v1`` — start a Crown session."""
    return _call_tool(hivemind_url, "hivemind.crown.session.start@v1", opts)


def crown_session_stop(hivemind_url: str) -> Any:
    """``hivemind.crown.session.stop@v1`` — stop the active Crown session."""
    return _call_tool(hivemind_url, "hivemind.crown.session.stop@v1")


def crown_events_list(hivemind_url: str, *, limit: int = 50) -> Any:
    """``hivemind.crown.events.list@v1`` — recent Crown events."""
    return _call_tool(hivemind_url, "hivemind.crown.events.list@v1", {"limit": limit})


def crown_marker_add(hivemind_url: str, *, label: str, **opts: Any) -> Any:
    """``hivemind.crown.marker.add@v1`` — push a labelled marker into the
    current Crown session timeline."""
    args = {"label": label, **opts}
    return _call_tool(hivemind_url, "hivemind.crown.marker.add@v1", args)


def crown_triggers_list(hivemind_url: str) -> Any:
    """``hivemind.crown.triggers.list@v1`` — list configured triggers."""
    return _call_tool(hivemind_url, "hivemind.crown.triggers.list@v1")


def crown_triggers_fire_test(hivemind_url: str, *, trigger_id: str) -> Any:
    """``hivemind.crown.triggers.fire_test@v1`` — dry-fire a trigger."""
    return _call_tool(hivemind_url, "hivemind.crown.triggers.fire_test@v1", {"trigger_id": trigger_id})


def crown_triggers_emergency_enable(hivemind_url: str) -> Any:
    """``hivemind.crown.triggers.emergency_enable@v1`` — emergency on switch
    for all triggers."""
    return _call_tool(hivemind_url, "hivemind.crown.triggers.emergency_enable@v1")


def crown_triggers_emergency_disable(hivemind_url: str) -> Any:
    """``hivemind.crown.triggers.emergency_disable@v1`` — emergency off
    switch for all triggers."""
    return _call_tool(hivemind_url, "hivemind.crown.triggers.emergency_disable@v1")


def crown_trigger_packs_list(hivemind_url: str) -> Any:
    """``hivemind.crown.trigger_packs.list@v1`` — available trigger packs."""
    return _call_tool(hivemind_url, "hivemind.crown.trigger_packs.list@v1")


def crown_trigger_packs_activate(hivemind_url: str, *, pack_id: str) -> Any:
    """``hivemind.crown.trigger_packs.activate@v1`` — activate a pack."""
    return _call_tool(hivemind_url, "hivemind.crown.trigger_packs.activate@v1", {"pack_id": pack_id})


def crown_combos_list(hivemind_url: str) -> Any:
    """``hivemind.crown.combos.list@v1`` — list combos."""
    return _call_tool(hivemind_url, "hivemind.crown.combos.list@v1")


def crown_trees_list(hivemind_url: str) -> Any:
    """``hivemind.crown.trees.list@v1`` — list decision trees."""
    return _call_tool(hivemind_url, "hivemind.crown.trees.list@v1")


def crown_calibration_profiles_list(hivemind_url: str) -> Any:
    """``hivemind.crown.calibration_profiles.list@v1`` — calibration profiles."""
    return _call_tool(hivemind_url, "hivemind.crown.calibration_profiles.list@v1")


def crown_calibration_profiles_activate(hivemind_url: str, *, profile_id: str) -> Any:
    """``hivemind.crown.calibration_profiles.activate@v1`` — activate one."""
    return _call_tool(
        hivemind_url, "hivemind.crown.calibration_profiles.activate@v1", {"profile_id": profile_id}
    )


# ===========================================================================
# Oracle — HiveMind's planning/coordination/reasoning surface
# ===========================================================================


def oracle_status(hivemind_url: str) -> Any:
    """``hivemind.oracle.status``  — Oracle planner state."""
    return _call_tool(hivemind_url, "hivemind.oracle.status")


def oracle_configure(hivemind_url: str, *, config: dict[str, Any]) -> Any:
    """``hivemind.oracle.configure`` — push Oracle planner config."""
    return _call_tool(hivemind_url, "hivemind.oracle.configure", {"config": config})


def oracle_chat(hivemind_url: str, *, message: str, **opts: Any) -> Any:
    """``hivemind.oracle.chat`` — ask Oracle to plan / reason about a
    request. Returns the planner's reply; MS4 surfaces this for the
    operator's "what should I do next?" workflows."""
    args: dict[str, Any] = {"message": message, **opts}
    return _call_tool(hivemind_url, "hivemind.oracle.chat", args, timeout=60)


# ===========================================================================
# Training — fine-tuning / LoRA / adapter training
# ===========================================================================


def training_backends(hivemind_url: str) -> Any:
    """``hivemind.training.backends@v1`` — available training backends
    (e.g. forge LoRA, peft, etc.)."""
    return _call_tool(hivemind_url, "hivemind.training.backends@v1")


def training_start(hivemind_url: str, *, recipe: dict[str, Any]) -> Any:
    """``hivemind.training.start@v1`` — start a training job.

    The live cluster contract is FLAT (req=['agent']; props include
    ``agent``, ``base_model``, ``data_file``, ``data_source``,
    ``preset``) — NOT a nested ``{recipe: {...}}`` envelope. MS4 keeps
    ``recipe`` as the operator-facing object and unwraps it to the flat
    wire shape here."""
    if not isinstance(recipe, dict) or not recipe:
        raise ValueError("recipe must be a non-empty object")
    return _call_tool(hivemind_url, "hivemind.training.start@v1", dict(recipe), timeout=30)


def training_status(hivemind_url: str, *, job_id: str | None = None) -> Any:
    """``hivemind.training.status@v1`` — current training jobs or a
    specific job's status."""
    args = {"job_id": job_id} if job_id else {}
    return _call_tool(hivemind_url, "hivemind.training.status@v1", args)


# ===========================================================================
# Adapters — LoRA / PEFT adapter lifecycle
# ===========================================================================


def adapters_list(hivemind_url: str) -> Any:
    """``hivemind.adapters.list@v1`` — list available trained adapters."""
    return _call_tool(hivemind_url, "hivemind.adapters.list@v1")


def adapters_deploy(hivemind_url: str, *, adapter_id: str, target_model: str | None = None) -> Any:
    """``hivemind.adapters.deploy@v1`` — deploy an adapter.

    Live contract: req=['name']; props=['backend','name','version'].
    MS4's ``adapter_id`` maps to the live ``name``. There is no live
    ``target_model`` concept (deploy registers the named adapter; the
    runtime loads it); it is forwarded only if supplied, for
    forward-compat, and ignored by the current cluster."""
    args: dict[str, Any] = {"name": adapter_id}
    if target_model:
        args["target_model"] = target_model
    return _call_tool(hivemind_url, "hivemind.adapters.deploy@v1", args, timeout=60)


# ===========================================================================
# Loadout — model loadout profiles (which models loaded together)
# ===========================================================================


def loadout_profiles(hivemind_url: str) -> Any:
    """``hivemind.loadout.profiles@v1`` — list configured loadout profiles."""
    return _call_tool(hivemind_url, "hivemind.loadout.profiles@v1")


def loadout_apply(hivemind_url: str, *, profile_id: str, quality: str | None = None) -> Any:
    """``hivemind.loadout.apply@v1`` — apply a loadout (load / unload
    models to match).

    Live contract: req=['tier']; props=['quality','tier']. The live
    loadout model is tier-based, so MS4's selected ``profile_id`` is
    sent as the live ``tier`` value, with optional ``quality``."""
    args: dict[str, Any] = {"tier": profile_id}
    if quality:
        args["quality"] = quality
    return _call_tool(hivemind_url, "hivemind.loadout.apply@v1", args, timeout=60)


# ===========================================================================
# Deploy — provision a GIM (general inference microservice)
# ===========================================================================


def deploy_gim(hivemind_url: str, *, gim_name: str, **opts: Any) -> Any:
    """``hivemind.deploy.gim@v1`` — deploy / start a GIM by name."""
    args = {"gim_name": gim_name, **opts}
    return _call_tool(hivemind_url, "hivemind.deploy.gim@v1", args, timeout=60)


# ===========================================================================
# Inference — direct chat + model catalog via MCP
# ===========================================================================


def inference_models(hivemind_url: str) -> Any:
    """``hivemind.inference.models@v1`` — model catalog via the native
    resilient handler (same data as ``hivemind.models.list@v1`` but
    sometimes returns a richer payload)."""
    return _call_tool(hivemind_url, "hivemind.inference.models@v1")


def inference_chat(
    hivemind_url: str,
    *,
    messages: list[dict[str, Any]],
    model: str,
    timeout: float = 120,
    **opts: Any,
) -> Any:
    """``hivemind.inference.chat@v1`` — direct chat completion through
    the MCP gateway. ``messages`` follows OpenAI shape.

    ``timeout`` bounds the HTTP round-trip (seconds) and is NOT sent as
    a tool argument — callers that need a short, fail-safe call (e.g.
    the Double Agent continuation classifier) pass a small value here.
    Other ``opts`` (``max_tokens``, ``temperature``, ``stream``) ARE
    forwarded as tool arguments."""
    args: dict[str, Any] = {"messages": messages, "model": model, **opts}
    return _call_tool(hivemind_url, "hivemind.inference.chat@v1", args, timeout=timeout)


# ===========================================================================
# Logos — prompt optimization (logos_machina @ port 6120)
# ===========================================================================


def logos_prompts_list(hivemind_url: str) -> Any:
    """``hivemind.logos.prompts.list@v1`` — list managed prompts."""
    return _call_tool(hivemind_url, "hivemind.logos.prompts.list@v1")


def logos_prompts_get(hivemind_url: str, *, prompt_id: str) -> Any:
    """``hivemind.logos.prompts.get@v1`` — fetch one prompt + history."""
    return _call_tool(hivemind_url, "hivemind.logos.prompts.get@v1", {"prompt_id": prompt_id})


def logos_prompts_fork(hivemind_url: str, *, prompt_id: str, **opts: Any) -> Any:
    """``hivemind.logos.prompts.fork@v1`` — fork a prompt for variant
    optimization."""
    args = {"prompt_id": prompt_id, **opts}
    return _call_tool(hivemind_url, "hivemind.logos.prompts.fork@v1", args)


def logos_optimize(hivemind_url: str, *, prompt_id: str, **opts: Any) -> Any:
    """``hivemind.logos.optimize@v1`` — run the optimizer."""
    args = {"prompt_id": prompt_id, **opts}
    return _call_tool(hivemind_url, "hivemind.logos.optimize@v1", args, timeout=120)


def logos_evaluate_generate(hivemind_url: str, *, prompt_id: str, **opts: Any) -> Any:
    """``hivemind.logos.evaluate.generate@v1`` — generate evaluation
    candidates for a prompt."""
    args = {"prompt_id": prompt_id, **opts}
    return _call_tool(hivemind_url, "hivemind.logos.evaluate.generate@v1", args, timeout=120)


def logos_candidates_promote(hivemind_url: str, *, candidate_id: str) -> Any:
    """``hivemind.logos.candidates.promote@v1`` — promote a candidate to
    canonical."""
    return _call_tool(hivemind_url, "hivemind.logos.candidates.promote@v1", {"candidate_id": candidate_id})


# ===========================================================================
# Services lifecycle + jobs.cancel + math
# ===========================================================================


def services_list(hivemind_url: str, *, filter: str = "all") -> Any:
    """``hivemind.services.list@v1`` — all Warden-managed services."""
    return _call_tool(hivemind_url, "hivemind.services.list@v1", {"filter": filter})


def services_enable(hivemind_url: str, *, service_name: str) -> Any:
    """``hivemind.services.enable@v1`` — enable a disabled service."""
    return _call_tool(hivemind_url, "hivemind.services.enable@v1", {"service_name": service_name})


def services_disable(hivemind_url: str, *, service_name: str) -> Any:
    """``hivemind.services.disable@v1`` — disable a running service."""
    return _call_tool(hivemind_url, "hivemind.services.disable@v1", {"service_name": service_name})


def services_restart(hivemind_url: str, *, service_name: str) -> Any:
    """``hivemind.services.restart@v1`` — restart a Warden-managed
    service via Warden's restart route."""
    return _call_tool(hivemind_url, "hivemind.services.restart@v1", {"service_name": service_name}, timeout=60)


def jobs_cancel(hivemind_url: str, *, job_id: str = "", reason: str | None = None) -> Any:
    """``hivemind.jobs.cancel@v1`` — cancel a specific inference job.

    Live contract: req=['job_id']; props=['job_id','reason']. Per-job
    cancel IS now implemented upstream (the older "resets ALL jobs
    regardless of job_id" behavior is no longer current), so a real
    ``job_id`` is required; an empty value surfaces the cluster's
    validation error."""
    args: dict[str, Any] = {"job_id": job_id}
    if reason:
        args["reason"] = reason
    return _call_tool(hivemind_url, "hivemind.jobs.cancel@v1", args)


def math_calculate(hivemind_url: str, *, expression: str) -> Any:
    """``hivemind.math.calculate@v1`` — server-side calculator."""
    return _call_tool(hivemind_url, "hivemind.math.calculate@v1", {"expression": expression})


# ===========================================================================
# Game session orchestration (May 26 2026 — Phase 1 dry-run)
# ===========================================================================
#
# HiveMind shipped a full game-session orchestrator that MS4 had
# planned for. Phase 1 is intentionally a DRY-RUN: every phase
# records "would_call <hivemind.vm.X@v1>" evidence into the in-
# memory ledger but never invokes a mutating endpoint. When
# HiveMind ships Phase 2 (real execution) MS4's wrappers will keep
# working unchanged.


def game_ensure_available(hivemind_url: str, *, game_id: str) -> Any:
    """``hivemind.game.ensure_available@v1`` — read-only availability
    check for a game. Resolves ``game_id`` (accepts aliases like
    ``cyberpunk-2077``) to an env override path, default install
    path, or a golden VHDX. Returns ``{available: bool, ...}`` plus
    structured remediation when missing. Never downloads or installs.
    """
    if not game_id:
        raise ValueError("game_id is required")
    return _call_tool(hivemind_url, "hivemind.game.ensure_available@v1", {"game_id": game_id})


def game_session_plan(
    hivemind_url: str,
    *,
    game: str,
    client: str | None = None,
    duration_hint: str | None = None,
    latency: str | None = None,
    quality: str | None = None,
    **extra: Any,
) -> Any:
    """``hivemind.game_session.plan@v1`` — produce a dry-run Plan for
    a workload intent. Plan never reserves resources; it probes
    ``hivemind.hosts.list`` + ``hivemind.gpu.availability`` +
    ``hivemind.vm.list`` when HLI/VM-Manager are reachable, falls
    back to a synthetic single-host demo plan otherwise. Returns
    ``{job_id, plan, ...}``.
    """
    if not game:
        raise ValueError("game is required")
    args: dict[str, Any] = {"game": game}
    if client:
        args["client"] = client
    if duration_hint:
        args["duration_hint"] = duration_hint
    if latency:
        args["latency"] = latency
    if quality:
        args["quality"] = quality
    args.update(extra)
    return _call_tool(hivemind_url, "hivemind.game_session.plan@v1", args, timeout=20)


def game_session_run(hivemind_url: str, *, job_id: str) -> Any:
    """``hivemind.game_session.run@v1`` — walk the simulated state
    machine for a planned game-session job. Returns at a terminal
    state (``COMPLETE`` | ``FAILED_*`` | ``CANCELLED``). Pure dry-run
    in Phase 1."""
    if not job_id:
        raise ValueError("job_id is required")
    return _call_tool(hivemind_url, "hivemind.game_session.run@v1", {"job_id": job_id}, timeout=60)


def game_session_status(hivemind_url: str, *, job_id: str) -> Any:
    """``hivemind.game_session.status@v1`` — current state-machine
    position for a job + per-phase transitions + dry_run flag.
    Read-only; safe to poll while ``run`` is walking."""
    if not job_id:
        raise ValueError("job_id is required")
    return _call_tool(hivemind_url, "hivemind.game_session.status@v1", {"job_id": job_id})


def game_session_evidence(hivemind_url: str, *, job_id: str) -> Any:
    """``hivemind.game_session.evidence@v1`` — full per-phase evidence
    ledger (transitions, simulated actions, planner inputs,
    last_error). Read-only; safe at any time including after a
    terminal state. In-memory only in Phase 1 — restart of
    ``menta_game_session`` loses the ledger."""
    if not job_id:
        raise ValueError("job_id is required")
    return _call_tool(hivemind_url, "hivemind.game_session.evidence@v1", {"job_id": job_id})


def game_session_cancel(hivemind_url: str, *, job_id: str) -> Any:
    """``hivemind.game_session.cancel@v1`` — move a job to CANCELLED.
    Idempotent. Dry-run only in Phase 1 (no real VM/stream
    teardown)."""
    if not job_id:
        raise ValueError("job_id is required")
    return _call_tool(hivemind_url, "hivemind.game_session.cancel@v1", {"job_id": job_id})


# ===========================================================================
# Image content fallthrough — for vm.screenshot etc. that emit BOTH a
# text JSON envelope AND an MCP ``image`` content block
# ===========================================================================


def call_tool_with_image(
    hivemind_url: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout: int = 30,
) -> dict[str, Any]:
    """Call a tool that may emit BOTH ``text`` and ``image`` content
    blocks (e.g. ``hivemind.vm.screenshot@v1``).

    Returns ``{"json": <text-payload-as-dict>, "image_base64": <bytes
    or None>, "image_mime_type": <str or None>}``. The JSON block is
    parsed when valid; the image block is returned as the raw base64
    string the MCP server provided so MS4 callers can either stream
    it to the browser or re-embed it elsewhere.

    Falls back to plain :func:`_call_tool` semantics when the tool
    only returns text (just a wrapped JSON dict).
    """
    payload = {
        "jsonrpc": "2.0",
        "id": f"ms4-tool-img-{uuid.uuid4().hex[:8]}",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments or {}},
    }
    try:
        envelope = post_mcp_envelope(hivemind_url, payload, timeout=timeout)
    except HivemindStateError as exc:
        raise HivemindToolError(f"{tool_name!r}: transport failure: {exc}") from exc
    if not isinstance(envelope, dict):
        raise HivemindToolError(f"{tool_name!r}: non-object JSON-RPC envelope")
    if envelope.get("error"):
        raise HivemindToolError(f"{tool_name!r}: protocol error: {envelope['error']}")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise HivemindToolError(f"{tool_name!r}: missing result block")
    if result.get("isError"):
        text_chunks = [
            c.get("text", "")
            for c in (result.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        raise HivemindToolError(f"{tool_name!r}: tool error: {' '.join(text_chunks)[:240]}")

    out: dict[str, Any] = {"json": None, "image_base64": None, "image_mime_type": None}
    for chunk in result.get("content") or []:
        if not isinstance(chunk, dict):
            continue
        ctype = chunk.get("type")
        if ctype == "text" and out["json"] is None:
            text = chunk.get("text") or ""
            try:
                parsed = json.loads(text)
                if isinstance(parsed, (dict, list)):
                    out["json"] = parsed
                else:
                    out["json"] = {"raw_text": text}
            except json.JSONDecodeError:
                out["json"] = {"raw_text": text}
        elif ctype == "image" and out["image_base64"] is None:
            # MCP 2025-11-25 image content shape.
            out["image_base64"] = chunk.get("data") or chunk.get("image_base64") or ""
            out["image_mime_type"] = chunk.get("mimeType") or chunk.get("mime_type") or "image/png"
    return out


# ===========================================================================
# PsyKyo bridge (Phase A, May 26 2026; renamed 2026-05-26 PM) -- HiveMind MCP
# proxies these to PsyKyo's local MCP gateway on port 6765. The actual
# benchmark.run can take ~64s + asset-compile slack on first launch, so the
# timeout ceiling is intentionally generous. PsyKyo's vlm_consensus_gate@v1
# enforces llama3.2-vision + qwen3-vl consensus before any result is
# "promoted"; a needs_consensus / blocked promotion_status surfaces here as
# a regular return value (NOT a HivemindToolError) so the caller can decide
# whether to surface it as a Failed* terminal in their own state machine.
#
# The HiveMind wrapper namespace is `hivemind.psykyo.*` (matching the
# upstream PsyKyo brand name); the upstream PsyKyo tool ids stay
# `psykyo.*`. The HiveMind MCP gateway dispatcher in
# menta_mcp_gateway/src/main.rs rejects any PSYKYO_MCP_BASE override that
# isn't a loopback-only http URL with `psykyo_mcp_base_rejected`; wrappers
# below don't need to do their own host validation because the server
# enforces it on dispatch.
# ===========================================================================


def psykyo_benchmark_run(
    hivemind_url: str,
    *,
    resolution: str = "3840x2160",
    preset: str = "ultra",
    ray_tracing: str = "off",
    upscaler: str = "dlss",
    upscaler_quality: str = "quality",
    runs: int = 1,
    allow_launch: bool = True,
    allow_foreground: bool = True,
    allow_capture: bool = True,
    confirm_actuation_token: str = "PSYKYO_EXECUTE_SAFE_AUTO",
) -> Any:
    """``hivemind.psykyo.benchmark.run@v1`` — run the PsyKyo Cyberpunk 2077
    benchmark operator. Launches the game, drives the in-game settings + Run
    Benchmark UI via PsyKyo's deterministic visual operator, waits ~64s, and
    returns the FPS/variance/thermals report through PsyKyo's
    ``vlm_consensus_gate`` promotion."""
    args = {
        "resolution": resolution,
        "preset": preset,
        "ray_tracing": ray_tracing,
        "upscaler": upscaler,
        "upscaler_quality": upscaler_quality,
        "runs": runs,
        "allow_launch": allow_launch,
        "allow_foreground": allow_foreground,
        "allow_capture": allow_capture,
        "confirm_actuation_token": confirm_actuation_token,
    }
    return _call_tool(
        hivemind_url, "hivemind.psykyo.benchmark.run@v1", args, timeout=600
    )


def psykyo_benchmark_gap(hivemind_url: str) -> Any:
    """``hivemind.psykyo.benchmark.gap@v1`` — pre-flight: confirm PsyKyo's
    benchmark operator is reachable + the visual boxes still match the live
    game UI. Idempotent and side-effect-free."""
    return _call_tool(hivemind_url, "hivemind.psykyo.benchmark.gap@v1", timeout=30)


def psykyo_benchmark_workqueue(hivemind_url: str) -> Any:
    """``hivemind.psykyo.benchmark.workqueue@v1`` — list queued PsyKyo gap-gate
    work items (boxes that need review or replay before benchmark.run can be
    trusted). Read-only."""
    return _call_tool(
        hivemind_url, "hivemind.psykyo.benchmark.workqueue@v1", timeout=10
    )


def psykyo_evidence_latest(hivemind_url: str) -> Any:
    """``hivemind.psykyo.evidence.latest@v1`` — stable validation pointers and
    latest evidence summary. Read-only; safe without confirm_actuation_token."""
    return _call_tool(
        hivemind_url, "hivemind.psykyo.evidence.latest@v1", timeout=5
    )


def psykyo_vlm_consensus(
    hivemind_url: str,
    *,
    evidence_ref: str,
    models: list[str] | None = None,
) -> Any:
    """``hivemind.psykyo.vlm_consensus@v1`` — reconcile saved HiveMind VLM/OCR
    sidecar artifacts across models (llama3.2-vision + qwen3-vl MUST agree)
    before semantic promotion. Read-only check on existing evidence."""
    args: dict[str, Any] = {"evidence_ref": evidence_ref}
    if models is not None:
        args["models"] = models
    return _call_tool(
        hivemind_url, "hivemind.psykyo.vlm_consensus@v1", args, timeout=30
    )
