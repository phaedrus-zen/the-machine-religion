"""Per-domain admin module tests + REST route tests.

Covers ``vm_admin``, ``app_admin``, ``storage_admin``, ``network_admin``,
``gpu_mode_admin``, ``voice_identity``, and ``human_approval`` with both
unit tests against the typed wrappers and route tests against the
``Ms4GatewayHandler``. Shares the in-memory MCP server fixture from
:mod:`tests.ms4_gateway.test_hivemind_tools` via direct import — keeps
infrastructure in one place.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway import (
    app_admin,
    gpu_mode_admin,
    human_approval,
    network_admin,
    server as srv_module,
    storage_admin,
    vm_admin,
    voice_identity,
)


# ---------------------------------------------------------------------------
# Shared fake MCP fixture (re-implemented here so this test module is
# self-contained; minor duplication is fine vs cross-test imports).
# ---------------------------------------------------------------------------


class _FakeMcp:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int = 0

    def set_response(self, tool: str, payload: Any) -> None:
        self.responses[tool] = payload

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = {}
                outer.calls.append({"body": body})
                tool_name = (body.get("params") or {}).get("name") if isinstance(body, dict) else None
                payload = outer.responses.get(tool_name or "", {})
                envelope = {
                    "jsonrpc": "2.0",
                    "id": body.get("id", 1),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload)}],
                        "isError": False,
                    },
                }
                wire = json.dumps(envelope).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def stop(self) -> None:
        if self.server:
            try:
                self.server.shutdown()
            except Exception:
                pass


@pytest.fixture
def fake_mcp(monkeypatch):
    server = _FakeMcp()
    server.start()
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"http://127.0.0.1:{server.port}/mcp")
    monkeypatch.delenv("MS4_HIVEMIND_API_KEY", raising=False)
    yield server
    server.stop()


def hurl(fake: _FakeMcp) -> str:
    return f"http://127.0.0.1:{fake.port}"


# ===========================================================================
# vm_admin
# ===========================================================================


def test_vm_admin_list_normalizes_wrapped_response(fake_mcp):
    fake_mcp.set_response(
        "hivemind.vm.list@v1",
        {"vms": [{"id": "vm-1", "state": "running"}, {"id": "vm-2", "state": "stopped"}]},
    )
    vms = vm_admin.list_vms(hurl(fake_mcp))
    assert len(vms) == 2
    assert vms[0]["id"] == "vm-1"


def test_vm_admin_list_with_gpu_assignments_fail_soft(fake_mcp):
    fake_mcp.set_response("hivemind.vm.list@v1", {"vms": [{"id": "vm-1"}]})
    # No response for vm.gpus → that tool call's body is empty {}, which
    # is fine (an empty body is still a non-error); the snapshot just
    # contains an empty gpus map.
    snap = vm_admin.list_with_gpu_assignments(hurl(fake_mcp))
    assert snap["schema"] == "Ms4VmSnapshot.v1"
    assert len(snap["vms"]) == 1
    assert isinstance(snap["errors"], list)


def test_force_stop_requires_confirm(fake_mcp):
    with pytest.raises(ValueError):
        vm_admin.force_stop_vm(hurl(fake_mcp), "vm-1", confirm=False)


def test_delete_vm_requires_confirm(fake_mcp):
    with pytest.raises(ValueError):
        vm_admin.delete_vm(hurl(fake_mcp), "vm-1", confirm=False)


# ===========================================================================
# app_admin
# ===========================================================================


def test_app_admin_list_with_status_and_metrics(fake_mcp):
    fake_mcp.set_response(
        "hivemind.app.list@v1",
        {"apps": [{"id": "a1", "name": "first"}, {"id": "a2"}]},
    )
    fake_mcp.set_response("hivemind.app.status@v1", {"state": "running"})
    fake_mcp.set_response("hivemind.app.metrics@v1", {"cpu_pct": 12.5})
    snap = app_admin.list_with_status_and_metrics(hurl(fake_mcp))
    assert snap["schema"] == "Ms4AppSnapshot.v1"
    assert len(snap["apps"]) == 2
    for entry in snap["apps"]:
        assert entry["status"] == {"state": "running"}
        assert entry["metrics"] == {"cpu_pct": 12.5}


# ===========================================================================
# storage_admin
# ===========================================================================


def test_storage_admin_combined_snapshot(fake_mcp):
    fake_mcp.set_response("hivemind.storage.status@v1", {"healthy": True})
    fake_mcp.set_response("hivemind.storage.pools@v1", {"pools": [{"id": "p1"}]})
    fake_mcp.set_response("hivemind.storage.volumes@v1", {"volumes": []})
    fake_mcp.set_response("hivemind.storage.snapshots@v1", {"snapshots": []})
    snap = storage_admin.combined_snapshot(hurl(fake_mcp))
    assert snap["schema"] == "Ms4StorageSnapshot.v1"
    assert snap["status"] == {"healthy": True}
    assert snap["pools"] == [{"id": "p1"}]
    assert snap["errors"] == []


@pytest.mark.parametrize("fn,kwargs", [
    (storage_admin.delete_volume, {"volume_id": "v-1"}),
    (storage_admin.delete_snapshot, {"snapshot_id": "s-1"}),
    (storage_admin.restore_snapshot, {"snapshot_id": "s-1"}),
])
def test_storage_destructive_requires_confirm(fake_mcp, fn, kwargs):
    with pytest.raises(ValueError):
        fn(hurl(fake_mcp), confirm=False, **kwargs)


# ===========================================================================
# network_admin
# ===========================================================================


def test_network_admin_combined_snapshot(fake_mcp):
    fake_mcp.set_response("hivemind.network.list@v1", {"networks": [{"id": "n-1"}]})
    fake_mcp.set_response("hivemind.network.bridges@v1", {"bridges": []})
    fake_mcp.set_response("hivemind.network.interfaces@v1", {"interfaces": []})
    fake_mcp.set_response("hivemind.network.attachments@v1", {"attachments": []})
    snap = network_admin.combined_snapshot(hurl(fake_mcp))
    assert snap["schema"] == "Ms4NetworkSnapshot.v1"
    assert snap["networks"] == [{"id": "n-1"}]
    assert snap["errors"] == []


def test_network_delete_requires_confirm(fake_mcp):
    with pytest.raises(ValueError):
        network_admin.delete_network(hurl(fake_mcp), "n-1", confirm=False)


# ===========================================================================
# gpu_mode_admin
# ===========================================================================


def test_gpu_mode_combined_snapshot(fake_mcp):
    fake_mcp.set_response(
        "hivemind.gpu_mode.capabilities@v1",
        {"modes": ["passthrough", "vgpu", "shared"]},
    )
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_status@v1", {"vgpus": []})
    fake_mcp.set_response("hivemind.gpu.availability@v1", {"free": 2})
    snap = gpu_mode_admin.combined_snapshot(hurl(fake_mcp))
    assert snap["schema"] == "Ms4GpuSnapshot.v1"
    assert snap["capabilities"] == {"modes": ["passthrough", "vgpu", "shared"]}
    assert snap["availability"] == {"free": 2}


# ===========================================================================
# voice_identity
# ===========================================================================


def test_voice_identity_enroll_rejects_empty_audio(fake_mcp):
    with pytest.raises(ValueError):
        voice_identity.enroll(hurl(fake_mcp), name="Alice", audio=b"")


def test_voice_identity_enroll_rejects_empty_name(fake_mcp):
    with pytest.raises(ValueError):
        voice_identity.enroll(hurl(fake_mcp), name="", audio=b"\x00\x01")


def test_voice_identity_enroll_base64_encodes_and_returns(fake_mcp):
    fake_mcp.set_response("hivemind.voice_identities.enroll@v1", {"identity_id": "id-A"})
    result = voice_identity.enroll(hurl(fake_mcp), name="Alice", audio=b"WAV-DATA")
    assert result["identity_id"] == "id-A"
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args["name"] == "Alice"
    assert args["audio_base64"] == base64.b64encode(b"WAV-DATA").decode("ascii")


def test_voice_identity_refine_forwards_only_current_hli_embedding_contract(fake_mcp):
    fake_mcp.set_response(
        "hivemind.voice_identities.refine@v1",
        {"ok": True, "name": "Alice", "refined": True},
    )
    result = voice_identity.refine(
        hurl(fake_mcp),
        name=" Alice ",
        embedding=[0.1, 0.2, 0.3],
        blend_alpha=0.1,
        metadata={"source": "fixture"},
    )
    assert result["refined"] is True
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args == {
        "name": "Alice",
        "embedding": [0.1, 0.2, 0.3],
        "blend_alpha": 0.1,
        "metadata": {"source": "fixture"},
    }
    assert "audio_base64" not in args


def test_voice_identity_refine_rejects_missing_or_invalid_embedding():
    with pytest.raises(ValueError, match="embedding array"):
        voice_identity.refine("http://hive", name="Alice", embedding=[])
    with pytest.raises(ValueError, match="finite numbers"):
        voice_identity.refine("http://hive", name="Alice", embedding=[float("nan")])


def test_identify_speaker_honors_hli_below_threshold(fake_mcp):
    fake_mcp.set_response(
        "hivemind.voice_identities.identify@v1",
        {
            "ok": True,
            "name": None,
            "confidence": 0.30,
            "threshold": 0.75,
            "below_threshold": True,
        },
    )
    result = voice_identity.identify_speaker_from_wav(hurl(fake_mcp), audio=b"WAV")
    assert result is None


def test_identify_speaker_consumes_hli_flat_accepted_result(fake_mcp):
    fake_mcp.set_response(
        "hivemind.voice_identities.identify@v1",
        {
            "ok": True,
            "name": "Alice",
            "confidence": 0.85,
            "threshold": 0.75,
            "below_threshold": False,
        },
    )
    result = voice_identity.identify_speaker_from_wav(hurl(fake_mcp), audio=b"WAV")
    assert result is not None
    assert result["accepted"] is True
    assert result["name"] == "Alice"
    assert result["score"] == 0.85
    assert result["confidence"] == 0.85
    assert result["threshold"] == 0.75
    assert result["below_threshold"] is False
    assert "identity_id" not in result
    assert "min_score" not in result


def test_identify_speaker_does_not_apply_a_second_local_threshold(fake_mcp):
    fake_mcp.set_response(
        "hivemind.voice_identities.identify@v1",
        {
            "ok": True,
            "name": "FixtureSpeaker",
            "confidence": 0.60,
            "threshold": 0.55,
            "below_threshold": False,
        },
    )
    result = voice_identity.identify_speaker_from_wav(hurl(fake_mcp), audio=b"WAV")
    assert result is not None
    assert result["name"] == "FixtureSpeaker"
    assert result["accepted"] is True


def test_identify_speaker_rejects_obsolete_matches_shape(fake_mcp):
    fake_mcp.set_response(
        "hivemind.voice_identities.identify@v1",
        {"matches": [{"identity_id": "legacy", "name": "Legacy", "score": 0.99}]},
    )
    assert voice_identity.identify_speaker_from_wav(hurl(fake_mcp), audio=b"WAV") is None


def test_identify_speaker_fail_soft_on_empty_audio(fake_mcp):
    assert voice_identity.identify_speaker_from_wav(hurl(fake_mcp), audio=b"") is None


def test_identify_speaker_forwards_timeout_and_cancellation(monkeypatch):
    observed: dict[str, object] = {}
    cancel_event = threading.Event()

    def fake_identify(_url, *, audio_base64, top_k, timeout, cancel_event):
        observed.update({
            "audio_base64": audio_base64,
            "top_k": top_k,
            "timeout": timeout,
            "cancel_event": cancel_event,
        })
        return {
            "ok": True,
            "name": None,
            "confidence": 0.0,
            "threshold": 0.75,
            "below_threshold": True,
        }

    monkeypatch.setattr(voice_identity.tools, "voice_identities_identify", fake_identify)
    assert voice_identity.identify_speaker_from_wav(
        "http://hive",
        b"WAV",
        timeout=0.25,
        cancel_event=cancel_event,
    ) is None
    assert observed["timeout"] == 0.25
    assert observed["cancel_event"] is cancel_event
    assert observed["audio_base64"] == base64.b64encode(b"WAV").decode("ascii")


# ===========================================================================
# human_approval
# ===========================================================================


def test_gate_action_returns_deny_when_no_action_id(fake_mcp):
    decision = human_approval.gate_action_with_human(hurl(fake_mcp), {"summary": "x"})
    assert decision["schema"] == "EthicsDecision.v1"
    assert decision["decision"] == "deny"
    assert "action_id missing" in decision["reason"]


def test_gate_action_approved_path(fake_mcp):
    fake_mcp.set_response(
        "hivemind.human.approval.request@v1",
        {"request_id": "req-1"},
    )
    fake_mcp.set_response(
        "hivemind.human.approval.status@v1",
        {"decision": "approved", "reason": "ok"},
    )
    intent = {"action_id": "act-1", "spirit_id": "sister", "action_type": "vm.start"}
    decision = human_approval.gate_action_with_human(hurl(fake_mcp), intent, poll_secs=0.05)
    assert decision["decision"] == "allow"
    assert decision["source"] == "human-approval"


def test_gate_action_rejected_path(fake_mcp):
    fake_mcp.set_response("hivemind.human.approval.request@v1", {"request_id": "req-2"})
    fake_mcp.set_response(
        "hivemind.human.approval.status@v1",
        {"decision": "rejected", "reason": "operator said no"},
    )
    intent = {"action_id": "act-2", "spirit_id": "sister", "action_type": "vm.delete"}
    decision = human_approval.gate_action_with_human(hurl(fake_mcp), intent, poll_secs=0.05)
    assert decision["decision"] == "deny"


# ===========================================================================
# REST route tests against Ms4GatewayHandler
# ===========================================================================


class _DummyRunner:
    ms3_url = "http://ms3:9080"
    default_model = "qwen3-coder-next:latest"
    face_lobe_chat = types.SimpleNamespace(_sessions={})

    def __init__(self, hivemind_url: str):
        self.hivemind_url = hivemind_url


def _make_handler(runner: _DummyRunner, method: str, path: str, body: bytes = b"") -> srv_module.Ms4GatewayHandler:
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = runner
    handler.command = method
    handler.path = path
    headers = {"Host": "127.0.0.1:9180"}
    if body:
        headers["Content-Length"] = str(len(body))
        headers["Content-Type"] = "application/json"
    handler.headers = types.SimpleNamespace(get=headers.get)
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"
    return handler


def _read_response(handler) -> tuple[int, dict[str, Any]]:
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split(b" ")[1])
    head, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8") or "{}")


def test_route_hivemind_time(fake_mcp):
    fake_mcp.set_response("hivemind.time.now@v1", {"iso": "2026-05-26T12:00:00Z"})
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/time")
    handler._hivemind_time_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["iso"] == "2026-05-26T12:00:00Z"


def test_route_hivemind_vms(fake_mcp):
    fake_mcp.set_response("hivemind.vm.list@v1", {"vms": [{"id": "vm-1"}]})
    fake_mcp.set_response("hivemind.vm.gpus@v1", {"gpus": []})
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/vms")
    handler._hivemind_vms_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4VmSnapshot.v1"
    assert body["vms"][0]["id"] == "vm-1"


def test_route_vm_force_stop_rejects_without_confirm(fake_mcp):
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "POST", "/hivemind/vms/vm-1/force_stop", body=b"{}")
    handled = handler._dispatch_vm_action("vm-1", "force_stop")
    assert handled is True
    status, body = _read_response(handler)
    assert status == 400
    assert "confirm" in body["error"]


def test_route_vm_force_stop_with_confirm(fake_mcp):
    fake_mcp.set_response("hivemind.vm.force_stop@v1", {"ok": True})
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(
        runner, "POST", "/hivemind/vms/vm-1/force_stop", body=b'{"confirm": true}'
    )
    handler._dispatch_vm_action("vm-1", "force_stop")
    status, body = _read_response(handler)
    assert status == 200
    assert body == {"ok": True}


def test_route_apps_snapshot(fake_mcp):
    fake_mcp.set_response("hivemind.app.list@v1", {"apps": [{"id": "a1"}]})
    fake_mcp.set_response("hivemind.app.status@v1", {"state": "running"})
    fake_mcp.set_response("hivemind.app.metrics@v1", {"cpu_pct": 5.0})
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/apps")
    handler._hivemind_apps_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4AppSnapshot.v1"
    assert len(body["apps"]) == 1


def test_route_voice_identities_list(fake_mcp):
    fake_mcp.set_response(
        "hivemind.voice_identities.list@v1",
        {"identities": [{"id": "v-1", "name": "Alice"}]},
    )
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/voice_identities")
    handler._hivemind_voice_identities_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4VoiceIdentitiesSnapshot.v1"
    assert body["identities"][0]["name"] == "Alice"


def test_route_voice_identity_refine_requires_embedding_not_audio(fake_mcp):
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(
        runner,
        "POST",
        "/hivemind/voice_identities/Alice/refine",
        body=b'{"audio_base64":"V0FW"}',
    )
    assert handler._dispatch_voice_identity_action("Alice", "refine") is True
    status, body = _read_response(handler)
    assert status == 400
    assert "embedding array required" in body["error"]
    assert fake_mcp.calls == []


def test_route_voice_identity_refine_forwards_embedding(monkeypatch, fake_mcp):
    fake_mcp.set_response(
        "hivemind.voice_identities.refine@v1",
        {"ok": True, "name": "Alice", "refined": True},
    )
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(srv_module, "append_event", lambda name, payload: events.append((name, payload)))
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(
        runner,
        "POST",
        "/hivemind/voice_identities/Alice/refine",
        body=b'{"embedding":[0.1,0.2,0.3],"blend_alpha":0.1,"metadata":{"source":"fixture"}}',
    )
    assert handler._dispatch_voice_identity_action("Alice", "refine") is True
    status, body = _read_response(handler)
    assert status == 200
    assert body["refined"] is True
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args == {
        "name": "Alice",
        "embedding": [0.1, 0.2, 0.3],
        "blend_alpha": 0.1,
        "metadata": {"source": "fixture"},
    }
    assert events == [(
        "hivemind_voice_identity_refine",
        {"name": "Alice", "embedding_dims": 3},
    )]


def test_route_approval_request_validates_required_fields(fake_mcp):
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(
        runner, "POST", "/hivemind/approval/request", body=b'{"summary":"x"}'
    )
    handler._hivemind_approval_request()
    status, body = _read_response(handler)
    assert status == 400
    assert "action_id" in body["error"]


def test_route_approval_status(fake_mcp):
    fake_mcp.set_response(
        "hivemind.human.approval.status@v1",
        {"decision": "pending", "request_id": "r-1"},
    )
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/approval/status/r-1")
    handler._hivemind_approval_status_get("r-1")
    status, body = _read_response(handler)
    assert status == 200
    assert body["decision"] == "pending"


def test_route_storage_combined(fake_mcp):
    fake_mcp.set_response("hivemind.storage.status@v1", {"healthy": True})
    fake_mcp.set_response("hivemind.storage.pools@v1", {"pools": []})
    fake_mcp.set_response("hivemind.storage.volumes@v1", {"volumes": []})
    fake_mcp.set_response("hivemind.storage.snapshots@v1", {"snapshots": []})
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/storage")
    handler._hivemind_storage_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4StorageSnapshot.v1"


def test_route_network_combined(fake_mcp):
    fake_mcp.set_response("hivemind.network.list@v1", {"networks": []})
    fake_mcp.set_response("hivemind.network.bridges@v1", {"bridges": []})
    fake_mcp.set_response("hivemind.network.interfaces@v1", {"interfaces": []})
    fake_mcp.set_response("hivemind.network.attachments@v1", {"attachments": []})
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/network")
    handler._hivemind_network_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4NetworkSnapshot.v1"


def test_route_gpu_combined(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.capabilities@v1", {"modes": []})
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_status@v1", {"vgpus": []})
    fake_mcp.set_response("hivemind.gpu.availability@v1", {"free": 0})
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/gpu")
    handler._hivemind_gpu_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4GpuSnapshot.v1"


def test_route_capability_matrix(fake_mcp):
    fake_mcp.set_response("hivemind.capability.matrix@v1", {"nodes": []})
    runner = _DummyRunner(hurl(fake_mcp))
    handler = _make_handler(runner, "GET", "/hivemind/capability_matrix")
    handler._hivemind_capability_matrix_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body == {"nodes": []}
