"""Request-local HLI recommend-lease propagation for exact Face warm+chat.

HLI contract (read-only source):
``menta_hli/gateway/api/src/routing/recommend_affinity.rs``
"""

from __future__ import annotations

import email.message
import http.client
import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from machine_spirit_4.double_agent.model_picker import (
    EXACT_FACE_CLUSTER_ADMISSION_SCHEMA,
    ForegroundModelUnavailable,
    admit_cluster_recommendation,
    choose_foreground_model,
    require_exact_face_cluster_admission,
)
from machine_spirit_4.double_agent import recommend_lease as lease_mod
from machine_spirit_4.gateway import face_lobe_chat as face_module
from machine_spirit_4.gateway import hivemind_tools
from machine_spirit_4.gateway import voice as voice_module
from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat, FaceLobeChatError


NOW_S = 1_700_000_000.0
LEASE_A = "lease-a-11111111"
LEASE_B = "lease-b-22222222"
OP_UUID = "018f0000-0000-7000-8000-0000000000aa"
OP_B_UUID = "018f0000-0000-7000-8000-0000000000bb"
HLI = "http://127.0.0.1:6089"
MODEL = "nemotron-3-nano:4b"
OWNER = "node-a"
ENDPOINT = "http://192.0.2.10:6089/v1/chat/completions"


def _typed_document(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": EXACT_FACE_CLUSTER_ADMISSION_SCHEMA,
        "capability": "chat",
        "recommended_model": MODEL,
        "model": MODEL,
        "backend": "ollama",
        "ready": True,
        "owner": OWNER,
        "node": OWNER,
        "endpoint": ENDPOINT,
        "snapshot_age_s": 1.0,
        "current_generation": 7,
        "lease": {
            "id": LEASE_A,
            "issued": NOW_S - 5,
            "expires": NOW_S + 30,
            "generation": 7,
        },
        "phase": "Active",
        "pending": False,
    }
    lease_override = overrides.pop("lease", None)
    body.update(overrides)
    if isinstance(lease_override, dict):
        merged = dict(body["lease"])
        merged.update(lease_override)
        body["lease"] = merged
    elif lease_override is not None:
        body["lease"] = lease_override
    return body


def _receipt() -> dict[str, Any]:
    return admit_cluster_recommendation(
        _typed_document(),
        catalog_ids={MODEL},
        requested_model=MODEL,
        now_s=NOW_S,
    )


class _TimeoutSock:
    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout


class _ResponseFp:
    def __init__(self, buf: io.BytesIO) -> None:
        self.raw = type("Raw", (), {"_sock": _TimeoutSock()})()
        self._buf = buf

    def read(self, *args: Any) -> bytes:
        return self._buf.read(*args)

    def readline(self, *args: Any) -> bytes:
        return self._buf.readline(*args)

    def close(self) -> None:
        return None


class _CapturedHTTPResponse:
    """urlopen-shaped response from an executed HTTPConnection.request."""

    def __init__(self, payload: bytes, status: int = 200, headers: dict | None = None):
        self.status = status
        self.code = status
        self.reason = "OK" if status == 200 else "Error"
        self.msg = self.reason
        self.headers = http.client.HTTPMessage()
        for key, value in (headers or {"Content-Type": "application/json"}).items():
            self.headers[key] = value
        self._buf = io.BytesIO(payload)
        self.fp = _ResponseFp(self._buf)
        self.length = len(payload)
        self.chunked = False
        self.will_close = True
        self.version = 11
        self.url = ""
        self.closed = False

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            return self._buf.read()
        return self._buf.read(amt)

    def readline(self, amt: int = -1) -> bytes:
        return self._buf.readline(amt)

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)

    def getheaders(self) -> list[tuple[str, str]]:
        return list(self.headers.items())

    def info(self) -> http.client.HTTPMessage:
        return self.headers

    def getcode(self) -> int:
        return self.code

    def close(self) -> None:
        self.closed = True

    def isclosed(self) -> bool:
        return self.closed

    def __enter__(self) -> "_CapturedHTTPResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _HliCapture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.recommend_bodies: list[Any] = []
        self.poll_documents: dict[str, list[Any]] = {}
        self.allow_job_cancel = False
        self.chat_replies: list[bytes] = [
            json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ready"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")
        ]
        self.stream_chunks: list[bytes] = [
            b'data: {"choices":[{"delta":{"content":"ready"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]

    def queue_poll(self, operation_id: str, document: Any) -> None:
        self.poll_documents.setdefault(str(operation_id), []).append(document)

    def _record_executed(self, recorded: dict[str, Any]) -> None:
        with self._lock:
            self.calls.append(recorded)

    def _respond(self, recorded: dict[str, Any]) -> _CapturedHTTPResponse:
        method = recorded["method"]
        path = recorded["path"]
        body = recorded["body"]
        with self._lock:
            if method == "GET" and path.rstrip("/") == "/v1/resources/recommend":
                payload = self.recommend_bodies.pop(0)
                return _CapturedHTTPResponse(json.dumps(payload).encode("utf-8"))
            if method == "GET" and path.startswith("/v1/resources/recommend/operations/"):
                operation_id = path.rstrip("/").rsplit("/", 1)[-1]
                queued = self.poll_documents.get(operation_id)
                if not queued:
                    raise AssertionError(
                        f"unexpected poll {method} {recorded['url']}"
                    )
                payload = queued.pop(0)
                return _CapturedHTTPResponse(json.dumps(payload).encode("utf-8"))
            if method == "POST" and path.endswith("/v1/chat/completions"):
                parsed_body = body if isinstance(body, dict) else {}
                if parsed_body.get("stream"):
                    return _CapturedHTTPResponse(
                        b"".join(self.stream_chunks),
                        headers={"Content-Type": "text/event-stream"},
                    )
                return _CapturedHTTPResponse(self.chat_replies[0])
            if method == "DELETE" and path.startswith("/jobs/"):
                if not self.allow_job_cancel:
                    raise AssertionError(
                        f"unexpected HLI call {method} {recorded['url']}"
                    )
                return _CapturedHTTPResponse(b"{}", 200)
        raise AssertionError(f"unexpected HLI call {method} {recorded['url']}")


def _install_hli(monkeypatch: pytest.MonkeyPatch, capture: _HliCapture) -> None:
    """Capture the HTTPConnection.request urllib actually executes."""

    real_http = http.client.HTTPConnection

    class FakeConnection:
        sock = None
        _get_content_length = staticmethod(real_http._get_content_length)

        def __init__(self, host, port=None, timeout=None, **_kwargs):
            if port is None and isinstance(host, str) and host.count(":") == 1:
                host_part, maybe_port = host.rsplit(":", 1)
                if maybe_port.isdigit():
                    host = host_part
                    port = int(maybe_port)
            self.host = host
            self.port = port
            self.timeout = timeout
            self.debuglevel = 0
            self._recorded: dict[str, Any] | None = None

        def set_debuglevel(self, level: int) -> None:
            self.debuglevel = level

        def set_tunnel(self, host, port=None, headers=None) -> None:
            raise AssertionError("lease harness must not open an HTTP tunnel")

        def connect(self) -> None:
            raise AssertionError("lease harness must not call connect()")

        def request(self, method, url, body=None, headers=None, *, encode_chunked=False):
            header_map = {
                str(key): str(value) for key, value in dict(headers or {}).items()
            }
            origin = f"http://{self.host}"
            if self.port is not None:
                origin = f"http://{self.host}:{self.port}"
            full_url = f"{origin}{url}"
            parsed = urllib.parse.urlparse(full_url)
            payload: Any = None
            if isinstance(body, bytes) and body:
                payload = json.loads(body.decode("utf-8"))
            elif isinstance(body, str) and body:
                payload = json.loads(body)
            self._recorded = {
                "method": method,
                "url": full_url,
                "path": parsed.path,
                "query": urllib.parse.parse_qs(parsed.query),
                "headers": {key.lower(): value for key, value in header_map.items()},
                "body": payload,
                "host": self.host,
                "port": self.port,
                "selector": url,
            }

        def getresponse(self) -> _CapturedHTTPResponse:
            recorded = self._recorded
            if recorded is None:
                raise AssertionError("getresponse() before request()")
            capture._record_executed(recorded)
            return capture._respond(recorded)

        def close(self) -> None:
            return None

    class FakeHTTPSConnection(FakeConnection):
        pass

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeHTTPSConnection)
    monkeypatch.setattr(lease_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.time.time",
        lambda: NOW_S,
    )


def _chat_lease_calls(capture: _HliCapture) -> list[dict[str, Any]]:
    return [c for c in capture.calls if c["path"].endswith("/v1/chat/completions")]


def _header_set(call: dict[str, Any]) -> dict[str, str]:
    headers = call["headers"]
    return {
        "lease": headers.get("x-hivemind-recommend-lease", ""),
        "owner": headers.get("x-hivemind-recommend-owner", ""),
        "endpoint": headers.get("x-hivemind-recommend-endpoint", ""),
        "generation": headers.get("x-hivemind-recommend-generation", ""),
        "issued": headers.get("x-hivemind-recommend-issued", ""),
        "expires": headers.get("x-hivemind-recommend-expires", ""),
    }


def test_exact_warm_and_production_chat_send_the_same_lease_headers(monkeypatch):
    capture = _HliCapture()
    capture.recommend_bodies = [_typed_document()]
    _install_hli(monkeypatch, capture)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker._http_get_models",
        lambda *_a, **_k: [{"id": MODEL, "object": "model", "created": 0, "owned_by": "org"}],
    )

    chat = FaceLobeChat(hivemind_url=HLI, http_timeout=5)
    result = voice_module.prewarm_face_lobe_model(
        hivemind_url=HLI,
        model=MODEL,
        exact_model=True,
        timeout=5,
        latency_budget_ms=15_000,
        face_chat=chat,
    )
    assert result["warmed"] is True
    chat.chat(
        "hello",
        session_id="face-1",
        model=MODEL,
        recommend_receipt=result["recommend"],
        recommend_validate_only=False,
    )
    chats = _chat_lease_calls(capture)
    assert len(chats) >= 3  # residency warm, TTFT probe, production chat
    header_rows = [_header_set(c) for c in chats]
    assert {row["lease"] for row in header_rows} == {LEASE_A}
    for row in header_rows:
        assert row["owner"] == OWNER
        assert row["endpoint"] == ENDPOINT
        assert row["generation"] == "7"
        assert row["issued"]
        assert row["expires"]
    warm = [c for c in chats if c["body"].get("prewarm") is True or c["body"].get("hivemind_prewarm") is True]
    consume = [c for c in chats if c["body"].get("prewarm") is not True and c["body"].get("hivemind_prewarm") is not True]
    assert warm, "residency/admission warm must present prewarm=true so HLI does not consume"
    assert consume, "production chat/TTFT must omit prewarm so HLI consumes on first visible token"
    assert all(c["method"] == "POST" for c in chats)
    assert all(c["host"] == "127.0.0.1" and c["port"] == 6089 for c in chats)
    assert not any(c["method"] == "DELETE" for c in capture.calls)
    assert not any("/resources/recommend/leases/" in c["url"] for c in capture.calls)
    assert not any(c["method"] == "POST" and "recommend" in c["path"] for c in capture.calls)


def test_pending_admission_polls_the_same_hli_operation_uuid_and_never_reposts(monkeypatch):
    capture = _HliCapture()
    capture.recommend_bodies = [
        {
            "pending": True,
            "poll": True,
            "phase": "Pending",
            "operation_id": OP_UUID,
            "poll_url": f"/v1/resources/recommend/operations/{OP_UUID}",
            "poll_path": f"/v1/resources/recommend/operations/{OP_UUID}",
        }
    ]
    capture.queue_poll(OP_UUID, _typed_document(phase="Active", pending=False))
    _install_hli(monkeypatch, capture)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker._http_get_models",
        lambda *_a, **_k: [{"id": MODEL}],
    )
    admitted = require_exact_face_cluster_admission(
        hivemind_url=HLI,
        requested_model=MODEL,
        now_s=NOW_S,
    )
    assert admitted["lease"]["id"] == LEASE_A
    methods_paths = [(c["method"], c["path"]) for c in capture.calls]
    assert methods_paths[0] == ("GET", "/v1/resources/recommend")
    assert methods_paths[1] == ("GET", f"/v1/resources/recommend/operations/{OP_UUID}")
    assert capture.calls[1]["selector"].endswith(OP_UUID)
    assert all(method == "GET" for method, _path in methods_paths)
    assert capture.calls[0]["query"]["schema"] == [EXACT_FACE_CLUSTER_ADMISSION_SCHEMA]
    assert "6110" not in capture.calls[0]["url"]
    assert "engines" not in capture.calls[1]["url"]
    assert capture.calls[1]["host"] == "127.0.0.1"
    assert capture.calls[1]["port"] == 6089


def test_pending_poll_url_for_a_different_operation_is_rejected(monkeypatch):
    capture = _HliCapture()
    capture.recommend_bodies = [
        {
            "pending": True,
            "poll": True,
            "phase": "Pending",
            "operation_id": OP_UUID,
            "poll_url": f"/v1/resources/recommend/operations/{OP_B_UUID}",
            "poll_path": f"/v1/resources/recommend/operations/{OP_B_UUID}",
        }
    ]
    capture.queue_poll(OP_B_UUID, _typed_document(phase="Active", pending=False))
    _install_hli(monkeypatch, capture)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker._http_get_models",
        lambda *_a, **_k: [{"id": MODEL}],
    )
    with pytest.raises(ForegroundModelUnavailable, match="operation"):
        require_exact_face_cluster_admission(
            hivemind_url=HLI,
            requested_model=MODEL,
            now_s=NOW_S,
        )
    assert capture.calls[0]["method"] == "GET"
    assert capture.calls[0]["path"] == "/v1/resources/recommend"
    assert all(OP_B_UUID not in c["url"] and OP_B_UUID not in c["selector"] for c in capture.calls)
    assert all("/operations/" not in c["path"] for c in capture.calls)
    assert all(c["method"] == "GET" for c in capture.calls)
    assert not any(c["method"] == "POST" for c in capture.calls)


def test_pending_poll_url_query_smuggle_of_original_uuid_is_rejected(monkeypatch):
    capture = _HliCapture()
    capture.recommend_bodies = [
        {
            "pending": True,
            "poll": True,
            "phase": "Pending",
            "operation_id": OP_UUID,
            "poll_url": (
                f"{HLI}/v1/resources/recommend/operations/{OP_B_UUID}"
                f"?operation_id={OP_UUID}"
            ),
            "poll_path": (
                f"/v1/resources/recommend/operations/{OP_B_UUID}"
                f"?operation_id={OP_UUID}"
            ),
        }
    ]
    capture.queue_poll(OP_B_UUID, _typed_document(phase="Active", pending=False))
    _install_hli(monkeypatch, capture)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker._http_get_models",
        lambda *_a, **_k: [{"id": MODEL}],
    )
    with pytest.raises(ForegroundModelUnavailable, match="operation"):
        require_exact_face_cluster_admission(
            hivemind_url=HLI,
            requested_model=MODEL,
            now_s=NOW_S,
        )
    assert capture.calls[0]["path"] == "/v1/resources/recommend"
    assert all(OP_B_UUID not in c["url"] and OP_B_UUID not in c["selector"] for c in capture.calls)
    assert all("/operations/" not in c["path"] for c in capture.calls)
    assert capture.poll_documents[OP_B_UUID], "OP-B bait document must remain unconsumed"


def test_pending_poll_url_must_stay_on_the_public_hli_origin(monkeypatch):
    capture = _HliCapture()
    capture.recommend_bodies = [
        {
            "pending": True,
            "poll": True,
            "phase": "Pending",
            "operation_id": OP_UUID,
            "poll_url": f"http://evil.example:6089/v1/resources/recommend/operations/{OP_UUID}",
        }
    ]
    capture.queue_poll(OP_UUID, _typed_document(phase="Active", pending=False))
    _install_hli(monkeypatch, capture)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker._http_get_models",
        lambda *_a, **_k: [{"id": MODEL}],
    )
    with pytest.raises(ForegroundModelUnavailable, match="origin"):
        require_exact_face_cluster_admission(
            hivemind_url=HLI,
            requested_model=MODEL,
            now_s=NOW_S,
        )
    assert all(c["host"] == "127.0.0.1" for c in capture.calls)
    assert all("evil.example" not in c["url"] for c in capture.calls)
    assert all("/operations/" not in c["path"] for c in capture.calls)


def test_prewarm_does_not_consume_and_visible_ttft_is_the_consume_path(monkeypatch):
    capture = _HliCapture()
    capture.recommend_bodies = [_typed_document()]
    _install_hli(monkeypatch, capture)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker._http_get_models",
        lambda *_a, **_k: [{"id": MODEL}],
    )
    chat = FaceLobeChat(hivemind_url=HLI, http_timeout=5)
    prewarm = voice_module.prewarm_face_lobe_model(
        hivemind_url=HLI,
        model=MODEL,
        exact_model=True,
        timeout=5,
        latency_budget_ms=15_000,
        face_chat=chat,
    )
    assert prewarm["warmed"] is True
    chats = _chat_lease_calls(capture)
    assert chats, "prewarm must actually call HLI chat"
    assert all(
        c["body"].get("prewarm") is True or c["body"].get("hivemind_prewarm") is True
        for c in chats
    )
    probe = chat.probe_first_token(
        "say r",
        model=MODEL,
        extra_system="probe",
        first_token_timeout=2.0,
        recommend_receipt=prewarm["recommend"],
        recommend_validate_only=False,
    )
    assert probe["first_token_observed"] is True
    assert probe["visible_tokens"] == 1
    production = _chat_lease_calls(capture)[-1]
    assert production["body"].get("prewarm") is not True
    assert production["body"].get("hivemind_prewarm") is not True
    assert _header_set(production)["lease"] == LEASE_A
    assert not any("consume" in c["url"] for c in capture.calls)
    assert not any(c["method"] == "DELETE" for c in capture.calls)


def test_sse_comments_role_reasoning_error_and_raw_bytes_are_not_first_token(monkeypatch):
    capture = _HliCapture()
    capture.stream_chunks = [
        b": keep-alive\n\n",
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n\n',
        b"\x00\x01not-json\n",
        b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    _install_hli(monkeypatch, capture)
    chat = FaceLobeChat(hivemind_url=HLI, http_timeout=5)
    probe = chat.probe_first_token(
        "hi",
        model=MODEL,
        extra_system="",
        first_token_timeout=2.0,
        recommend_receipt=_receipt(),
        recommend_validate_only=False,
    )
    assert probe["text"] == "Hi"
    assert probe["visible_tokens"] == 1
    assert probe["first_token_observed"] is True
    assert not any(c["method"] == "DELETE" for c in capture.calls)

    capture.stream_chunks = [
        b'data: {"error":{"message":"nope"}}\n\n',
    ]
    capture.allow_job_cancel = True
    error_probe = chat.probe_first_token(
        "hi",
        model=MODEL,
        extra_system="",
        first_token_timeout=2.0,
        recommend_receipt=_receipt(),
        recommend_validate_only=False,
    )
    assert error_probe["first_token_observed"] is False
    assert not str(error_probe.get("text") or "").strip()
    deletes = [c for c in capture.calls if c["method"] == "DELETE"]
    assert len(deletes) == 1
    assert deletes[0]["path"].startswith("/jobs/")
    assert deletes[0]["query"].get("reason") == ["MS4 Face Lobe upstream SSE error"]


def test_stale_mismatch_expired_and_replay_fail_closed(monkeypatch):
    receipt = _receipt()
    with pytest.raises(ForegroundModelUnavailable, match="stale"):
        admit_cluster_recommendation(
            _typed_document(snapshot_age_s=61),
            catalog_ids={MODEL},
            requested_model=MODEL,
            now_s=NOW_S,
        )
    with pytest.raises(ForegroundModelUnavailable, match="mismatch"):
        admit_cluster_recommendation(
            _typed_document(owner="node-b"),
            catalog_ids={MODEL},
            requested_model=MODEL,
            expected_owner=OWNER,
            now_s=NOW_S,
        )
    with pytest.raises(ForegroundModelUnavailable, match="expired"):
        admit_cluster_recommendation(
            _typed_document(lease={"expires": NOW_S - 1}),
            catalog_ids={MODEL},
            requested_model=MODEL,
            now_s=NOW_S,
        )

    def boom(req, timeout=None):
        payload = json.dumps(
            {
                "error": {
                    "type": "recommend_lease_refresh_required",
                    "schema": EXACT_FACE_CLUSTER_ADMISSION_SCHEMA,
                    "message": "recommend lease already consumed",
                    "retryable": True,
                }
            }
        ).encode("utf-8")
        hdrs = email.message.Message()
        hdrs["Content-Type"] = "application/json"
        raise urllib.error.HTTPError(
            req.full_url, 409, "Conflict", hdrs, io.BytesIO(payload)
        )

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    chat = FaceLobeChat(hivemind_url=HLI, http_timeout=5)
    with pytest.raises(FaceLobeChatError, match="consumed|refresh|409") as exc:
        chat.chat(
            "hello",
            session_id="replay",
            model=MODEL,
            recommend_receipt=receipt,
            recommend_validate_only=False,
        )
    assert exc.value.fail_closed is True


def test_terminal_pending_admission_fails_closed_without_repost(monkeypatch):
    capture = _HliCapture()
    capture.recommend_bodies = [
        {
            "pending": True,
            "poll": True,
            "phase": "Pending",
            "operation_id": OP_UUID,
            "poll_url": f"/v1/resources/recommend/operations/{OP_UUID}",
        }
    ]
    capture.queue_poll(
        OP_UUID,
        {
            "pending": False,
            "phase": "Failed",
            "operation_id": OP_UUID,
            "ready": False,
            "error": {"type": "cluster recommendation owner/endpoint unavailable"},
        },
    )
    _install_hli(monkeypatch, capture)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker._http_get_models",
        lambda *_a, **_k: [{"id": MODEL}],
    )
    with pytest.raises(ForegroundModelUnavailable):
        require_exact_face_cluster_admission(
            hivemind_url=HLI,
            requested_model=MODEL,
            now_s=NOW_S,
        )
    assert all(c["method"] == "GET" for c in capture.calls)
    assert not any(c["method"] == "POST" for c in capture.calls)
    assert capture.calls[1]["path"] == f"/v1/resources/recommend/operations/{OP_UUID}"


def test_concurrent_two_lease_isolation(monkeypatch):
    capture = _HliCapture()
    _install_hli(monkeypatch, capture)
    errors: list[BaseException] = []

    def run(receipt: dict[str, Any], session: str) -> None:
        try:
            chat = FaceLobeChat(hivemind_url=HLI, http_timeout=5)
            chat.chat(
                "hello",
                session_id=session,
                model=MODEL,
                recommend_receipt=receipt,
                recommend_validate_only=False,
            )
        except BaseException as exc:  # noqa: BLE001 — test barrier
            errors.append(exc)

    receipt_a = admit_cluster_recommendation(
        _typed_document(lease={"id": LEASE_A}),
        catalog_ids={MODEL},
        requested_model=MODEL,
        now_s=NOW_S,
    )
    receipt_b = admit_cluster_recommendation(
        _typed_document(
            owner="node-b",
            node="node-b",
            endpoint="http://192.0.2.11:6089/v1/chat/completions",
            lease={"id": LEASE_B, "generation": 7},
        ),
        catalog_ids={MODEL},
        requested_model=MODEL,
        now_s=NOW_S,
    )
    threads = [
        threading.Thread(target=run, args=(receipt_a, "s-a")),
        threading.Thread(target=run, args=(receipt_b, "s-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert errors == []
    chats = _chat_lease_calls(capture)
    leases = sorted(_header_set(c)["lease"] for c in chats)
    assert leases == [LEASE_A, LEASE_B]
    owners = {_header_set(c)["owner"] for c in chats}
    assert owners == {OWNER, "node-b"}
    assert not any(c["method"] == "DELETE" for c in capture.calls)


def test_ordinary_chat_without_lease_is_unchanged(monkeypatch):
    capture = _HliCapture()
    _install_hli(monkeypatch, capture)
    chat = FaceLobeChat(hivemind_url=HLI, http_timeout=5)
    result = chat.chat("hello", session_id="ordinary", model=MODEL)
    assert result["text"]
    chats = _chat_lease_calls(capture)
    assert len(chats) == 1
    headers = chats[0]["headers"]
    assert "x-hivemind-recommend-lease" not in headers
    assert "x-hivemind-recommend-owner" not in headers
    assert chats[0]["body"].get("prewarm") is None
    assert not any(c["method"] == "DELETE" for c in capture.calls)
    monkeypatch.delenv("MS4_FOREGROUND_MODEL", raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("MS4_FACE_PROFILE", "latency")
    monkeypatch.setattr(hivemind_tools, "models_recommend", lambda *_a, **_k: {
        "capability": "chat",
        "recommended_model": MODEL,
        "backend": "ollama",
    })
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker._http_get_models",
        lambda *_a, **_k: [
            {
                "id": MODEL,
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            }
        ],
    )
    from machine_spirit_4.double_agent import model_picker

    model_picker._clear_cache_for_tests()
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == MODEL
    assert choice.source == "loaded"


def test_public_evidence_binds_owner_endpoint_generation_lease_without_secrets():
    evidence = lease_mod.public_recommend_evidence(_receipt())
    dumped = json.dumps(evidence).lower()
    assert evidence["owner"] == OWNER
    assert evidence["endpoint"] == ENDPOINT
    assert evidence["generation"] == 7
    assert evidence["lease_id"] == LEASE_A
    assert "authorization" not in dumped
    assert "bearer" not in dumped
    assert "identity_proof" not in dumped
    assert "api_key" not in dumped


def test_prewarm_fail_closed_keeps_failed_not_sent_contract(monkeypatch):
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.require_exact_face_cluster_admission",
        lambda **_k: (_ for _ in ()).throw(
            ForegroundModelUnavailable("cluster recommendation lease expired")
        ),
    )
    result = voice_module.prewarm_face_lobe_model(
        hivemind_url=HLI,
        model=MODEL,
        exact_model=True,
    )
    assert result["warmed"] is False
    assert result.get("fail_closed") is True
    assert "expired" in str(result.get("error") or "")


def test_face_lobe_has_one_binder_and_one_probe_attach():
    source = Path(face_module.__file__).read_text(encoding="utf-8")
    assert source.count("def bind_recommend_receipt(") == 1
    assert source.count("def recommend_receipt_for(") == 1
    probe = source.split("def probe_first_token(", 1)[1].split("\n    def chat(", 1)[0]
    assert probe.count("attach_recommend_lease(") == 1
