"""Focused tests for MS4's default-off voice-candidate review consumer.

This is a dream-cadence / background-review intent owned by the MS4
gateway. It does not claim MS3 dream-scheduler integration, live
biometric retention, or automatic action authority.
"""

from __future__ import annotations

import http.client
import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from machine_spirit_4.gateway.voice_candidate_reconcile import (
    DEFAULT_INTERVAL_SECS,
    DEFAULT_TIMEOUT_SECS,
    FORBIDDEN_KEYS,
    MAX_STORE_ITEMS,
    STORE_SCHEMA,
    VoiceCandidateReconcileConfig,
    VoiceCandidateReconciler,
    VoiceCandidateReviewStore,
)


CANDIDATES_PATH = "/v1/audio/voice-identities/candidates"
RECONCILE_PATH = "/v1/audio/voice-identities/candidates/reconcile"
CONSEQUENCE_MARKERS = (
    "/label",
    "/merge",
    "/ignore",
    "/enroll",
    "/identify",
    "/refine",
)

SYNTHETIC_CANDIDATE_ID = "cand-7f3a9c2e-synth"
SYNTHETIC_TARGET_ID = "profile-0b17d4aa-synth"
FIXED_NOW = "2026-08-18T18:04:00+00:00"
FOREIGN_REDIRECT_LOCATION = "https://example.invalid/collect"
AUTH_TOKEN = "test-secret"
AUTH_HEADER = f"Bearer {AUTH_TOKEN}"


class FakeTransport:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "timeout": timeout,
                "headers": dict(headers or {}),
            }
        )
        if not self.script:
            raise AssertionError(f"unexpected extra transport call: {method} {url}")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _list_payload(*, retention_enabled: bool, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "ok": True,
        "object": "voice_identity.candidate_list",
        "count": 0,
        "retention_enabled": retention_enabled,
        "db_path": "E:/HiveMind/forbidden/candidates.json",
        "candidates": [],
    }
    if extras:
        payload.update(extras)
    return payload


def _proposal(**overrides: Any) -> dict[str, Any]:
    payload = {
        "candidate_id": SYNTHETIC_CANDIDATE_ID,
        "action": "merge",
        "target_kind": "profile",
        "target_id": SYNTHETIC_TARGET_ID,
        "score": 0.91,
        "reason": "same-model cosine to a named profile cleared the identify threshold",
    }
    payload.update(overrides)
    return payload


def _reconcile_payload(proposals: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    payload = {
        "ok": True,
        "object": "voice_identity.candidate_reconciliation",
        "applied": False,
        "count": len(proposals),
        "proposals": proposals,
    }
    payload.update(overrides)
    return payload


def _walk(obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def assert_representation_clean(obj: Any) -> None:
    serialized = json.dumps(obj, ensure_ascii=False)
    lowered = serialized.lower()
    for key in FORBIDDEN_KEYS:
        assert f'"{key}"' not in lowered
        assert f"'{key}'" not in lowered
    for key, value in _walk(obj):
        assert str(key).lower() not in FORBIDDEN_KEYS
        if isinstance(value, str):
            assert value.lower() not in FORBIDDEN_KEYS


def _make_reconciler(
    tmp_path: Path,
    *,
    enabled: bool = True,
    hivemind_url: str = "http://127.0.0.1:6089",
    transport: FakeTransport | None = None,
    use_default_transport: bool = False,
    replace=os.replace,
    wait=None,
    thread_factory=None,
    auth_headers_fn=None,
) -> tuple[VoiceCandidateReconciler, Path, list[dict[str, Any]], FakeTransport]:
    store_path = tmp_path / "voice_candidate_review_items.json"
    audits: list[dict[str, Any]] = []
    fake_transport = transport or FakeTransport([])
    config = VoiceCandidateReconcileConfig(
        enabled=enabled,
        interval_secs=DEFAULT_INTERVAL_SECS,
        timeout_secs=DEFAULT_TIMEOUT_SECS,
        store_path=store_path,
        hivemind_url=hivemind_url,
    )
    store = VoiceCandidateReviewStore(store_path, replace=replace)
    reconciler = VoiceCandidateReconciler(
        config=config,
        transport=None if use_default_transport else fake_transport,
        store=store,
        audit_sink=audits.append,
        wait=wait,
        now=lambda: FIXED_NOW,
        monotonic=lambda: 0.0,
        thread_factory=thread_factory,
        auth_headers_fn=auth_headers_fn or (lambda: {"Authorization": AUTH_HEADER}),
        join_timeout_secs=0.2,
    )
    return reconciler, store_path, audits, fake_transport


def _seed_store(path: Path, items: list[dict[str, Any]]) -> bytes:
    payload = {"schema": STORE_SCHEMA, "items": items}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    path.write_bytes(data)
    return data


def _pending_item(candidate_id: str, *, index: int = 0) -> dict[str, Any]:
    return {
        "item_id": f"vcr_seed_{index:03d}",
        "candidate_id": candidate_id,
        "action": "ignore",
        "target_kind": None,
        "target_id": None,
        "score": None,
        "reason": "seeded pending review",
        "first_seen": "2026-08-18T00:00:00+00:00",
        "last_seen": "2026-08-18T00:00:00+00:00",
        "observation_count": 1,
        "status": "pending_operator_review",
    }


class _FakeHTTPResponse:
    """Minimal http.client.HTTPResponse stand-in. Never wraps a socket."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes, reason: str = "Found"):
        self.status = status
        self.code = status
        self.reason = reason
        self.msg = reason
        self.headers = http.client.HTTPMessage()
        for key, value in headers.items():
            self.headers[key] = value
        self._body = io.BytesIO(body)
        self.fp = self._body
        self.closed = False
        self.length = len(body)
        self.version = 11
        self.chunked = False
        self.will_close = True
        self.url = ""

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            return self._body.read()
        return self._body.read(amt)

    def readinto(self, b) -> int:
        return self._body.readinto(b)

    def info(self) -> http.client.HTTPMessage:
        return self.headers

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)

    def getheaders(self) -> list[tuple[str, str]]:
        return list(self.headers.items())

    def getcode(self) -> int:
        return self.code

    def geturl(self) -> str:
        return self.url

    def close(self) -> None:
        self.closed = True

    def isclosed(self) -> bool:
        return self.closed

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False


def _install_scripted_http(
    monkeypatch: pytest.MonkeyPatch,
    script: list[tuple[int, dict[str, str], bytes, str]],
) -> list[dict[str, Any]]:
    """Record every urllib connection/request without opening a socket."""

    attempts: list[dict[str, Any]] = []
    remaining = list(script)
    real_http = http.client.HTTPConnection

    class FakeConnection:
        sock = None
        _get_content_length = staticmethod(real_http._get_content_length)

        def __init__(self, host, port=None, timeout=None, **_kwargs):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.debuglevel = 0

        def set_debuglevel(self, level: int) -> None:
            self.debuglevel = level

        def set_tunnel(self, host, port=None, headers=None) -> None:
            raise AssertionError("reconciler transport must not open an HTTP tunnel")

        def connect(self) -> None:
            raise AssertionError("reconciler transport must not call connect()")

        def request(self, method, url, body=None, headers=None, *, encode_chunked=False):
            self._recorded = {
                "host": self.host,
                "port": self.port,
                "method": method,
                "selector": url,
                "body": body,
                "headers": {str(key): str(value) for key, value in dict(headers or {}).items()},
            }

        def getresponse(self) -> _FakeHTTPResponse:
            recorded = getattr(self, "_recorded", None)
            if recorded is None:
                raise AssertionError("getresponse() before request()")
            attempts.append(recorded)
            if remaining:
                status, headers, body, reason = remaining.pop(0)
                return _FakeHTTPResponse(status, headers, body, reason)
            # A follow-up hop that urllib constructed after a 30x. Returning a
            # successful retention payload makes a missed reject visible as a
            # non-fail-soft cycle instead of a harness error.
            return _FakeHTTPResponse(
                200,
                {"Content-Type": "application/json"},
                json.dumps(_list_payload(retention_enabled=False)).encode("utf-8"),
                "OK",
            )

        def close(self) -> None:
            return None

    class FakeHTTPSConnection(FakeConnection):
        pass

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeHTTPSConnection)
    return attempts


def _is_loopback_attempt(attempt: dict[str, Any]) -> bool:
    host = str(attempt.get("host") or "").split("%", 1)[0].strip().lower()
    hostname = host.rsplit(":", 1)[0] if host.count(":") == 1 else host.strip("[]")
    if hostname.startswith("[") and hostname.endswith("]"):
        hostname = hostname[1:-1]
    return hostname in {"127.0.0.1", "::1", "localhost"}


def _serialized_without_secrets(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_and_clamps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MS4_VOICE_CANDIDATE_RECONCILE_ENABLED", raising=False)
    monkeypatch.delenv("MS4_VOICE_CANDIDATE_RECONCILE_SECS", raising=False)
    monkeypatch.delenv("MS4_VOICE_CANDIDATE_RECONCILE_TIMEOUT_SECS", raising=False)
    monkeypatch.delenv("MS4_VOICE_CANDIDATE_REVIEW_STORE", raising=False)
    monkeypatch.delenv("MS4_HIVEMIND_URL", raising=False)
    monkeypatch.delenv("MS4_HIVEMIND_HLI_URL", raising=False)

    cfg = VoiceCandidateReconcileConfig.from_env()
    assert cfg.enabled is False
    assert cfg.interval_secs == 3600
    assert cfg.timeout_secs == 3
    assert cfg.store_path == (
        Path(__file__).resolve().parents[2]
        / "machine_spirit_4"
        / "runtime"
        / "voice_candidate_review_items.json"
    )
    assert cfg.hivemind_url == "http://127.0.0.1:6089"

    monkeypatch.setenv("MS4_VOICE_CANDIDATE_RECONCILE_ENABLED", "true")
    monkeypatch.setenv("MS4_VOICE_CANDIDATE_RECONCILE_SECS", "10")
    monkeypatch.setenv("MS4_VOICE_CANDIDATE_RECONCILE_TIMEOUT_SECS", "0")
    monkeypatch.setenv("MS4_VOICE_CANDIDATE_REVIEW_STORE", str(tmp_path / "q.json"))
    monkeypatch.setenv("MS4_HIVEMIND_URL", "http://localhost:6089")
    cfg = VoiceCandidateReconcileConfig.from_env()
    assert cfg.enabled is True
    assert cfg.interval_secs == 60
    assert cfg.timeout_secs == 1
    assert cfg.store_path == tmp_path / "q.json"
    assert cfg.hivemind_url == "http://localhost:6089"

    monkeypatch.setenv("MS4_VOICE_CANDIDATE_RECONCILE_SECS", "not-a-number")
    monkeypatch.setenv("MS4_VOICE_CANDIDATE_RECONCILE_TIMEOUT_SECS", "nope")
    cfg = VoiceCandidateReconcileConfig.from_env()
    assert cfg.interval_secs == 3600
    assert cfg.timeout_secs == 3
    monkeypatch.setenv("MS4_VOICE_CANDIDATE_RECONCILE_SECS", "999999")
    monkeypatch.setenv("MS4_VOICE_CANDIDATE_RECONCILE_TIMEOUT_SECS", "99")
    cfg = VoiceCandidateReconcileConfig.from_env()
    assert cfg.interval_secs == 86400
    assert cfg.timeout_secs == 10


# ---------------------------------------------------------------------------
# Default-off / lifecycle
# ---------------------------------------------------------------------------


def test_default_off_start_has_zero_side_effects(tmp_path: Path) -> None:
    created: list[Any] = []

    def thread_factory(*args: Any, **kwargs: Any):
        created.append((args, kwargs))
        raise AssertionError("disabled start must not create a thread")

    transport = FakeTransport([(200, {"ok": True, "retention_enabled": True})])
    reconciler, store_path, audits, _ = _make_reconciler(
        tmp_path,
        enabled=False,
        transport=transport,
        thread_factory=thread_factory,
    )
    reconciler.start()
    reconciler.start()
    reconciler.stop()

    assert created == []
    assert transport.calls == []
    assert audits == []
    assert not store_path.exists()
    assert not any(path.name.startswith("voice_candidate") for path in tmp_path.iterdir())


def test_start_is_idempotent_and_stop_join_is_bounded(tmp_path: Path) -> None:
    threads: list[Any] = []

    class FakeThread:
        def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
            self.target = target
            self.name = name
            self.daemon = daemon
            self.started = 0
            self.join_calls: list[float | None] = []
            self._alive = False
            threads.append(self)

        def start(self) -> None:
            self.started += 1
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)
            self._alive = False

    wait_calls: list[float] = []

    def wait(event, timeout: float) -> bool:
        wait_calls.append(timeout)
        return event.is_set()

    reconciler, store_path, audits, transport = _make_reconciler(
        tmp_path,
        thread_factory=FakeThread,
        wait=wait,
    )
    reconciler.start()
    reconciler.start()
    assert len(threads) == 1
    assert threads[0].started == 1
    assert threads[0].name == "ms4-voice-candidate-reconcile"
    assert threads[0].daemon is True

    reconciler.stop()
    assert threads[0].join_calls == [0.2]
    assert not threads[0].is_alive()
    assert transport.calls == []
    assert audits == []
    assert not store_path.exists()
    assert wait_calls == []  # loop never entered; FakeThread does not run target


def test_real_thread_stop_unblocks_wait_without_interval_sleep(tmp_path: Path) -> None:
    entered = []

    def wait(event, timeout: float) -> bool:
        entered.append(timeout)
        return event.wait()

    reconciler, _, _, transport = _make_reconciler(tmp_path, wait=wait)
    reconciler.start()
    reconciler.stop()
    assert entered == [float(DEFAULT_INTERVAL_SECS)]
    assert transport.calls == []
    assert reconciler.is_alive() is False


# ---------------------------------------------------------------------------
# HLI contract
# ---------------------------------------------------------------------------


def test_retention_off_gets_only_and_preserves_store_bytes(tmp_path: Path) -> None:
    prior = _seed_store(tmp_path / "voice_candidate_review_items.json", [_pending_item("cand-keep")])
    transport = FakeTransport([(200, _list_payload(retention_enabled=False))])
    reconciler, store_path, audits, _ = _make_reconciler(tmp_path, transport=transport)

    result = reconciler.run_once()

    assert result["outcome"] == "retention_off"
    assert store_path.read_bytes() == prior
    assert [call["method"] for call in transport.calls] == ["GET"]
    assert CANDIDATES_PATH in transport.calls[0]["url"]
    assert "return_embeddings=false" in transport.calls[0]["url"]
    assert "include_ignored=false" in transport.calls[0]["url"]
    assert RECONCILE_PATH not in transport.calls[0]["url"]
    assert len(audits) == 1
    assert audits[0]["outcome"] == "retention_off"
    assert_representation_clean(audits[0])
    assert "db_path" not in json.dumps(audits[0])
    assert AUTH_TOKEN not in json.dumps(audits[0])


def test_non_loopback_and_route_unavailable_do_not_post(tmp_path: Path) -> None:
    prior = _seed_store(tmp_path / "voice_candidate_review_items.json", [_pending_item("cand-keep")])
    transport = FakeTransport([(200, _list_payload(retention_enabled=True))])
    reconciler, store_path, audits, _ = _make_reconciler(
        tmp_path,
        hivemind_url="http://hive.internal:6089",
        transport=transport,
    )
    result = reconciler.run_once()
    assert result["outcome"] == "route_unavailable"
    assert result["reason"] == "loopback_required"
    assert transport.calls == []
    assert store_path.read_bytes() == prior
    assert audits[0]["reason"] == "loopback_required"
    assert_representation_clean(audits[0])

    transport = FakeTransport([TimeoutError("slow")])
    reconciler, store_path, audits, _ = _make_reconciler(tmp_path, transport=transport)
    _seed_store(store_path, [_pending_item("cand-keep")])
    prior = store_path.read_bytes()
    result = reconciler.run_once()
    assert result["outcome"] == "route_unavailable"
    assert [call["method"] for call in transport.calls] == ["GET"]
    assert all(call["method"] != "POST" for call in transport.calls)
    assert store_path.read_bytes() == prior

    transport = FakeTransport([(503, {"ok": False})])
    reconciler, store_path, audits, _ = _make_reconciler(tmp_path, transport=transport)
    _seed_store(store_path, [_pending_item("cand-keep")])
    prior = store_path.read_bytes()
    result = reconciler.run_once()
    assert result["outcome"] == "route_unavailable"
    assert all(call["method"] != "POST" for call in transport.calls)
    assert store_path.read_bytes() == prior


def test_default_transport_rejects_get_redirect_before_foreign_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prior = _seed_store(tmp_path / "voice_candidate_review_items.json", [_pending_item("cand-keep")])
    raw_body = b'{"ok":true,"embedding":[1.2],"authorization":"leak-me"}'
    attempts = _install_scripted_http(
        monkeypatch,
        [
            (
                302,
                {"Location": FOREIGN_REDIRECT_LOCATION},
                raw_body,
                "Found",
            )
        ],
    )
    caplog.set_level(logging.DEBUG)
    reconciler, store_path, audits, _ = _make_reconciler(
        tmp_path,
        use_default_transport=True,
    )

    result = reconciler.run_once()

    loopback = [item for item in attempts if _is_loopback_attempt(item)]
    redirected = [item for item in attempts if not _is_loopback_attempt(item)]
    assert result["outcome"] == "route_unavailable"
    assert result.get("recorded_count") == 0
    assert len(loopback) == 1
    assert loopback[0]["method"] == "GET"
    assert CANDIDATES_PATH in loopback[0]["selector"]
    assert loopback[0]["headers"].get("Authorization") == AUTH_HEADER
    assert redirected == []
    assert all(
        "example.invalid" not in str(item.get("host") or "").lower()
        and AUTH_HEADER not in json.dumps(item.get("headers") or {})
        for item in redirected
    )
    assert all(item["method"] != "POST" for item in attempts)
    assert store_path.read_bytes() == prior
    assert len(audits) == 1
    assert audits[0]["outcome"] == "route_unavailable"
    assert_representation_clean(result)
    assert_representation_clean(audits[0])
    combined = _serialized_without_secrets({"result": result, "audit": audits, "store": json.loads(prior)})
    assert FOREIGN_REDIRECT_LOCATION not in combined
    assert "example.invalid" not in combined
    assert AUTH_TOKEN not in combined
    assert "leak-me" not in combined
    assert FOREIGN_REDIRECT_LOCATION not in caplog.text
    assert "example.invalid" not in caplog.text
    assert AUTH_TOKEN not in caplog.text
    assert "leak-me" not in caplog.text


def test_default_transport_rejects_post_redirect_before_foreign_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prior = _seed_store(tmp_path / "voice_candidate_review_items.json", [_pending_item("cand-keep")])
    attempts = _install_scripted_http(
        monkeypatch,
        [
            (
                200,
                {"Content-Type": "application/json"},
                json.dumps(_list_payload(retention_enabled=True)).encode("utf-8"),
                "OK",
            ),
            (
                302,
                {"Location": FOREIGN_REDIRECT_LOCATION},
                b'{"ok":true,"raw":"post-body-must-not-persist"}',
                "Found",
            ),
        ],
    )
    caplog.set_level(logging.DEBUG)
    reconciler, store_path, audits, _ = _make_reconciler(
        tmp_path,
        use_default_transport=True,
    )

    result = reconciler.run_once()

    loopback = [item for item in attempts if _is_loopback_attempt(item)]
    redirected = [item for item in attempts if not _is_loopback_attempt(item)]
    assert result["outcome"] == "route_unavailable"
    assert result.get("recorded_count") == 0
    assert [item["method"] for item in loopback] == ["GET", "POST"]
    assert CANDIDATES_PATH in loopback[0]["selector"]
    assert loopback[1]["selector"].endswith(RECONCILE_PATH)
    assert loopback[1]["headers"].get("Authorization") == AUTH_HEADER
    assert redirected == []
    assert all("example.invalid" not in str(item.get("host") or "").lower() for item in attempts)
    assert all(
        AUTH_TOKEN not in json.dumps(item.get("headers") or {})
        for item in redirected
    )
    assert store_path.read_bytes() == prior
    assert audits[0]["outcome"] == "route_unavailable"
    assert_representation_clean(result)
    assert_representation_clean(audits[0])
    combined = _serialized_without_secrets({"result": result, "audit": audits})
    assert FOREIGN_REDIRECT_LOCATION not in combined
    assert "example.invalid" not in combined
    assert AUTH_TOKEN not in combined
    assert "post-body-must-not-persist" not in combined
    assert FOREIGN_REDIRECT_LOCATION not in caplog.text
    assert AUTH_TOKEN not in caplog.text
    assert "post-body-must-not-persist" not in caplog.text


def test_successful_proposals_upsert_stable_pending_items(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            (200, _list_payload(retention_enabled=True)),
            (200, _reconcile_payload([_proposal()])),
        ]
    )
    reconciler, store_path, audits, _ = _make_reconciler(tmp_path, transport=transport)
    result = reconciler.run_once()
    assert result["outcome"] == "proposals_recorded"
    assert result["proposal_count"] == 1
    assert result["recorded_count"] == 1

    stored = json.loads(store_path.read_text(encoding="utf-8"))
    assert stored["schema"] == STORE_SCHEMA
    assert len(stored["items"]) == 1
    item = stored["items"][0]
    assert item["candidate_id"] == SYNTHETIC_CANDIDATE_ID
    assert item["action"] == "merge"
    assert item["target_kind"] == "profile"
    assert item["target_id"] == SYNTHETIC_TARGET_ID
    assert item["score"] == 0.91
    assert item["status"] == "pending_operator_review"
    assert item["first_seen"] == FIXED_NOW
    assert item["last_seen"] == FIXED_NOW
    assert item["observation_count"] == 1
    assert item["item_id"]
    assert "approved" not in json.dumps(item)
    assert_representation_clean(stored)
    assert_representation_clean(audits[0])


def test_repeated_cycles_do_not_duplicate_items(tmp_path: Path) -> None:
    def script() -> FakeTransport:
        return FakeTransport(
            [
                (200, _list_payload(retention_enabled=True)),
                (200, _reconcile_payload([_proposal(score=0.91)])),
            ]
        )

    transport = script()
    reconciler, store_path, _, _ = _make_reconciler(tmp_path, transport=transport)
    first = reconciler.run_once()
    item_id = json.loads(store_path.read_text(encoding="utf-8"))["items"][0]["item_id"]

    transport2 = script()
    reconciler2, _, _, _ = _make_reconciler(tmp_path, transport=transport2)
    second = reconciler2.run_once()
    stored = json.loads(store_path.read_text(encoding="utf-8"))
    assert first["outcome"] == second["outcome"] == "proposals_recorded"
    assert len(stored["items"]) == 1
    assert stored["items"][0]["item_id"] == item_id
    assert stored["items"][0]["observation_count"] == 2
    assert stored["items"][0]["first_seen"] == FIXED_NOW
    assert stored["items"][0]["status"] == "pending_operator_review"


def test_malformed_or_applied_true_does_not_mutate_store(tmp_path: Path) -> None:
    prior = _seed_store(tmp_path / "voice_candidate_review_items.json", [_pending_item("cand-keep")])
    cases = [
        FakeTransport(
            [
                (200, _list_payload(retention_enabled=True)),
                (200, _reconcile_payload([_proposal()], applied=True)),
            ]
        ),
        FakeTransport(
            [
                (200, _list_payload(retention_enabled=True)),
                (200, _reconcile_payload([_proposal()], object="voice_identity.candidate_merge")),
            ]
        ),
        FakeTransport(
            [
                (200, _list_payload(retention_enabled=True)),
                (200, _reconcile_payload([_proposal(action="enroll")])),
            ]
        ),
        FakeTransport(
            [
                (200, _list_payload(retention_enabled=True)),
                (200, _reconcile_payload([_proposal() for _ in range(65)])),
            ]
        ),
        FakeTransport(
            [
                (200, {"ok": True, "retention_enabled": "yes"}),
            ]
        ),
        FakeTransport(
            [
                (200, _list_payload(retention_enabled=True)),
                (200, "not-json-object"),
            ]
        ),
    ]
    for transport in cases:
        reconciler, store_path, audits, _ = _make_reconciler(tmp_path, transport=transport)
        result = reconciler.run_once()
        assert result["outcome"] == "contract_violation"
        assert store_path.read_bytes() == prior
        assert all(call["method"] != "DELETE" for call in transport.calls)
        assert_representation_clean(audits[-1])


def test_hostile_extras_cannot_reach_store_or_audit(tmp_path: Path) -> None:
    hostile = _proposal(
        embedding=[0.1, 0.2, 0.3],
        audio="RIFF",
        audio_data="AAAA",
        audio_base64="c2VjcmV0",
        transcript="hello operator",
        text="named human",
        nested={
            "embedding": [9.9],
            "audio": {"bytes": "nope"},
            "transcript": "inner",
            "text": "inner-text",
            "audio_data": "x",
            "audio_base64": "y",
        },
        headers={"Authorization": "Bearer leaked"},
        db_path="C:/secret/db.json",
        url="http://127.0.0.1:6089/v1/audio/voice-identities/candidates/c/label",
    )
    transport = FakeTransport(
        [
            (
                200,
                _list_payload(
                    retention_enabled=True,
                    extras={
                        "embedding": [1.0, 2.0],
                        "audio": "wav",
                        "transcript": "list-transcript",
                        "text": "list-text",
                    },
                ),
            ),
            (200, _reconcile_payload([hostile])),
        ]
    )
    reconciler, store_path, audits, _ = _make_reconciler(tmp_path, transport=transport)
    result = reconciler.run_once()
    assert result["outcome"] == "proposals_recorded"
    stored = json.loads(store_path.read_text(encoding="utf-8"))
    assert_representation_clean(stored)
    assert_representation_clean(audits)
    combined = json.dumps({"store": stored, "audit": audits, "result": result})
    for token in ("embedding", "audio_data", "audio_base64", "transcript"):
        assert f'"{token}"' not in combined.lower()
    assert '"text"' not in combined.lower()
    assert "c2VjcmV0" not in combined
    assert "named human" not in combined
    assert "Bearer leaked" not in combined
    assert "hello operator" not in combined


def test_forced_atomic_replace_failure_preserves_prior_file(tmp_path: Path) -> None:
    store_path = tmp_path / "voice_candidate_review_items.json"
    prior = _seed_store(store_path, [_pending_item("cand-keep")])

    def boom(_src: str | os.PathLike[str], _dst: str | os.PathLike[str]) -> None:
        raise OSError("simulated replace failure")

    transport = FakeTransport(
        [
            (200, _list_payload(retention_enabled=True)),
            (200, _reconcile_payload([_proposal()])),
        ]
    )
    reconciler, _, audits, _ = _make_reconciler(tmp_path, transport=transport, replace=boom)
    result = reconciler.run_once()
    assert result["outcome"] == "store_unavailable"
    assert store_path.read_bytes() == prior
    leftover_tmps = [path for path in tmp_path.iterdir() if path.suffix == ".tmp" or ".tmp." in path.name]
    assert leftover_tmps == []
    assert_representation_clean(audits[0])


def test_full_store_preserves_pending_items_and_reports_capacity(tmp_path: Path) -> None:
    items = [_pending_item(f"cand-full-{index:03d}", index=index) for index in range(MAX_STORE_ITEMS)]
    store_path = tmp_path / "voice_candidate_review_items.json"
    prior = _seed_store(store_path, items)
    transport = FakeTransport(
        [
            (200, _list_payload(retention_enabled=True)),
            (200, _reconcile_payload([_proposal()])),
        ]
    )
    reconciler, _, audits, _ = _make_reconciler(tmp_path, transport=transport)
    result = reconciler.run_once()
    assert result["outcome"] == "store_capacity"
    assert store_path.read_bytes() == prior
    stored = json.loads(store_path.read_text(encoding="utf-8"))
    assert len(stored["items"]) == MAX_STORE_ITEMS
    assert all(item["status"] == "pending_operator_review" for item in stored["items"])
    assert audits[0]["outcome"] == "store_capacity"


def test_captured_urls_prove_only_reconcile_post_and_no_consequence_routes(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            (200, _list_payload(retention_enabled=True)),
            (200, _reconcile_payload([_proposal(action="ignore", target_kind=None, target_id=None, score=None)])),
        ]
    )
    reconciler, _, _, _ = _make_reconciler(tmp_path, transport=transport)
    reconciler.run_once()
    methods_and_urls = [(call["method"], call["url"]) for call in transport.calls]
    assert [item[0] for item in methods_and_urls] == ["GET", "POST"]
    get_url = methods_and_urls[0][1]
    post_url = methods_and_urls[1][1]
    assert get_url.endswith(f"{CANDIDATES_PATH}?return_embeddings=false&include_ignored=false")
    assert post_url.endswith(RECONCILE_PATH)
    assert transport.calls[1]["body"] == b"{}"
    for method, url in methods_and_urls:
        lowered = url.lower()
        for marker in CONSEQUENCE_MARKERS:
            if marker == "/ignore" and "include_ignored=false" in lowered:
                assert url.rstrip("/").split("?")[0].endswith(CANDIDATES_PATH)
                continue
            assert marker not in lowered
        assert method != "DELETE"


def test_localhost_and_ipv6_loopback_are_accepted(tmp_path: Path) -> None:
    for url in ("http://localhost:6089", "http://[::1]:6089"):
        transport = FakeTransport(
            [
                (200, _list_payload(retention_enabled=False)),
            ]
        )
        reconciler, _, _, _ = _make_reconciler(tmp_path, hivemind_url=url, transport=transport)
        result = reconciler.run_once()
        assert result["outcome"] == "retention_off"
        assert len(transport.calls) == 1


def test_no_proposals_does_not_create_store(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            (200, _list_payload(retention_enabled=True)),
            (200, _reconcile_payload([])),
        ]
    )
    reconciler, store_path, audits, _ = _make_reconciler(tmp_path, transport=transport)
    result = reconciler.run_once()
    assert result["outcome"] == "no_proposals"
    assert not store_path.exists()
    assert audits[0]["outcome"] == "no_proposals"


def test_import_does_not_start_reconciler() -> None:
    import machine_spirit_4.gateway.voice_candidate_reconcile as module

    assert getattr(module, "_IMPORT_STARTED", False) is False


# ---------------------------------------------------------------------------
# Gateway lifecycle wiring
# ---------------------------------------------------------------------------


class _DoubleAgentRunner:
    def recover_on_startup(self) -> None:
        pass


def test_gateway_lifecycle_starts_and_finally_stops_without_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    from machine_spirit_4.gateway import server as srv

    order: list[str] = []
    heartbeat_calls: list[str] = []

    class SpyReconciler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            order.append("construct")
            self.kwargs = kwargs

        def start(self) -> None:
            order.append("start")

        def stop(self, timeout: float | None = None) -> None:
            order.append("stop")

    class FakeServer:
        def __init__(self, _address: Any, _handler: Any) -> None:
            order.append("server")

        def serve_forever(self) -> None:
            order.append("serve")

        def server_close(self) -> None:
            order.append("close")

    class CapturedThread:
        def __init__(self, *, target=None, args=(), kwargs=None, daemon=None, name=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            self.name = name

        def start(self) -> None:
            return None

    monkeypatch.setattr(srv, "VoiceCandidateReconciler", SpyReconciler)
    monkeypatch.setattr(srv, "require_contained_runtime", lambda _service: None)
    monkeypatch.setattr(
        srv,
        "build_runner",
        lambda: SimpleNamespace(hivemind_url="http://127.0.0.1:6089", ms3_url="http://127.0.0.1:9080"),
    )
    monkeypatch.setattr(srv.hermes_admin, "initialize_state", lambda: None)
    monkeypatch.setattr(srv, "default_runner", _DoubleAgentRunner)
    monkeypatch.setattr(srv, "_set_hivemind_url_for_auth", lambda _url: None)
    monkeypatch.setattr(srv, "start_heartbeat_thread", lambda url: heartbeat_calls.append(url))
    monkeypatch.setattr(srv, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(srv.threading, "Thread", CapturedThread)
    monkeypatch.setenv("MS4_DA_CONTINUATION_CLASSIFIER", "0")
    monkeypatch.setenv("MS4_VOICE_TTS_KEEPWARM_SECS", "0")
    monkeypatch.setenv("MS4_VOICE_FACE_KEEPWARM_SECS", "0")
    monkeypatch.setenv("MS4_VOICE_TTS_AUTOSCALE", "0")
    monkeypatch.setenv("MS4_VOICE_CANDIDATE_RECONCILE_ENABLED", "1")

    srv.run(host="127.0.0.1", port=0)

    assert heartbeat_calls == ["http://127.0.0.1:9080"]
    assert order == ["server", "construct", "start", "serve", "stop", "close"]
