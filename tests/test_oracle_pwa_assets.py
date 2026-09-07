from __future__ import annotations

import io
import json
from pathlib import Path

from machine_spirit_4.gateway import server


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "machine_spirit_4" / "web" / "index.html"


def _handler(path: str) -> server.Ms4GatewayHandler:
    item = server.Ms4GatewayHandler.__new__(server.Ms4GatewayHandler)
    item.path = path
    item.status = None
    item.headers_out: dict[str, str] = {}
    item.wfile = io.BytesIO()
    item.send_response = lambda status: setattr(item, "status", status)
    item.send_header = lambda name, value: item.headers_out.__setitem__(name, value)
    item.end_headers = lambda: None
    return item


def _dispatch(path: str) -> server.Ms4GatewayHandler:
    item = _handler(path)
    item.do_GET()
    return item


def _serve_private_helper(path: str) -> server.Ms4GatewayHandler:
    item = _handler(path)
    item._serve_static()
    return item


def test_public_gateway_dispatches_root_pwa_assets_with_safe_headers() -> None:
    manifest = _dispatch("/manifest.webmanifest?v=ignored")
    assert manifest.status == 200
    assert manifest.headers_out["Content-Type"].startswith("application/manifest+json")
    assert manifest.headers_out["X-Content-Type-Options"] == "nosniff"
    assert json.loads(manifest.wfile.getvalue())["scope"] == "/"

    worker = _dispatch("/service-worker.js?build=reviewed")
    assert worker.status == 200
    assert worker.headers_out["Service-Worker-Allowed"] == "/"
    assert worker.headers_out["Cache-Control"] == "no-store, max-age=0"
    assert worker.headers_out["Content-Type"].startswith("application/javascript")

    icon = _dispatch("/static/oracle-icon.svg")
    assert icon.status == 200
    assert icon.headers_out["Content-Type"].startswith("image/svg+xml")


def test_static_route_rejects_unlisted_root_and_traversal() -> None:
    assert _dispatch("/oracle_remote_pwa.js").status == 404
    assert _serve_private_helper("/static/../gateway/server.py").status == 403


def test_installability_and_visible_truth_are_in_the_served_dom() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for required in (
        'rel="manifest" href="/manifest.webmanifest"',
        'id="oracleConnectionState"',
        'id="oracleLatencyState"',
        'id="oraclePwaState"',
        'id="oracleBuildTruth"',
        "window.__ORACLE_PWA_EXPECTED_BUILD = 'oracle-pwa-stage-a/2026-08-11.v5'",
        'src="/static/ms4_voice_dsp.js?build=oracle-pwa-stage-a%2F2026-08-11.v5"',
        'src="/static/voice_input_session.js?build=oracle-pwa-stage-a%2F2026-08-11.v5"',
        'src="/static/oracle_remote_pwa.js?build=oracle-pwa-stage-a%2F2026-08-11.v5"',
        "window.OracleRemotePwa.bootstrap(window, document)",
    ):
        assert required in html


def test_pwa_addition_cannot_open_media_and_hooks_only_follow_voice_state() -> None:
    pwa = (ROOT / "machine_spirit_4" / "web" / "oracle_remote_pwa.js").read_text(
        encoding="utf-8"
    )
    for forbidden in ("getUserMedia", "enumerateDevices", "getDisplayMedia"):
        assert forbidden not in pwa

    html = INDEX.read_text(encoding="utf-8")
    enabled = html[html.index("async function enableFullDuplex"):html.index("async function disableFullDuplex")]
    disabled = html[html.index("async function disableFullDuplex"):html.index("function vadPushAudio")]
    assert enabled.count("wakeLock.setDesired(true)") == 1
    assert enabled.index("wakeLock.setDesired(true)") > enabled.index("vadState.enabled = true")
    assert disabled.count("wakeLock.setDesired(false)") == 1
    assert disabled.index("wakeLock.setDesired(false)") > disabled.index("vadState.enabled = false")


def test_build_mismatch_fences_media_before_inline_runtime() -> None:
    html = INDEX.read_text(encoding="utf-8")
    bootstrap = "window.__oraclePwa = window.OracleRemotePwa.bootstrap(window, document);"
    inline_anchor = "const messages = document.getElementById('messages');"
    assert html.count(bootstrap) == 1
    assert html.index(bootstrap) < html.index(inline_anchor)
    authority = html[
        html.index("const oraclePwaMediaAuthority"):
        html.index("// Oracle exact physical device binding")
    ]
    assert "boot.status !== 'blocked_build_mismatch'" in authority
    assert "boot.build === expectedBuild" in authority
    assert "boot.expectedBuild === expectedBuild" in authority

    direct_recording_start = html.index("async function startRecording()")
    direct_recording = html[
        direct_recording_start:
        html.index("if (recorder) return", direct_recording_start)
    ]
    assert "oraclePwaMediaAuthority.require('start-recording')" in direct_recording

    mic_acquire = html[
        html.index("async function acquireMicStream()"):
        html.index("function micErrorMessage")
    ]
    assert mic_acquire.index("oraclePwaMediaAuthority.require('microphone-acquire')") < mic_acquire.index("getUserMedia")

    exact_discovery = html[
        html.index("async function oracleDiscoverExactDevices()"):
        html.index("async function oracleEnsureExactBinding()")
    ]
    assert exact_discovery.count("oraclePwaMediaAuthority.require(") >= 4
    assert exact_discovery.index("oraclePwaMediaAuthority.require(") < exact_discovery.index("enumerateDevices()")

    room_capture = html[
        html.index("function startRoomRecorder()"):
        html.index("// ---- scoped SSE tee")
    ]
    assert room_capture.index("oraclePwaMediaAuthority.require('room-recorder')") < room_capture.index("getDisplayMedia")

    capture_driver = html[
        html.index("function runCapture(prompt, runMarker)"):
        html.index("runOracleTurn: function", html.index("function runCapture(prompt, runMarker)"))
    ]
    assert capture_driver.index("oraclePwaMediaAuthority.require('capture-driver')") < capture_driver.index("__oracleCaptureDriver")

    mic = html[html.index("function micPressedDown()"):html.index("function micReleased()")]
    assert "oraclePwaMediaAuthority.require('push-to-talk')" in mic
    assert mic.index("oraclePwaMediaAuthority.require(") < mic.index("beginActiveOracleVoiceInputSession")

    enable = html[html.index("async function enableFullDuplex()"):html.index("async function disableFullDuplex")]
    assert "oraclePwaMediaAuthority.require('full-duplex-enable')" in enable
    assert enable.index("oraclePwaMediaAuthority.require(") < enable.index("cancelActiveOracleVoiceInputSession")

    deferred = html[
        html.index("function runDeferredFullDuplexAutoArm"):
        html.index("window.__ms4RunDeferredFullDuplexAutoArm")
    ]
    assert "oraclePwaMediaAuthority.require('full-duplex-auto-arm')" in deferred
    assert deferred.index("oraclePwaMediaAuthority.require(") < deferred.index("enableFullDuplex()")

    pwa = (ROOT / "machine_spirit_4" / "web" / "oracle_remote_pwa.js").read_text(
        encoding="utf-8"
    )
    mismatch = pwa[pwa.index("if (expectedBuild !== BUILD)"):pwa.index("const retry =")]
    for control in (
        "messageInput",
        "micButton",
        "sendButton",
        "oracleStartHandsfree",
        "settingsFullDuplex",
    ):
        assert control in mismatch


def test_depth_wait_cue_is_exact_session_job_and_turn_owned() -> None:
    html = INDEX.read_text(encoding="utf-8")
    block = html[html.index("function daTrackDispatchedJob"):html.index("function daResolveTrackedJob")]
    for required in (
        "const firstSeen = !daSeenInFlight.has(safeJobId);",
        "const cueTurnId = currentVoiceTurnId;",
        "validateOwner: () => (",
        "daAnnouncedSession === safeSessionId",
        "String(sessionId || '') === safeSessionId",
        "daSeenInFlight.has(safeJobId)",
        "currentVoiceTurnId === cueTurnId",
    ):
        assert required in block
    assert "getUserMedia" not in block


def test_service_worker_has_no_runtime_cache_write_path() -> None:
    worker = (ROOT / "machine_spirit_4" / "web" / "service-worker.js").read_text(
        encoding="utf-8"
    )
    assert "if (url.pathname !== '/' || url.search) return;" in worker
    assert "if (url.searchParams.toString() !== BUILD_QUERY) return;" in worker
    assert "fetch(request).catch(() => caches.match(request))" in worker
    assert "if (url.search || !STATIC_PATHS.has(url.pathname)) return;" in worker
    assert "cache.put" not in worker
    install = worker[worker.index("self.addEventListener('install'"):worker.index("self.addEventListener('activate'")]
    assert ".then(() => self.skipWaiting())" in install
    assert "self.addEventListener('message'" in install
    assert "data.type !== 'oracle-activate-build'" in install
    assert "data.build !== BUILD" in install
    for forbidden in ("/chat", "/voice/turn", "/api/", "/memory", "/conversations/"):
        assert forbidden not in worker


def test_runbook_reports_current_startup_watchdog_lifecycle() -> None:
    runbook = (
        ROOT / "machine_spirit_4" / "docs" / "ORACLE_PRIVATE_REMOTE_PWA_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    for required in (
        "per-user",
        "`HiveMind Oracle.lnk` Startup watchdog",
        "supervise_ms4.py --watch --interval-seconds 300",
        "MS3 `:9080`",
        "MS4 gateway",
        "`:9180`",
        "MS4 MCP `:9181`",
        "warden_service.json` definition remains reference-only",
        "does not claim Warden-managed 24/7 availability",
    ):
        assert required in runbook
