import io
import hashlib
import json
from pathlib import Path
import threading
import types

from machine_spirit_4.gateway import server as srv_module
from machine_spirit_4.gateway.voice import VoiceUnavailable


ROOT = Path(__file__).resolve().parents[2]


class _VoiceRunner:
    ms3_url = "http://ms3:9080"
    hivemind_url = "http://hive:6089"


def _voice_handler() -> srv_module.Ms4GatewayHandler:
    boundary = "----VOICEBOUNDARY"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="t.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\nWAVDATA\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = _VoiceRunner()
    handler.command = "POST"
    handler.path = "/voice/turn/stream"
    handler.headers = types.SimpleNamespace(get={
        "Host": "127.0.0.1:9180",
        "Content-Length": str(len(body)),
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }.get)
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.requestline = "POST /voice/turn/stream HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"
    return handler


def _voice_json_handler(
    payload: object | None = None,
    *,
    raw: bytes | None = None,
    content_type: str = "application/json",
) -> srv_module.Ms4GatewayHandler:
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = _VoiceRunner()
    handler.command = "POST"
    handler.path = "/voice/turn/stream"
    headers = {
        "Host": "127.0.0.1:9180",
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
    }
    handler.headers = types.SimpleNamespace(get=headers.get)
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.requestline = "POST /voice/turn/stream HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"
    return handler


def _static_handler(path: str) -> tuple[srv_module.Ms4GatewayHandler, list[tuple[str, str]]]:
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.path = path
    handler.wfile = io.BytesIO()
    handler.status_code = None
    headers: list[tuple[str, str]] = []
    handler.send_response = lambda status: setattr(handler, "status_code", status)
    handler.send_header = lambda name, value: headers.append((name, value))
    handler.end_headers = lambda: None
    return handler, headers


def test_do_get_static_index_routes_on_parsed_path_and_disables_http_cache(
    monkeypatch, tmp_path
):
    expected = b"<html>reviewed oracle candidate</html>"
    (tmp_path / "index.html").write_bytes(expected)
    monkeypatch.setattr(srv_module, "WEB_ROOT", tmp_path)
    expected_sha256 = hashlib.sha256(expected).hexdigest().upper()
    handler, headers = _static_handler(f"/?v={expected_sha256}")

    handler.do_GET()

    assert handler.status_code == 200
    assert handler.wfile.getvalue() == expected
    assert ("Content-Type", "text/html; charset=utf-8") in headers
    assert ("Cache-Control", "no-store, max-age=0") in headers
    assert ("Pragma", "no-cache") in headers
    assert ("Expires", "0") in headers


def test_do_get_static_asset_query_serves_exact_bytes(monkeypatch, tmp_path):
    expected = b"window.reviewedAsset = true;\n"
    (tmp_path / "reviewed.js").write_bytes(expected)
    monkeypatch.setattr(srv_module, "WEB_ROOT", tmp_path)
    handler, headers = _static_handler("/static/reviewed.js?v=exact-sha256")

    handler.do_GET()

    assert handler.status_code == 200
    assert handler.wfile.getvalue() == expected
    assert ("Content-Type", "application/javascript; charset=utf-8") in headers
    assert ("Content-Length", str(len(expected))) in headers


def test_do_get_static_query_does_not_weaken_traversal_rejection(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(srv_module, "WEB_ROOT", tmp_path)
    handler, _headers = _static_handler("/static/%2e%2e/outside.txt?v=reviewed")
    response = {}
    monkeypatch.setattr(
        srv_module,
        "_json_response",
        lambda _handler, status, payload: response.update(
            status=status, payload=payload
        ),
    )

    handler.do_GET()

    assert response == {"status": 403, "payload": {"error": "forbidden"}}


def test_do_get_preserves_non_static_query_route_behavior(monkeypatch):
    handler, _headers = _static_handler("/audit?limit=7&view=compact")
    observed_limits = []
    response = {}
    monkeypatch.setattr(
        srv_module,
        "read_events",
        lambda *, limit: observed_limits.append(limit) or [{"event": "kept"}],
    )
    monkeypatch.setattr(
        srv_module,
        "_json_response",
        lambda _handler, status, payload: response.update(
            status=status,
            payload=payload,
        ),
    )

    handler.do_GET()

    assert observed_limits == [7]
    assert response == {
        "status": 200,
        "payload": {"events": [{"event": "kept"}]},
    }


def test_server_exposes_required_routes():
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")

    for route in (
        '"/health"',
        '"/healthcheck/basic"',
        '"/api/v1/ms4_gateway/healthcheck/basic"',
        '"/api/v1/ms4_gateway/status"',
        '"/chat"',
        '"/chat/stream"',
        '"/sessions"',
        '"/deps/status"',
        '"/hermes/tools"',
        '"/hermes/tool"',
        '"/vision/analyze-local"',
        '"/models"',
        '"/voice/status"',
        '"/voice/synthesize/stream"',
    ):
        assert route in server
    for hermes_route in (
        "/api/v1/hermes/version",
        "/api/v1/hermes/releases",
        "/api/v1/hermes/update",
        "/api/v1/hermes/update/status",
    ):
        assert hermes_route in server, f"missing route {hermes_route}"
    for da_route in (
        "/api/v1/double-agent/jobs",
        "/api/v1/double-agent/jobs/",
        "/api/v1/double-agent/conversations/",
    ):
        assert da_route in server, f"missing route fragment {da_route}"
    for da_handler in (
        "_double_agent_submit",
        "_double_agent_list",
        "_double_agent_get",
        "_double_agent_cancel",
        "_double_agent_mark_stale",
        "_double_agent_events",
        "_double_agent_revision_bump",
        "_double_agent_revision_get",
    ):
        assert da_handler in server, f"missing handler {da_handler}"
    for voice_route in (
        "/voice/transcribe",
        "/voice/synthesize",
        "/voice/turn",
    ):
        assert voice_route in server, f"missing voice route {voice_route}"
    assert "_voice_transcribe" in server
    assert "_voice_synthesize" in server
    assert "_voice_turn" in server
    # The UI appends a query string (?session_id=...) to /voice/turn; the
    # POST handler must strip the query string before matching, or it'll
    # 404 on every voice request. See the "Voice turn failed: not found"
    # regression. We assert the path-normalization call exists.
    assert 'path_only = self.path.split("?", 1)[0]' in server
    assert "hermes_admin.trigger_update" in server
    assert "hermes_admin.initialize_state" in server
    assert "require_contained_runtime(SERVICE_NAME)" in server
    assert "default_runner()" in server
    assert "recover_on_startup" in server
    # The gateway must NOT set a chat_runner_factory on the default Double
    # Agent runner; doing so flips workers back into in-thread mode and
    # defeats the subprocess-cancel fix.
    assert "set_chat_runner_factory(" not in server
    assert "Ms4HermesRunner" in server
    assert "service_info" in server


def test_mcp_server_refuses_uncontained_runtime():
    mcp = (ROOT / "machine_spirit_4" / "mcp" / "server.py").read_text(encoding="utf-8")
    assert "from machine_spirit_4.contained import require_contained_runtime" in mcp
    assert "require_contained_runtime(SERVER_NAME)" in mcp


def test_ms4_web_entrypoint_uses_gateway_routes():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    assert 'id="chatForm"' in html
    assert "/chat" in html
    assert "/chat/stream" in html
    assert "streamToggle" in html
    assert "/models" in html
    assert "/voice/status" in html
    assert "Hermes tool trace" in html
    assert "Hermes" in html
    assert "64K context" in html
    assert "return 'done'" in html
    assert "if (terminal) return" in html


def test_ms4_web_renders_hermes_update_banner():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    for required in (
        'id="hermesUpdateBanner"',
        'id="hermesUpdateBtn"',
        'id="hermesUpdatePinBtn"',
        'id="hermesPinDialog"',
        'id="hermesPinSelect"',
        'id="hermesVersionCurrent"',
        "/api/v1/hermes/version",
        "/api/v1/hermes/releases",
        "/api/v1/hermes/update",
        "refreshHermesVersion",
        "triggerHermesUpdate",
        "loadHermesReleases",
        "latest_signature_state",
        "official_tag_unsigned",
        "Failed target:",
        "unsigned - not offered",
        "hermesLastInfo",
    ):
        assert required in html, f"missing UI hook: {required}"


def test_ms4_web_has_voice_ptt_controls():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        'id="micButton"',
        'id="voiceStatusLine"',
        'id="voiceReplyAudio"',
        "/voice/turn",
        "startRecording",
        "stopRecordingAndSend",
        "getUserMedia",
        # MS4 captures WAV directly in the browser (AudioContext + ScriptProcessor
        # + on-the-fly 16-bit PCM encode) so the HiveMind ASR pipeline never needs
        # FFmpeg to decode WebM/Opus blobs. Pin those bits so a regression to
        # MediaRecorder doesn't reintroduce the "FFmpeg not found" 400.
        "AudioContext",
        "encodeWavBlob",
        "TARGET_SAMPLE_RATE",
        "mic.wav",
    ):
        assert required in html, f"missing voice UI hook: {required}"
    microphone_capture = html.split(
        "// We deliberately do NOT use MediaRecorder.", 1
    )[1].split("/* >>> ORACLE CAPTURE HOOKS SEAM 7", 1)[0]
    room_evidence_recorder = html.split(
        "// ---- room recorder: REAL screen+audio recapture", 1
    )[1].split("// ---- scoped SSE tee", 1)[0]

    # Microphone capture must remain direct PCM/WAV. The separate room-evidence
    # path legitimately records display video plus audio as WebM.
    assert "new MediaRecorder(" not in microphone_capture
    assert "getDisplayMedia" in room_evidence_recorder
    assert "new MediaRecorder(stream" in room_evidence_recorder


def test_ms4_web_loads_bounded_voice_input_session_before_oracle_runtime():
    web = ROOT / "machine_spirit_4" / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    core_path = web / "voice_input_session.js"
    assert core_path.is_file(), "the accepted bounded session core must remain a real static asset"
    core = core_path.read_text(encoding="utf-8")
    assert "window.MS4VoiceInputSession = api" in core
    assert "createVoiceInputSession" in core
    assert "onFinalizing({ sessionId: sessionId, reason: reason });" in core

    dsp_tag = '<script src="/static/ms4_voice_dsp.js?build=oracle-pwa-stage-a%2F2026-08-11.v5"></script>'
    session_tag = '<script src="/static/voice_input_session.js?build=oracle-pwa-stage-a%2F2026-08-11.v5"></script>'
    inline_anchor = "const messages = document.getElementById('messages');"
    assert session_tag in html, "the served browser path must load the accepted bounded session core"
    assert html.index(dsp_tag) < html.index(session_tag) < html.index(inline_anchor)

    runtime = html[html.index(session_tag):]
    for required in (
        "createOracleVoiceInputSession",
        "Ms4VoiceTranscriptTurn.v1",
        "/voice/transcribe",
        "bounded-session",
        "voice_input_session",
        "onFinalizing() {",
        "mode === 'vad' && ownerTurnId === currentVoiceTurnId",
    ):
        assert required in runtime, f"missing bounded browser adapter hook: {required}"
    assert "let recordedChunks = []" not in runtime
    assert "vadState.recordedChunks.push" not in runtime


def test_server_chat_stream_has_cooperative_disconnect_guard():
    """Text /chat/stream should not keep consuming Face Lobe tokens after
    the browser SSE client disconnects. The worker API is cooperative
    today: failed SSE writes clear client_alive, then stream_callback()
    returns False so FaceLobeChat can close its upstream response.
    """
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")
    stream_chat = server.split("def _stream_chat", 1)[1].split("def _serve_static", 1)[0]
    assert "client_alive = threading.Event()" in stream_chat
    assert "client_alive.set()" in stream_chat
    assert "def stream_callback(delta: str) -> bool:" in stream_chat
    assert "if not client_alive.is_set():" in stream_chat
    assert "return False" in stream_chat
    assert "client_alive.clear()" in stream_chat
    assert stream_chat.count("client_alive.clear()") >= 4


def test_server_voice_stream_closes_on_terminal_event():
    """After the voice SSE handler emits done/error it must close the HTTP
    response. Otherwise clients can receive a valid done event but keep
    waiting forever on a keep-alive connection."""
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")
    voice_stream = server.split("def _voice_turn_stream", 1)[1].split("    # ------------------------------------------------------------------", 1)[0]

    assert "if item.terminal:" in voice_stream
    assert "self.close_connection = True" in voice_stream


def test_voice_stream_zero_audio_failure_emits_error_once_and_never_done(monkeypatch):
    writes: list[tuple[str, dict]] = []

    def fake_stream(**_kwargs):
        raise VoiceUnavailable("voice output unavailable: zero client-written audio")

    def fake_sse(_handler, event, payload, **_kwargs):
        writes.append((event, dict(payload)))
        return True

    monkeypatch.setattr(srv_module, "voice_ptt_turn_stream", fake_stream)
    monkeypatch.setattr(srv_module, "_sse_event", fake_sse)
    handler = _voice_handler()
    handler._voice_turn_stream()

    terminal = [(event, payload) for event, payload in writes if event in {"done", "error"}]
    assert len(terminal) == 1
    assert terminal[0][0] == "error"
    assert terminal[0][1]["fail_closed"] is True
    assert not any(event == "done" for event, _payload in writes)


def test_voice_stream_write_failure_nacks_producer_and_joins_worker(monkeypatch):
    writes: list[str] = []
    producer_observed: dict[str, bool] = {}

    def fake_stream(**kwargs):
        producer_observed["accepted"] = kwargs["emit"](
            "audio_chunk",
            {"index": 0, "audio_base64": "V0FW", "audio_mime": "audio/wav"},
        )
        producer_observed["cancelled"] = kwargs["cancel_event"].is_set()
        raise VoiceUnavailable("client write failed")

    def fake_sse(_handler, event, _payload, **_kwargs):
        writes.append(event)
        if event == "audio_chunk":
            return False
        return True

    monkeypatch.setattr(srv_module, "voice_ptt_turn_stream", fake_stream)
    monkeypatch.setattr(srv_module, "_sse_event", fake_sse)
    handler = _voice_handler()
    handler._voice_turn_stream()

    assert producer_observed == {"accepted": False, "cancelled": True}
    assert writes.count("audio_chunk") == 1
    assert writes.count("error") == 1
    assert writes.count("done") == 0
    assert not any(
        thread.name == "ms4-voice-stream" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_voice_stream_accepts_one_strict_transcript_ready_envelope(monkeypatch):
    transcript = "alpha sentinel repeated phrase repeated phrase omega sentinel"
    envelope = {
        "schema": "Ms4VoiceTranscriptTurn.v1",
        "transcript": transcript,
        "transcription_model": "bounded-session",
        "source": "voice_input_session",
    }
    stream_calls: list[dict] = []
    transcribe_calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    network_asr_calls = 0

    def forbidden_network_asr(**_kwargs):
        nonlocal network_asr_calls
        network_asr_calls += 1
        raise AssertionError("transcript-ready input must not call network ASR")

    def fake_stream(**kwargs):
        stream_calls.append(kwargs)
        result = kwargs["transcribe_fn"](
            hivemind_url=kwargs["runner"].hivemind_url,
            audio=kwargs["audio"],
            filename=kwargs["filename"],
            model=kwargs["transcribe_model"],
        )
        transcribe_calls.append(result)
        assert kwargs["emit"](
            "transcript",
            {"text": result["text"], "raw_text": result["text"], "asr_ms": 0, "model": result["model"]},
        )
        return {
            "transcript": result["text"],
            "raw_transcript": result["text"],
            "reply_text": "",
            "session_id": "strict-json-session",
            "foreground_model": None,
            "router": None,
            "dispatched_job": None,
            "grounding_source": None,
            "transcription_model": result["model"],
            "tts_model": "tts-1",
            "metrics": {"audio_chunks": 0, "audio_errors": 0},
        }

    def fake_sse(_handler, event, payload, **_kwargs):
        events.append((event, dict(payload)))
        return True

    monkeypatch.setattr(srv_module, "transcribe", forbidden_network_asr)
    monkeypatch.setattr(srv_module, "voice_ptt_turn_stream", fake_stream)
    monkeypatch.setattr(srv_module, "_sse_start", lambda _handler: None)
    monkeypatch.setattr(srv_module, "_sse_event", fake_sse)

    handler = _voice_json_handler(envelope)
    handler._voice_turn_stream()

    assert len(stream_calls) == 1
    call = stream_calls[0]
    assert call["audio"] == b""
    assert call["filename"] == "transcript.json"
    assert call["transcribe_model"] == "bounded-session"
    assert callable(call["transcribe_fn"])
    assert transcribe_calls == [{"text": transcript, "model": "bounded-session"}]
    assert network_asr_calls == 0
    assert [event for event, _payload in events].count("transcript") == 1
    assert [event for event, _payload in events].count("done") == 1


def test_voice_stream_rejects_ambiguous_or_oversize_transcript_envelopes(monkeypatch):
    exact = {
        "schema": "Ms4VoiceTranscriptTurn.v1",
        "transcript": "bounded transcript",
        "transcription_model": "bounded-session",
        "source": "voice_input_session",
    }
    responses: list[tuple[int, dict]] = []
    stream_calls = 0

    def fake_response(_handler, status, payload):
        responses.append((status, dict(payload)))

    def forbidden_stream(**_kwargs):
        nonlocal stream_calls
        stream_calls += 1
        return {
            "transcript": "must not run",
            "reply_text": "",
            "metrics": {},
        }

    monkeypatch.setattr(srv_module, "_json_response", fake_response)
    monkeypatch.setattr(srv_module, "voice_ptt_turn_stream", forbidden_stream)

    cases: list[tuple[str, srv_module.Ms4GatewayHandler]] = []
    cases.append(("unknown schema", _voice_json_handler({**exact, "schema": "Ms4VoiceTranscriptTurn.v2"})))
    cases.append(("unknown field", _voice_json_handler({**exact, "unexpected": True})))
    cases.append(("mixed audio+json", _voice_json_handler({**exact, "audio_base64": "V0FW"})))
    cases.append(("empty transcript", _voice_json_handler({**exact, "transcript": ""})))
    cases.append(("whitespace transcript", _voice_json_handler({**exact, "transcript": " \r\n\t"})))
    cases.append(("non-string transcript", _voice_json_handler({**exact, "transcript": ["not", "text"]})))
    cases.append(("oversize transcript", _voice_json_handler({**exact, "transcript": "x" * 65_537})))
    cases.append(("missing source", _voice_json_handler({key: value for key, value in exact.items() if key != "source"})))
    cases.append((
        "duplicate transcript",
        _voice_json_handler(
            raw=(
                b'{"schema":"Ms4VoiceTranscriptTurn.v1",'
                b'"transcript":"first","transcript":"second",'
                b'"transcription_model":"bounded-session","source":"voice_input_session"}'
            ),
        ),
    ))
    cases.append(("malformed json", _voice_json_handler(raw=b'{"schema":')))

    boundary = "----MIXEDVOICE"
    mixed_body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="t.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\nWAVDATA\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="transcript"\r\n\r\n'
        "ambiguous transcript\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    cases.append((
        "multipart audio+transcript",
        _voice_json_handler(
            raw=mixed_body,
            content_type=f"multipart/form-data; boundary={boundary}",
        ),
    ))

    for label, handler in cases:
        responses.clear()
        handler._voice_turn_stream()
        assert responses, f"{label}: request did not fail before SSE/chat/TTS"
        assert responses[-1][0] == 400, f"{label}: expected HTTP 400, got {responses[-1]}"

    assert stream_calls == 0, "ambiguous or malformed admission reached chat/TTS"


def test_ms4_web_renders_oracle_voice_front_door():
    """MS4's first screen should be the Oracle/machine-spirit voice
    front door, with the FleetOps/MDM honeycomb face language wired to
    real voice lifecycle states instead of living only in Settings.
    """
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    for required in (
        'id="oracleStage"',
        'id="oracleMirror"',
        'id="oracleSubtitle"',
        'id="oracleTranscriptLine"',
        'id="oraclePresenceChip"',
        'id="oracleTimingLine"',
        'id="oracleLobeLine"',
        'id="oracleLobesLine"',
        'id="oracleHandsfreePrompt"',
        'id="oracleStartHandsfree"',
        'id="oracleKeepPushToTalk"',
        'id="oracleVoiceReadiness"',
        'id="oracleVoiceReadinessText"',
        'id="oracleProvisionVoiceBtn"',
        'id="developerViewToggle"',
        'id="chatForm"',
        'id="messageInput"',
        'aria-label="Oracle voice front door"',
        'data-audio-reactive="false"',
        "Backend lobes active:",
        "oracle-mirror-cell",
        "mirror-speaking-bars",
        "mirror-speaking-cell",
        "mirror-wave-cell",
        "oraclePresenceAnimationFrame",
        "oraclePlaybackAnalyser",
        "oraclePlaybackSignal",
        "ensureOraclePlaybackAnalyser",
        "oraclePlaybackNodeForSource",
        "getByteFrequencyData",
        "getByteTimeDomainData",
            "source.connect(onsetAnalyser || oraclePlaybackNodeForSource(ctx));",
        "#chatForm { display: grid; grid-template-columns: minmax(0, 1fr) 48px auto;",
        "#chatForm { grid-template-columns: 56px minmax(0, 1fr);",
        "#messageInput { min-width: 0; width: 100%; justify-self: stretch; }",
        "#messageInput { grid-column: 1 / -1; }",
        "input, select, button { box-sizing: border-box;",
        "renderOracleMirror",
        "oracleMirrorCellSpecs",
        "setOracleStageState",
        "setOracleStageTranscript",
        "setOracleLobeState",
        "setOracleStageFromVoiceStatus",
        "scheduleOracleIdleAfterPlayback",
        "speakTextChatReply",
        "TEXT_CHAT_SPEAK_KEY",
        "ms4_text_chat_speak_replies",
        "await speakTextChatReply(data.text, 'Face Lobe text chat reply spoken.', turnId);",
        "await speakTextChatReply(state.text || eventData.text, 'Face Lobe text chat reply spoken.', state.turnId);",
        "Transcribing your voice.",
        "Oracle is composing.",
        "Oracle is speaking.",
        "Full-duplex listening.",
        "Start hands-free conversation",
        "Provision voice now",
        "renderOracleVoiceReadiness",
        "triggerVoiceServiceProvision(service, oracleProvisionVoiceBtn)",
        "setDeveloperView",
        "ms4_developer_view",
        "main:not(.developer-view) .dev-only",
        'class="grid dev-only"',
        'id="doubleAgentPanel" class="dev-only"',
        'id="messages"',
        "Reply became audible in ${onsetMs}ms.",
    ):
        assert required in html, f"missing Oracle front-door hook: {required}"
    assert 'id="messages" class="dev-only"' not in html

    # The Oracle face should be rendered from hive cells only. Keep the
    # old circle/teardrop guide geometry out of the active SVG template.
    for removed in (
        "mirror-aura",
        "mirror-core",
        "mirror-face-energy",
        "mirror-face-outline",
        "mirror-eyes",
        "mirror-eye-pupil",
        "mirror-eye-rings",
        "shape-brow",
        "<circle ",
    ):
        assert removed not in html, f"Oracle hive face should not render {removed}"
    assert "shape-eye" in html
    assert "shape-socket" in html
    assert "shape-pupil" in html
    assert "cy > 332" in html
    assert "oracleMirrorFaceHalfWidth" in html
    assert "oracleMirrorPointInPiercingEyeZone" in html
    assert "oracleMirrorPiercingEyeCellSpecs" in html
    assert "const eyeOffset = 42" in html
    assert "const eyeY = 207.5" in html
    assert "const detailRadius = 4.55" in html
    assert "const ringRadius = 15.3" in html
    assert "const socketRadiusX = 37" in html
    assert "const socketRadiusY = 24" in html
    assert "const ringBands = [" in html
    assert "{rad: ringRadius, count: 12, cellRadius: detailRadius * 0.50" in html
    assert "{rad: ringRadius * 0.55, count: 8, cellRadius: detailRadius * 0.35" in html
    assert "detailRadius * 0.98" in html
    assert "const eyeOffset = 47" not in html
    assert "const eyeY = 218.5" not in html
    assert "shape === 'brow'" not in html
    assert "detailRadius = 3.2" not in html
    assert "oracleMirrorEyeSocketDetailCellSpecs" not in html
    assert "oracleMirrorPointInEyeSocketDetailZone" not in html
    voice_section = html.split("const voiceBars =", 1)[1].split("const waveCells =", 1)[0]
    wave_section = html.split("const waveCells =", 1)[1].split("oracleMirror.innerHTML", 1)[0]
    assert "const mouthCenterY = 258.5" in html
    assert "oracleMirrorPointInMouthZone" not in html
    assert "shape === 'voice'" not in html
    assert ".mirror-cell.shape-voice" not in html
    assert "if (cy > 238) return 'jaw';" in html
    assert "const voiceColumnCount = 25" in html
    assert "const voiceRows = 5" in html
    assert "Math.hypot((x - mouthCenterX) / 58, (y - mouthCenterY) / 18) > 1" in html
    assert "--voice-core-rgb: 140, 240, 0;" in html
    assert "--hivemind-breath-rgb: 168, 85, 255;" in html
    assert "--hivemind-breath-hot-rgb: 205, 133, 255;" in html
    assert "--hivemind-breath-deep-rgb: 82, 30, 170;" in html
    assert "oracle-hivemind-breath" in html
    assert 'data-presence="speaking"] .mirror-cell.shape-eye.is-active' in html
    assert 'data-presence="speaking"] .mirror-cell.shape-pupil.is-active' in html
    assert 'data-presence="speaking"] .oracle-mirror-cell .mirror-cell-link.is-face' in html
    assert "if (presence === 'speaking') return false;" in html
    base_bars = html.split('.oracle-mirror-cell .mirror-speaking-bars {', 1)[1].split('}', 1)[0]
    base_wave = html.split('.oracle-mirror-cell .mirror-wave {', 1)[1].split('}', 1)[0]
    assert "opacity: 0;" in base_bars
    assert "opacity: 0;" in base_wave
    assert '.oracle-stage[data-presence="speaking"] .mirror-speaking-bars' in html
    assert '.oracle-stage[data-presence="speaking"] .mirror-wave' in html
    assert '.oracle-stage[data-presence="reflex"] .mirror-speaking-bars' in html
    assert '.oracle-stage[data-presence="reflex"] .mirror-wave' in html
    assert '.oracle-stage[data-presence="listening"] .mirror-wave' not in html
    assert "Array.from({ length: voiceColumnCount }" in voice_section
    assert "Array.from({ length: voiceRows }" in voice_section
    assert "cellIndex - Math.floor(voiceRows / 2)" in voice_section
    assert "const rowDistance = Math.abs(rowOffset);" in voice_section
    assert "const isSignalNode = (" in voice_section
    assert "distance >= 8 && rowDistance <= 1" in voice_section
    assert "distance >= 6 && rowDistance === 2" in voice_section
    assert "is-hivemind-breath" in voice_section
    assert "is-signal-node" in voice_section
    assert "is-voice-core" not in voice_section
    assert "mouthCell(" in voice_section
    assert "isSignalNode ? 1.86 : 2.08" in voice_section
    assert "Array.from({ length: 41 }" in wave_section
    assert "'mirror-wave-cell is-hivemind-breath'" in wave_section
    assert "mouthCell(" in wave_section
    assert "1.82," in wave_section
    assert "const mouthCenterY = 205.5" not in html
    assert "const mouthCenterY = 244.5" not in html
    assert "const mouthCenterY = 269.5" not in html
    assert "cy > 252 && cy < 277 && ax < 37" not in html
    assert "Array.from({ length: 9 }" not in voice_section
    assert "Array.from({ length: 3 }" not in voice_section
    assert "shape === 'voice' && Math.abs(col) <= 2" not in html
    assert "cellIndex - 1" not in voice_section
    assert "cellIndex - 3" not in voice_section
    # The visible VAD default must match the runtime default. This was
    # previously 700 ms in markup while the code used 1400 ms, making
    # timing feel random before the user touched Settings.
    assert 'id="settingsVadHangoverMs" type="range" min="200" max="2000" step="50" value="1400"' in html
    assert 'id="settingsVadHangoverMsLabel" class="muted">1400 ms</span>' in html
    assert "Server now defaults to ws_super" not in html
    assert 'id="messageInput" autocomplete="off" placeholder="Talk to MS4 through Hermes..." autofocus' not in html


def test_ms4_web_renders_full_duplex_vad_panel():
    """The Settings dialog must surface the full-duplex (VAD barge-in)
    panel: enable toggle + ack toggle + half-context toggle + three
    sliders (energy threshold, min speech duration, hangover) + live
    mic-level meter. The hands-free PTT loop in the JS depends on these
    DOM ids being present, so this test guards the contract."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    # The PTT-extraction refactor moved the /voice/turn/stream SSE
    # consumer into submitWavBlobAsVoiceTurn(). Both `myTurnId` (turn
    # race protection) and `turnStart` (used to compute first_audio_ms)
    # MUST be declared inside that function, not just outside it —
    # otherwise we hit "myTurnId is not defined" at runtime the moment
    # the user opens a voice turn. Lock those locals in.
    submit_fn = html.split("async function submitWavBlobAsVoiceTurn(", 1)[1]
    submit_fn = submit_fn.split("\n// ----- Full-duplex voice", 1)[0]
    assert "const myTurnId = currentVoiceTurnId;" in submit_fn, \
        "submitWavBlobAsVoiceTurn must declare myTurnId before the SSE loop"
    assert "const turnStart = Date.now();" in submit_fn, \
        "submitWavBlobAsVoiceTurn must declare turnStart for first_audio_ms math"

    for required in (
        # Settings DOM ids
        'id="settingsFullDuplex"',
        'id="settingsVadAckOnOnset"',
        'id="settingsHalfContext"',
        'id="settingsVadSpeechThreshold"',
        'id="settingsVadMinSpeechMs"',
        'id="settingsVadHangoverMs"',
        'id="settingsVadMicMeter"',
        'id="settingsVadMicMeterBar"',
        'id="settingsVadMicMeterThreshold"',
        'id="settingsVadState"',
        # JS state + functions
        "vadState",
        "enableFullDuplex",
        "disableFullDuplex",
        "vadTick",
        "vadComputeRms",
        "onVadSpeechOnset",
        "onVadSpeechOffset",
        "submitWavBlobAsVoiceTurn",
        "vadPushAudio",
        "playRandomReflexAck",
        # Reflex ack ids — must match the ones canned_reflexes.py
        # pre-renders, or the instant ack will silently fall through
        # to nothing on speech onset.
        "'ack_mhm'",
        "'ack_yes'",
        "'ack_go_ahead'",
        "'ack_listening'",
        # localStorage keys for full-duplex settings
        "ms4_voice_full_duplex",
        "updateHandsfreePrompt",
        "localStorage.getItem(FULL_DUPLEX_KEY) !== null",
        "localStorage.setItem(FULL_DUPLEX_KEY, 'true')",
        "ms4_voice_vad_ack",
        "ms4_voice_vad_halfctx",
        "ms4_voice_vad_speech_threshold",
        "ms4_voice_vad_min_speech_ms",
        "ms4_voice_vad_hangover_ms",
        # Half-context audio behavior
        "HALF_CONTEXT_TAIL_MS",
        "halfContext",
        "playAck",
        # The hands-free loop must use the same /voice/turn/stream
        # endpoint as PTT so the server doesn't need a separate path.
        "submitWavBlobAsVoiceTurn",
    ):
        assert required in html, f"missing full-duplex UI hook: {required}"

    reflex_wrapper = html.split("const _coreBargeIn = bargeIn;", 1)[1]
    reflex_wrapper = reflex_wrapper.split("// ----- Voice services lifecycle", 1)[0]
    assert "patchedBargeIn(reason, options)" in reflex_wrapper
    assert "_coreBargeIn(reason, opts);" in reflex_wrapper
    # The delayed ack is turn-owned: it respects an explicit playAck:false, never
    # speaks for an internal/null reason, binds to the turn it created, and
    # cancels stale callbacks (at most one owned ack per current turn).
    assert "opts.playAck === false" in reflex_wrapper
    assert "if (reason == null) return;" in reflex_wrapper
    assert "if (ackTurnId !== currentVoiceTurnId) return;" in reflex_wrapper


def test_ms4_web_renders_hivemind_admin_panels():
    """The May-25 2026 HiveMind catalog expansion landed UI panels for
    VMs, apps, voice identities, storage, and an approval queue, plus
    top-level approval + maintenance banners. Lock the DOM ids + JS
    function names in so a future refactor can't silently drop them.
    """
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    for required in (
        # Settings DOM ids
        'id="settingsHivemindVms"',
        'id="settingsHivemindVmsRefresh"',
        'id="settingsHivemindApps"',
        'id="settingsHivemindAppsRefresh"',
        'id="settingsVoiceIdentities"',
        'id="settingsVoiceIdentityEnroll"',
        'id="settingsVoiceIdentityName"',
        'id="settingsHivemindStorage"',
        'id="settingsHivemindStorageRefresh"',
        'id="settingsApprovalQueue"',
        'id="settingsApprovalCheck"',
        # Top-level banners
        'id="approvalBanner"',
        'id="approvalBannerApprove"',
        'id="approvalBannerReject"',
        'id="maintenanceBanner"',
        # JS functions
        "refreshHivemindVms",
        "refreshHivemindApps",
        "refreshVoiceIdentities",
        "enrollVoiceIdentity",
        "refreshHivemindStorage",
        "checkApprovalStatus",
        "refreshApprovalQueue",
        "showApprovalBanner",
        "updateMaintenanceBanner",
        # vmAction destructive confirms
        "Force-stop ${name}?",
        "Delete VM ${name}?",
        # Routes the UI calls
        "/hivemind/vms",
        "/hivemind/apps",
        "/hivemind/voice_identities",
        "/hivemind/voice_identities/enroll",
        "/hivemind/storage",
        "/hivemind/approval/status/",
        # Speaker pill rendering
        "payload.speaker",
        "speakerName",
        "🎤 ${speakerName}:",
    ):
        assert required in html, f"missing HiveMind-admin UI hook: {required}"


def test_ms4_web_renders_may26_admin_expansion_panels():
    """May 26 2026 fill-all-gaps round added four new Settings panels
    (Oracle / Training / Adapters / Loadout) plus the VM screenshot
    now uses the Ms4VmScreenshot.v1 shape with image_base64 +
    mime_type. Lock the DOM ids + function names + URL contracts in
    so a future refactor can't silently drop them."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    for required in (
        # Settings DOM ids
        'id="settingsHivemindOracle"',
        'id="settingsHivemindOracleRefresh"',
        'id="settingsOracleReadinessStatus"',
        'id="settingsOracleChatInput"',
        'id="settingsOracleChatBtn"',
        'id="settingsHivemindTraining"',
        'id="settingsHivemindTrainingRefresh"',
        'id="settingsHivemindAdapters"',
        'id="settingsHivemindAdaptersRefresh"',
        'id="settingsHivemindLoadout"',
        'id="settingsHivemindLoadoutRefresh"',
        # JS functions
        "renderOracleReadiness",
        "refreshHivemindOracle",
        "askOracle",
        "refreshHivemindTraining",
        "refreshHivemindAdapters",
        "deployAdapter",
        "refreshHivemindLoadout",
        "applyLoadout",
        # Routes the panels call
        "/hivemind/oracle/readiness",
        "/hivemind/oracle/chat",
        "/hivemind/training",
        "/hivemind/adapters",
        "/hivemind/adapters/deploy",
        "/hivemind/loadout",
        "/hivemind/loadout/apply",
        # VM screenshot updated shape: accept image_base64 + mime_type
        "body.image_base64",
        "body.mime_type",
    ):
        assert required in html, f"missing May-26 admin UI hook: {required}"


def test_ms4_web_renders_gpu_passthrough_panel():
    """The May-27 2026 GPU passthrough round added a Settings panel
    for GPU-P (consumer Windows 11 path) + DDA (wired for future
    Windows Server license) + vGPU (NVIDIA license). Lock the DOM
    ids + JS function names + URL contracts so a future refactor
    can't silently drop them."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    for required in (
        'id="settingsGpuPassRefresh"',
        'id="settingsGpuPassStatus"',
        'id="settingsGpuPassSnapshot"',
        'id="settingsGpuPassVmName"',
        'id="settingsGpuPassCreateVmBtn"',
        'id="settingsGpuPassPrepPciId"',
        'id="settingsGpuPassPrepMode"',
        'id="settingsGpuPassPrepBtn"',
        'id="settingsGpuPassVgpuPciId"',
        'id="settingsGpuPassVgpuProfile"',
        'id="settingsGpuPassVgpuCount"',
        'id="settingsGpuPassVgpuBtn"',
        "refreshGpuPassthrough",
        "createGpuPGameStreamVm",
        "prepareGpuPassthroughMode",
        "createVgpuMdev",
        "/hivemind/gpu/passthrough/snapshot",
        "/hivemind/gpu/passthrough/prepare",
        "/hivemind/gpu/passthrough/vgpu",
        "/hivemind/gpu/passthrough/game-stream-vm",
        "GPU passthrough (GPU-P / DDA / vGPU)",
        # Make the DDA caveat explicit in the panel:
        "Windows Server",
        # And remind the operator that GPU-P uses the prebuilt template:
        "windows_game_stream_prebuilt",
    ):
        assert required in html, f"missing GPU-passthrough UI hook: {required}"


def test_server_exposes_gpu_passthrough_routes():
    """Lock the GPU passthrough REST surface in: 4 routes + 4
    handler methods + the error class hookup in the admin error
    mapper."""
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")

    for handler in (
        "_hivemind_gpu_passthrough_snapshot",
        "_hivemind_gpu_passthrough_prepare",
        "_hivemind_gpu_passthrough_vgpu",
        "_hivemind_gpu_passthrough_game_stream_vm",
    ):
        assert handler in server, f"missing GPU passthrough handler: {handler}"
    for route in (
        '"/hivemind/gpu/passthrough/snapshot"',
        '"/hivemind/gpu/passthrough/prepare"',
        '"/hivemind/gpu/passthrough/vgpu"',
        '"/hivemind/gpu/passthrough/game-stream-vm"',
    ):
        assert route in server, f"missing GPU passthrough route: {route}"
    assert "GpuPassthroughError" in server


def test_ms4_web_renders_game_session_panel():
    """The May-26 2026 game-session round added a Phase-1 dry-run
    panel for HiveMind's ``hivemind.game_session.*`` orchestrator.
    Lock the DOM ids + JS function names + URL contracts in so a
    future refactor can't silently drop them."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    for required in (
        'id="settingsGameId"',
        'id="settingsGameAvailBtn"',
        'id="settingsGamePlanRunBtn"',
        'id="settingsGameStatus"',
        'id="settingsGameSnapshot"',
        "checkGameAvailability",
        "planAndRunGameSession",
        "/hivemind/games/",
        "/availability",
        "/hivemind/game-sessions/plan-run",
        "Game sessions (Phase 1 dry-run)",
        "would_call",
    ):
        assert required in html, f"missing game-session UI hook: {required}"


def test_server_exposes_game_session_routes():
    """The game admin module exposes a small REST surface (six routes).
    Lock them in so server.py refactors can't silently drop the
    dispatch table that the UI + MCP proxies depend on."""
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")

    for handler in (
        "_hivemind_game_availability_get",
        "_hivemind_game_session_plan",
        "_hivemind_game_session_run",
        "_hivemind_game_session_status_get",
        "_hivemind_game_session_evidence_get",
        "_hivemind_game_session_cancel",
        "_hivemind_game_session_plan_run",
    ):
        assert handler in server, f"missing game handler: {handler}"
    for route in (
        '"/hivemind/games/"',
        '"/availability"',
        '"/hivemind/game-sessions/plan"',
        '"/hivemind/game-sessions/run"',
        '"/hivemind/game-sessions/plan-run"',
        '"/cancel"',
        '"/status"',
        '"/evidence"',
    ):
        assert route in server, f"missing game route fragment: {route}"


def test_ms4_web_renders_double_agent_panel():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        'id="doubleAgentPanel"',
        'id="daJobList"',
        'id="daRevisionId"',
        'id="daSubmitBtn"',
        'id="daSubmitDialog"',
        'id="daSubmitConfirm"',
        'id="daEmptyState"',
        "/api/v1/double-agent/jobs",
        "/api/v1/double-agent/conversations/",
        "refreshDoubleAgent",
        "cancelDoubleAgentJob",
        "submitDoubleAgentJob",
        "Depth Lobe dispatched",
        'value="oracle_deep_chat"',
        'value="oracle_deep_coder"',
        'value="oracle_diagnostic"',
        'value="oracle_planner"',
        'value="oracle_research"',
        'value="oracle_verifier"',
        "/deep",
        "/direct",
    ):
        assert required in html, f"missing Double Agent UI hook: {required}"


def test_ms4_web_has_voice_feedback_overhaul():
    """Jun 1 2026 voice overhaul: soft onset tick (not a spoken ack),
    context-aware `reflex` event, high-fidelity thinking ambience, and strict
    replacement-turn abort ownership for long-utterance ASR."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "function playOnsetTick(",            # soft non-speech onset cue
        "function startThinkingAmbient(",     # high-fidelity dead-air ambience
        "function stopThinkingAmbient(",
        "pulse: {frequencies: [523.25, 783.99]", # perceptible HLI-derived progress tone
        "function scheduleOracleThinkingBridgePulse(",
        "function oracleThinkingBridgeOwns(",
        "function oracleThinkingBridgeAdmissionValidator(",
        "validateOwner: validateAdmission",
        "beginOracleThinkingBridge(ownerTurnId, {playTransition: true});",
        "pauseOracleThinkingBridge(myTurnId);",
        "resumeOracleThinkingBridge(myTurnId, {immediate: true});",
        "eventName === 'reflex'",             # context-aware canned response
        "const reflexSelection = beginVoiceReflexTelemetry(",
        "const reflexPlayed = reflexSelection.selectedId && await playReflex(",
        "recordVoiceReflexAudible(turnState.telemetry, reflexSelection);",
        "recordVoiceReflexScheduled(turnState.telemetry, reflexSelection);",
        "onEnded: ({entry} = {}) => {",
        "entry.preemptedByGeneratedReply === true",
        "stopOracleThinkingBridge(turnState.turnId);",
        "firstSseReceivedMs: null",
        "firstDecodedMs: null",
        "firstScheduledMs: null",
        "firstNonSilentOnsetMs: null",
        "audibleCompletionMs: null",
        "client_first_audible_ms",
        "client_first_reply_audio_ms",
        "client_full_utterance_completion_ms",
        "abortActiveBrowserTurns(reason)",    # abort fetch and established reader
        "turnState.reader = reader",          # explicit response-body ownership
        "currentVoiceTurnId += 1",            # invalidate late SSE/audio work
        "|| '1400'",                          # raised end-of-turn hangover default
    ):
        assert required in html, f"missing voice-overhaul hook: {required}"
    assert "keepStream" not in html
    assert "VAD_CONTINUATION_WINDOW_MS" not in html
    first_audio_receipt = html.split(
        "if (turnState.telemetry.firstSseReceivedMs == null) {", 1
    )[1].split("if (payload.engine)", 1)[0]
    assert "stopOracleOptionalAudio()" not in first_audio_receipt
    assert "stopOracleThinkingBridge(" not in first_audio_receipt
    # The onset must no longer fire a spoken ack reflex (it talked over you).
    assert "playAck: isVadAckEnabled()" not in html, "onset should no longer play a spoken ack"


def test_ms4_web_has_latency_pass_hooks():
    """Jun 1 2026 ChatGPT-level latency pass (client side): a curated
    'Fast' model group for sub-second voice tokens, and the late
    `speaker` event handler that decorates the user bubble AFTER the
    transcript ships (speaker ID now runs off the critical path)."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "FAST_FACE_MODELS",                          # curated fast small models
        r"Fast \u2014 best for voice latency",       # optgroup label (JS-escaped in source)
        "eventName === 'speaker'",                   # late async speaker-ID handler
        "voiceUserBubble",                           # decorates the existing bubble in place
    ):
        assert required in html, f"missing latency-pass hook: {required}"


def test_server_provisions_tts_replicas_for_concurrency():
    """Jun 2 2026: MS4 calls HiveMind's POST /provision/tts/scale on boot
    (and periodically) so per-chunk TTS fans across one GPU per replica —
    the fix for serialized TTS, scaling with nodes/GPUs."""
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")
    assert "_tts_autoscale_loop" in server
    assert "provision_tts_replicas" in server
    assert "MS4_VOICE_TTS_AUTOSCALE" in server
    assert "MS4_VOICE_TTS_REPLICA_TARGET" in server  # pack replicas (GEN_LOCK serializes per GIM)
    voice = (ROOT / "machine_spirit_4" / "gateway" / "voice.py").read_text(encoding="utf-8")
    assert "def provision_tts_replicas" in voice
    assert "/provision/tts/scale" in voice


def test_server_autoprovisions_asr_before_prewarm():
    """ASR boot prewarm must ensure the service is actually provisioned.
    A silent transcription alone can pass through HiveMind's silence gate
    without loading Whisper, which leaves voice input broken on first use."""
    voice = (ROOT / "machine_spirit_4" / "gateway" / "voice.py").read_text(encoding="utf-8")
    assert "MS4_VOICE_ASR_AUTOPROVISION" in voice
    assert "request_voice_service(hivemind_url, \"ASR\"" in voice
    assert "poll_until_healthy(" in voice
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")
    assert "ms4-asr-prewarm" in server


def test_server_has_face_keepwarm_and_residency_bias():
    """Server-side latency pass: periodic Face keep-warm loop (idle
    cold-start fix) + the Ollama keep_alive residency bias (anti-thrash)
    + fully-async speaker ID (off the critical path)."""
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")
    assert "_face_keepwarm_loop" in server
    assert "MS4_VOICE_FACE_KEEPWARM_SECS" in server
    assert "ms4-flb-keepwarm" in server
    assert "last_face_model()" in server
    flc = (ROOT / "machine_spirit_4" / "gateway" / "face_lobe_chat.py").read_text(encoding="utf-8")
    assert "MS4_FACE_KEEP_ALIVE" in flc
    assert '"keep_alive"' in flc
    assert "_is_local_ollama_model" in flc
    voice = (ROOT / "machine_spirit_4" / "gateway" / "voice.py").read_text(encoding="utf-8")
    assert "MS4_VOICE_SPEAKER_ID_ASYNC" in voice
    assert "def record_face_model" in voice
    assert "def _identify_speaker_async" in voice


def test_gateway_defaults_face_lobe_to_fast_chat_model():
    """The Oracle voice front door must not cold-start on a heavy coder
    model when the operator has not picked a Face Lobe override.
    """
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")
    voice = (ROOT / "machine_spirit_4" / "gateway" / "voice.py").read_text(encoding="utf-8")
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    config = (ROOT / "machine_spirit_4" / "config" / "ms4.runtime.example.yaml").read_text(encoding="utf-8")

    assert "default_model=depth_fallback_model()" in server
    assert "choose_foreground_model" in voice
    assert 'model = os.environ.get("MS4_DEFAULT_MODEL"' not in voice
    assert "face_lobe.configured_default = ${data.face_lobe?.configured_default_model}" in html
    assert "face_lobe.effective_fallback = ${data.face_lobe?.fallback_model || '(unavailable)'}" in html
    assert "face_lobe.fallback_source    = ${data.face_lobe?.fallback_source || '(unknown)'}" in html
    assert "face_lobe.serving_scope      = ${data.face_lobe?.serving_scope || '(unknown)'}" in html
    assert "const depthSelection = data.depth_lobe?.automatic_selection || {};" in html
    assert "depth_lobe.preferred_cluster_target = ${data.depth_lobe?.preferred_cluster_target || '(unavailable)'}" in html
    assert "depth_lobe.quality_target_model      = ${data.depth_lobe?.quality_target_model || '(unavailable)'}" in html
    assert "depth_lobe.current_auto_model       = ${depthSelection.model_id || '(unavailable)'}" in html
    assert "depth_lobe.current_auto_source      = ${depthSelection.source || '(unknown)'}" in html
    assert "depth_lobe.current_auto_policy_tier = ${depthSelection.policy_tier || '(unknown)'}" in html
    assert "depth_lobe.current_auto_detail      = ${depthSelection.detail || '(none)'}" in html
    assert "depth_lobe.configured_override = ${data.depth_lobe?.configured_override || '(none)'}" in html
    assert "depth_lobe.fallback_policy_tier = ${data.depth_lobe?.fallback_policy_tier || '(unknown)'}" in html
    assert "depth_lobe.minimum_target_params_b = ${data.depth_lobe?.minimum_target_total_parameters_b}" in html
    assert "depth_lobe.fallback_minimum_params_b = ${data.depth_lobe?.fallback_minimum_total_parameters_b}" in html
    assert "depth_lobe.serving_scope       = ${data.depth_lobe?.serving_scope || '(unknown)'}" in html
    # effective_fallback must read the runtime fallback_model (rendered as
    # '(unavailable)' when the picker fails), never the raw configured default;
    # otherwise a rejected heavy default would be presented to the operator as active.
    assert "face_lobe.effective_fallback = ${data.face_lobe?.configured_default_model" not in html
    assert "model: nemotron-3-nano:4b" in config
    assert "model: qwen2.5-coder:32b" not in config


def test_ms4_web_speaks_deep_job_completions_proactively():
    """Jun 1 2026: in a voice conversation the agent must SPEAK deep-job
    completions on its own — the silent chat bubble isn't enough when
    you're hands-free. The spoken delivery is gated on voice mode + a
    Settings toggle and must not talk over the user."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "voiceModeActive",                       # only speak when in a voice convo
        "function buildCompletionSpeech(",       # natural announcement text
        "function speakAnnouncement(",           # synth + play the announcement
        "function speakCompletionWhenQuiet(",    # defer until not talking over the user
        "function playNotifyChime(",             # soft "something's ready" earcon
        "_isVoiceBusy(",                         # the don't-talk-over-you guard
        "isSpeakCompletionsEnabled(",            # Settings toggle gate
        "settingsSpeakCompletions",              # the toggle element
        "ms4_da_speak_completions",              # persisted key (default on)
    ):
        assert required in html, f"missing proactive-completion hook: {required}"
    # It must be wired into the terminal-job announcement path.
    assert "queueCompletionSpeech(speechText)" in html
    assert "fetch('/voice/synthesize/stream'" in html


def test_ms4_web_has_adaptive_endpoint_vad():
    """Jun 1 2026: adaptive end-of-turn — respond sooner after a long,
    complete utterance, but keep the full hangover for short/mid-thought
    fragments so we never clip someone gathering their thoughts."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "function vadEffectiveHangoverMs(",
        "isVadAdaptiveEnabled(",
        "VAD_ADAPTIVE_LONG_SPEECH_MS",
        "VAD_ADAPTIVE_FLOOR_MS",
        "settingsVadAdaptive",
        "ms4_voice_vad_adaptive",
    ):
        assert required in html, f"missing adaptive-VAD hook: {required}"
    # The state machine must use the adaptive hangover, not the raw one.
    assert "vadEffectiveHangoverMs()" in html


def test_ms4_web_rest_engine_labeled_recommended():
    """REST must be labeled recommended/fastest and ws_super as slower, plus
    the one-time auto-clear of a pinned ws_super engine."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    assert "recommended, fastest here" in html       # REST option
    assert "SLOWER on this cluster" in html           # ws_super option, honestly labeled
    assert "ms4_engine_migrated_v2" in html           # one-time ws_super -> REST clear


def test_ms4_web_has_canned_speech_qa_and_ambience_toggle():
    """Reflex QA plus production-reachable optional-sound accessibility controls."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "settingsReflexValidate",      # the "Validate canned speech" button
        "/reflexes/validate",          # endpoint it calls
        "QA \u2713",                   # validated badge
        "function isThinkingAmbienceEnabled(",
        "settingsThinkingAmbience",    # explicit operator toggle
        "ms4_thinking_ambience",       # persisted key (default on)
        "function oracleOptionalSoundEnabled(",
        "function playOracleProceduralCue(",
        "function oracleOptionalAudioState(",
        'id="settingsOracleOptionalSound" type="checkbox" checked',
        "settingsOracleOptionalSoundToggle.addEventListener('change'",
        "localStorage.setItem(ORACLE_OPTIONAL_SOUND_KEY",
    ):
        assert required in html, f"missing QA/ambience hook: {required}"
    assert 'id="settingsThinkingAmbience" type="checkbox" checked' in html
    assert "localStorage.getItem(THINKING_AMBIENCE_KEY) !== 'false'" in html
    reduced_motion = html.split("@media (prefers-reduced-motion: reduce)", 1)[1].split("}", 2)[0]
    for selector in (
        ".oracle-stage::before",
        ".oracle-stage::after",
        ".oracle-mirror-cell .mirror-cell.is-active",
        ".oracle-mirror-cell .mirror-cell.is-status",
        ".oracle-stage .mirror-speaking-cell.is-hivemind-breath",
        ".oracle-stage .mirror-wave-cell",
    ):
        assert selector in reduced_motion, f"reduced-motion override missing {selector}"
    assert "animation: none !important" in reduced_motion


def test_ms4_web_has_coherent_oracle_feedback_and_accessible_audio_controls():
    """Oracle feedback must be legible, adjustable, and keyboard-operable."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        '--mirror-a-rgb: 168, 85, 255;',
        '--mirror-b-rgb: 140, 240, 0;',
        '--voice-core-rgb: 140, 240, 0;',
        "stroke='%23a855ff' stroke-width='4' stroke-linecap='round'",
        'id="voiceStatusLine" class="muted" role="status" aria-live="polite" aria-atomic="true"',
        'id="micButton" type="button"',
        'aria-keyshortcuts="Space Enter"',
        'id="settingsOracleMasterMute" type="checkbox"',
        'id="settingsOracleAudioVolume" type="range" min="0" max="1" step="0.05"',
        'id="settingsOracleAudioVolumeLabel"',
        "const ORACLE_OPTIONAL_AUDIO_MUTED_KEY = 'ms4_oracle_optional_audio_muted';",
        "const ORACLE_OPTIONAL_AUDIO_VOLUME_KEY = 'ms4_oracle_optional_audio_volume';",
        "function oracleOptionalAudioVolume()",
        "menu: {frequencies:",
        "const effectivePeakGain = profile.peakGain * oracleOptionalAudioVolume();",
        "0.022 * oracleOptionalAudioVolume()",
        "playOracleProceduralCue('menu'",
        "settingsOracleMasterMuteToggle.addEventListener('change'",
        "settingsOracleAudioVolumeSlider.addEventListener('input'",
        "function micKeyboardPressed(event)",
        "function micKeyboardReleased(event)",
        "micButton.addEventListener('keydown', micKeyboardPressed);",
        "micButton.addEventListener('keyup', micKeyboardReleased);",
        "micButton.addEventListener('blur', micKeyboardReleased);",
    ):
        assert required in html, f"missing Oracle feedback/accessibility hook: {required}"

    base_palette = html.split('.oracle-stage {', 1)[1].split('}', 1)[0]
    reflex_palette = html.split('.oracle-stage[data-presence="reflex"] {', 1)[1].split('}', 1)[0]
    assert '--mirror-b-rgb: 163, 230, 53;' in reflex_palette
    assert '76, 201, 240' not in base_palette
    assert '125, 211, 252' not in base_palette
    assert "const respectAmbience = kind === 'pending';" not in html


def test_ms4_web_has_honorable_easter_egg():
    """The 'Honorable -> WE ON GO' easter egg: an `egg` SSE handler that
    plays the clip from /easter/honorable."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "function playEggClip(",
        "eventName === 'egg'",
        "/easter/honorable",
    ):
        assert required in html, f"missing honorable easter-egg hook: {required}"


def test_ms4_web_has_robust_mic_acquire():
    """Mic acquisition retries with relaxed constraints and reports
    accurate, actionable errors (not a blanket 'permission denied')."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "function acquireMicStream(",
        "function micErrorMessage(",
        "{audio: true}",              # relaxed-constraint fallback
        "No microphone found",        # NotFoundError guidance
        "Microphone is busy",         # NotReadableError guidance
    ):
        assert required in html, f"missing robust mic-acquire hook: {required}"
    # The old misleading blanket message must be gone.
    assert "Microphone permission denied:" not in html


def test_ms4_web_wires_exact_device_binding_and_static_dsp_asset():
    """The physical Oracle gate must bind the reviewed devices fail-closed.

    Normal, unarmed voice keeps its existing relaxed-mic behavior. Once armed,
    browser-origin-scoped IDs/groups are discovered from the live inventory,
    frozen into the canonical binding authority, and every playback start goes
    through the awaited sink gate. The dependency-free DSP helper is served as a
    classic script immediately before the existing inline runtime.
    """
    web = ROOT / "machine_spirit_4" / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    dsp_path = web / "ms4_voice_dsp.js"

    assert dsp_path.is_file(), "the gateway's /static mapping needs a real ms4_voice_dsp.js asset"
    dsp = dsp_path.read_text(encoding="utf-8")
    assert "window.MS4DSP = api" in dsp

    dsp_tag = '<script src="/static/ms4_voice_dsp.js?build=oracle-pwa-stage-a%2F2026-08-11.v5"></script>'
    inline_anchor = "const messages = document.getElementById('messages');"
    assert dsp_tag in html
    assert inline_anchor in html
    assert html.index(dsp_tag) < html.index(inline_anchor)

    for required in (
        'id="oracleDeviceBindingStatus"',
        "Microphone (Razer Seiren V3 Chroma)",
        "Odyssey Ark (NVIDIA High Definition Audio)",
        "async function oracleEnsureExactBinding(",
        "navigator.mediaDevices.enumerateDevices()",
        "const ORACLE_BINDING_AUTHORITY = Object.freeze",
        "createOracleExactBinding({",
        "Object.defineProperty(window, '__oracleExactBindingState'",
        "async function oracleAwaitSinkReadyBeforeStart(ctx)",
        "await oracleExactBinding.routePlaybackSink(ctx);",
        "await oracleExactBinding.sinkReady();",
        "function vadAdaptiveThresholds(",
        "function vadReadPlaybackForEcho(",
        "window.MS4DSP.effectiveOnsetThreshold(",
        "dsp.updateNoiseFloor(",
        "dsp.updateEchoReference(",
        "dsp.echoGuardOnset(",
        "dsp.echoGuardTransition(",
        "window.MS4DSP.shouldSubmitUtterance(",
        "Full-duplex unavailable: audio safety module is not ready",
    ):
        assert required in html, f"missing exact-device integration hook: {required}"

    acquire = html.split("async function acquireMicStream()", 1)[1].split("\n}", 1)[0]
    assert acquire.index("oracleExactBinding.isRequired()") < acquire.index(
        "oracleExactBinding.acquireBoundCaptureStream()"
    ) < acquire.index("MIC_PREFERRED_CONSTRAINTS")


def test_ms4_web_has_audio_prebuffer():
    """The voice player prebuffers the first chunk(s) so a tiny first
    chunk doesn't leave an audible gap before the second word."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "PREBUFFER_MIN_CHUNKS",
        "PREBUFFER_TARGET_MS",
        "PREBUFFER_MAX_MS",
        "function flushPrebuffer(",
        "function scheduleDecodedChunk(",
        "playbackRunStarted",
    ):
        assert required in html, f"missing audio prebuffer hook: {required}"


def test_ms4_web_separates_voice_receipt_decode_schedule_and_onset():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "client_first_sse_audio_ms",
        "client_first_decoded_audio_ms",
        "client_first_webaudio_scheduled_ms",
        "client_first_non_silent_onset_ms",
        "client_full_utterance_completion_ms",
        "metrics.client_first_audible_ms = telemetry.firstNonSilentOnsetMs",
        "metrics.client_full_utterance_completion_ms = telemetry.audibleCompletionMs",
        "VOICE_ONSET_RMS_THRESHOLD",
        "function startVoiceTurnOnsetMonitor(",
        "function recordVoiceTurnAudibleCompletion(",
        "function oracleVoiceAcceptanceState(",
        "window.__ms4OracleVoiceAcceptanceState",
        "activeGeneratedSources",
        "prebufferDepth",
        "activeVoiceControllerPresent",
        "function finalizeAudibleVerdict(",
        "awaiting_audible",
        "blocked_zero_audio",
        "degraded_silent",
        "degraded_audio_error",
    ):
        assert required in html, f"missing voice acceptance telemetry hook: {required}"
    assert "metrics.client_first_audible_ms = firstAudioAt" not in html
    done_block = html.split("} else if (eventName === 'done') {", 1)[1].split(
        "} else if (eventName === 'error') {", 1
    )[0]
    assert "generatedAudioCount = turnState.telemetry.scheduledChunks" in done_block
    # Finding 3: scheduled chunks are NOT audible success. The done handler
    # blocks on zero audio, degrades on an audio error, and DEFERS the audible
    # verdict (awaiting_audible -> the onset monitor) for clean scheduled audio,
    # so decoded silence can never read as ordinary done/complete.
    assert "terminal = 'blocked_zero_audio'" in done_block
    assert "terminal = 'degraded_audio_error'" in done_block
    assert "terminal = 'awaiting_audible'" in done_block
    assert "finalizeAudibleVerdict(turnState)" in done_block
    assert "turnState.terminalKind = zeroAudioBlocked ? 'blocked_zero_audio' : 'done'" not in done_block
    assert "const generatedAudioCount = Math.max(" not in done_block
    assert "payload.metrics?.audio_client_written || payload.metrics?.audio_chunks" not in done_block


def test_ms4_web_records_full_playback_completion_at_owned_source_drain():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    scheduler = html.split("async function scheduleDecodedChunk(", 1)[1].split(
        "async function stopRecordingAndSend(", 1
    )[0]
    onended = scheduler.split("source.onended = () => {", 1)[1].split("};", 1)[0]
    removal = "activeAudioSources = activeAudioSources.filter"
    completion = "recordVoiceTurnAudibleCompletion(turnState);"
    finalization = "finalizeAudibleVerdict(turnState);"
    assert removal in onended
    assert completion in onended
    assert finalization in onended
    assert onended.index(removal) < onended.index(completion) < onended.index(finalization)

    completion_helper = html.split(
        "function recordVoiceTurnAudibleCompletion(turnState) {", 1
    )[1].split("function voiceTurnAcceptanceSnapshot(", 1)[0]
    assert "activeAudioSources.some(entry => entry.turnState === turnState)" in completion_helper
    assert "turnState.completed !== true" in completion_helper
    assert "telemetry.firstNonSilentOnsetMs == null" in completion_helper
    assert "telemetry.audibleCompletionMs = Math.max(" in completion_helper
    assert "syncVoiceClientMetrics(turnState);" in completion_helper
    assert "recordVoiceTurnAcceptance(turnState);" in completion_helper
    assert html.count("recordVoiceTurnAudibleCompletion(turnState);") >= 2

    assert "['server audio emission', fmtMs(data.first_audio_chunk_ms)]" in html
    assert "['response onset', fmtMs(data.client_first_non_silent_onset_ms ?? data.client_first_audible_ms)]" in html
    assert "['playback complete', fmtMs(data.client_full_utterance_completion_ms)]" in html
    assert "server audio emission ${vm.first_audio_chunk_ms}ms" in html
    assert "response onset ${metrics.client_first_non_silent_onset_ms}ms" in html
    assert "playback complete ${metrics.client_full_utterance_completion_ms}ms" in html
    assert "['first audio', fmtMs(data.first_audio_chunk_ms)]" not in html
    assert "parts.push(`first audio ${vm.first_audio_chunk_ms}ms`)" not in html


def test_ms4_web_timeout_copy_distinguishes_zero_and_partial_audio_progress():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    turn_state_block = html.split("const turnState = {", 1)[1].split(
        "activeVoiceStreamController = streamController;", 1
    )[0]
    assert "nonAudioPayloadFrames: 0" in turn_state_block

    payload_progress_block = html.split(
        "try { payload = dataText ? JSON.parse(dataText) : {}; }", 1
    )[1].split("if (eventName === 'status') {", 1)[0]
    for progress_event in ("status", "transcript", "text_delta"):
        assert f"eventName === '{progress_event}'" in payload_progress_block
    assert (
        "turnState.telemetry.nonAudioPayloadFrames += 1;"
        in payload_progress_block
    )

    timeout_block = html.split(
        "if (turnState.timedOut && myTurnId === currentVoiceTurnId) {", 1
    )[1].split("recordVoiceTurnAcceptance(turnState);", 1)[0]

    assert "const partialAudioChunks = turnState.telemetry.scheduledChunks;" in timeout_block
    assert (
        "turnState.telemetry.firstNonSilentOnsetMs != null"
        in timeout_block
    )
    assert "partialAudioChunks > 0" in timeout_block
    assert (
        "const hadNonAudioPayloadProgress = "
        "turnState.telemetry.nonAudioPayloadFrames > 0;"
        in timeout_block
    )
    partial_branch, remaining_branches = timeout_block.split(
        "if (hadPartialAudioProgress) {", 1
    )[1].split("} else if (hadNonAudioPayloadProgress) {", 1)
    non_audio_branch, zero_branch = remaining_branches.split("} else {", 1)

    cases = (
        (
            {
                "firstNonSilentOnsetMs": None,
                "scheduledChunks": 0,
                "nonAudioPayloadFrames": 0,
            },
            zero_branch,
            "No audio or data arrived before the client liveness deadline.",
            (
                "Voice turn aborted after partial audio.",
                "response data was received.",
            ),
        ),
        (
            {
                "firstNonSilentOnsetMs": None,
                "scheduledChunks": 0,
                "nonAudioPayloadFrames": 3,
            },
            non_audio_branch,
            "No audio arrived before the client liveness deadline, but response data was received.",
            (
                "Voice turn aborted after partial audio.",
                "No audio or data arrived before the client liveness deadline.",
            ),
        ),
        (
            {
                "firstNonSilentOnsetMs": 240,
                "scheduledChunks": 2,
                "nonAudioPayloadFrames": 3,
            },
            partial_branch,
            "Voice turn aborted after partial audio.",
            (
                "No audio or data arrived before the client liveness deadline.",
                "response data was received.",
            ),
        ),
    )
    for telemetry, expected_branch, expected_copy, forbidden_copies in cases:
        had_partial_audio_progress = (
            telemetry["firstNonSilentOnsetMs"] is not None
            and telemetry["scheduledChunks"] > 0
        )
        had_non_audio_payload_progress = telemetry["nonAudioPayloadFrames"] > 0
        selected_branch = (
            partial_branch
            if had_partial_audio_progress
            else non_audio_branch
            if had_non_audio_payload_progress
            else zero_branch
        )
        assert selected_branch is expected_branch
        assert expected_copy in selected_branch
        for forbidden_copy in forbidden_copies:
            assert forbidden_copy not in selected_branch

    assert "Voice turn timed out: no response before the client deadline." in zero_branch
    assert "Voice turn timed out before audio arrived" in non_audio_branch
    assert "Voice turn timed out after partial audio" in partial_branch
    assert "Partial reply audio became audible" in partial_branch
    assert "${partialAudioChunks}" in partial_branch


def test_ms4_web_has_both_lobe_model_selectors():
    """Both Face and Depth lobe models are UI-selectable and sent to the
    backend (chat, voice params, deep-submit)."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        'id="modelSelect"',         # Face Lobe model
        'id="depthModelSelect"',    # Depth Lobe model (new)
        "Face Lobe model",
        "Depth Lobe model",
        "ms4_depth_model_id",       # persisted
        "depth_model_id",           # sent to chat + deep-submit
        "params.set('depth_model_id'",  # sent on voice turns
    ):
        assert required in html, f"missing per-lobe model selector hook: {required}"


def test_ms4_web_filters_non_chat_models_from_lobe_dropdowns():
    """Face/Depth dropdowns must not expose TTS/ASR/VLM-only catalog entries.

    This pins the rev47 Qwen3-TTS class of bug at the UI layer: `/models` is
    already chat-filtered server-side, and the browser keeps the same guard
    before writing either lobe selector or restoring localStorage choices.
    """
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "NON_CHAT_MODEL_ID_RES",
        "function isChatCapableModel(",
        "const models = (data.models || []).filter(isChatCapableModel);",
        "localStorage.removeItem('ms4_model_id')",
        "localStorage.removeItem('ms4_depth_model_id')",
        r"/(^|[-_:])tts($|[-_:])/",
        r"/(^|[-_:])asr($|[-_:])/",
        r"/(^|[-_:])vlm($|[-_:])/",
    ):
        assert required in html, f"missing non-chat model filter hook: {required}"


def test_ms4_web_announces_deep_job_completion():
    """The UI automatically delivers a watched job's full verified answer."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "announceDoubleAgentCompletion",
        "daAnnounceTerminalJobs",
        "daSeenInFlight",          # only announce jobs we watched run
        "ms4_da_announced_",       # localStorage guard against re-announce
        "Depth answer",
        "/deliver",
        "data.result",
        "data.history_bound === true",
    ):
        assert required in html, f"missing Double Agent completion-delivery hook: {required}"
    assert "Show details" not in html


def test_ms4_web_surfaces_double_agent_event_excerpts():
    """Running/stalled jobs need event detail before a terminal result exists."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "toggleDoubleAgentEvents",
        "renderDoubleAgentEventLines",
        "/events?limit=100",
        "payload.result_excerpt",
        "result: ${payload.result_excerpt}",
        "Hide events",
    ):
        assert required in html, f"missing Double Agent event-detail hook: {required}"


def test_ms4_voice_playback_paths_wait_for_measured_onset_and_route_cues():
    """Speech-like paths become active only after measured onset; generated
    cues use the non-mouth analyser and remain explicitly cancellable."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for function_name in ("playReflex", "speakAnnouncement", "playEggClip"):
        body = html.split(f"function {function_name}", 1)[1].split("\n}", 1)[0]
        assert "activeAudioSources.push" in body, f"{function_name} must track playback source"
        assert "watchOracleAuxSpeechOnset(" in body, (
            f"{function_name} must wait for analyser-measured non-silent onset"
        )
        assert "markOraclePlaybackActive();" not in body, (
            f"{function_name} must not claim speaking before measured onset"
        )
    generated = html.split("function scheduleDecodedChunk", 1)[1].split("\n}", 1)[0]
    assert "activeAudioSources.push" in generated
    assert "startVoiceTurnOnsetMonitor(turnState)" in generated
    assert "firstNonSilentOnsetMs" in html
    cue = html.split("function playOracleProceduralCue", 1)[1].split("\n}", 1)[0]
    assert "oraclePlaybackNodeForSource(ctx, 'cue')" in cue
    assert "activeOracleCueSources" in cue
    assert "oraclePlaybackNodeForSource(ctx, 'speech')" not in cue
    assert "playOracleProceduralCue('transition', myTurnId)" in html


def test_fusion_validator_reads_sse_incrementally():
    validator = (ROOT / "machine_spirit_4" / "scripts" / "validate_ms4_fusion.py").read_text(encoding="utf-8")
    post_sse = validator.split("def post_sse", 1)[1].split("def main", 1)[0]

    assert ".readline()" in post_sse
    assert "response.read().decode" not in post_sse


def test_local_vision_bridge_prefers_hivemind_mcp_vlm_tool():
    vision = (ROOT / "machine_spirit_4" / "gateway" / "vision.py").read_text(encoding="utf-8")

    assert "hivemind.vlm.describe_image@v1" in vision
    assert "/v1/chat/completions" in vision
    assert "analysis_source" in vision
    assert "empty_visible_text" in vision
    assert "llama3.2-vision:11b-instruct-q4_K_M" in vision
