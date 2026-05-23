from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
    # And make sure we did NOT regress to MediaRecorder for capture.
    assert "new MediaRecorder(" not in html, "MediaRecorder must not be reintroduced for voice capture"


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
        "/deep",
        "/direct",
    ):
        assert required in html, f"missing Double Agent UI hook: {required}"


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
