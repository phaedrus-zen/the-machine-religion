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
        'aria-label="Oracle voice front door"',
        "Face Lobe front door",
        "Backend lobes can run deep chat",
        "oracle-mirror-cell",
        "mirror-speaking-bars",
        "mirror-speaking-cell",
        "mirror-wave-cell",
        "renderOracleMirror",
        "oracleMirrorCellSpecs",
        "setOracleStageState",
        "setOracleStageTranscript",
        "setOracleLobeState",
        "setOracleStageFromVoiceStatus",
        "Transcribing your voice.",
        "Oracle is composing.",
        "Oracle is speaking.",
        "Full-duplex listening.",
        "First audio in ${firstAudioAt}ms",
    ):
        assert required in html, f"missing Oracle front-door hook: {required}"

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
    context-aware `reflex` event, high-fidelity thinking ambience, and the
    raised VAD hangover / continuation guard for long-utterance ASR."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "function playOnsetTick(",            # soft non-speech onset cue
        "function startThinkingAmbient(",     # high-fidelity dead-air ambience
        "function stopThinkingAmbient(",
        "eventName === 'reflex'",             # context-aware canned response
        "VAD_CONTINUATION_WINDOW_MS",         # continuation merge guard
        "keepStream",                          # don't drop prior transcript on re-onset
        "|| '1400'",                          # raised end-of-turn hangover default
    ):
        assert required in html, f"missing voice-overhaul hook: {required}"
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
    assert "speakCompletionWhenQuiet(buildCompletionSpeech(job)" in html


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
    """Reflex QA report hooks + the (default-off) thinking-ambience toggle."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "settingsReflexValidate",      # the "Validate canned speech" button
        "/reflexes/validate",          # endpoint it calls
        "QA \u2713",                   # validated badge
        "function isThinkingAmbienceEnabled(",
        "settingsThinkingAmbience",    # the opt-in toggle
        "ms4_thinking_ambience",       # persisted key (default off)
    ):
        assert required in html, f"missing QA/ambience hook: {required}"


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


def test_ms4_web_announces_deep_job_completion():
    """Proactive completion delivery: the UI announces a watched deep
    job's terminal transition inline with a Show-details expander."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "announceDoubleAgentCompletion",
        "daAnnounceTerminalJobs",
        "daSeenInFlight",          # only announce jobs we watched run
        "ms4_da_announced_",       # localStorage guard against re-announce
        "Deep job done",
        "Show details",
        "data.result",             # pulls full result.text from the detail endpoint
    ):
        assert required in html, f"missing Double Agent completion-delivery hook: {required}"


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
