import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_normal_oracle_view_keeps_the_durable_transcript_visible() -> None:
    html = _source("machine_spirit_4/web/index.html")

    assert '<section id="messages"></section>' in html
    assert '<section id="messages" class="dev-only"></section>' not in html


def test_terminal_delivery_fetches_exact_job_and_renders_full_success_text() -> None:
    html = _source("machine_spirit_4/web/index.html")
    delivery = _between(
        html,
        "async function announceDoubleAgentCompletion(job)",
        "async function daAnnounceTerminalJobs(jobs)",
    )

    fetch = "await fetch(`/api/v1/double-agent/jobs/${encodeURIComponent(job.job_id)}/deliver`"
    assert fetch in delivery
    assert delivery.index(fetch) < delivery.index("messages.appendChild(el)")
    assert "method: 'POST'" in delivery
    assert "body: JSON.stringify({conversation_id: sessionId})" in delivery
    assert "data.history_bound === true" in delivery
    assert "const resultText = String(result.text || '').trim();" in delivery
    assert "answer.textContent = resultText;" in delivery
    assert "buildCompletionSpeech(job, resultText)" in delivery
    assert "job.job_id" in delivery
    assert "addEventListener('click', async" not in delivery


def test_terminal_delivery_never_labels_failed_or_partial_output_successful() -> None:
    html = _source("machine_spirit_4/web/index.html")
    delivery = _between(
        html,
        "async function announceDoubleAgentCompletion(job)",
        "async function daAnnounceTerminalJobs(jobs)",
    )

    assert "job.state === 'completed'" in delivery
    assert "result.status === 'success'" in delivery
    assert "resultText.length > 0" in delivery
    assert "Depth job did not complete successfully" in delivery
    assert "Partial output (not a successful result)" in delivery


def test_failed_exact_result_fetch_remains_eligible_for_automatic_retry() -> None:
    html = _source("machine_spirit_4/web/index.html")
    refresh = _between(
        html,
        "async function refreshDoubleAgent()",
        "async function cancelDoubleAgentJob(jobId)",
    )

    assert "const needsCompletionDelivery = jobs.some" in refresh
    assert "daSeenInFlight.has(j.job_id)" in refresh
    assert "!daAnnounced.has(j.job_id)" in refresh
    assert "const shouldPoll = anyInFlight || needsCompletionDelivery;" in refresh


def test_completion_speech_uses_result_content_without_a_details_teaser() -> None:
    html = _source("machine_spirit_4/web/index.html")
    speech = _between(
        html,
        "function buildCompletionSpeech(job, resultText)",
        "const ORACLE_OPTIONAL_SOUND_KEY",
    )
    lowered = speech.lower()

    assert "resulttext" in lowered
    assert "The Depth answer is complete. ${fullText}" in speech
    assert ".slice(" not in speech
    assert "digest" not in lowered
    for teaser in ("want the full details", "want the details", "would you like", "if you want"):
        assert teaser not in lowered

    announcement = _between(
        html,
        "async function speakAnnouncement(text, turnId = currentVoiceTurnId, idleAfterPlayback = null)",
        "const TEXT_CHAT_SPEAK_KEY",
    )
    assert "fetch('/voice/synthesize/stream'" in announcement
    assert "expectedIndex" in announcement
    assert "Number(payload.audio_chunks) !== expectedIndex" in announcement
    assert "synthesize stream ended before terminal done" in announcement


def test_chat_stream_eof_before_done_is_an_explicit_failure() -> None:
    html = _source("machine_spirit_4/web/index.html")
    streaming = _between(
        html,
        "async function sendStreamingChat(text, turnId, hooks = {})",
        "document.getElementById('chatForm')",
    )

    marker = "markStreamingAssistantIncomplete(state, 'stream ended before terminal done.')"
    assert marker in streaming
    assert streaming.index("if (done) break;") < streaming.index(
        marker
    )


def test_blocking_chat_never_renders_or_speaks_unverified_completion() -> None:
    html = _source("machine_spirit_4/web/index.html")
    blocking = _between(
        html,
        "async function sendBlockingChat(text, turnId, hooks = {})",
        "function markStreamingAssistantIncomplete",
    )

    guard = "if (data.completed !== true || data.cancelled === true)"
    assert guard in blocking
    assert blocking.index(guard) < blocking.index("addMessage('assistant', data.text, data)")
    assert blocking.index(guard) < blocking.index("await speakTextChatReply")
    assert "Partial text was not committed or spoken." in blocking


def test_completion_deliveries_are_serialized_in_terminal_order() -> None:
    html = _source("machine_spirit_4/web/index.html")
    deliveries = _between(
        html,
        "async function daAnnounceTerminalJobs(jobs)",
        "async function refreshDoubleAgent()",
    )

    assert "const orderedJobs = [...jobs].sort" in deliveries
    assert "for (const job of orderedJobs)" in deliveries
    assert "await announceDoubleAgentCompletion(job)" in deliveries
    assert "Promise.all" not in deliveries


def test_dispatched_job_is_persisted_before_first_terminal_poll() -> None:
    html = _source("machine_spirit_4/web/index.html")
    adoption = _between(html, "function adoptAssistantRoutingMeta", "function renderAssistantMeta")
    add_message = _between(html, "function addMessage", "function createStreamingAssistantMessage")
    incomplete = _between(html, "if (eventName === 'incomplete')", "if (eventName === 'error')")
    tracking = _between(
        html,
        "function daAnnouncedKey(sid)",
        "// ---- Proactive SPOKEN delivery",
    )
    deliveries = _between(
        html,
        "async function daAnnounceTerminalJobs(jobs)",
        "async function refreshDoubleAgent()",
    )

    seed = "daTrackDispatchedJob(meta.dispatched_job.job_id, sessionId);"
    assert seed in adoption
    assert adoption.index(seed) < adoption.index("refreshDoubleAgent();")
    assert "adoptAssistantRoutingMeta(meta);" in add_message
    assert "adoptAssistantRoutingMeta(eventData);" in incomplete
    assert "function daPendingKey" in tracking
    assert "localStorage.getItem(daPendingKey(sid))" in tracking
    assert "daPersistPending(safeSessionId)" in tracking
    assert "daResolveTrackedJob(job.job_id, sessionId)" in deliveries


def test_incomplete_face_stream_preserves_depth_dispatch_identity() -> None:
    server = _source("machine_spirit_4/gateway/server.py")
    incomplete = _between(
        server,
        'events.put(("incomplete", {',
        "except ValueError as exc:",
    )

    assert '"session_id": result.get("session_id")' in incomplete
    assert '"dispatched_job": result.get("dispatched_job")' in incomplete
    assert '"router": result.get("router")' in incomplete
    assert '"depth_lobe_model": result.get("depth_lobe_model")' in incomplete


def test_manual_depth_submit_persists_job_before_first_refresh() -> None:
    html = _source("machine_spirit_4/web/index.html")
    submit = _between(
        html,
        "async function submitDoubleAgentJob()",
        "daSubmitBtn.addEventListener",
    )

    tracking = "daTrackDispatchedJob(data.job_id, sessionId);"
    assert "if (!data.job_id)" in submit
    assert tracking in submit
    assert submit.index(tracking) < submit.index("daSubmitDialog.close();")
    assert submit.index(tracking) < submit.index("refreshDoubleAgent();")


def test_announcement_failure_aborts_fetch_and_cancels_reader() -> None:
    html = _source("machine_spirit_4/web/index.html")
    announcement = _between(
        html,
        "async function speakAnnouncement(text, turnId = currentVoiceTurnId, idleAfterPlayback = null)",
        "const TEXT_CHAT_SPEAK_KEY",
    )

    assert "let reader = null;" in announcement
    assert "if (!controller.signal.aborted) controller.abort();" in announcement
    assert "await reader.cancel()" in announcement


def test_voice_guidance_keeps_requested_substance_and_never_offers_to_continue() -> None:
    runner = _source("machine_spirit_4/gateway/hermes_runner.py")
    voice = _between(runner, "_VOICE_BREVITY_DIRECTIVE = (", "_VOICE_INLINE_MAX_CHARS")
    face = _source("machine_spirit_4/gateway/face_lobe_chat.py")
    base = _between(face, "FACE_LOBE_SYSTEM_PROMPT = (", "HTTP_TIMEOUT_DEFAULT")

    assert "at most 2-3 short sentences" not in voice
    assert "offer to send the full detail" not in voice
    assert "Do not omit requested substance" in voice
    assert "180 to 240 visible words" in voice
    assert "150 is a hard floor and 380 is a hard ceiling" in voice
    assert "single-fact lookups may be shorter" in voice
    assert "complete answer is visible" in voice
    assert "answer every requested part" in base
    assert "verified completed Depth Lobe result" in base
    assert "offer to continue" not in base.lower()


def test_structured_depth_result_grounds_a_referential_json_followup(tmp_path) -> None:
    from machine_spirit_4.double_agent import (
        Blackboard,
        JobEnvelope,
        JobResult,
        build_face_lobe_context_block,
    )
    from machine_spirit_4.double_agent import safety
    from machine_spirit_4.gateway.face_lobe_chat import apply_face_lobe_output_guard

    board = Blackboard(tmp_path / "structured-followup.sqlite3")
    job_id = safety.new_job_id()
    envelope = JobEnvelope(
        job_id=job_id,
        parent_conversation_id="structured-followup",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="List each requested component.",
        internal_goal="List each requested component.",
    )
    envelope.validate()
    board.insert_job(envelope)
    result_text = json.dumps(
        {"components": [{"name": "alpha"}, {"name": "beta"}]},
        indent=2,
    )
    board.insert_result(
        JobResult(
            job_id=job_id,
            status="success",
            summary="Two components found.",
            text=result_text,
            confidence="high",
            conversation_revision_id=1,
        )
    )
    board.update_job_state(job_id, state="completed")
    context = build_face_lobe_context_block(
        conversation_id="structured-followup",
        blackboard=board,
    )

    guarded, verdict = apply_face_lobe_output_guard(
        result_text,
        message="What were those components again?",
        extra_system=context,
    )

    assert guarded == result_text
    assert verdict["applied"] is False
