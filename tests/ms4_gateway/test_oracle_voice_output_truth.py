"""Fail-first: Oracle recent-turns, pipeline stages, and readiness truth.

Live 9180 (2026-08-26): /voice/status and /voice/services return 200, ASR is
configured_not_provisioned, chat plane is ready, and /voice/recent-turns?limit=5
returns an empty turns list even though the audit log holds 1552 voice_turn_*
rows buried under later chat/quartermaster events. Input readiness must not
be treated as output delivery; Oracle chat readiness must not be collapsed
into ASR provisioning.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import types
from pathlib import Path

from machine_spirit_4.gateway import audit, oracle_admin
from machine_spirit_4.gateway import server as srv_module

ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "machine_spirit_4" / "web" / "index.html"

_HEALTHY_CHAT_LIFECYCLE = {
    "source": "menta_hli.cluster.lifecycle_state.v1",
    "ai_plane_ready": True,
    "boot_stage": "ready",
    "workload_phase": "idle",
    "signals": {"loaded_models_count": 2},
    "capabilities_available": ["chat", "embedding"],
    "capabilities_unavailable": [],
}


def _patch_healthy_foreground_model(monkeypatch) -> None:
    monkeypatch.setattr(
        oracle_admin,
        "choose_foreground_model",
        lambda **_kwargs: types.SimpleNamespace(
            model_id="nemotron-3-nano:4b",
            source="loaded",
            detail="test catalog admission",
        ),
    )


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )


def _voice_complete(transcript: str = "hello oracle") -> dict:
    return {
        "timestamp": "2026-08-15T02:17:41.857686+00:00",
        "event_type": "voice_turn_complete",
        "transcript": transcript,
        "reply_text_preview": "hello human",
        "audio_chunks": 3,
        "delivery": {"client_written": True, "tts_bytes": 2048},
    }


def _padded_chat(n: int, pad_size: int = 1024) -> dict:
    return {
        "timestamp": f"2026-08-26T12:{n // 60:02d}:{n % 60:02d}.000000+00:00",
        "event_type": "chat_turn",
        "n": n,
        "pad": "x" * pad_size,
    }


EQUAL_SIZE_BYTES = 204800


def _exact_size_audit_bytes(size: int, *, trailing: dict | None, pad_char: bytes = b"x") -> bytes:
    tail = b""
    if trailing is not None:
        tail = (json.dumps(trailing, ensure_ascii=False) + "\n").encode("utf-8")
    remain = size - len(tail)
    prefix = b'{"event_type":"chat_turn","pad":"'
    suffix = b'"}\n'
    pad = remain - len(prefix) - len(suffix)
    if pad < 1:
        raise AssertionError(f"size {size} too small for exact audit fixture")
    raw = prefix + (pad_char * pad) + suffix + tail
    if len(raw) != size:
        raise AssertionError(f"expected {size} bytes, got {len(raw)}")
    return raw


def _rewrite_same_inode(path: Path, raw: bytes) -> None:
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(raw)
        handle.truncate(len(raw))


def _forbid_full_audit_read(monkeypatch, audit_path: Path) -> None:
    original_read_text = Path.read_text

    def boom(self, *args, **kwargs):
        if self == audit_path:
            raise AssertionError("recent-turns must not read the whole audit file")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boom)


def test_read_events_matching_finds_buried_voice_turns_without_full_read_text(
    tmp_path, monkeypatch
):
    audit_path = tmp_path / "ms4_audit.jsonl"
    events = [_voice_complete("hello oracle")] + [
        {"timestamp": f"2026-08-26T00:00:{i:02d}Z", "event_type": "chat_turn", "n": i}
        for i in range(80)
    ]
    _write_jsonl(audit_path, events)
    _forbid_full_audit_read(monkeypatch, audit_path)

    result = audit.read_matching_events(
        event_types=("voice_turn_complete", "voice_turn_failed"),
        limit=5,
        audit_path=audit_path,
        max_bytes=256_000,
        timeout_s=2.0,
    )
    assert result["complete"] is True
    assert result["reason"] is None
    turns = result["events"]
    assert turns
    assert turns[0]["event_type"] == "voice_turn_complete"
    assert turns[0]["transcript"] == "hello oracle"


def test_read_matching_events_times_out_with_last_good_or_explicit_error(tmp_path):
    audit_path = tmp_path / "ms4_audit.jsonl"
    # One known-good turn followed by enough later noise that a tiny timeout
    # plus a slow decode still has to fail closed with an explicit error.
    events = [
        {
            "timestamp": "2026-08-15T00:00:00Z",
            "event_type": "voice_turn_complete",
            "transcript": "last-good",
        }
    ] + [
        {"timestamp": f"2026-08-26T01:00:{i:02d}Z", "event_type": "chat_turn", "pad": "x" * 200}
        for i in range(40)
    ]
    _write_jsonl(audit_path, events)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64,
        timeout_s=0.0,
    )
    assert payload["schema"] == "Ms4VoiceRecentTurns.v1"
    assert payload["turns"] or payload.get("error")
    if payload["turns"]:
        assert payload["turns"][0]["transcript"] == "last-good"
    else:
        assert payload.get("error")


def test_append_event_mirrors_voice_turns_into_bounded_ring(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    monkeypatch.setenv("MS4_AUDIT_LOG", str(audit_path))
    audit.append_event(
        "chat_turn",
        {"session_id": "s1"},
        audit_path=audit_path,
    )
    complete = audit.append_event(
        "voice_turn_complete",
        {
            "transcript": "ping",
            "reply_text": "pong",
            "audio_chunks": 1,
            "tts_bytes": 512,
            "delivery": {"client_written": True},
        },
        audit_path=audit_path,
    )
    payload = audit.read_recent_voice_turns(limit=5, audit_path=audit_path)
    assert payload["turns"]
    assert payload["turns"][0]["event_type"] == "voice_turn_complete"
    assert payload["turns"][0]["transcript"] == "ping"
    assert complete["event_type"] == "voice_turn_complete"
    # Ring must not require scanning unrelated chat_turn rows.
    assert all(
        t.get("event_type") in {"voice_turn_complete", "voice_turn_failed"}
        for t in payload["turns"]
    )


def test_voice_recent_turns_handler_uses_matching_reader_not_mixed_tail():
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(
        encoding="utf-8"
    )
    body = server.split("def _voice_recent_turns_get", 1)[1].split(
        "    # ------------------------------------------------------------------", 1
    )[0]
    assert "read_recent_voice_turns" in body
    assert "read_events(limit=limit * 8)" not in body
    assert 'e.get("event_type")' in body or "read_recent_voice_turns" in body


def test_classify_voice_pipeline_truth_never_marks_input_ready_as_output():
    from machine_spirit_4.gateway.voice_pipeline_truth import classify_voice_pipeline_truth

    ready_only = classify_voice_pipeline_truth(
        input_ready=True,
        asr_transcript="",
        reasoning_result="",
        tts_bytes=b"",
        playback_receipt=None,
        persisted_turn=None,
        ui_status="voice ready",
    )
    assert ready_only["input_ready"] is True
    assert ready_only["asr_ok"] is False
    assert ready_only["reasoning_ok"] is False
    assert ready_only["tts_bytes_ok"] is False
    assert ready_only["playback_ok"] is False
    assert ready_only["persisted"] is False
    assert ready_only["output_delivered"] is False

    delivered = classify_voice_pipeline_truth(
        input_ready=True,
        asr_transcript="hello",
        reasoning_result="hello back",
        tts_bytes=b"RIFF" + b"\x00" * 64,
        playback_receipt={"client_written": True, "bytes_played": 68},
        persisted_turn={"event_type": "voice_turn_complete"},
        ui_status="speaking",
    )
    assert delivered["asr_ok"] is True
    assert delivered["reasoning_ok"] is True
    assert delivered["tts_bytes_ok"] is True
    assert delivered["playback_ok"] is True
    assert delivered["persisted"] is True
    assert delivered["output_delivered"] is True

    silent_tts = classify_voice_pipeline_truth(
        input_ready=True,
        asr_transcript="hello",
        reasoning_result="hello back",
        tts_bytes=b"",
        playback_receipt=None,
        persisted_turn={"event_type": "voice_turn_complete"},
        ui_status="voice ready",
    )
    assert silent_tts["output_delivered"] is False


def test_isolated_voice_turn_persists_distinct_pipeline_stages(tmp_path, monkeypatch):
    from machine_spirit_4.gateway.voice_pipeline_truth import classify_voice_pipeline_truth

    audit_path = tmp_path / "ms4_audit.jsonl"
    monkeypatch.setenv("MS4_AUDIT_LOG", str(audit_path))
    asr = "what time is it"
    reasoning = "it is Wednesday"
    tts_bytes = b"RIFF" + b"\x01" * 128
    playback = {"client_written": True, "bytes_played": len(tts_bytes)}
    audit.append_event(
        "voice_turn_complete",
        {
            "transcript": asr,
            "reply_text": reasoning,
            "reply_text_preview": reasoning,
            "tts_bytes": len(tts_bytes),
            "audio_chunks": 2,
            "delivery": playback,
        },
        audit_path=audit_path,
    )
    payload = audit.read_recent_voice_turns(limit=5, audit_path=audit_path)
    turn = payload["turns"][0]
    truth = classify_voice_pipeline_truth(
        input_ready=True,
        asr_transcript=turn.get("transcript"),
        reasoning_result=turn.get("reply_text") or turn.get("reply_text_preview"),
        tts_bytes=tts_bytes if int(turn.get("tts_bytes") or 0) > 0 else b"",
        playback_receipt=turn.get("delivery"),
        persisted_turn=turn,
        ui_status="done",
    )
    assert truth["input_ready"] is True
    assert truth["asr_ok"] is True
    assert truth["reasoning_ok"] is True
    assert truth["tts_bytes_ok"] is True
    assert truth["playback_ok"] is True
    assert truth["persisted"] is True
    assert truth["output_delivered"] is True
    assert payload["source"] in {"ring", "audit-scan", "last-good"}


def test_oracle_chat_readiness_stays_ready_when_only_asr_is_unprovisioned(monkeypatch):
    _patch_healthy_foreground_model(monkeypatch)
    monkeypatch.setattr(
        oracle_admin,
        "status",
        lambda *_args, **_kwargs: {"schema": "Ms4OracleSnapshot.v1", "healthy": True},
    )
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_combined_snapshot",
        lambda *_args, **_kwargs: {
            "mcp_base_url": "http://hivemind.test:6105/mcp",
            "active_jobs": {"total_active": 0},
            "cluster_load": {},
            "service_health": {
                "menta_hli": {"name": "menta_hli", "healthy": True},
                "ASR": {
                    "name": "ASR",
                    "healthy": False,
                    "provisioning_state": "configured_not_provisioned",
                    "detail": "No endpoints configured",
                },
            },
            "errors": [],
        },
    )
    monkeypatch.setattr(oracle_admin.hivemind_state, "hivemind_auth_configured", lambda: True)
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_lifecycle_state",
        lambda *_args, **_kwargs: dict(_HEALTHY_CHAT_LIFECYCLE),
    )
    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)
    assert snap["chat_plane"]["status"] == "ready"
    assert snap["voice_input"]["ready"] is False
    assert snap["voice_input"]["issues"]
    assert snap["readiness"] == "ready"
    assert snap["ready"] is True
    assert any(check["name"] == "voice_input" and check["state"] == "fail" for check in snap["checks"])
    # Output delivery is not implied by chat readiness or input failure.
    assert snap.get("output_delivery", {}).get("delivered") is not True


def test_ui_projects_event_type_and_last_good_recent_turns():
    html = HTML.read_text(encoding="utf-8")
    assert "function renderVoiceTurnRow(turn)" in html
    row_fn = html[html.index("function renderVoiceTurnRow"): html.index("async function refreshVoiceTurns")]
    assert "event_type" in row_fn
    refresh_fn = html[html.index("async function refreshVoiceTurns"): html.index("// ----- HiveMind VMs panel")]
    assert "AbortSignal" in refresh_fn or "signal:" in refresh_fn
    assert "lastGood" in refresh_fn or "last-good" in refresh_fn or "lastGoodVoiceTurns" in html
    # Front door must not treat voice_input_ready as audible output.
    assert "input ready; output blocked" in html
    assert "Never mark input-ready as output-delivered" in html or "output_delivered" in html or "latchedVoiceOutputFailure" in html


def test_post_decode_ownership_guard_precedes_blocked_ui_write():
    html = HTML.read_text(encoding="utf-8")
    stream = html.split("async function submitWavBlobAsVoiceTurn", 1)[1]
    parsed = stream.index("payload = dataText ? JSON.parse(dataText) : {}")
    identity = stream.index("bindVoiceServerTurnId(turnState, payload.turn_id)")
    audio_event = stream.index("eventName === 'audio_chunk'")
    assert parsed < identity < audio_event
    assert "failVoiceTurnIdentityMismatch(turnState, payload.turn_id)" in stream

    audio_branch = html.split(
        "} else if (eventName === 'audio_chunk') {",
        1,
    )[1].split("} else if (eventName === 'audio_error') {", 1)[0]

    enqueue = audio_branch.index("await enqueueAudioChunk(")
    ownership = audio_branch.index(
        "if (streamController.signal.aborted || myTurnId !== currentVoiceTurnId)"
    )
    decode_failure = audio_branch.index("if (!decoded)")
    blocked = audio_branch.index(
        "setOracleStageState('blocked', 'Reply audio could not be decoded.'"
    )

    assert enqueue < ownership < decode_failure < blocked


def test_voice_turn_identity_survives_recent_turn_reload_and_ui_projection(
    tmp_path,
    monkeypatch,
):
    audit_path = tmp_path / "ms4_audit.jsonl"
    monkeypatch.setenv("MS4_AUDIT_LOG", str(audit_path))
    turn_id = "ms4-turn-0123456789abcdef"
    audit.append_event(
        "voice_turn_complete",
        {
            "turn_id": turn_id,
            "revision_id": 7,
            "transcript": "first visible turn",
            "reply_text_preview": "first visible reply",
            "dispatched_job": {"job_id": "da-identity-proof"},
        },
        audit_path=audit_path,
    )

    reloaded = audit.read_recent_voice_turns(limit=5, audit_path=audit_path)
    row = reloaded["turns"][0]
    assert row["turn_id"] == turn_id
    assert row["revision_id"] == 7
    assert row["dispatched_job"]["job_id"] == "da-identity-proof"

    rendered = _run_refresh_voice_turns_vm(reloaded)
    assert f"turn={turn_id}" in rendered["text"]
    assert "revision=7" in rendered["text"]
    assert "first visible turn" in rendered["text"]
    assert "first visible reply" in rendered["text"]

    html = HTML.read_text(encoding="utf-8")
    row_fn = html[
        html.index("function renderVoiceTurnRow"):
        html.index("async function refreshVoiceTurns")
    ]
    assert "data.turn_id" in row_fn
    assert "data.revision_id" in row_fn


def test_buried_voice_turn_beyond_byte_budget_is_not_empty_success(tmp_path, monkeypatch):
    """Verifier probe: 1 buried voice_turn_complete + 256 later 1 KiB chat rows.

    A 64 KiB reverse-scan budget must not report {turns:[], source:audit-scan,
    error:null} — that is indistinguishable from proven empty.
    """
    audit_path = tmp_path / "ms4_audit.jsonl"
    events = [_voice_complete("buried-hello")] + [_padded_chat(i) for i in range(256)]
    _write_jsonl(audit_path, events)
    assert audit_path.stat().st_size > 64 * 1024
    _forbid_full_audit_read(monkeypatch, audit_path)

    scan = audit.read_matching_events(
        event_types=("voice_turn_complete", "voice_turn_failed"),
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert scan["complete"] is False
    assert scan["reason"] == "scan_budget_exhausted"
    assert scan["events"] == []
    assert scan["reached_start"] is False

    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert payload["schema"] == "Ms4VoiceRecentTurns.v1"
    assert payload["error"] == "scan_budget_exhausted"
    assert payload["complete"] is False
    assert payload["source"] == "incomplete-audit-scan"
    assert payload["turns"] == []
    assert payload.get("stale") is not True
    assert not (payload["turns"] == [] and payload["source"] == "audit-scan" and payload["error"] is None)


def test_fully_scanned_empty_audit_is_proven_empty_success(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    _write_jsonl(audit_path, [_padded_chat(i, pad_size=16) for i in range(8)])
    _forbid_full_audit_read(monkeypatch, audit_path)

    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert payload["turns"] == []
    assert payload["source"] == "audit-scan"
    assert payload["error"] is None
    assert payload["complete"] is True
    assert payload.get("stale") is not True


def test_budget_exhaustion_returns_last_good_as_stale_incomplete(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    monkeypatch.setenv("MS4_VOICE_TURNS_LOG", str(tmp_path / "ring-unused.jsonl"))
    _write_jsonl(
        audit_path,
        [_voice_complete("last-good-cache")] + [_padded_chat(i) for i in range(256)],
    )
    seeded = audit.read_recent_voice_turns(
        limit=5, audit_path=audit_path, max_bytes=2_000_000, timeout_s=2.0
    )
    assert seeded["turns"]
    assert seeded["turns"][0]["transcript"] == "last-good-cache"

    _forbid_full_audit_read(monkeypatch, audit_path)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert payload["source"] == "last-good"
    assert payload["stale"] is True
    assert payload["complete"] is False
    assert payload["error"] in {"scan_budget_exhausted", "generation_changed"}
    assert payload["turns"]
    assert payload["turns"][0]["transcript"] == "last-good-cache"
    assert payload["source"] != "audit-scan" or payload["error"] is not None


def test_budget_exhaustion_without_last_good_is_explicit_error(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    _write_jsonl(
        audit_path,
        [_voice_complete("too-deep")] + [_padded_chat(i) for i in range(256)],
    )
    _forbid_full_audit_read(monkeypatch, audit_path)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert payload["turns"] == []
    assert payload["error"] == "scan_budget_exhausted"
    assert payload["source"] == "incomplete-audit-scan"
    assert payload["complete"] is False
    assert payload.get("stale") is not True


def test_corrupt_and_truncated_tail_still_projects_valid_event_type(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    voice = _voice_complete("after-corrupt")
    audit_path.write_bytes(
        b'{"event_type":"chat_turn","truncated":true'
        + b"\n{not json}\n"
        + (json.dumps(voice, ensure_ascii=False) + "\n").encode("utf-8")
        + b'{"event_type":"chat_turn","pad":'
    )
    _forbid_full_audit_read(monkeypatch, audit_path)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64_000,
        timeout_s=2.0,
    )
    assert payload["turns"]
    assert payload["turns"][0]["event_type"] == "voice_turn_complete"
    assert payload["turns"][0]["transcript"] == "after-corrupt"
    assert payload["error"] is None
    assert payload["complete"] is True
    assert all(
        t.get("event_type") in {"voice_turn_complete", "voice_turn_failed"}
        for t in payload["turns"]
    )


def test_log_rotation_scans_current_file_not_rotated_generation(tmp_path, monkeypatch):
    current = tmp_path / "ms4_audit.jsonl"
    rotated = tmp_path / "ms4_audit.jsonl.1"
    _write_jsonl(rotated, [_voice_complete("old-generation")] + [_padded_chat(i) for i in range(256)])
    _write_jsonl(current, [_padded_chat(i, pad_size=16) for i in range(5)])
    opened: list[str] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=current,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert payload["turns"] == []
    assert payload["error"] is None
    assert payload["complete"] is True
    assert payload["source"] == "audit-scan"
    assert not any(str(rotated) == path or path.endswith("ms4_audit.jsonl.1") for path in opened)

    _write_jsonl(current, [_voice_complete("post-rotate")])
    payload2 = audit.read_recent_voice_turns(
        limit=5,
        audit_path=current,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert payload2["turns"]
    assert payload2["turns"][0]["transcript"] == "post-rotate"
    assert payload2["error"] is None
    assert payload2["complete"] is True


def test_ring_file_success_without_unbounded_audit_read(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    ring_path = tmp_path / "ms4_voice_turns.jsonl"
    monkeypatch.setenv("MS4_VOICE_TURNS_LOG", str(ring_path))
    _write_jsonl(ring_path, [_voice_complete("from-ring")])
    _write_jsonl(
        audit_path,
        [_voice_complete("from-audit-buried")] + [_padded_chat(i) for i in range(256)],
    )
    _forbid_full_audit_read(monkeypatch, audit_path)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert payload["source"] == "ring"
    assert payload["turns"]
    assert payload["turns"][0]["transcript"] == "from-ring"
    assert payload["turns"][0]["event_type"] == "voice_turn_complete"
    assert payload.get("stale") is not True
    # Historical audit was not fully examined; ring rows must not mask that.
    assert payload["complete"] is False
    assert payload["error"] == "scan_budget_exhausted"


def test_ui_renders_incomplete_stale_distinct_from_proven_empty():
    html = HTML.read_text(encoding="utf-8")
    refresh_fn = html[
        html.index("async function refreshVoiceTurns") : html.index("// ----- HiveMind VMs panel")
    ]
    assert "scan_budget_exhausted" in refresh_fn
    assert "complete" in refresh_fn
    assert "stale" in refresh_fn
    assert "No voice turns logged yet." in refresh_fn
    assert "Voice recent-turns incomplete:" in refresh_fn
    assert "stale/incomplete" in refresh_fn
    # Incomplete/error must not be concatenated onto the proven-empty sentence.
    assert "No voice turns logged yet. Try a voice turn and refresh.${err}" not in refresh_fn
    assert "No voice turns logged yet. Try a voice turn and refresh.${data.error" not in refresh_fn
    assert "incomplete-audit-scan" in refresh_fn or "scan_budget_exhausted" in refresh_fn


def _voice_handler(path: str = "/voice/recent-turns?limit=5"):
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.path = path
    handler.wfile = io.BytesIO()
    handler.status_code = None
    headers: list[tuple[str, str]] = []
    handler.send_response = lambda status: setattr(handler, "status_code", status)
    handler.send_header = lambda name, value: headers.append((name, value))
    handler.end_headers = lambda: None
    return handler


def _install_chunk_hook(monkeypatch, hook):
    monkeypatch.setattr(audit, "_SCAN_CHUNK_HOOK", hook)


def test_concurrent_posix_rename_discards_old_row_from_replaced_generation(
    tmp_path, monkeypatch
):
    audit_path = tmp_path / "ms4_audit.jsonl"
    _write_jsonl(
        audit_path,
        [_padded_chat(i) for i in range(256)] + [_voice_complete("old-sentinel")],
    )
    replacement = tmp_path / "ms4_audit.jsonl.new"
    _write_jsonl(replacement, [_padded_chat(0, pad_size=16)])
    fired = {"n": 0}

    def hook(path, _handle):
        if fired["n"]:
            return
        fired["n"] += 1
        try:
            os.replace(replacement, path)
        except OSError:
            _write_jsonl(path, [_padded_chat(0, pad_size=16)])

    _install_chunk_hook(monkeypatch, hook)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=2_000_000,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert "old-sentinel" not in transcripts
    assert audit_path.read_text(encoding="utf-8").find("old-sentinel") < 0
    if payload["complete"] is True:
        assert payload["turns"] == []
        assert payload["error"] is None
        assert payload["source"] == "audit-scan"
    else:
        assert payload["error"] in {"generation_changed", "scan_budget_exhausted"}
        assert payload["complete"] is False


def test_equal_size_truncate_regrow_discards_old_sentinel(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    original = _exact_size_audit_bytes(
        EQUAL_SIZE_BYTES, trailing=_voice_complete("old-sentinel"), pad_char=b"x"
    )
    replacement = _exact_size_audit_bytes(
        EQUAL_SIZE_BYTES, trailing=None, pad_char=b"y"
    )
    audit_path.write_bytes(original)
    assert audit_path.stat().st_size == EQUAL_SIZE_BYTES
    fired = {"n": 0}

    def hook(path, _handle):
        if fired["n"]:
            return
        fired["n"] += 1
        _rewrite_same_inode(path, replacement)

    _install_chunk_hook(monkeypatch, hook)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=2_000_000,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert "old-sentinel" not in transcripts
    assert b"old-sentinel" not in audit_path.read_bytes()
    assert not (
        payload["complete"] is True
        and payload.get("stale") is not True
        and payload.get("error") is None
        and payload.get("source") == "audit-scan"
        and "old-sentinel" in transcripts
    )
    if payload["complete"] is True:
        assert payload["turns"] == []
        assert payload["error"] is None
        assert payload.get("stale") is not True
    else:
        assert payload["complete"] is False
        assert payload["error"] in {"generation_changed", "scan_budget_exhausted"}


def test_larger_regrow_discards_old_sentinel(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    original = _exact_size_audit_bytes(
        EQUAL_SIZE_BYTES, trailing=_voice_complete("old-sentinel"), pad_char=b"x"
    )
    replacement = _exact_size_audit_bytes(
        EQUAL_SIZE_BYTES + 65536, trailing=None, pad_char=b"z"
    )
    audit_path.write_bytes(original)
    fired = {"n": 0}

    def hook(path, _handle):
        if fired["n"]:
            return
        fired["n"] += 1
        _rewrite_same_inode(path, replacement)

    _install_chunk_hook(monkeypatch, hook)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=2_000_000,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert "old-sentinel" not in transcripts
    assert b"old-sentinel" not in audit_path.read_bytes()
    if payload["complete"] is True:
        assert payload["turns"] == []
        assert payload["error"] is None
        assert payload.get("stale") is not True
    else:
        assert payload["complete"] is False
        assert payload["error"] in {"generation_changed", "scan_budget_exhausted"}


def test_in_place_truncation_never_mixes_old_and_new_rows(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    _write_jsonl(
        audit_path,
        [_padded_chat(i) for i in range(256)] + [_voice_complete("old-sentinel")],
    )
    fired = {"n": 0}

    def hook(path, _handle):
        if fired["n"]:
            return
        fired["n"] += 1
        _write_jsonl(path, [_voice_complete("new-generation")])

    _install_chunk_hook(monkeypatch, hook)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=2_000_000,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert not ("old-sentinel" in transcripts and "new-generation" in transcripts)
    if payload["complete"] is True:
        assert transcripts == ["new-generation"]
        assert payload["error"] is None
        assert payload.get("stale") is not True
    else:
        assert "old-sentinel" not in transcripts
        assert payload["complete"] is False
        assert payload["error"] in {"generation_changed", "scan_budget_exhausted"}


def test_generation_retry_exhaustion_is_incomplete_not_old_success(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    _write_jsonl(
        audit_path,
        [_padded_chat(i) for i in range(256)] + [_voice_complete("old-sentinel")],
    )
    n = {"i": 0}

    def hook(path, _handle):
        n["i"] += 1
        # Keep the replacement larger than one 64 KiB chunk so the scan cannot
        # finish coherently, but always smaller than the opened generation.
        rows = max(90, 240 - n["i"] * 8)
        _write_jsonl(path, [_padded_chat(i) for i in range(rows)])

    _install_chunk_hook(monkeypatch, hook)
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=2_000_000,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert "old-sentinel" not in transcripts
    assert payload["complete"] is False
    assert payload["error"] == "generation_changed"
    assert payload.get("stale") is not True or payload["source"] == "last-good"
    if payload["source"] != "last-good":
        assert payload["turns"] == []


def test_ring_row_with_buried_history_keeps_incomplete_audit_diagnostic(
    tmp_path, monkeypatch
):
    audit_path = tmp_path / "ms4_audit.jsonl"
    ring_path = tmp_path / "ms4_voice_turns.jsonl"
    monkeypatch.setenv("MS4_VOICE_TURNS_LOG", str(ring_path))
    _write_jsonl(ring_path, [_voice_complete("from-ring")])
    _write_jsonl(
        audit_path,
        [_voice_complete("from-audit-buried")] + [_padded_chat(i) for i in range(256)],
    )
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    assert payload["turns"]
    assert payload["turns"][0]["transcript"] == "from-ring"
    assert payload["source"] == "ring"
    assert payload["complete"] is False
    assert payload["error"] == "scan_budget_exhausted"
    assert payload.get("stale") is not True


def test_last_good_is_isolated_across_audit_paths(tmp_path, monkeypatch):
    path_a = tmp_path / "source-a.jsonl"
    path_b = tmp_path / "source-b.jsonl"
    monkeypatch.setenv("MS4_VOICE_TURNS_LOG", str(tmp_path / "ring-unused.jsonl"))
    _write_jsonl(path_a, [_voice_complete("from-a")])
    seeded = audit.read_recent_voice_turns(limit=5, audit_path=path_a, max_bytes=64 * 1024)
    assert seeded["turns"][0]["transcript"] == "from-a"

    _write_jsonl(
        path_b,
        [_voice_complete("too-deep")] + [_padded_chat(i) for i in range(256)],
    )
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=path_b,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert "from-a" not in transcripts
    assert payload["source"] != "last-good"
    assert payload["error"] == "scan_budget_exhausted"
    assert payload["complete"] is False
    assert payload["turns"] == []


def test_equal_size_rewrite_invalidates_last_good_cache(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    monkeypatch.setenv("MS4_VOICE_TURNS_LOG", str(tmp_path / "ring-unused.jsonl"))
    original = _exact_size_audit_bytes(
        EQUAL_SIZE_BYTES, trailing=_voice_complete("old-sentinel"), pad_char=b"x"
    )
    replacement = _exact_size_audit_bytes(
        EQUAL_SIZE_BYTES, trailing=None, pad_char=b"y"
    )
    audit_path.write_bytes(original)
    seeded = audit.read_recent_voice_turns(
        limit=5, audit_path=audit_path, max_bytes=2_000_000, timeout_s=2.0
    )
    assert seeded["turns"][0]["transcript"] == "old-sentinel"
    generation_before = audit._scope_generation(audit_path)
    _rewrite_same_inode(audit_path, replacement)
    assert audit_path.stat().st_size == EQUAL_SIZE_BYTES
    generation_after = audit._scope_generation(audit_path)
    assert generation_before != generation_after
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert "old-sentinel" not in transcripts
    assert payload.get("stale") is not True or payload["source"] != "last-good"
    if payload["complete"] is True:
        assert payload["turns"] == []
        assert payload["error"] is None
    else:
        assert payload["complete"] is False
        assert payload["error"] in {"generation_changed", "scan_budget_exhausted"}
        assert payload["source"] != "last-good"


def test_larger_regrow_invalidates_last_good_cache(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    monkeypatch.setenv("MS4_VOICE_TURNS_LOG", str(tmp_path / "ring-unused.jsonl"))
    original = _exact_size_audit_bytes(
        EQUAL_SIZE_BYTES, trailing=_voice_complete("old-sentinel"), pad_char=b"x"
    )
    replacement = _exact_size_audit_bytes(
        EQUAL_SIZE_BYTES + 65536, trailing=None, pad_char=b"z"
    )
    audit_path.write_bytes(original)
    seeded = audit.read_recent_voice_turns(
        limit=5, audit_path=audit_path, max_bytes=2_000_000, timeout_s=2.0
    )
    assert seeded["turns"][0]["transcript"] == "old-sentinel"
    generation_before = audit._scope_generation(audit_path)
    _rewrite_same_inode(audit_path, replacement)
    generation_after = audit._scope_generation(audit_path)
    assert generation_before != generation_after
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert "old-sentinel" not in transcripts
    assert payload["source"] != "last-good"
    if payload["complete"] is True:
        assert payload["turns"] == []
        assert payload["error"] is None
    else:
        assert payload["complete"] is False
        assert payload["error"] in {"generation_changed", "scan_budget_exhausted"}


def test_equal_size_ring_rewrite_invalidates_last_good_cache(tmp_path, monkeypatch):
    audit_path = tmp_path / "ms4_audit.jsonl"
    ring_path = tmp_path / "ms4_voice_turns.jsonl"
    monkeypatch.setenv("MS4_VOICE_TURNS_LOG", str(ring_path))
    ring_path.write_bytes(
        _exact_size_audit_bytes(
            EQUAL_SIZE_BYTES, trailing=_voice_complete("from-ring"), pad_char=b"x"
        )
    )
    _write_jsonl(audit_path, [_padded_chat(i) for i in range(256)])
    seeded = audit.read_recent_voice_turns(
        limit=5, audit_path=audit_path, max_bytes=64 * 1024, timeout_s=2.0
    )
    assert seeded["turns"][0]["transcript"] == "from-ring"
    generation_before = audit._scope_generation(audit_path)
    _rewrite_same_inode(
        ring_path,
        _exact_size_audit_bytes(EQUAL_SIZE_BYTES, trailing=None, pad_char=b"y"),
    )
    generation_after = audit._scope_generation(audit_path)
    assert generation_before != generation_after
    payload = audit.read_recent_voice_turns(
        limit=5,
        audit_path=audit_path,
        max_bytes=64 * 1024,
        timeout_s=2.0,
    )
    transcripts = [t.get("transcript") for t in payload["turns"]]
    assert "from-ring" not in transcripts
    assert payload["source"] != "last-good"


def test_voice_recent_turns_handler_exception_payload_is_fail_closed(monkeypatch):
    handler = _voice_handler("/voice/recent-turns?limit=5")
    captured: dict = {}

    def fake_response(_handler, status, payload):
        captured["status"] = status
        captured["payload"] = payload

    monkeypatch.setattr(srv_module, "_json_response", fake_response)
    monkeypatch.setattr(
        srv_module,
        "read_recent_voice_turns",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("scan exploded")),
    )
    handler._voice_recent_turns_get()
    payload = captured["payload"]
    assert captured["status"] == 200
    assert payload["schema"] == "Ms4VoiceRecentTurns.v1"
    assert payload["turns"] == []
    assert payload["source"] == "error"
    assert payload["error"] == "scan exploded"
    assert payload["complete"] is False
    assert payload["stale"] is False
    assert "limit" in payload


def test_ui_proven_empty_requires_complete_true_and_error_cache_is_stale():
    html = HTML.read_text(encoding="utf-8")
    refresh_fn = html[
        html.index("async function refreshVoiceTurns") : html.index("// ----- HiveMind VMs panel")
    ]
    assert "data.complete === true" in refresh_fn
    assert "!incomplete" in refresh_fn
    assert "source !== 'error'" in refresh_fn or 'source !== "error"' in refresh_fn
    assert "generation_changed" in refresh_fn
    assert "source === 'error'" in refresh_fn or 'source === "error"' in refresh_fn
    proven_idx = refresh_fn.index("No voice turns logged yet.")
    complete_idx = refresh_fn.index("data.complete === true")
    incomplete_gate_idx = refresh_fn.index("!incomplete")
    # Proven-empty rendering must be gated on complete===true and !incomplete.
    assert complete_idx < proven_idx
    assert incomplete_gate_idx < proven_idx
    assert "stale/incomplete" in refresh_fn
    assert "Voice recent-turns incomplete:" in refresh_fn


_PROVEN_EMPTY_SENTENCE = "No voice turns logged yet. Try a voice turn and refresh."
_INCOMPLETE_SENTENCE = "Voice recent-turns incomplete:"
_STALE_WARNING = "stale/incomplete"

_REFRESH_VM_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const htmlPath = process.argv[2];
const spec = JSON.parse(fs.readFileSync(0, 'utf8'));
const html = fs.readFileSync(htmlPath, 'utf8');
const start = html.indexOf('function fmtMs');
const end = html.indexOf('// ----- HiveMind VMs panel');
if (start < 0 || end < 0 || end <= start) {
  throw new Error('refreshVoiceTurns extract failed');
}
let source = html.slice(start, end);
source = source.replace(
  'let lastGoodVoiceTurns = [];',
  'var lastGoodVoiceTurns = ' + JSON.stringify(spec.lastGood || []) + ';'
);

function el(tag) {
  const children = [];
  let text = '';
  let htmlValue = '';
  return {
    tagName: tag,
    className: '',
    style: {},
    children,
    appendChild(child) { children.push(child); return child; },
    get textContent() {
      if (children.length) return children.map((c) => c.textContent).join('');
      return text;
    },
    set textContent(v) {
      text = String(v ?? '');
      children.length = 0;
      htmlValue = '';
    },
    get innerHTML() { return htmlValue; },
    set innerHTML(v) {
      htmlValue = String(v ?? '');
      if (htmlValue === '') {
        children.length = 0;
        text = '';
      }
    },
  };
}

function visible(node) {
  const parts = [];
  if (node.textContent) parts.push(node.textContent);
  if (node.innerHTML) parts.push(String(node.innerHTML).replace(/<[^>]+>/g, ' '));
  for (const child of node.children || []) parts.push(visible(child));
  return parts.join('\n');
}

const settingsVoiceTurns = el('div');
const settingsVoiceTurnsStatus = el('span');
const context = vm.createContext({
  fetch: async () => ({ ok: true, json: async () => spec.payload }),
  AbortController,
  setTimeout,
  clearTimeout,
  document: { createElement: (tag) => el(tag) },
  settingsVoiceTurns,
  settingsVoiceTurnsStatus,
  settingsVoiceTurnsRefresh: { addEventListener() {} },
  console,
});

vm.runInContext(source, context);
vm.runInContext('refreshVoiceTurns()', context).then(() => {
  process.stdout.write(JSON.stringify({
    text: visible(settingsVoiceTurns),
    status: settingsVoiceTurnsStatus.textContent,
  }));
}).catch((err) => {
  console.error(err);
  process.exit(1);
});
"""


def _run_refresh_voice_turns_vm(payload: dict, *, last_good: list | None = None) -> dict:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is required for refreshVoiceTurns vm regression")
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "refresh_voice_turns_vm.cjs"
        harness.write_text(_REFRESH_VM_HARNESS, encoding="utf-8")
        completed = subprocess.run(
            [node, str(harness), str(HTML)],
            input=json.dumps({"payload": payload, "lastGood": last_good or []}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    if completed.returncode != 0:
        raise AssertionError(
            "refreshVoiceTurns vm harness failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def test_ui_error_complete_true_empty_is_incomplete_not_proven_empty():
    """Cross-version UI: contradictory error+complete:true must not look proven-empty."""
    rendered = _run_refresh_voice_turns_vm(
        {
            "source": "error",
            "complete": True,
            "stale": False,
            "error": None,
            "turns": [],
        }
    )
    assert _PROVEN_EMPTY_SENTENCE not in rendered["text"]
    assert _INCOMPLETE_SENTENCE in rendered["text"]
    assert "incomplete" in rendered["status"]


def test_ui_vm_proven_empty_missing_complete_and_cached_error_states():
    proven = _run_refresh_voice_turns_vm(
        {
            "source": "audit-scan",
            "complete": True,
            "stale": False,
            "error": None,
            "turns": [],
        }
    )
    assert _PROVEN_EMPTY_SENTENCE in proven["text"]
    assert _INCOMPLETE_SENTENCE not in proven["text"]
    assert "0 turn" in proven["status"]
    assert "incomplete" not in proven["status"]
    assert "stale" not in proven["status"]

    missing_complete = _run_refresh_voice_turns_vm(
        {"source": "audit-scan", "turns": []}
    )
    assert _PROVEN_EMPTY_SENTENCE not in missing_complete["text"]
    assert _INCOMPLETE_SENTENCE in missing_complete["text"]
    assert "incomplete" in missing_complete["status"]

    cached_error = _run_refresh_voice_turns_vm(
        {
            "source": "error",
            "complete": False,
            "stale": False,
            "error": "scan exploded",
            "turns": [],
        },
        last_good=[
            {
                "event_type": "voice_turn_complete",
                "transcript": "cached-hello",
                "reply_text_preview": "cached-reply",
            }
        ],
    )
    assert _PROVEN_EMPTY_SENTENCE not in cached_error["text"]
    assert _STALE_WARNING in cached_error["text"]
    assert "cached-hello" in cached_error["text"]
    assert "stale" in cached_error["status"] or "incomplete" in cached_error["status"]
