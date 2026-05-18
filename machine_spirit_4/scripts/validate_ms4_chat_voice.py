from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests


HIVEMIND_URL = os.environ.get("MS4_HIVEMIND_BASE_URL", "http://127.0.0.1:6089").rstrip("/")
MS3_URL = os.environ.get("MS4_MS3_SIDECAR_URL", "http://127.0.0.1:9080").rstrip("/")
CHAT_MODEL = os.environ.get("MS4_CHAT_TEST_MODEL", "qwen2.5:0.5b")


@dataclass
class CheckResult:
    name: str
    ok: bool
    status: int | None
    elapsed_ms: int
    detail: str


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _result(name: str, ok: bool, status: int | None, start: float, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=ok, status=status, elapsed_ms=_elapsed_ms(start), detail=detail[:600])


def _post_json(name: str, url: str, payload: dict[str, Any], timeout: int, contains: str | None = None) -> CheckResult:
    start = time.monotonic()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        text = response.text
        ok = response.status_code == 200 and (contains is None or contains in text)
        return _result(name, ok, response.status_code, start, text)
    except Exception as exc:
        return _result(name, False, None, start, repr(exc))


def _get_json(name: str, url: str, timeout: int, predicate) -> CheckResult:
    start = time.monotonic()
    try:
        response = requests.get(url, timeout=timeout)
        data = response.json()
        ok = response.status_code == 200 and bool(predicate(data))
        return _result(name, ok, response.status_code, start, json.dumps(data, ensure_ascii=True)[:600])
    except Exception as exc:
        return _result(name, False, None, start, repr(exc))


def _make_silence_wav() -> Path:
    path = Path(tempfile.gettempdir()) / "ms4_true_silence.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes([0, 0]) * 16000)
    return path


def check_hivemind_chat() -> CheckResult:
    return _post_json(
        "hivemind_chat_completions",
        f"{HIVEMIND_URL}/v1/chat/completions",
        {
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": "Reply with exactly MS4_CHAT_OK."},
                {"role": "user", "content": "validation"},
            ],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        },
        timeout=120,
        contains="MS4_CHAT_OK",
    )


def check_hivemind_tts() -> CheckResult:
    start = time.monotonic()
    try:
        response = requests.post(
            f"{HIVEMIND_URL}/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "MS4 voice validation.",
                "voice": "alloy",
                "response_format": "wav",
            },
            timeout=90,
        )
        content_type = response.headers.get("content-type", "")
        ok = response.status_code == 200 and content_type.startswith("audio/") and len(response.content) > 1000
        return _result("hivemind_tts_speech", ok, response.status_code, start, f"{content_type}; bytes={len(response.content)}")
    except Exception as exc:
        return _result("hivemind_tts_speech", False, None, start, repr(exc))


def check_hivemind_asr_silence_gate() -> CheckResult:
    start = time.monotonic()
    path = _make_silence_wav()
    try:
        with path.open("rb") as handle:
            response = requests.post(
                f"{HIVEMIND_URL}/v1/audio/transcriptions",
                files={"file": ("ms4_true_silence.wav", handle, "audio/wav")},
                data={"model": "whisper-1"},
                timeout=20,
            )
        text = response.text
        ok = response.status_code == 200 and '"no_speech_detected":true' in text.replace(" ", "")
        return _result("hivemind_asr_silence_gate", ok, response.status_code, start, text)
    except Exception as exc:
        return _result("hivemind_asr_silence_gate", False, None, start, repr(exc))


def check_hivemind_mcp_jsonrpc() -> CheckResult:
    start = time.monotonic()
    url = f"{HIVEMIND_URL}/v1/mcp"
    try:
        initialize = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "ms4-validator", "version": "0.1.0"},
                },
            },
            timeout=20,
        )
        init_json = initialize.json()
        tools = requests.post(
            url,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            timeout=30,
        )
        tools_json = tools.json()
        tool_list = tools_json.get("result", {}).get("tools", [])
        ok = (
            initialize.status_code == 200
            and tools.status_code == 200
            and "result" in init_json
            and isinstance(tool_list, list)
            and len(tool_list) > 0
        )
        detail = {
            "initialize": init_json,
            "tools_count": len(tool_list) if isinstance(tool_list, list) else 0,
            "first_tool": tool_list[0].get("name") if tool_list and isinstance(tool_list[0], dict) else None,
        }
        return _result("hivemind_mcp_jsonrpc", ok, tools.status_code, start, json.dumps(detail, ensure_ascii=True))
    except Exception as exc:
        return _result("hivemind_mcp_jsonrpc", False, None, start, repr(exc))


def check_ms3_interact() -> CheckResult:
    return _post_json(
        "ms3_interact_chat",
        f"{MS3_URL}/interact",
        {"text": "Reply with exactly MS3_INTERACT_OK.", "personality_id": "sister"},
        timeout=120,
        contains="MS3_INTERACT_OK",
    )


def check_ms3_voice_status() -> CheckResult:
    return _get_json(
        "ms3_voice_status",
        f"{MS3_URL}/voice/status",
        timeout=15,
        predicate=lambda data: data.get("schema") == "VoiceReadiness.v1" and "voice_input_ready" in data,
    )


def check_ms3_voice_interact_fast_fail() -> CheckResult:
    start = time.monotonic()
    path = _make_silence_wav()
    try:
        response = requests.post(
            f"{MS3_URL}/voice-interact?personality_id=sister",
            data=path.read_bytes(),
            headers={"Content-Type": "audio/wav"},
            timeout=25,
        )
        text = response.text
        ok = (
            response.status_code in {200, 500, 503}
            and _elapsed_ms(start) < 25_000
            and ("ASR" in text or "transcript" in text or "no_speech" in text)
        )
        return _result("ms3_voice_interact_contract", ok, response.status_code, start, text)
    except Exception as exc:
        return _result("ms3_voice_interact_contract", False, None, start, repr(exc))


def main() -> int:
    checks = [
        _get_json("hivemind_health", f"{HIVEMIND_URL}/v1/health", 10, lambda data: data.get("status") == "ok"),
        check_hivemind_chat(),
        check_hivemind_mcp_jsonrpc(),
        check_hivemind_tts(),
        check_hivemind_asr_silence_gate(),
        _get_json("ms3_health", f"{MS3_URL}/health", 10, lambda data: data.get("service") == "Machine Spirit 3"),
        check_ms3_interact(),
        check_ms3_voice_status(),
        check_ms3_voice_interact_fast_fail(),
    ]
    print(json.dumps([asdict(check) for check in checks], indent=2))
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
