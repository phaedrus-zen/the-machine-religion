"""Chat submit must not look delivered before prewarm; 503 is not-sent."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "machine_spirit_4" / "web" / "index.html"
HARNESS = Path(__file__).with_name("chat_prewarm_pending_bubble.mjs")


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def _submit() -> str:
    html = _html()
    start = html.index("async function submitChatTurn(")
    end = html.index("const hermesBanner = document.getElementById('hermesUpdateBanner')")
    return html[start:end]


def _streaming() -> str:
    html = _html()
    start = html.index("async function sendStreamingChat(")
    end = html.index("document.getElementById('chatForm').addEventListener('submit'")
    return html[start:end]


def test_chat_submit_renders_user_pending_before_prewarm():
    submit = _submit()
    streaming = _streaming()
    assert "addUserMessagePending(" in submit
    assert "addMessage('user', text)" not in submit
    assert submit.index("addUserMessagePending(") < submit.index("sendStreamingChat(")
    assert streaming.index("await awaitSelectedFaceModelPrewarm()") < streaming.index(
        "createStreamingAssistantMessage("
    )
    assert streaming.index("await awaitSelectedFaceModelPrewarm()") < streaming.index(
        "fetch('/chat/stream'"
    )


def test_prewarm_failure_marks_failed_not_sent_without_assistant_copy():
    html = _html()
    submit = _submit()
    assert "failed / not sent" in html
    assert "markUserMessageFailedNotSent(" in submit
    assert "dataset.sendState" in html
    fail_branch = submit.split("catch (error)")[1]
    assert "markUserMessageFailedNotSent(" in fail_branch
    assert "addMessage('assistant'" not in fail_branch
    assert "speakTextChatReply" not in fail_branch
    assert "/voice/recent-turns" not in submit
    assert "/voice/recent-turns" not in _streaming()


def _run_harness(mode: str) -> dict:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is required for chat prewarm pending-bubble DOM tests")
    completed = subprocess.run(
        [node, str(HARNESS), str(HTML), mode],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "chat prewarm harness failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def test_prewarm_503_pending_to_failed_not_sent_zero_chat_tts_recent():
    result = _run_harness("prewarm-503")
    assert result["chatCalls"] == 0
    assert result["ttsCalls"] == 0
    assert result["recentCalls"] == 0
    assert result["assistantCount"] == 0
    assert result["userState"] == "failed-not-sent"
    assert "failed / not sent" in result["userText"]
    assert result["userLooksDelivered"] is False


def test_successful_terminal_flow_commits_user_and_assistant():
    result = _run_harness("success")
    assert result["chatCalls"] == 1
    assert result["prewarmCalls"] == 1
    assert result["recentCalls"] == 0
    assert result["userState"] == "sent"
    assert result["assistantCount"] == 1
    assert result["assistantText"] == "Exact warm route."
    assert result["terminal"] == "done"
    assert "failed / not sent" not in result["userText"]
