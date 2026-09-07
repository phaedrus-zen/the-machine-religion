"""Fake chat runner factories importable from the worker subprocess.

The Double Agent worker subprocess (``_worker_entry.py``) honors
``MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER=module:factory`` to bypass the real
Hermes path during tests. This module exposes a few deterministic
factories used by ``test_runner_subprocess.py``.
"""

from __future__ import annotations

import time


def quick_factory():
    def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback):
        for word in ("hello", "from", "the", "subprocess"):
            stream_callback(word)
        return {"text": "Subprocess worker completed cleanly."}

    return _call


def sleeping_factory():
    """Sleeps forever; useful for cancellation tests. The subprocess
    needs to die because of a SIGTERM / CTRL_BREAK_EVENT, not because
    the chat returned."""

    def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback):
        while True:
            time.sleep(0.2)
            stream_callback("tick")

    return _call


def tool_calling_factory():
    def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback):
        tool_start_callback("tc1", "read_file", {"path": "worker.rs"})
        tool_complete_callback("tc1", "read_file", {"path": "worker.rs"}, "ok")
        stream_callback("done")
        return {"text": "Did the read."}

    return _call


def hivemind_tool_then_sleep_factory():
    def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback):
        tool_start_callback("tc-hm", "hivemind_cluster_summary", {})
        tool_complete_callback(
            "tc-hm",
            "hivemind_cluster_summary",
            {},
            '{"healthy":true,"cluster_statistics":{"total_nodes":4,"active_nodes":4}}',
        )
        while True:
            time.sleep(0.2)
            stream_callback("waiting")

    return _call


def hivemind_tool_then_hard_hang_factory():
    def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback):
        tool_start_callback("tc-hm", "hivemind_cluster_summary", {})
        tool_complete_callback(
            "tc-hm",
            "hivemind_cluster_summary",
            {},
            '{"healthy":true,"cluster_statistics":{"total_nodes":4,"active_nodes":4}}',
        )
        while True:
            time.sleep(0.2)

    return _call


def read_file_tool_then_hard_hang_factory():
    def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback):
        tool_start_callback("tc-file", "read_file", {"path": "secret.txt"})
        tool_complete_callback("tc-file", "read_file", {"path": "secret.txt"}, "file contents")
        while True:
            time.sleep(0.2)

    return _call


def hivemind_tool_then_second_tool_hang_factory():
    def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback):
        tool_start_callback("tc-hm", "hivemind_cluster_summary", {})
        tool_complete_callback(
            "tc-hm",
            "hivemind_cluster_summary",
            {},
            '{"healthy":true,"cluster_statistics":{"total_nodes":4,"active_nodes":4}}',
        )
        tool_start_callback("tc-file", "read_file", {"path": "followup.txt"})
        while True:
            time.sleep(0.2)

    return _call
