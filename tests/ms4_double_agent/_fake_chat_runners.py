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
