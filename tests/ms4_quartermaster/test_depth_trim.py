"""Phase E — Depth Lobe tool-context trimming.

Covers the conservative intent->Hermes-toolset mapping, the
ResourceRequest.enabled_toolsets schema field, and the end-to-end
plumbing: Ms4HermesRunner.new_background_agent / _construct_agent pass
enabled_toolsets to the Hermes AIAgent, and the worker forwards the
envelope's toolsets only when set.
"""

from __future__ import annotations

import pytest

from machine_spirit_4.gateway.quartermaster import hermes_toolsets_for_query
from machine_spirit_4.double_agent.schemas import ResourceRequest


# ---------------------------------------------------------------------------
# Conservative intent -> Hermes toolset mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("generate an image of a cat", ["image_tools"]),
    ("search the web for python news", ["web_tools"]),
    ("browse to the stripe homepage and click pricing", ["browser_tools", "web_tools"]),
])
def test_narrow_intents_map_to_toolsets(query, expected):
    assert hermes_toolsets_for_query(query) == expected


@pytest.mark.parametrize("query", [
    "debug this module and run the tests",        # broadening: debug/run
    "read the config file and edit it",            # broadening: read/file/edit
    "research competitors and implement a clone",  # broadening: and / implement-ish
    "install the package and build it",            # broadening: install/build
    "do everything needed to ship this",           # broadening: everything
    "",                                            # empty
    "what is the meaning of recursion",            # no clean mapping
])
def test_broadening_or_unclear_returns_none(query):
    # None = full catalog (escape hatch) — never starve open-ended work.
    assert hermes_toolsets_for_query(query) is None


def test_broadening_signal_overrides_narrow_match():
    # Has an image phrase BUT also a broadening 'script' signal -> None.
    assert hermes_toolsets_for_query("generate an image then run a script on it") is None


# ---------------------------------------------------------------------------
# ResourceRequest schema field
# ---------------------------------------------------------------------------


def test_resource_request_enabled_toolsets_roundtrip():
    rr = ResourceRequest(model_override="m", enabled_toolsets=["web_tools", "browser_tools"])
    d = rr.to_dict()
    assert d["enabled_toolsets"] == ["web_tools", "browser_tools"]
    rr2 = ResourceRequest.from_dict(d)
    assert rr2.enabled_toolsets == ["web_tools", "browser_tools"]


def test_resource_request_default_none():
    rr = ResourceRequest()
    assert rr.enabled_toolsets is None
    assert rr.to_dict()["enabled_toolsets"] is None
    assert ResourceRequest.from_dict({}).enabled_toolsets is None


def test_resource_request_sanitizes_bad_toolsets():
    rr = ResourceRequest.from_dict({"enabled_toolsets": ["ok", 123, "", "x" * 200]})
    # Non-strings, empty, and over-long are dropped.
    assert rr.enabled_toolsets == ["ok"]
    rr2 = ResourceRequest.from_dict({"enabled_toolsets": [123, ""]})
    assert rr2.enabled_toolsets is None  # nothing valid -> None


# ---------------------------------------------------------------------------
# Agent construction passes enabled_toolsets through
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")


def _runner(tmp_path):
    from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner

    class _FakeFace:
        def __init__(self, **_):
            self._sessions = {}

        def chat(self, *a, **k):
            return {"text": "x", "session_id": "s", "model": "m", "completed": True}

        def sessions(self):
            return []

    return Ms4HermesRunner(
        hermes_dir=str(tmp_path), agent_cls=_FakeAgent, face_lobe_chat=_FakeFace(),
    )


def test_new_background_agent_passes_enabled_toolsets(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    agent = runner.new_background_agent(
        session_id="da-1", model="m",
        tool_start_callback=lambda *_: None, tool_complete_callback=lambda *_: None,
        enabled_toolsets=["web_tools"],
    )
    assert agent.kwargs.get("enabled_toolsets") == ["web_tools"]


def test_new_background_agent_omits_toolsets_when_none(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    agent = runner.new_background_agent(
        session_id="da-2", model="m",
        tool_start_callback=lambda *_: None, tool_complete_callback=lambda *_: None,
        enabled_toolsets=None,
    )
    # Full catalog escape hatch: kwarg not passed -> Hermes default.
    assert "enabled_toolsets" not in agent.kwargs


# ---------------------------------------------------------------------------
# Worker forwards envelope toolsets only when set
# ---------------------------------------------------------------------------


def test_worker_forwards_enabled_toolsets_when_set(tmp_path):
    """build_real_chat_runner._call accepts + forwards enabled_toolsets
    to new_background_agent."""
    from machine_spirit_4.double_agent.worker import build_real_chat_runner

    captured = {}

    class _Runner:
        default_model = "m"

        def new_background_agent(self, **kwargs):
            captured.update(kwargs)

            class _Agent:
                def run_conversation(self, message, **_k):
                    return {"final_response": "ok", "completed": True}

            return _Agent()

    call = build_real_chat_runner(_Runner())
    call(
        message="go", session_id="da-x", model="m",
        stream_callback=None,
        tool_start_callback=lambda *_: None,
        tool_complete_callback=lambda *_: None,
        enabled_toolsets=["image_tools"],
    )
    assert captured.get("enabled_toolsets") == ["image_tools"]
