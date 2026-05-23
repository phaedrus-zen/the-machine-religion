"""Face Lobe model pinning + meta-grounding tests.

These pin three behaviors we caught failing live:

1. Once a session is created, subsequent turns reuse the pinned model
   even if the auto-picker would now choose a different one. The old
   behavior wiped session history on model change, which made turn 2
   act like a fresh boot.
2. Questions about cluster/GPU state ("do you see any GPUs?") trigger
   the HiveMind inventory grounding so the Face Lobe answers from
   live MCP data, not invention.
3. Questions about MS4 capabilities ("what tools do you have?")
   trigger the MS4 tools-list grounding so the model cannot invent
   tool names like ``browser_snapshot`` or ``skills_list``.
"""

from __future__ import annotations

import pytest

from machine_spirit_4.gateway.context import (
    build_grounded_user_message,
    is_inventory_question,
    is_tools_question,
)
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner
from tests.ms4_gateway.test_hermes_runner import FakeFaceLobeChat, FakeHermesAgent


def test_session_model_is_pinned_across_turns(monkeypatch, tmp_path):
    """The picker must not be re-consulted after a session exists; turn 2
    onward keeps using the first turn's pinned model."""
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=fake)

    picker_calls = {"n": 0}

    def picker_first(**_kwargs):
        picker_calls["n"] += 1
        if picker_calls["n"] == 1:
            return type("C", (), {
                "model_id": "qwen2.5:0.5b",
                "to_dict": lambda self: {"model_id": "qwen2.5:0.5b", "source": "loaded"},
            })()
        return type("C", (), {
            "model_id": "llama3.1:8b",
            "to_dict": lambda self: {"model_id": "llama3.1:8b", "source": "loaded"},
        })()

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_foreground_model",
        picker_first,
    )

    r1 = runner.chat("hello", session_id="s-pin")
    r2 = runner.chat("again", session_id="s-pin")
    r3 = runner.chat("third", session_id="s-pin")

    assert r1["model"] == "qwen2.5:0.5b"
    assert r2["model"] == "qwen2.5:0.5b", "model must be pinned per session"
    assert r3["model"] == "qwen2.5:0.5b"
    assert picker_calls["n"] == 1, "picker should only run on the first turn of a session"
    assert r2["face_lobe_model"]["source"] == "session_pinned"
    assert r3["face_lobe_model"]["source"] == "session_pinned"
    # FaceLobeChat session keeps history across turns (3 user + 3 assistant = 6 messages).
    state = fake._sessions["s-pin"]
    assert len(state.messages) == 6


def test_per_turn_model_override_still_wins(monkeypatch, tmp_path):
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=fake)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_foreground_model",
        lambda **_: type("C", (), {
            "model_id": "qwen2.5:0.5b",
            "to_dict": lambda self: {"model_id": "qwen2.5:0.5b", "source": "loaded"},
        })(),
    )
    r1 = runner.chat("first turn", session_id="s-explicit")
    r2 = runner.chat("second turn", session_id="s-explicit", model="phi4-mini")
    assert r1["model"] == "qwen2.5:0.5b"
    assert r2["model"] == "phi4-mini", "explicit per-turn override must beat session pinning"
    assert r2["face_lobe_model"]["source"] == "per_turn_override"


# ----- meta-question detectors ---------------------------------------------


@pytest.mark.parametrize("msg", [
    "do you see any GPUs?",
    "any GPUs around?",
    "tell me about the hivemind cluster",
    "show me the nodes",
    "what nodes are in the cluster",
    "describe the hardware available",
    "what hosts are reachable",
])
def test_is_inventory_question_catches_natural_phrasing(msg):
    assert is_inventory_question(msg), msg


@pytest.mark.parametrize("msg", [
    "what tools do you have?",
    "list your tools",
    "any tools available?",
    "what mcp servers can you call?",
    "what capabilities do you have",
    "what can you do",
])
def test_is_tools_question_catches_natural_phrasing(msg):
    assert is_tools_question(msg), msg


@pytest.mark.parametrize("msg", [
    "hello",
    "thanks",
    "what time is it?",
    "implement an audit of the deadlock",
])
def test_meta_detectors_do_not_overreach(msg):
    assert not is_tools_question(msg)


def test_tools_grounding_block_includes_real_tool_names(monkeypatch):
    """The grounded message must enumerate real MS4 tool ids — never
    invented names like browser_snapshot / skills_list."""
    # Mock HiveMind MCP catalog call so we don't depend on it being up.
    import urllib.request

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b'{"jsonrpc":"2.0","result":{"tools":[{"name":"x"},{"name":"y"}]}}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())

    grounded, source = build_grounded_user_message("what tools do you have?", "http://hive")
    assert source == "ms4-tools-grounding"
    assert "MS4 capability surface" in grounded
    # Real MS4 tools we know exist in the manifest:
    assert "ms4.hivemind.inventory@v1" in grounded
    assert "ms4.hermes.tools.list@v1" in grounded
    assert "ms4.double_agent.submit@v1" in grounded
    # Invented names that the operator saw in the failing transcript:
    for invented in ("browser_snapshot", "skills_list", "hivemind list-tools"):
        # Grounding mentions them only inside the "do not invent" block;
        # we still want them present once as guard rails. Just assert the
        # grounding warns against inventing them.
        assert invented in grounded


def test_tools_question_routes_through_tools_grounding(monkeypatch):
    import urllib.request

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b'{"jsonrpc":"2.0","result":{"tools":[]}}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())
    grounded, source = build_grounded_user_message("any mcp tools?", "http://hive")
    assert source == "ms4-tools-grounding"
    assert "MS4 capability surface" in grounded
