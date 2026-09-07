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

import threading

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


def test_per_turn_override_does_not_replace_session_pin(monkeypatch, tmp_path):
    """A one-turn override must not silently repin later model-less turns."""
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )
    picker_calls = {"n": 0}

    def pick_auto(**_kwargs):
        picker_calls["n"] += 1
        return type(
            "Choice",
            (),
            {
                "model_id": "auto-A",
                "to_dict": lambda self: {"model_id": "auto-A", "source": "loaded"},
            },
        )()

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_foreground_model",
        pick_auto,
    )

    first = runner.chat("first", session_id="s-one-turn")
    override = runner.chat("override", session_id="s-one-turn", model="explicit-B")
    resumed = runner.chat("resume", session_id="s-one-turn")

    assert [first["model"], override["model"], resumed["model"]] == [
        "auto-A",
        "explicit-B",
        "auto-A",
    ]
    assert picker_calls["n"] == 1
    assert resumed["face_lobe_model"]["source"] == "session_pinned"


def test_explicit_first_turn_does_not_prevent_later_automatic_pin(monkeypatch, tmp_path):
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )
    picker_calls = {"n": 0}

    def pick_auto(**_kwargs):
        picker_calls["n"] += 1
        return type(
            "Choice",
            (),
            {
                "model_id": "auto-after-explicit",
                "to_dict": lambda self: {
                    "model_id": "auto-after-explicit",
                    "source": "loaded",
                },
            },
        )()

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_foreground_model",
        pick_auto,
    )

    explicit = runner.chat("first", session_id="s-explicit-first", model="explicit-X")
    automatic = runner.chat("second", session_id="s-explicit-first")
    resumed = runner.chat("third", session_id="s-explicit-first")

    assert explicit["model"] == "explicit-X"
    assert automatic["model"] == "auto-after-explicit"
    assert resumed["model"] == "auto-after-explicit"
    assert picker_calls["n"] == 1
    assert resumed["face_lobe_model"]["source"] == "session_pinned"


def test_concurrent_first_turns_converge_on_one_session_pin(monkeypatch, tmp_path):
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )
    barrier = threading.Barrier(2)
    counter_lock = threading.Lock()
    picker_calls = {"n": 0}

    def pick_concurrently(**_kwargs):
        with counter_lock:
            picker_calls["n"] += 1
            model_id = f"auto-{picker_calls['n']}"
        barrier.wait(timeout=5)
        return type(
            "Choice",
            (),
            {
                "model_id": model_id,
                "to_dict": lambda self: {"model_id": model_id, "source": "loaded"},
            },
        )()

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_foreground_model",
        pick_concurrently,
    )
    results = []
    errors = []

    def run_turn(message):
        try:
            results.append(runner.chat(message, session_id="s-concurrent-first"))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=run_turn, args=("first-a",)),
        threading.Thread(target=run_turn, args=("first-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len(results) == 2
    winning_models = {result["model"] for result in results}
    assert len(winning_models) == 1
    winner = winning_models.pop()
    assert winner in {"auto-1", "auto-2"}
    assert runner._face_model_pins["s-concurrent-first"] == winner
    assert picker_calls["n"] == 2


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
