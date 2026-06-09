"""Confirmation-triggered dispatch: when the Face Lobe OFFERS a deep
action and the next turn is an affirmation ('okay do that'), MS4 actually
dispatches it instead of the model falsely claiming a dispatch.

Regression for the live failure: 'okay can you do that for me please'
routed direct (no deep signals), nothing dispatched, yet the model said
'dispatching a Depth Lobe job now'.
"""

from __future__ import annotations

import types

import pytest

import machine_spirit_4.gateway.hermes_runner as hr
from machine_spirit_4.gateway.hermes_runner import (
    Ms4HermesRunner,
    _is_confirmation,
    _looks_like_deep_offer,
)


class _ConfigurableFace:
    def __init__(self, **_):
        self.reply_text = "ok"
        self.calls: list[dict] = []
        self._sessions: dict = {}  # chat() reads this for session-pinned model resolution

    def chat(self, message, *, session_id, model, stream_callback=None, extra_system=None):
        self.calls.append({"message": message, "extra_system": extra_system})
        return {
            "text": self.reply_text,
            "session_id": session_id or "s",
            "model": model,
            "completed": True,
            "api_calls": 1,
            "metrics": {"schema": "Ms4TurnMetrics.v1"},
        }

    def sessions(self):
        return []


class _FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")


def _patch_network_seams(monkeypatch):
    """Make chat() fully hermetic — it otherwise reaches hive:6089 for
    the foreground model pick, grounding, and cluster time, which can
    hang against an unreachable cluster."""
    monkeypatch.setattr(hr, "router_route", lambda m: types.SimpleNamespace(
        kind="direct", confidence=0.0, source="default",
        cleaned_message=m, goal="", to_dict=lambda: {"kind": "direct"},
    ))
    monkeypatch.setattr(hr, "choose_depth_model", lambda **k: types.SimpleNamespace(
        model_id="depth-model", to_dict=lambda: {"model_id": "depth-model"},
    ))
    monkeypatch.setattr(hr, "choose_foreground_model", lambda **k: types.SimpleNamespace(
        model_id="face-model", to_dict=lambda: {"schema": "Ms4ForegroundModel.v1", "model_id": "face-model", "source": "test"},
    ))
    monkeypatch.setattr(hr, "build_grounded_user_message", lambda msg, url: (msg, None))


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    monkeypatch.setenv("MS4_DA_CONFIRM_DISPATCH", "1")
    _patch_network_seams(monkeypatch)
    face = _ConfigurableFace()
    r = Ms4HermesRunner(
        hermes_dir=str(tmp_path), hivemind_url="http://hive:6089",
        agent_cls=_FakeAgent, face_lobe_chat=face,
    )
    monkeypatch.setattr(r, "_try_cluster_time", lambda: None)
    r._face = face
    return r


# ---------------------------------------------------------------------------
# _is_confirmation / _looks_like_deep_offer units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "okay can you do that for me please",
    "yeah do it",
    "go ahead",
    "ok do that",
    "yes please",
    "do it",
    "sure, go for it",
    "I need you to do that for me. I'm only able to talk to you verbally.",  # the live voice case
    "please do that",
    "run it",
])
def test_is_confirmation_positive(msg):
    assert _is_confirmation(msg) is True


@pytest.mark.parametrize("msg", [
    "okay what is the weather",          # affirmation start but new topic, no backref
    "what tools do you have",
    "tell me about hivemind",
    "",
    "can you also launch cyberpunk and run a benchmark suite please",  # new request
    "no, don't do that",                 # explicit decline
    "never mind, do it later",           # decline
])
def test_is_confirmation_negative(msg):
    assert _is_confirmation(msg) is False


@pytest.mark.parametrize("text,expected", [
    ("Sure—just prefix /deep and I'll dispatch a Depth Lobe job.", True),
    ("I'll dispatch a background job to fetch that.", True),
    ("Want me to spin one up?", True),
    ("HiveMind is a distributed platform.", False),
    ("", False),
])
def test_looks_like_deep_offer(text, expected):
    assert _looks_like_deep_offer(text) is expected


# ---------------------------------------------------------------------------
# chat() integration
# ---------------------------------------------------------------------------


def test_offer_then_affirmation_dispatches(runner):
    # Turn 1: the Face Lobe OFFERS deep work (its reply mentions /deep).
    runner._face.reply_text = "Sure—prefix /deep and I'll dispatch a Depth Lobe job to fetch the GPUs."
    r1 = runner.chat("are you sure? can you consult a deep lobe", session_id="c1")
    assert r1["dispatched_job"] is None       # the offer turn itself doesn't dispatch
    assert r1["confirm_dispatch"] is False

    # Turn 2: the user affirms -> we dispatch the offered goal.
    runner._face.reply_text = "On it."
    r2 = runner.chat("okay can you do that for me please", session_id="c1")
    assert r2["confirm_dispatch"] is True
    assert r2["dispatched_job"] is not None
    assert r2["dispatched_job"].get("job_id")


def test_affirmation_without_offer_does_not_dispatch(runner):
    runner._face.reply_text = "Here's some info, no tools needed."  # no offer
    runner.chat("what is hivemind", session_id="c2")
    r2 = runner.chat("okay do that", session_id="c2")
    assert r2["confirm_dispatch"] is False
    assert r2["dispatched_job"] is None


def test_topic_change_after_offer_clears_pending(runner):
    runner._face.reply_text = "Sure—I'll dispatch a Depth Lobe job if you confirm."
    runner.chat("check the gpus", session_id="c3")           # sets pending
    runner._face.reply_text = "It is sunny."
    r2 = runner.chat("okay what is the weather", session_id="c3")  # not a confirmation
    assert r2["confirm_dispatch"] is False
    assert r2["dispatched_job"] is None
    # pending was cleared; a later affirmation must NOT resurrect it
    r3 = runner.chat("ok do that", session_id="c3")
    assert r3["confirm_dispatch"] is False


def test_voice_mode_injects_brevity_directive(runner):
    """voice_mode=True appends the spoken-style brevity directive to the
    Face Lobe's extra_system; voice_mode=False (text chat) does not."""
    runner._face.calls.clear()
    runner.chat("what's in hivemind", session_id="v1", voice_mode=True)
    assert any("VOICE MODE" in (c.get("extra_system") or "") for c in runner._face.calls), \
        "voice turn should inject the brevity directive"

    runner._face.calls.clear()
    runner.chat("what's in hivemind", session_id="v2", voice_mode=False)
    assert not any("VOICE MODE" in (c.get("extra_system") or "") for c in runner._face.calls), \
        "text chat must NOT inject the voice brevity directive"


def test_depth_model_obeyed_in_dispatch(tmp_path, monkeypatch):
    """An explicit depth_model must flow verbatim to the dispatched
    JobEnvelope's resource_request.model_override."""
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    # Route deep so a job is dispatched.
    monkeypatch.setattr(hr, "router_route", lambda m: types.SimpleNamespace(
        kind="deep", confidence=0.9, source="heuristic",
        cleaned_message=m, goal=m, to_dict=lambda: {"kind": "deep"},
    ))
    monkeypatch.setattr(hr, "choose_foreground_model", lambda **k: types.SimpleNamespace(
        model_id="face-model", to_dict=lambda: {"model_id": "face-model", "source": "test"}))
    monkeypatch.setattr(hr, "build_grounded_user_message", lambda msg, url: (msg, None))

    captured = {}
    real_submit = hr.default_runner().submit

    def spy_submit(envelope):
        captured["model_override"] = envelope.resource_request.model_override
        return real_submit(envelope)

    monkeypatch.setattr(hr.default_runner(), "submit", spy_submit)

    face = _ConfigurableFace()
    r = Ms4HermesRunner(hermes_dir=str(tmp_path), hivemind_url="http://hive:6089",
                        agent_cls=_FakeAgent, face_lobe_chat=face)
    monkeypatch.setattr(r, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(r, "_try_quartermaster_inline", lambda m: (None, None))  # skip inline -> depth
    resp = r.chat("run a big analysis", session_id="cdep", depth_model="Qwen3-Coder-30B")
    assert resp["dispatched_job"] is not None
    assert captured["model_override"] == "Qwen3-Coder-30B"


def test_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_DA_CONFIRM_DISPATCH", "0")
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    _patch_network_seams(monkeypatch)
    face = _ConfigurableFace()
    r = Ms4HermesRunner(hermes_dir=str(tmp_path), hivemind_url="http://hive:6089",
                        agent_cls=_FakeAgent, face_lobe_chat=face)
    monkeypatch.setattr(r, "_try_cluster_time", lambda: None)
    face.reply_text = "I'll dispatch a Depth Lobe job."
    r.chat("check gpus", session_id="c4")
    face.reply_text = "On it."
    r2 = r.chat("okay do that", session_id="c4")
    assert r2["confirm_dispatch"] is False
    assert r2["dispatched_job"] is None
