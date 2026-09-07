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

    def chat(
        self,
        message,
        *,
        session_id,
        model,
        stream_callback=None,
        extra_system=None,
        substantive_word_range=None,
    ):
        self.calls.append(
            {
                "message": message,
                "extra_system": extra_system,
                "substantive_word_range": substantive_word_range,
            }
        )
        return {
            "text": self.reply_text,
            "session_id": session_id or "s",
            "model": model,
            "completed": True,
            "api_calls": 1,
            "metrics": {"schema": "Ms4TurnMetrics.v1"},
        }

    def chat_authoritative(self, message, *, authoritative_text, session_id, model, **kwargs):
        self.calls.append({"message": message, "authoritative_text": authoritative_text})
        return {
            "text": authoritative_text,
            "session_id": session_id or "s",
            "model": model,
            "completed": True,
            "api_calls": 0,
            "metrics": {"schema": "Ms4TurnMetrics.v1"},
        }

    def sessions(self):
        return []


GATED_CANONICAL = "hivemind.services.enable@v1"
GATED_GOAL = f"enable service menta_hli with {GATED_CANONICAL}"


class _SeamCounters:
    def __init__(self):
        self.inline = 0
        self.submit = 0
        self.direct = 0
        self.envelopes: list = []


def _gated_outcome(query: str, canonical: str = GATED_CANONICAL) -> dict:
    return {
        "schema": "Ms4ToolRouteDecision.v1",
        "verdict": "depth",
        "query": query,
        "reason": f"top tool {canonical} is not inline-eligible",
        "resolution": {
            "schema": "Ms4ToolResolution.v1",
            "query": query,
            "tier": "deterministic",
            "confidence": 1.0,
            "toolboxes": ["services"],
            "tools": [{
                "name": canonical,
                "toolbox": "services",
                "cluster": "operations",
                "score": 1.0,
                "kind": "hivemind_native",
                "inline_eligible": False,
                "source": "hivemind",
                "gated_by": ["operator-level"],
            }],
            "catalog_version": "test-catalog",
            "fallback_reason": None,
        },
        "inline_tool": None,
        "inline_toolbox": None,
        "ethics": None,
    }


def _gated_catalog():
    from machine_spirit_4.gateway.quartermaster.catalog import (
        KIND_HIVEMIND_NATIVE,
        SOURCE_HIVEMIND,
        Catalog,
        ToolEntry,
    )

    entry = ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=GATED_CANONICAL,
        toolbox="services",
        cluster="operations",
        description="enable a service",
        source=SOURCE_HIVEMIND,
        kind=KIND_HIVEMIND_NATIVE,
        input_schema={"type": "object", "required": ["service_name"]},
        gated_by=("operator-level",),
    )
    return Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version="test",
        built_at="2026-08-17T00:00:00Z",
        hivemind_url="http://hive:6089",
        tools=(entry,),
        toolboxes={"services": (entry,)},
        sources={"hivemind": 1},
        errors=(),
    )


def _exact_read_catalog():
    from machine_spirit_4.gateway.quartermaster.catalog import (
        KIND_HIVEMIND_NATIVE,
        SOURCE_HIVEMIND,
        Catalog,
        ToolEntry,
    )

    entry = ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name="hivemind.app.get@v1",
        toolbox="apps",
        cluster="applications",
        description="get an app by id",
        source=SOURCE_HIVEMIND,
        kind=KIND_HIVEMIND_NATIVE,
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    )
    return Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version="test",
        built_at="2026-08-17T00:00:00Z",
        hivemind_url="http://hive:6089",
        tools=(entry,),
        toolboxes={"apps": (entry,)},
        sources={"hivemind": 1},
        errors=(),
    )


def _install_submit_counter(monkeypatch, counters: _SeamCounters):
    class _CountingRunner:
        def submit(self, envelope):
            counters.submit += 1
            counters.envelopes.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(hr, "default_runner", lambda: _CountingRunner())


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
    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.catalog.get_catalog",
        lambda _url, **_k: _gated_catalog(),
    )
    face = _ConfigurableFace()
    r = Ms4HermesRunner(
        hermes_dir=str(tmp_path), hivemind_url="http://hive:6089",
        agent_cls=_FakeAgent, face_lobe_chat=face,
    )
    monkeypatch.setattr(r, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        r,
        "_try_quartermaster_inline",
        lambda message: (
            (None, _gated_outcome(message))
            if GATED_CANONICAL in message
            else (None, None)
        ),
    )
    r._face = face
    r._counters = _SeamCounters()
    _install_submit_counter(monkeypatch, r._counters)
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
    r1 = runner.chat(GATED_GOAL, session_id="c1")
    assert r1["dispatched_job"] is None       # the offer turn itself doesn't dispatch
    assert r1["confirm_dispatch"] is False
    assert runner._counters.inline == 0
    assert runner._counters.submit == 0
    assert runner._counters.direct == 0

    # Turn 2: the user affirms -> we dispatch the offered goal.
    runner._face.reply_text = "On it."
    r2 = runner.chat("okay can you do that for me please", session_id="c1")
    assert r2["confirm_dispatch"] is True
    assert r2["dispatched_job"] is not None
    assert r2["dispatched_job"].get("job_id")
    assert runner._counters.inline == 0
    assert runner._counters.submit == 1
    assert runner._counters.direct == 0
    assert runner._counters.envelopes[0].resource_request.enabled_toolsets == [
        "mcp-hivemind-exact-gated"
    ]


def test_affirmation_without_offer_does_not_dispatch(runner):
    runner._face.reply_text = "Here's some info, no tools needed."  # no offer
    runner.chat("what is hivemind", session_id="c2")
    r2 = runner.chat("okay do that", session_id="c2")
    assert r2["confirm_dispatch"] is False
    assert r2["dispatched_job"] is None


def test_topic_change_after_offer_clears_pending(runner):
    runner._face.reply_text = "Sure—I'll dispatch a Depth Lobe job if you confirm."
    runner.chat(GATED_GOAL, session_id="c3")           # sets pending
    runner._face.reply_text = "It is sunny."
    r2 = runner.chat("okay what is the weather", session_id="c3")  # not a confirmation
    assert r2["confirm_dispatch"] is False
    assert r2["dispatched_job"] is None
    # pending was cleared; a later affirmation must NOT resurrect it
    r3 = runner.chat("ok do that", session_id="c3")
    assert r3["confirm_dispatch"] is False
    assert runner._counters.submit == 0


def test_voice_mode_injects_brevity_directive(runner):
    """voice_mode=True appends the spoken-style brevity directive to the
    Face Lobe's extra_system; voice_mode=False (text chat) does not."""
    runner._face.calls.clear()
    runner.chat("what's in hivemind", session_id="v1", voice_mode=True)
    assert any("VOICE MODE" in (c.get("extra_system") or "") for c in runner._face.calls), \
        "voice turn should inject the brevity directive"
    assert any(
        "180 to 240 visible words" in (c.get("extra_system") or "")
        and "150 is a hard floor and 380 is a hard ceiling" in (c.get("extra_system") or "")
        for c in runner._face.calls
    ), "substantive voice turns need an explicit bounded word target"
    assert all(
        c.get("substantive_word_range") == (150, 380)
        for c in runner._face.calls
    )

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
    monkeypatch.setattr(
        r,
        "_try_quartermaster_inline",
        lambda m: (None, {
            "schema": "Ms4ToolRouteDecision.v1",
            "verdict": "depth",
            "query": m,
            "reason": "top tool hivemind.app.get@v1 requires args ['id']",
            "resolution": {
                "schema": "Ms4ToolResolution.v1",
                "query": m,
                "tier": "deterministic",
                "confidence": 1.0,
                "toolboxes": ["apps"],
                "tools": [{
                    "name": "hivemind.app.get@v1",
                    "toolbox": "apps",
                    "cluster": "applications",
                    "score": 1.0,
                    "kind": "hivemind_native",
                    "inline_eligible": True,
                    "source": "hivemind",
                    "gated_by": [],
                }],
                "catalog_version": "test-catalog",
                "fallback_reason": None,
            },
            "inline_tool": None,
            "inline_toolbox": None,
            "ethics": None,
        }),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.catalog.get_catalog",
        lambda _url, **_k: _exact_read_catalog(),
    )
    resp = r.chat("fetch app hivemind.app.get@v1 id=demo", session_id="cdep", depth_model="Qwen3-Coder-30B")
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


def test_unconfirmed_gated_intent_is_zero_effect(runner):
    runner._face.reply_text = "Sure—I'll dispatch a Depth Lobe job if you confirm."
    r1 = runner.chat(GATED_GOAL, session_id="c-unconfirmed")
    assert r1["dispatched_job"] is None
    assert r1["confirm_dispatch"] is False
    assert runner._counters.inline == 0
    assert runner._counters.submit == 0
    assert runner._counters.direct == 0


def test_cancelled_gated_intent_is_zero_effect(runner):
    runner._face.reply_text = "Sure—I'll dispatch a Depth Lobe job if you confirm."
    runner.chat(GATED_GOAL, session_id="c-cancel")
    runner._face.reply_text = "Okay, I will not do that."
    r2 = runner.chat("never mind, cancel that", session_id="c-cancel")
    assert r2["confirm_dispatch"] is False
    assert r2["dispatched_job"] is None
    assert runner._counters.inline == 0
    assert runner._counters.submit == 0
    assert runner._counters.direct == 0
    r3 = runner.chat("ok do that", session_id="c-cancel")
    assert r3["dispatched_job"] is None
    assert runner._counters.submit == 0


def test_first_confirmation_then_repeat_stays_one_submit(runner):
    runner._face.reply_text = "Sure—prefix /deep and I'll dispatch a Depth Lobe job."
    runner.chat(GATED_GOAL, session_id="c-repeat")
    assert runner._counters.submit == 0
    runner._face.reply_text = "On it."
    r2 = runner.chat("yes please", session_id="c-repeat")
    assert r2["confirm_dispatch"] is True
    assert runner._counters.submit == 1
    assert runner._counters.envelopes[0].resource_request.enabled_toolsets == [
        "mcp-hivemind-exact-gated"
    ]
    runner._face.reply_text = "Already running."
    r3 = runner.chat("yes please", session_id="c-repeat")
    assert r3["confirm_dispatch"] is False
    assert runner._counters.submit == 1
    assert runner._counters.inline == 0
    assert runner._counters.direct == 0
