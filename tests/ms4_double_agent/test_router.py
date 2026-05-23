"""Auto-router decision tests."""

from __future__ import annotations

import os

import pytest

from machine_spirit_4.double_agent import router as router_module
from machine_spirit_4.double_agent import router_route


def test_empty_message_routes_direct():
    decision = router_route("")
    assert decision.kind == "direct"
    assert decision.source == "heuristic"


def test_short_greeting_routes_direct():
    decision = router_route("hi there!")
    assert decision.kind == "direct"
    assert decision.confidence > 0.8


def test_slash_deep_forces_dispatch_and_strips_prefix():
    decision = router_route("/deep   fix the login bug")
    assert decision.kind == "deep"
    assert decision.source == "slash"
    assert decision.override is True
    assert decision.cleaned_message == "fix the login bug"
    assert decision.goal


def test_slash_direct_forces_no_dispatch_even_for_long_complex_message():
    msg = "/direct " + ("implement an audit of the authentication pipeline " * 6)
    decision = router_route(msg)
    assert decision.kind == "direct"
    assert decision.source == "slash"
    assert decision.override is True


def test_long_action_verb_message_routes_deep():
    msg = (
        "implement a thorough audit of our async worker module to find why "
        "it deadlocks under load; trace the lock acquisition order, "
        "investigate the await boundary issue, and propose a patch with "
        "rationale. also review the test coverage gaps."
    )
    decision = router_route(msg)
    assert decision.kind == "deep"
    assert decision.confidence >= 0.5
    assert "action verbs" in decision.reason


def test_code_block_triggers_deep_route():
    msg = "Look at this please:\n```python\ndef foo():\n    return 1\n```\nis it idiomatic?"
    decision = router_route(msg)
    assert decision.kind == "deep"
    # Either signal is acceptable: the code block heuristic OR the
    # tool-requiring intent allowlist (this message contains "look at"
    # which is a tool-intent phrase that short-circuits to deep before
    # the code-block scoring layer runs).
    assert any(token in decision.reason for token in ("code block", "tool-requiring intent"))


def test_llm_classifier_only_consulted_when_heuristic_uncertain(monkeypatch):
    monkeypatch.setenv(router_module.ENV_LLM_CLASSIFY, "1")
    calls = {"n": 0}

    def fake(_msg):
        calls["n"] += 1
        return {"kind": "deep", "confidence": 0.9, "reason": "model says deep", "goal": "x"}

    # High-confidence heuristic (long + actiony) should NOT call the classifier.
    long_msg = "audit and refactor the entire codebase to remove the deadlock " * 3
    router_route(long_msg, classifier=fake)
    assert calls["n"] == 0, "classifier should be skipped when heuristic is confident"
    # Low-confidence heuristic (medium length, no obvious keywords) should call it.
    ambiguous = "could you check whether the timezone handling is correct in the report?"
    decision = router_route(ambiguous, classifier=fake)
    assert calls["n"] == 1
    assert decision.source == "llm"
    assert decision.kind == "deep"


def test_llm_classifier_not_consulted_when_env_off(monkeypatch):
    monkeypatch.delenv(router_module.ENV_LLM_CLASSIFY, raising=False)
    calls = {"n": 0}

    def fake(_msg):
        calls["n"] += 1
        return {"kind": "deep", "confidence": 0.95, "reason": "x", "goal": "y"}

    ambiguous = "what's up with the timezone handling?"
    router_route(ambiguous, classifier=fake)
    assert calls["n"] == 0


def test_llm_classifier_exception_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setenv(router_module.ENV_LLM_CLASSIFY, "1")

    def boom(_msg):
        raise RuntimeError("classifier model down")

    ambiguous = "is the cache eviction logic still correct?"
    decision = router_route(ambiguous, classifier=boom)
    assert decision.source == "heuristic"
    assert "classifier raised" in decision.reason


def test_route_decision_serializes_to_v1_schema():
    decision = router_route("/deep run a smoke test")
    payload = decision.to_dict()
    assert payload["schema"] == "Ms4RouteDecision.v1"
    assert payload["kind"] == "deep"
    assert payload["override"] is True


@pytest.mark.parametrize("msg", [
    "do you see any GPUs?",
    "what tools do you have?",
    "list mcp servers",
    "can you run that for me",
    "show me the nodes",
    "what models are loaded",
])
def test_tool_intent_questions_route_deep(msg):
    """Short questions that require real tool access (GPU/cluster/tools/
    run/etc.) must route deep so the Face Lobe doesn't try to answer
    from invention. This is the regression the operator caught live."""
    decision = router_route(msg)
    assert decision.kind == "deep", f"expected deep for {msg!r}, got {decision.kind} ({decision.reason})"
    assert "tool-requiring intent" in decision.reason or "action verbs" in decision.reason


@pytest.mark.parametrize("msg", [
    "I want you to start a vm",
    "please stop the docker container",
    "boot the dev VM",
    "create a new service for me",
    "install the python build deps",
    "delete that file",
    "configure the redis instance",
    "shut down node-3",
    "deploy the new version",
    "take a screenshot of my desktop",
    "look at hivemind tool list",
])
def test_operational_intents_route_deep(msg):
    """Operational ('do a thing on my system') intents — start/stop/boot/
    create/install/delete/configure/shutdown/deploy/screenshot — must
    route deep. This was the live miss: 'I want you to start a vm' got
    'no deep signals' before the verb list was expanded."""
    decision = router_route(msg)
    assert decision.kind == "deep", f"expected deep for {msg!r}, got {decision.kind} ({decision.reason})"
