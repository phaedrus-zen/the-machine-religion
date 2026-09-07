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


def test_compound_analytical_demo_prompt_is_reasoning_only() -> None:
    prompt = (
        "Explain why a 35B Depth lobe might outperform a 4B Face lobe when "
        "diagnosing an intermittent distributed-inference failure. Give me the "
        "recommendation, mechanism, tradeoffs, and a concrete example."
    )

    assert router_module.is_reasoning_only_analytical(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Research the best local models, compare their tradeoffs, and recommend one.",
        "Benchmark these two architectures, assess the tradeoffs, and recommend an approach.",
        (
            "Diagnose why the inference worker crashed; recommend a fix, explain the "
            "mechanism, and give a concrete example."
        ),
        "Investigate the intermittent timeout, compare likely causes, and recommend a strategy.",
        "Please research the candidates, compare their tradeoffs, and recommend one.",
        "Could you benchmark both designs, assess the tradeoffs, and recommend an approach?",
        "Diagnosing the crash, explain the mechanism and recommend a concrete fix.",
        "Now investigating the timeout, compare likely causes and recommend a strategy.",
        "Debug the crash, explain the mechanism, and recommend a concrete fix.",
        "Please profile both architectures, assess the tradeoffs, and recommend one.",
        "Trace the intermittent request, compare likely causes, and recommend a strategy.",
        "Could you start by debugging the crash and recommend a concrete fix?",
        (
            "Compare the two architectures and recommend one, then benchmark the result and "
            "explain the tradeoffs with a concrete example."
        ),
        (
            "Compare the candidate designs, and then investigate the intermittent timeout "
            "before giving a recommendation and concrete example."
        ),
    ],
)
def test_request_form_operational_analysis_retains_tools(prompt) -> None:
    assert router_module.is_reasoning_only_analytical(prompt) is False


@pytest.mark.parametrize(
    "prompt",
    [
        (
            "Do not use tools. Compare the two designs and give me the recommendation, "
            "mechanism, tradeoffs, and a concrete example."
        ),
        (
            "Answer only from the supplied context; do not search skills or sessions. "
            "Compare the two designs and give me the recommendation, mechanism, tradeoffs, "
            "and a concrete example."
        ),
        (
            "Without browsing or external sources, compare these approaches and give me the "
            "recommendation, mechanism, tradeoffs, and a concrete example."
        ),
        (
            "Do not use any tools. Compare the two designs and give me the recommendation, "
            "mechanism, tradeoffs, and a concrete example."
        ),
        (
            "Do not browse the web. Compare the two designs and give me the recommendation, "
            "mechanism, tradeoffs, and a concrete example."
        ),
        (
            "Don't search the web. Compare the two designs and give me the recommendation, "
            "mechanism, tradeoffs, and a concrete example."
        ),
        (
            "No tools or web browsing. Compare the two designs and give me the recommendation, "
            "mechanism, tradeoffs, and a concrete example."
        ),
        (
            "Do not use tools or browse the web. Compare the two designs and give me the "
            "recommendation, mechanism, tradeoffs, and a concrete example."
        ),
        (
            "Explain the recommendation, mechanism, tradeoffs, and a concrete example for a "
            "Depth lobe when diagnosing an intermittent inference failure."
        ),
    ],
)
def test_negated_tool_language_and_descriptive_diagnosing_remain_reasoning_only(
    prompt,
) -> None:
    assert router_module.is_reasoning_only_analytical(prompt) is True


def test_negated_tool_clause_does_not_hide_later_affirmative_web_request() -> None:
    prompt = (
        "Do not use tools, but search the web for current evidence. Compare the two designs "
        "and give me the recommendation, mechanism, tradeoffs, and a concrete example."
    )

    assert router_module.is_reasoning_only_analytical(prompt) is False


@pytest.mark.parametrize(
    "prompt",
    [
        "Inspect the current cluster logs, recommend a fix, and explain the tradeoffs.",
        "Search the web for the latest evidence, compare the options, and recommend one.",
        "Run a benchmark, explain the mechanism, and give me a concrete example.",
        "Use the authoritative HiveMind GPU availability tool to compare the current nodes.",
        "Diagnose the issue.",
    ],
)
def test_reasoning_only_guard_retains_tools_for_external_effect_or_ambiguous_jobs(
    prompt,
) -> None:
    assert router_module.is_reasoning_only_analytical(prompt) is False


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


@pytest.mark.parametrize(
    "msg",
    [
        "use the authoritative HiveMind GPU availability tool",
        "use the authoritative HiveMine GPU availability tool",
        "use the authoritative Hive Mine GPU availability tool",
        "Now, use the authoritative HiveMind GPU availability tool",
        "Then use the authoritative HiveMind GPU availability tool",
        "Hey, use the authoritative HiveMind GPU availability tool",
        "Can you please use the authoritative HiveMind GPU availability tool",
        (
            "Oracle, use the authoritative HiveMind GPU availability tool. "
            "Tell me exactly which GPUs are available right now and do not guess."
        ),
        (
            "Oracle, use the authoritative HiveMine GPU availability tool. "
            "Tell me exactly which GPUs are available right now and do not guess."
        ),
        (
            "Hey Oracle, use the authoritative HiveMind GPU availability tool. "
            "Tell me exactly which GPUs are available right now and do not guess."
        ),
    ],
)
def test_explicit_authoritative_gpu_tool_request_routes_deep(msg):
    decision = router_route(msg)
    assert decision.kind == "deep"
    assert decision.source == "heuristic"
    assert "explicit named-tool request" in decision.reason


@pytest.mark.parametrize(
    "msg",
    [
        "The authoritative HiveMind GPU availability documentation is useful.",
        "Is GPU availability a useful metric?",
        "HiveMind has GPUs available.",
        "Use the word tool in this sentence.",
        "Do not use the authoritative HiveMind GPU availability tool",
        "Quote: use the authoritative HiveMind GPU availability tool verbatim",
        "Use the word authoritative tool in this sentence.",
        "Use the sentence authoritative tool in this example.",
        "Use the example authoritative tool in this sentence.",
        "Use the quote authoritative tool in this sentence.",
        "Use the quotation authoritative tool in this sentence.",
        "Oracle, do not use the authoritative HiveMind GPU availability tool",
        "Oracle, quote: use the authoritative HiveMind GPU availability tool verbatim",
        "Oracle, describe how to use the authoritative HiveMind GPU availability tool",
        "Oracle, use this sentence as an authoritative tool",
    ],
)
def test_explicit_tool_request_rule_keeps_near_misses_direct(msg):
    decision = router_route(msg)
    assert decision.kind == "direct", (
        f"near miss changed route for {msg!r}: {decision.kind} ({decision.reason})"
    )


@pytest.mark.parametrize(
    "demonstrative,template",
    [
        ("this", "Use {demonstrative} {meta_noun} as an authoritative tool"),
        ("that", "Use {demonstrative} {meta_noun} as an authoritative tool"),
        (
            "the following",
            "Use {demonstrative} {meta_noun} authoritative tool in this example",
        ),
    ],
)
@pytest.mark.parametrize(
    "meta_noun",
    ["sentence", "phrase", "word", "example", "quote", "quotation"],
)
def test_explicit_tool_request_rejects_demonstrative_meta_constructions(
    demonstrative, template, meta_noun
):
    decision = router_route(
        template.format(demonstrative=demonstrative, meta_noun=meta_noun)
    )
    assert decision.kind == "direct"


@pytest.mark.parametrize(
    "msg",
    [
        "/direct use the authoritative HiveMind GPU availability tool",
        "/direct Oracle, use the authoritative HiveMind GPU availability tool",
    ],
)
def test_explicit_tool_request_preserves_slash_direct_priority(msg):
    decision = router_route(msg)
    assert decision.kind == "direct"
    assert decision.source == "slash"
    assert decision.override is True
