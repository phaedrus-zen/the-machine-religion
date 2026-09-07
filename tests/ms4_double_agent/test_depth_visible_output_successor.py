from __future__ import annotations

import threading

import pytest

from machine_spirit_4.double_agent.worker import (
    WorkerCanceled,
    _depth_answer_quality_error,
    _depth_explicit_contract_error,
    _depth_final_visible_text,
    _depth_tool_contract_error,
    _goal_with_resolved_context_reference,
    build_real_chat_runner,
)
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner


GOAL = (
    "Using the remembered facts, create a complete recovery plan. "
    "Call hivemind_time_now exactly once. Use exactly these headings: "
    "SUMMARY, CAUSES, ACTIONS, RISKS, VALIDATION. ACTIONS must contain "
    "exactly four numbered actions; Action 2 must mitigate risk ALPHA. "
    "RISKS exactly two bullets. No owner supplied, so write OWNER=UNKNOWN. "
    "VALIDATION includes VERIFY-SUCCESSOR01 and real UTC. Do not ask whether I want more."
)


def _answer(*, owner: bool = True, actions: int = 4, risks: int = 2) -> str:
    owner_line = " OWNER=UNKNOWN." if owner else ""
    action_lines = [
        "1. Flush stale route entries, reload the authoritative table, and record the starting UTC timestamp.",
        "2. Mitigate risk ALPHA with a thirty-second TTL guard, quarantine stale paths, and verify a healthy fallback.",
        "3. Run end-to-end probes for latency, packet loss, route ownership, and durable job accounting.",
        "4. Freeze the recovered state, preserve the audit evidence, and compare it with the pre-recovery baseline.",
    ][:actions]
    risk_lines = [
        "- A missed recovery window can let stale routes propagate into sessions that require a controlled flush.",
        "- Aggressive TTL enforcement can add latency while legitimate long-lived paths are revalidated.",
    ][:risks]
    return (
        "SUMMARY\n\n"
        "Project AMBER-LATTICE needs a bounded recovery within twelve minutes. The plan restores "
        "authoritative routing, preserves evidence, and keeps every mutation proposal behind the "
        "operator's existing authority boundary."
        f"{owner_line}\n\n"
        "CAUSES\n\n"
        "The primary cause is a failed routing-table refresh that left expired next hops in the "
        "forwarder mesh. Missing TTL validation allowed those entries to survive, and slow failover "
        "detection kept degraded paths eligible. The recovery therefore has to repair state, verify "
        "freshness, and prove end-to-end delivery rather than trust one health response.\n\n"
        "ACTIONS\n\n"
        + "\n".join(action_lines)
        + "\n\nRISKS\n\n"
        + "\n".join(risk_lines)
        + "\n\nVALIDATION\n\n"
        "VERIFY-SUCCESSOR01 must record real UTC from the verified HiveMind time result, show that "
        "all four actions completed inside 720 seconds, and reconcile the route table, probes, logs, "
        "job state, and visible answer. A failed check keeps the recovery held instead of converting "
        "an incomplete draft into a successful result."
    )


class _Agent:
    def __init__(self, result, *, on_start, on_complete, call_tool):
        self.result = result
        self.on_start = on_start
        self.on_complete = on_complete
        self.call_tool = call_tool
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def run_conversation(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if self.call_tool:
            for index in range(self.call_tool):
                call_id = f"call-time-{index}"
                self.on_start(call_id, "hivemind_time_now", {})
                self.on_complete(
                    call_id,
                    "hivemind_time_now",
                    {},
                    {"utc": "2026-08-11T04:30:00+00:00"},
                )
        return self.result


class _Runner:
    default_model = "qwen3.6:35b_ollama"

    def __init__(self, results, *, first_tool_calls=1):
        self.results = iter(results)
        self.first_tool_calls = first_tool_calls
        self.agent_kwargs: list[dict] = []
        self.agents: list[_Agent] = []

    def new_background_agent(self, **kwargs):
        index = len(self.agents)
        self.agent_kwargs.append(kwargs)
        agent = _Agent(
            next(self.results),
            on_start=kwargs["tool_start_callback"],
            on_complete=kwargs["tool_complete_callback"],
            call_tool=self.first_tool_calls if index == 0 else 0,
        )
        self.agents.append(agent)
        return agent


def _run(runner: _Runner):
    starts: list[tuple[str, str]] = []
    completes: list[tuple[str, str]] = []
    history = [
        {
            "role": "user",
            "content": "Remember project AMBER-LATTICE, a twelve-minute window, and risk ALPHA.",
        },
        {"role": "assistant", "content": "READY"},
    ]
    response = build_real_chat_runner(runner)(
        message=GOAL,
        session_id="da-worker-da-11111111-1111-4111-8111-111111111111",
        model="qwen3.6:35b_ollama",
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda call_id, name, _args: starts.append((call_id, name)),
        tool_complete_callback=lambda call_id, name, _args, _result: completes.append(
            (call_id, name)
        ),
        enabled_toolsets=["hermes-cli", "mcp-hivemind"],
        conversation_history=history,
    )
    return response, starts, completes, history


def test_exact_live_failure_class_is_rejected_before_publication() -> None:
    assert _depth_answer_quality_error(GOAL, _answer(owner=False)) == (
        "model_result_missing_required_literal"
    )


def test_same_job_no_tool_nonthinking_repair_preserves_tool_and_context_identity() -> (
    None
):
    runner = _Runner(
        [
            {"final_response": _answer(owner=False), "completed": True},
            {"final_response": _answer(owner=True), "completed": True},
        ]
    )
    response, starts, completes, history = _run(runner)

    assert response["completed"] is True
    assert response["text"] == _answer(owner=True)
    assert response["quality_repair_attempted"] is True
    assert response["quality_repair_mode"] == "same_job_no_tools_non_thinking"
    assert response["quality_repair_reason"] == "model_result_missing_required_literal"
    assert starts == [("call-time-0", "hivemind_time_now")]
    assert completes == starts
    assert len(runner.agents) == 2
    assert runner.agent_kwargs[0]["session_id"] == runner.agent_kwargs[1]["session_id"]
    assert runner.agent_kwargs[1]["enabled_toolsets"] == []
    assert runner.agent_kwargs[1]["reasoning_config_override"] == {"enabled": False}
    repair_prompt = runner.agents[1].prompts[0]
    assert _answer(owner=False) in repair_prompt
    assert "call-time-0" in repair_prompt
    assert "hivemind_time_now" in repair_prompt
    assert "2026-08-11T04:30:00+00:00" in repair_prompt
    assert runner.agents[1].kwargs[0]["conversation_history"] == history
    assert (
        "tools and hidden reasoning are disabled"
        in runner.agents[1].kwargs[0]["system_message"]
    )


def test_second_structurally_bad_draft_fails_closed() -> None:
    runner = _Runner(
        [
            {"final_response": _answer(owner=False), "completed": True},
            {"final_response": _answer(owner=False), "completed": True},
        ]
    )
    response, starts, completes, _history = _run(runner)

    assert response["completed"] is False
    assert response["error"] == "model_result_missing_required_literal"
    assert response["quality_repair_attempted"] is True
    assert starts == completes == [("call-time-0", "hivemind_time_now")]
    assert len(runner.agents) == 2


def test_requested_tool_count_mismatch_fails_without_text_only_repair() -> None:
    runner = _Runner(
        [{"final_response": _answer(owner=True), "completed": True}],
        first_tool_calls=2,
    )
    response, starts, completes, _history = _run(runner)

    assert response["completed"] is False
    assert response["error"] == "model_result_requested_tool_count_mismatch"
    assert response["quality_repair_attempted"] is False
    assert len(starts) == len(completes) == 2
    assert len(runner.agents) == 1


def test_all_machine_checkable_invariants_have_distinct_failure_gates() -> None:
    assert (
        _depth_answer_quality_error(
            GOAL, _answer(owner=True).replace("CAUSES", "ROOT CAUSES")
        )
        == "model_result_exact_headings_mismatch"
    )
    assert _depth_answer_quality_error(GOAL, _answer(owner=True, actions=3)) == (
        "model_result_numbered_count_mismatch"
    )
    assert _depth_answer_quality_error(GOAL, _answer(owner=True, risks=1)) == (
        "model_result_bullet_count_mismatch"
    )
    assert (
        _depth_answer_quality_error(
            GOAL,
            _answer(owner=True).replace("Mitigate risk ALPHA", "Document risk ALPHA"),
        )
        == "model_result_numbered_item_constraint_mismatch"
    )
    assert (
        _depth_answer_quality_error(
            GOAL, _answer(owner=True).replace("VERIFY-SUCCESSOR01", "VERIFY-WRONG")
        )
        == "model_result_missing_required_marker"
    )


def test_background_agent_can_explicitly_disable_reasoning_for_repair(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MS4_DA_MAX_TOKENS", raising=False)
    runner = object.__new__(Ms4HermesRunner)
    runner.ensure_hermes_path = lambda: None
    runner.require_plugin = lambda: None
    captured = {}
    runner._construct_agent = lambda **kwargs: captured.update(kwargs) or kwargs

    result = runner.new_background_agent(
        session_id="da-worker-da-22222222-2222-4222-8222-222222222222",
        model="qwen3.6:35b_ollama",
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
        enabled_toolsets=[],
        reasoning_config_override={"enabled": False},
    )

    assert result["reasoning_config"] == {"enabled": False}
    assert result["max_tokens"] == 4096
    assert captured["enabled_toolsets"] == []


def test_explicit_parser_ignores_negated_literal_and_plain_lowercase_prose() -> None:
    assert (
        _depth_explicit_contract_error(
            "Do not write OWNER=UNKNOWN; explain the owner state in prose.",
            "The owner is unavailable in the supplied context.",
        )
        is None
    )
    assert (
        _depth_explicit_contract_error(
            "The validation includes additional narrative context.",
            "Validation includes the available narrative context.",
        )
        is None
    )
    assert (
        _depth_explicit_contract_error(
            "Do not use exactly these headings: SUMMARY, ACTIONS, RISKS.",
            "A normal answer without those headings.",
        )
        is None
    )
    assert (
        _depth_tool_contract_error(
            "Do not call hivemind_time_now exactly once; explain the tool only.",
            [],
            [],
        )
        is None
    )


@pytest.mark.parametrize(
    "text",
    (
        "SUMMARY\nDone.\nRISKS\nNone.\nACTIONS\n1. Act.",
        "SUMMARY\nDone.\nACTIONS\n1. Act.\nSUMMARY\nRepeated.\nRISKS\nNone.",
        (
            "SUMMARY\nDone.\nACTIONS\n1. Act.\nRISKS\nNone.\n"
            "Verified HiveMind tool results:\n- hivemind_time_now: verified UTC"
        ),
    ),
)
def test_exact_headings_reject_reordering_duplicates_and_extra_headings(
    text: str,
) -> None:
    goal = "Use exactly these headings: SUMMARY, ACTIONS, RISKS."
    assert _depth_explicit_contract_error(goal, text) == (
        "model_result_exact_headings_mismatch"
    )


def test_numbered_contract_rejects_gap_even_when_count_matches() -> None:
    goal = (
        "Use exactly these headings: SUMMARY, ACTIONS, RISKS. "
        "ACTIONS must contain exactly three numbered actions."
    )
    text = "SUMMARY\nDone.\nACTIONS\n1. First.\n2. Second.\n4. Fourth.\nRISKS\nNone."
    assert _depth_explicit_contract_error(goal, text) == (
        "model_result_numbered_count_mismatch"
    )


def test_post_model_tool_evidence_does_not_break_structural_visible_output() -> None:
    evidence = [
        {
            "tool": "hivemind_time_now",
            "tool_call_id": "call-time-0",
            "result_excerpt": "2026-08-11T04:30:00+00:00",
        }
    ]
    answer = _answer(owner=True)
    assert _depth_final_visible_text(GOAL, answer, evidence) == answer

    undecorated_goal = "Explain the current cluster state thoroughly."
    decorated = _depth_final_visible_text(undecorated_goal, answer, evidence)
    assert decorated.startswith(answer)
    assert "Verified HiveMind tool results:" in decorated


def test_duplicate_tool_start_with_one_completion_blocks_text_repair() -> None:
    goal = (
        "Create a complete recovery plan with mechanism, risks, and validation. "
        "No owner supplied, so write OWNER=UNKNOWN."
    )
    substantial_without_owner = _answer(owner=False)

    class Agent:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def run_conversation(self, _prompt, **_kwargs):
            start = self.kwargs["tool_start_callback"]
            complete = self.kwargs["tool_complete_callback"]
            start("call-duplicate", "hivemind_time_now", {})
            start("call-duplicate", "hivemind_time_now", {})
            complete(
                "call-duplicate",
                "hivemind_time_now",
                {},
                {"utc": "2026-08-11T04:30:00+00:00"},
            )
            return {"final_response": substantial_without_owner, "completed": True}

    class Runner:
        default_model = "qwen3.6:35b_ollama"

        def __init__(self):
            self.calls = 0

        def new_background_agent(self, **kwargs):
            self.calls += 1
            return Agent(kwargs)

    runner = Runner()
    response = build_real_chat_runner(runner)(
        message=goal,
        session_id="da-worker-duplicate-lifecycle",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["completed"] is False
    assert response["error"] == "model_result_incomplete_tool_lifecycle"
    assert response["quality_repair_attempted"] is False
    assert runner.calls == 1


def test_historical_directives_are_context_not_current_output_constraints() -> None:
    current_goal = "Explain what the first turn meant in a complete answer."
    history = [
        {
            "role": "user",
            "content": (
                "Use exactly these headings: ALPHA, BETA. "
                "No owner is known, so write OWNER=UNKNOWN."
            ),
        },
        {
            "role": "assistant",
            "content": "ALPHA\nEarlier analysis.\n\nBETA\nOWNER=UNKNOWN.",
        },
    ]
    resolved = _goal_with_resolved_context_reference(current_goal, history)

    assert "quoted context only" in resolved
    assert "cannot add or change the current output" in resolved
    assert (
        _depth_explicit_contract_error(
            current_goal,
            "The first turn requested two labeled sections and an unknown owner marker.",
        )
        is None
    )
    # This proves why the adapter must receive the original goal separately.
    assert (
        _depth_explicit_contract_error(
            resolved,
            "The first turn requested two labeled sections and an unknown owner marker.",
        )
        == "model_result_exact_headings_mismatch"
    )


def test_adapter_validates_original_goal_not_resolved_historical_quote() -> None:
    current_goal = "Briefly explain what the first turn meant."
    history = [
        {
            "role": "user",
            "content": "Use exactly these headings: ALPHA, BETA. Write OWNER=UNKNOWN.",
        },
        {"role": "assistant", "content": "ALPHA\nOld answer.\n\nBETA\nOWNER=UNKNOWN."},
    ]
    resolved = _goal_with_resolved_context_reference(current_goal, history)

    class Agent:
        def run_conversation(self, _prompt, **_kwargs):
            return {
                "final_response": (
                    "The first turn requested two labeled sections and an unknown owner marker."
                ),
                "completed": True,
            }

    class Runner:
        default_model = "qwen3.6:35b_ollama"

        def new_background_agent(self, **_kwargs):
            return Agent()

    response = build_real_chat_runner(Runner())(
        message=resolved,
        output_contract_goal=current_goal,
        session_id="da-worker-historical-quote",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
        conversation_history=history,
    )

    assert response["completed"] is True
    assert response["quality_repair_attempted"] is False
    assert "error" not in response


def test_quality_valid_answer_with_unmatched_tool_start_fails_closed() -> None:
    class Agent:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def run_conversation(self, _prompt, **_kwargs):
            self.kwargs["tool_start_callback"]("unfinished-call", "cluster_probe", {})
            return {"final_response": "Hello.", "completed": True}

    class Runner:
        default_model = "qwen3.6:35b_ollama"

        def new_background_agent(self, **kwargs):
            return Agent(kwargs)

    response = build_real_chat_runner(Runner())(
        message="Say hello.",
        session_id="da-worker-unmatched-start",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["completed"] is False
    assert response["error"] == "model_result_incomplete_tool_lifecycle"
    assert response["quality_repair_attempted"] is False


def test_quality_valid_answer_with_unmatched_tool_completion_fails_closed() -> None:
    class Agent:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def run_conversation(self, _prompt, **_kwargs):
            self.kwargs["tool_complete_callback"](
                "completion-only", "cluster_probe", {}, {"healthy": True}
            )
            return {"final_response": "Hello.", "completed": True}

    class Runner:
        default_model = "qwen3.6:35b_ollama"

        def new_background_agent(self, **kwargs):
            return Agent(kwargs)

    response = build_real_chat_runner(Runner())(
        message="Say hello.",
        session_id="da-worker-unmatched-completion",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["completed"] is False
    assert response["error"] == "model_result_incomplete_tool_lifecycle"
    assert response["quality_repair_attempted"] is False


def test_repair_receives_complete_long_tool_result_within_budget() -> None:
    tail = "TAIL-SENTINEL-MUST-BE-PRESERVED"
    tool_result = "A" * 2100 + tail

    class Agent:
        def __init__(self, result, kwargs, *, emit_tool):
            self.result = result
            self.kwargs = kwargs
            self.emit_tool = emit_tool
            self.prompts = []

        def run_conversation(self, prompt, **_kwargs):
            self.prompts.append(prompt)
            if self.emit_tool:
                self.kwargs["tool_start_callback"]("long-call", "cluster_summary", {})
                self.kwargs["tool_complete_callback"](
                    "long-call", "cluster_summary", {}, tool_result
                )
            return self.result

    class Runner:
        default_model = "qwen3.6:35b_ollama"

        def __init__(self):
            self.agents = []

        def new_background_agent(self, **kwargs):
            first = not self.agents
            agent = Agent(
                {
                    "final_response": (
                        "The owner remains unspecified." if first else "OWNER=UNKNOWN"
                    ),
                    "completed": True,
                },
                kwargs,
                emit_tool=first,
            )
            self.agents.append(agent)
            return agent

    runner = Runner()
    response = build_real_chat_runner(runner)(
        message="No owner is known, so write OWNER=UNKNOWN.",
        session_id="da-worker-long-result",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["completed"] is True
    assert response["quality_repair_attempted"] is True
    repair_prompt = runner.agents[1].prompts[0]
    assert tail in repair_prompt
    assert '"result_truncated": false' in repair_prompt


def test_repair_fails_closed_when_aggregate_tool_evidence_exceeds_budget() -> None:
    class Agent:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def run_conversation(self, _prompt, **_kwargs):
            self.kwargs["tool_start_callback"]("huge-call", "cluster_summary", {})
            self.kwargs["tool_complete_callback"](
                "huge-call", "cluster_summary", {}, "Z" * 33_000
            )
            return {
                "final_response": "The owner remains unspecified.",
                "completed": True,
            }

    class Runner:
        default_model = "qwen3.6:35b_ollama"

        def __init__(self):
            self.calls = 0

        def new_background_agent(self, **kwargs):
            self.calls += 1
            return Agent(kwargs)

    runner = Runner()
    response = build_real_chat_runner(runner)(
        message="No owner is known, so write OWNER=UNKNOWN.",
        session_id="da-worker-oversize-result",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["completed"] is False
    assert response["error"] == "model_result_repair_tool_evidence_truncated"
    assert response["quality_repair_attempted"] is False
    assert runner.calls == 1


def test_repair_completion_callback_without_start_fails_closed() -> None:
    class CompletionOnlyRepairRunner(_Runner):
        def new_background_agent(self, **kwargs):
            index = len(self.agents)
            self.agent_kwargs.append(kwargs)
            if index == 0:
                agent = _Agent(
                    next(self.results),
                    on_start=kwargs["tool_start_callback"],
                    on_complete=kwargs["tool_complete_callback"],
                    call_tool=1,
                )
            else:
                result = next(self.results)

                class RepairAgent(_Agent):
                    def run_conversation(self, prompt, **run_kwargs):
                        self.prompts.append(prompt)
                        self.kwargs.append(run_kwargs)
                        self.on_complete(
                            "call-forged",
                            "hivemind_time_now",
                            {},
                            {"utc": "2026-08-11T04:31:00+00:00"},
                        )
                        return self.result

                agent = RepairAgent(
                    result,
                    on_start=kwargs["tool_start_callback"],
                    on_complete=kwargs["tool_complete_callback"],
                    call_tool=0,
                )
            self.agents.append(agent)
            return agent

    runner = CompletionOnlyRepairRunner(
        [
            {"final_response": _answer(owner=False), "completed": True},
            {"final_response": _answer(owner=True), "completed": True},
        ]
    )
    response, _starts, _completes, _history = _run(runner)

    assert response["completed"] is False
    assert response["error"] == "model_result_repair_tool_activity"
    assert response["text"] == ""


def test_cancel_during_corrective_pass_interrupts_the_active_repair_agent() -> None:
    cancel_event = threading.Event()
    repair_entered = threading.Event()
    repair_released = threading.Event()
    interrupted: list[str] = []

    class Agent:
        def __init__(self, name: str, result: dict):
            self.name = name
            self.result = result

        def run_conversation(self, _prompt, **_kwargs):
            if self.name == "repair":
                repair_entered.set()
                assert repair_released.wait(timeout=2.0)
            return self.result

        def interrupt(self, _reason: str):
            interrupted.append(self.name)
            if self.name == "repair":
                repair_released.set()

    class Runner:
        default_model = "qwen3.6:35b_ollama"

        def __init__(self):
            self.agents = [
                Agent(
                    "initial",
                    {"final_response": _answer(owner=False), "completed": True},
                ),
                Agent(
                    "repair", {"final_response": _answer(owner=True), "completed": True}
                ),
            ]

        def new_background_agent(self, **_kwargs):
            return self.agents.pop(0)

    outcome: dict[str, object] = {}

    def invoke() -> None:
        try:
            build_real_chat_runner(Runner())(
                message=GOAL.replace("Call hivemind_time_now exactly once. ", ""),
                session_id="da-worker-cancel-active-repair",
                model=None,
                stream_callback=lambda _chunk: None,
                tool_start_callback=lambda *_args: None,
                tool_complete_callback=lambda *_args: None,
                cancel_event=cancel_event,
            )
        except (
            BaseException
        ) as exc:  # capture from the worker thread for the assertion below
            outcome["error"] = exc

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    assert repair_entered.wait(timeout=1.0)
    cancel_event.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), WorkerCanceled)
    assert interrupted == ["repair"]


class _LifecycleAgent:
    def __init__(self, script, kwargs):
        self.script = script
        self.kwargs = kwargs

    def run_conversation(self, _prompt, **_kwargs):
        for kind, call_id, tool in self.script["events"]:
            if kind == "start":
                self.kwargs["tool_start_callback"](call_id, tool, {})
            else:
                self.kwargs["tool_complete_callback"](call_id, tool, {}, "ok")
        return self.script["response"]


class _LifecycleRunner:
    default_model = "qwen3.6:35b_ollama"

    def __init__(self, scripts):
        self.scripts = iter(scripts)
        self.calls = 0

    def new_background_agent(self, **kwargs):
        self.calls += 1
        return _LifecycleAgent(next(self.scripts), kwargs)


@pytest.mark.parametrize(
    "events",
    (
        (("complete", "call-a", "cluster_probe"), ("start", "call-a", "cluster_probe")),
        (
            ("start", "call-a", "cluster_probe"),
            ("complete", "call-a", "cluster_probe"),
            ("start", "call-a", "cluster_probe"),
            ("complete", "call-a", "cluster_probe"),
        ),
        (("start", "", ""), ("complete", "", "")),
    ),
    ids=("completion-before-start", "balanced-call-id-reuse", "blank-identity"),
)
@pytest.mark.parametrize("quality_valid", (True, False), ids=("valid", "invalid"))
def test_counter_balanced_invalid_lifecycle_fails_before_publication_or_repair(
    events, quality_valid
):
    goal = "Say hello." if quality_valid else "Write OWNER=UNKNOWN."
    draft = "Hello." if quality_valid else "The owner is unspecified."
    runner = _LifecycleRunner(
        [
            {
                "events": events,
                "response": {"final_response": draft, "completed": True},
            },
            {
                "events": (),
                "response": {"final_response": "OWNER=UNKNOWN", "completed": True},
            },
        ]
    )

    response = build_real_chat_runner(runner)(
        message=goal,
        output_contract_goal=goal,
        session_id="da-worker-v3-lifecycle",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["completed"] is False
    assert response["error"] == "model_result_incomplete_tool_lifecycle"
    assert response["quality_repair_attempted"] is False
    assert runner.calls == 1


def test_overlapping_unique_tool_calls_may_complete_in_either_order() -> None:
    runner = _LifecycleRunner(
        [
            {
                "events": (
                    ("start", "call-a", "probe_a"),
                    ("start", "call-b", "probe_b"),
                    ("complete", "call-b", "probe_b"),
                    ("complete", "call-a", "probe_a"),
                ),
                "response": {"final_response": "Hello.", "completed": True},
            }
        ]
    )

    response = build_real_chat_runner(runner)(
        message="Say hello.",
        output_contract_goal="Say hello.",
        session_id="da-worker-v3-overlap",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["completed"] is True
    assert "error" not in response


def test_fenced_uppercase_data_is_not_a_document_heading() -> None:
    goal = "Use exactly these headings: SUMMARY, ACTIONS, RISKS."
    text = (
        "SUMMARY\nThe probe returned this literal data token:\n```text\nHTTP_STATUS\n```\n"
        "ACTIONS\n1. Preserve the token as data.\nRISKS\nNo material risks."
    )

    assert _depth_explicit_contract_error(goal, text) is None
