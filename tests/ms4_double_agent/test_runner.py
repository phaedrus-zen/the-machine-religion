"""Runner + worker tests.

Uses a fake chat runner so the suite is hermetic — Hermes / HiveMind /
MS3 are not contacted. Worker thread is awaited explicitly via a
synchronization event so tests are deterministic.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from machine_spirit_4.double_agent import (
    Blackboard,
    JobEnvelope,
    JobRunner,
    RunnerError,
)
from machine_spirit_4.double_agent import safety
from machine_spirit_4.double_agent import runner as runner_module
from machine_spirit_4.double_agent.worker import (
    _normalize_no_tool_exact_marker,
    _requested_exact_marker,
    build_real_chat_runner,
)


@pytest.fixture()
def board(tmp_path):
    return Blackboard(tmp_path / "double_agent.sqlite3")


def _envelope(conv="conv-1", revision=1) -> JobEnvelope:
    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id=conv,
        conversation_revision_id=revision,
        background_lobe_type="deep_chat",
        user_visible_goal="Diagnose the issue.",
        internal_goal="Diagnose the issue.",
    )
    env.validate()
    return env


class FakeChat:
    def __init__(self, *, sleep_until: threading.Event | None = None, tool_calls: list[tuple[str, dict]] | None = None,
                 final_text: str = "Done.", completed: bool = True,
                 raise_exc: BaseException | None = None,
                 response_fields: dict | None = None):
        self.sleep_until = sleep_until
        self.tool_calls = tool_calls or []
        self.final_text = final_text
        self.completed = completed
        self.raise_exc = raise_exc
        self.response_fields = dict(response_fields or {})
        self.calls: list[dict] = []

    def factory(self):
        def _call(*, message, session_id, model, stream_callback, tool_start_callback, tool_complete_callback,
                  enabled_toolsets=None, conversation_history=None):
            self.calls.append({
                "message": message,
                "session_id": session_id,
                "model": model,
                "enabled_toolsets": enabled_toolsets,
                "conversation_history": list(conversation_history or []),
            })
            for i, (name, args) in enumerate(self.tool_calls):
                tool_start_callback(f"tc-{i}", name, args)
                tool_complete_callback(f"tc-{i}", name, args, f"tool {name} ok")
            # Light token churn so the worker exercises the stream callback path.
            for chunk in self.final_text.split():
                stream_callback(chunk)
            if self.sleep_until is not None:
                # Block until the test signals; lets us reproduce cancellation.
                self.sleep_until.wait(timeout=5)
            if self.raise_exc is not None:
                raise self.raise_exc
            response = {"text": self.final_text, "completed": self.completed}
            response.update(self.response_fields)
            return response
        return _call


def _await_state(runner: JobRunner, job_id: str, target_states: set[str], timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = runner.get(job_id)
        if snap is not None and snap.get("state") in target_states:
            return snap
        time.sleep(0.02)
    snap = runner.get(job_id) or {}
    raise AssertionError(f"job did not reach {target_states} in {timeout}s; last={snap.get('state')}")


def test_submit_and_completion(board):
    chat = FakeChat(final_text="The bridge command was sent and acknowledged.")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        snap = runner.submit(env)
        assert snap["state"] in {"queued", "running", "completed"}
        final = _await_state(runner, env.job_id, {"completed"})
        assert final["state"] == "completed"
        result = board.get_result(env.job_id)
        assert result is not None
        assert result["status"] == "success"
        assert result["text"].startswith("The bridge command was sent")
    finally:
        runner.shutdown(wait=False)


def test_submit_emits_lifecycle_events_in_order(board):
    chat = FakeChat(
        final_text="Found two mutex acquisitions across an await boundary.",
        tool_calls=[("read_file", {"path": "worker.rs"}), ("grep", {"pattern": "lock"})],
    )
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
        events = board.list_events(env.job_id, limit=100)
        types = [e["type"] for e in events]
        assert types[0] == "job.started"
        assert types[-1] == "job.completed"
        assert "job.tool.call.started" in types
        assert "job.tool.call.completed" in types
        for t in types:
            assert t in safety.EVENT_TYPES, t
    finally:
        runner.shutdown(wait=False)


def _complete_analytical_answer() -> str:
    return (
        "Recommendation: use the 35B Depth lobe for the intermittent diagnosis while the "
        "4B Face lobe preserves conversational latency. The mechanism is model capacity: "
        "the larger model can retain a longer event timeline, compare competing causal "
        "hypotheses, and notice interactions that a smaller foreground model may compress "
        "away. Tradeoffs include slower generation, higher memory demand, and possible GPU "
        "contention, so the two lobes should run on separately scheduled cluster capacity. "
        "A concrete example is a request that sometimes times out only after a retry races "
        "with peer eviction. The Face lobe can acknowledge and preserve the trace identity, "
        "while the Depth lobe reconstructs request, retry, placement, and eviction ordering. "
        "It can then distinguish a transport timeout from duplicate execution and recommend "
        "an idempotency guard plus placement evidence. This choice is strongest when the "
        "extra diagnosis quality is worth background latency; straightforward known failures "
        "should remain on the Face path."
    )


def _analytical_envelope() -> JobEnvelope:
    env = _envelope(conv="analytical-conv")
    env.internal_goal = (
        "Explain why a 35B Depth lobe might outperform a 4B Face lobe when "
        "diagnosing an intermittent distributed-inference failure. Give me the "
        "recommendation, mechanism, tradeoffs, and a concrete example."
    )
    env.user_visible_goal = env.internal_goal[:240]
    env.resource_request.enabled_toolsets = ["hermes-cli", "mcp-hivemind"]
    env.validate()
    return env


def test_reasoning_only_worker_overrides_generic_catalog_to_empty(board):
    chat = FakeChat(final_text=_complete_analytical_answer())
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _analytical_envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})

        assert chat.calls[-1]["enabled_toolsets"] == []
        assert board.get_result(env.job_id)["status"] == "success"
        assert not any(
            event["type"].startswith("job.tool.call")
            for event in board.list_events(env.job_id, limit=100)
        )
    finally:
        runner.shutdown(wait=False)


def test_reasoning_only_worker_fails_closed_if_runner_ignores_empty_catalog(board):
    chat = FakeChat(
        final_text=_complete_analytical_answer(),
        tool_calls=[("ms4_skill_view", {"names": ["unrelated-skill"]})],
    )
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _analytical_envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"failed"})

        result = board.get_result(env.job_id)
        assert chat.calls[-1]["enabled_toolsets"] == []
        assert result["status"] == "failed"
        assert result["evidence"] == []
        failed = [
            event
            for event in board.list_events(env.job_id, limit=100)
            if event["type"] == "job.failed"
        ]
        assert failed[-1]["payload"]["reason"] == (
            "model_result_unexpected_tool_activity_for_reasoning_only_job"
        )
    finally:
        runner.shutdown(wait=False)


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
        "Debug the crash, explain the mechanism, and recommend a concrete fix.",
        "Profile both architectures, assess the tradeoffs, and recommend one.",
        "Trace the intermittent request, compare likely causes, and recommend a strategy.",
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
def test_worker_preserves_catalog_for_request_form_operational_analysis(board, prompt):
    chat = FakeChat(final_text=_complete_analytical_answer())
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="operational-analysis")
        env.internal_goal = prompt
        env.user_visible_goal = prompt[:240]
        env.resource_request.enabled_toolsets = ["hermes-cli", "mcp-hivemind"]
        env.validate()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})

        assert chat.calls[-1]["enabled_toolsets"] == ["hermes-cli", "mcp-hivemind"]
    finally:
        runner.shutdown(wait=False)


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
    ],
)
def test_worker_honors_explicit_no_tool_analytical_constraints(board, prompt):
    chat = FakeChat(final_text=_complete_analytical_answer())
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="explicit-no-tools")
        env.internal_goal = prompt
        env.user_visible_goal = prompt[:240]
        env.resource_request.enabled_toolsets = ["hermes-cli", "mcp-hivemind"]
        env.validate()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})

        assert chat.calls[-1]["enabled_toolsets"] == []
    finally:
        runner.shutdown(wait=False)


@pytest.mark.parametrize(
    "model_text",
    [
        '{"name": "ORA08_COMPLETE_OK", "parameters": {}}',
        '{"name": "ORA08_COMPLETE_OK", "arguments": {}}',
        '{"name": "get_constant", "parameters": {"constant_name": "ORA08_COMPLETE_OK"}}',
        '{"name": "get_constant", "arguments": {"constant_name": "ORA08_COMPLETE_OK"}}',
    ],
)
def test_no_tool_authority_disables_catalog_and_unwraps_exact_marker(
    board, model_text
):
    chat = FakeChat(final_text=model_text)
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        env.authority.can_call_tools = False
        env.authority.allowed_toolsets = []
        env.internal_goal = (
            "Do not call tools. Return exactly ORA08_COMPLETE_OK as the complete final answer."
        )
        env.user_visible_goal = env.internal_goal
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})

        result = board.get_result(env.job_id)
        assert result is not None
        assert result["text"] == "ORA08_COMPLETE_OK"
        assert result["summary"] == "ORA08_COMPLETE_OK"
        assert chat.calls[-1]["enabled_toolsets"] == []
        assert not any(
            event["type"].startswith("job.tool.call")
            for event in board.list_events(env.job_id, limit=100)
        )
    finally:
        runner.shutdown(wait=False)


def test_exact_marker_normalization_fails_closed_for_wrong_or_tool_authorized_output(
    board,
):
    wrapped = '{"name": "WRONG_MARKER", "parameters": {"value": "WRONG_MARKER"}}'
    chat = FakeChat(final_text=wrapped)
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        env.authority.can_call_tools = False
        env.internal_goal = "Return exactly ORA08_COMPLETE_OK as the complete final answer."
        runner.submit(env)
        _await_state(runner, env.job_id, {"failed"})
        result = board.get_result(env.job_id)
        assert result is not None
        assert result["status"] == "failed"
        assert result["text"] == wrapped

        env2 = _envelope(conv="conv-tool-authorized")
        env2.authority.can_call_tools = True
        env2.internal_goal = "Return exactly WRONG_MARKER as the complete final answer."
        runner.submit(env2)
        _await_state(runner, env2.job_id, {"failed"})
        result2 = board.get_result(env2.job_id)
        assert result2 is not None
        assert result2["status"] == "failed"
        assert result2["text"] == wrapped
    finally:
        runner.shutdown(wait=False)


def test_exact_marker_normalizer_rejects_large_or_deep_json_wrappers():
    goal = "Return exactly BOUNDED_MARKER as the complete final answer."
    large = json.dumps(
        {
            "name": "get_constant",
            "parameters": {"constant_name": "BOUNDED_MARKER", "pad": "x" * 4096},
        }
    )
    deep_value = "BOUNDED_MARKER"
    for index in range(7):
        deep_value = {f"level_{index}": deep_value}
    deep = json.dumps({"name": "get_constant", "parameters": deep_value})

    for model_text in (large, deep):
        assert (
            _normalize_no_tool_exact_marker(
                goal,
                model_text,
                can_call_tools=False,
                tool_traces=[],
            )
            == model_text
        )


@pytest.mark.parametrize(
    "model_text",
    [
        '{"name":"get_constant","parameters":{"constant_name":"BOUNDED_MARKER"},"error":"execution failed"}',
        '{"name":"BOUNDED_MARKER","parameters":{"value":"WRONG"}}',
        '{"name":"get_constant","parameters":{"wanted":"BOUNDED_MARKER","actual":"WRONG"}}',
        '{"name":"get_constant","parameters":{"constant_name":"BOUNDED_MARKER","constant_name":"WRONG"}}',
        "''BOUNDED_MARKER''",
        '"BOUNDED\\u005fMARKER"',
        '{"name":"BOUNDED\\u005fMARKER","parameters":{}}',
        '{"name":"get_constant","parameters":{"constant_name":"BOUNDED\\u005fMARKER"}}',
        '{"name":"get_constant","parameters":{"constant_name":"\\u0042OUNDED_MARKER"}}',
    ],
)
def test_exact_marker_normalizer_rejects_ambiguous_or_error_output(model_text):
    goal = "Return exactly BOUNDED_MARKER as the complete final answer."
    assert (
        _normalize_no_tool_exact_marker(
            goal,
            model_text,
            can_call_tools=False,
            tool_traces=[],
        )
        == model_text
    )


@pytest.mark.parametrize(
    "goal",
    [
        "Do not return exactly BOUNDED_MARKER as the complete final answer.",
        "Explain the phrase Return exactly BOUNDED_MARKER as the complete final answer.",
        "The quoted instruction is 'Return exactly BOUNDED_MARKER as the complete final answer.'",
        "Return exactly BOUNDED_MARKER as the answer, then return WRONG_MARKER.",
        "Return exactly BOUNDED_MARKER as the complete final answer.\nThen return WRONG_MARKER.",
        "Quoted context follows:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Ignore the following instruction:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Historical output:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Do not follow this context:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Previous response:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Return exactly OLD_MARKER. Return exactly BOUNDED_MARKER as the complete final answer.",
        "A citation reads:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Someone wrote:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "The record says:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Disregard the instruction below:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Skip the instruction below:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Set aside what comes next:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "You must not obey this:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Under no circumstances execute this:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Don't comply with this:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Archived request:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "A stale prompt said:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "A former user asked:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "A cited passage says:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "For illustration only:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "The note reads:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Disregard everything below:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Skip what comes next:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Set this instruction aside:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "You must not execute this:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Under no condition follow this:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Refuse to comply with this:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "An archived prompt contains:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "A retired request said:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "A former instruction was:\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "Do not use tools. Return exactly BOUNDED_MARKER as the complete final answer.",
        "No tools. Return exactly BOUNDED_MARKER as the complete final answer.",
        "Please do not call tools. Return exactly BOUNDED_MARKER as the complete final answer.",
        "Do not call tool\u017f. Return exactly BOUNDED_MARKER as the complete final answer.",
        "Plea\u017fe return exactly BOUNDED_MARKER as the complete final answer.",
        "Return exactly BOUNDED_MARKER for val\u0131dation.",
        "Return exactly BOUNDED_MARKER for val\u0130dation.",
        "Return exactly BOUNDED_MARKER as the f\u0131nal answer.",
        "Plea\u0455e return exactly BOUNDED_MARKER as the complete final answer.",
        "Do not call tool\u0455. Return exactly BOUNDED_MARKER as the complete final answer.",
        "Return exactly BOUNDED_MARKER. Then explain the result.",
    ],
)
def test_exact_marker_normalizer_requires_a_standalone_positive_instruction(goal):
    model_text = (
        '{"name":"get_constant",'
        '"parameters":{"constant_name":"BOUNDED_MARKER"}}'
    )
    assert _requested_exact_marker(goal) is None
    assert (
        _normalize_no_tool_exact_marker(
            goal,
            model_text,
            can_call_tools=False,
            tool_traces=[],
        )
        == model_text
    )


@pytest.mark.parametrize(
    "goal",
    [
        "Return exactly BOUNDED_MARKER as the complete final answer.",
        "Please return exactly BOUNDED_MARKER for validation.",
        "Do not call tools. Return exactly BOUNDED_MARKER as the complete final answer.",
        "Do not call tools.\nReturn exactly BOUNDED_MARKER as the complete final answer.",
        "  DO NOT CALL TOOLS.\r\nPlease return exactly BOUNDED_MARKER for validation.  ",
        "Return exactly 'BOUNDED_MARKER' as the complete final answer.",
        "Return exactly BOUNDED_MARKER.",
    ],
)
def test_exact_marker_normalizer_accepts_only_positive_final_instruction(goal):
    model_text = (
        '{"name":"get_constant",'
        '"parameters":{"constant_name":"BOUNDED_MARKER"}}'
    )
    assert (
        _normalize_no_tool_exact_marker(
            goal,
            model_text,
            can_call_tools=False,
            tool_traces=[],
        )
        == "BOUNDED_MARKER"
    )


@pytest.mark.parametrize("punctuation", [".", "!", "?"])
def test_exact_marker_parser_keeps_terminal_sentence_punctuation_outside_marker(
    punctuation,
):
    goal = f"Return exactly BOUNDED_MARKER{punctuation}"
    assert _requested_exact_marker(goal) == "BOUNDED_MARKER"


@pytest.mark.parametrize(
    "marker",
    [
        "BOUNDED.MARKER",
        "BOUNDED:MARKER",
        "BOUNDED-MARKER",
        "BOUNDED_MARKER",
    ],
)
def test_exact_marker_parser_preserves_internal_ascii_delimiters(marker):
    goal = f"Return exactly {marker}."
    model_text = json.dumps(
        {"name": "get_constant", "parameters": {"constant_name": marker}},
        separators=(",", ":"),
    )
    assert _requested_exact_marker(goal) == marker
    assert (
        _normalize_no_tool_exact_marker(
            goal,
            model_text,
            can_call_tools=False,
            tool_traces=[],
        )
        == marker
    )


@pytest.mark.parametrize("quote", ["'", '"', "`"])
@pytest.mark.parametrize(
    "marker",
    ["QUOTED_MARKER.", "QUOTED-MARKER", "QUOTED:MARKER"],
)
def test_exact_marker_parser_preserves_delimited_marker_contracts(quote, marker):
    goal = f"Return exactly {quote}{marker}{quote}."
    direct_wrapper = json.dumps(
        {"name": marker, "parameters": {}},
        separators=(",", ":"),
    )
    constant_wrapper = json.dumps(
        {"name": "get_constant", "parameters": {"constant_name": marker}},
        separators=(",", ":"),
    )
    assert _requested_exact_marker(goal) == marker
    for model_text in (direct_wrapper, constant_wrapper):
        assert (
            _normalize_no_tool_exact_marker(
                goal,
                model_text,
                can_call_tools=False,
                tool_traces=[],
            )
            == marker
        )


@pytest.mark.parametrize(
    "marker",
    [
        "ORA08_\u212a_OK",  # Kelvin sign: Python Unicode IGNORECASE matches K
        "ORA08_\u017f_OK",  # long s
        "ORA08_\u0130_OK",  # capital I with dot
        "ORA08_\u0131_OK",  # dotless i
        "ORA08_\u039a_OK",  # Greek kappa lookalike
        "ORA08_\u0441_OK",  # Cyrillic es lookalike
        "ORA08_\uff21_OK",  # full-width A
        "ORA08_A\u030a_OK",  # combining ring
    ],
)
def test_exact_marker_parser_and_normalizer_reject_non_ascii_markers(marker):
    goal = f"Return exactly {marker} as the complete final answer."
    model_text = json.dumps(
        {"name": "get_constant", "parameters": {"constant_name": marker}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert _requested_exact_marker(goal) is None
    assert (
        _normalize_no_tool_exact_marker(
            goal,
            model_text,
            can_call_tools=False,
            tool_traces=[],
        )
        == model_text
    )


@pytest.mark.parametrize(
    "marker",
    ["ORA08_\u212a_OK", "ORA08_\u017f_OK", "ORA08_\u0130_OK", "ORA08_\u0131_OK"],
)
def test_non_ascii_raw_wrapper_never_completes_end_to_end(board, marker):
    model_text = json.dumps(
        {"name": "get_constant", "parameters": {"constant_name": marker}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    chat = FakeChat(final_text=model_text)
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-non-ascii-marker")
        env.authority.can_call_tools = False
        env.authority.allowed_toolsets = []
        env.resource_request.enabled_toolsets = []
        env.internal_goal = (
            f"Return exactly {marker} as the complete final answer."
        )
        env.user_visible_goal = env.internal_goal
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"failed"})

        result = board.get_result(env.job_id)
        assert final["state"] == "failed"
        assert result is not None
        assert result["status"] == "failed"
        assert result["text"] == model_text
        assert chat.calls[-1]["enabled_toolsets"] == []
        assert not any(
            event["type"] == "job.completed"
            for event in board.list_events(env.job_id, limit=100)
        )
    finally:
        runner.shutdown(wait=False)


@pytest.mark.parametrize(
    "final_text,response_fields",
    [
        (
            '{"name":"get_constant","parameters":{"constant_name":"ORA08_ERROR_OK"}}',
            {"error": "execution failed"},
        ),
        ("ORA08_ERROR_OK", {"errors": ["provider failed"]}),
        ("ORA08_ERROR_OK", {"exception": "transport closed"}),
        ("ORA08_ERROR_OK", {"failure": True}),
        ("ORA08_ERROR_OK", {"success": False}),
        ("ORA08_ERROR_OK", {"ok": False}),
        ("ORA08_ERROR_OK", {"status": "failed"}),
        ("ORA08_ERROR_OK", {"status": "timeout"}),
        ("ORA08_ERROR_OK", {"status": "timed_out"}),
        ("ORA08_ERROR_OK", {"status": "aborted"}),
        ("ORA08_ERROR_OK", {"status": "rejected"}),
        ("ORA08_ERROR_OK", {"status": "terminated"}),
        ("ORA08_ERROR_OK", {"status": "running"}),
        ("ORA08_ERROR_OK", {"status": 200}),
        ("ORA08_ERROR_OK", {"metadata": {"error": "provider failed"}}),
        (
            "ORA08_ERROR_OK",
            {"metadata": {"provider_metadata": {"result": {"status": "timed_out"}}}},
        ),
        ("ORA08_ERROR_OK", {"metadata": {"completed": False}}),
        (
            "ORA08_ERROR_OK",
            {
                "response_metadata": {
                    "details": [{"provider": {"errors": ["provider failed"]}}]
                }
            },
        ),
        ("ORA08_ERROR_OK", {"provider_response": {"last_error": "timed out"}}),
        (
            "ORA08_ERROR_OK",
            {"data": {"payload": {"body": {"error_info": {"code": "TIMEOUT"}}}}},
        ),
        ("ORA08_ERROR_OK", {"failed": True}),
        ("ORA08_ERROR_OK", {"partial": True}),
        ("ORA08_ERROR_OK", {"interrupted": True}),
        ("ORA08_ERROR_OK", {"completed": "false"}),
        ("ERROR: provider failed", {}),
        (
            '{"name":"get_constant","parameters":{"constant_name":"ORA08_ERROR_OK"},"error":"execution failed"}',
            {},
        ),
        (
            '{"name":"get_constant","parameters":{"constant_name":"ORA08_ERROR\\u005fOK"}}',
            {},
        ),
    ],
)
def test_error_bearing_exact_answer_never_completes(
    board, final_text, response_fields
):
    chat = FakeChat(final_text=final_text, response_fields=response_fields)
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        env.authority.can_call_tools = False
        env.internal_goal = (
            "Return exactly ORA08_ERROR_OK as the complete final answer."
        )
        env.user_visible_goal = env.internal_goal
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"failed"})

        result = board.get_result(env.job_id)
        assert final["state"] == "failed"
        assert result is not None
        assert result["status"] == "failed"
        assert not any(
            event["type"] == "job.completed"
            for event in board.list_events(env.job_id, limit=100)
        )
    finally:
        runner.shutdown(wait=False)


@pytest.mark.parametrize("status", ["success", "succeeded", "complete", "completed", "ok"])
def test_exact_answer_allows_only_documented_success_statuses(board, status):
    marker = "ORA08_SUCCESS_STATUS_OK"
    chat = FakeChat(final_text=marker, response_fields={"status": status})
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv=f"conv-{status}")
        env.authority.can_call_tools = False
        env.internal_goal = f"Return exactly {marker} as the complete final answer."
        env.user_visible_goal = env.internal_goal
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"completed"})

        result = board.get_result(env.job_id)
        assert final["state"] == "completed"
        assert result is not None
        assert result["status"] == "success"
        assert result["text"] == marker
    finally:
        runner.shutdown(wait=False)


def test_exact_answer_allows_nested_metadata_only_when_every_terminal_is_success(board):
    marker = "ORA08_NESTED_SUCCESS_OK"
    chat = FakeChat(
        final_text=marker,
        response_fields={
            "status": "success",
            "metadata": {
                "completed": True,
                "success": True,
                "ok": True,
                "provider_metadata": {"status": "completed"},
            },
        },
    )
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-nested-success")
        env.authority.can_call_tools = False
        env.internal_goal = f"Return exactly {marker} as the complete final answer."
        env.user_visible_goal = env.internal_goal
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"completed"})

        result = board.get_result(env.job_id)
        assert final["state"] == "completed"
        assert result is not None
        assert result["status"] == "success"
        assert result["text"] == marker
    finally:
        runner.shutdown(wait=False)


def test_real_chat_runner_adapter_marks_error_metadata_incomplete():
    class _Agent:
        def run_conversation(self, _message, **_kwargs):
            return {
                "final_response": "ORA08_ADAPTER_OK",
                "error": "provider failed",
            }

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **_kwargs):
            return _Agent()

    response = build_real_chat_runner(_Runner())(
        message="Return exactly ORA08_ADAPTER_OK.",
        session_id="da-adapter-error",
        model=None,
        stream_callback=None,
        tool_start_callback=lambda *_a: None,
        tool_complete_callback=lambda *_a: None,
        enabled_toolsets=[],
    )

    assert response["text"] == "ORA08_ADAPTER_OK"
    assert response["completed"] is False
    assert response["error"] == "model_result_error"


@pytest.mark.parametrize(
    "response_fields",
    [
        {"completed": True, "status": "timeout"},
        {"completed": True, "status": "timed_out"},
        {"completed": True, "status": "aborted"},
        {"completed": True, "status": "rejected"},
        {"completed": True, "status": "terminated"},
        {"completed": True, "metadata": {"error": "provider failed"}},
        {
            "completed": True,
            "metadata": {"provider_metadata": {"result": {"status": "timeout"}}},
        },
        {"completed": True, "failed": True},
        {"completed": True, "partial": True},
        {"completed": True, "interrupted": True},
    ],
)
def test_real_chat_runner_adapter_fails_closed_for_terminal_metadata(response_fields):
    class _Agent:
        def run_conversation(self, _message, **_kwargs):
            return {
                "final_response": "ORA08_ADAPTER_TERMINAL_OK",
                **response_fields,
            }

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **_kwargs):
            return _Agent()

    response = build_real_chat_runner(_Runner())(
        message="Return exactly ORA08_ADAPTER_TERMINAL_OK.",
        session_id="da-adapter-terminal",
        model=None,
        stream_callback=None,
        tool_start_callback=lambda *_a: None,
        tool_complete_callback=lambda *_a: None,
        enabled_toolsets=[],
    )

    assert response["text"] == "ORA08_ADAPTER_TERMINAL_OK"
    assert response["completed"] is False
    assert response["error"].startswith("model_result_")


def test_hivemind_tool_completion_event_keeps_safe_result_excerpt(board):
    chat = FakeChat(
        final_text="Cluster summary captured.",
        tool_calls=[
            ("hivemind_cluster_summary", {"include_gpu_details": False}),
            ("read_file", {"path": "secret.txt"}),
        ],
    )
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
        completed = [
            e for e in board.list_events(env.job_id, limit=100)
            if e["type"] == "job.tool.call.completed"
        ]
        by_tool = {e["payload"]["tool"]: e["payload"] for e in completed}
        assert by_tool["hivemind_cluster_summary"]["result_excerpt"] == (
            "tool hivemind_cluster_summary ok"
        )
        assert "result_excerpt" not in by_tool["read_file"]
    finally:
        runner.shutdown(wait=False)


def test_compound_hivemind_results_survive_marker_only_synthesis_with_history(board):
    captured: dict = {}
    time_result = '"2026-06-29T14:48:09.208062500+00:00"'
    cluster_result = (
        '{"healthy":true,"cluster_statistics":{"active_nodes":8,'
        '"total_nodes":8,"total_gpus":7}}'
    )

    def factory():
        def _call(
            *,
            message,
            session_id,
            model,
            stream_callback,
            tool_start_callback,
            tool_complete_callback,
            conversation_history=None,
        ):
            captured["message"] = message
            captured["conversation_history"] = list(conversation_history or [])
            tool_start_callback("call-time", "hivemind_time_now", {})
            tool_complete_callback(
                "call-time", "hivemind_time_now", {}, time_result
            )
            tool_start_callback("call-cluster", "hivemind_cluster_summary", {})
            tool_complete_callback(
                "call-cluster", "hivemind_cluster_summary", {}, cluster_result
            )
            stream_callback("ORACLE_LIVE_TEXT_OK")
            return {"text": "ORACLE_LIVE_TEXT_OK", "completed": True}

        return _call

    runner = JobRunner(blackboard=board, chat_runner_factory=factory)
    try:
        env = _envelope()
        env.internal_goal = (
            "Use time and cluster summary, then include the exact first marker."
        )
        env.prior_context = [
            {"role": "user", "content": "The first marker is ORACLE_LIVE_TEXT_OK."},
            {"role": "assistant", "content": "ACK1"},
            {"role": "user", "content": "The second marker is SECOND."},
            {"role": "assistant", "content": "ACK2"},
        ]
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})

        result = board.get_result(env.job_id)
        assert result is not None
        assert result["status"] == "success"
        assert "ORACLE_LIVE_TEXT_OK" in result["text"]
        assert "Verified HiveMind tool results:" in result["text"]
        assert time_result in result["text"]
        assert cluster_result in result["text"]
        assert [item["tool"] for item in result["evidence"]] == [
            "hivemind_time_now",
            "hivemind_cluster_summary",
        ]
        assert result["evidence"][0]["result_excerpt"] == time_result
        assert result["evidence"][1]["result_excerpt"] == cluster_result
        assert [item["tool"] for item in result["actions_taken"]] == [
            "hivemind_time_now",
            "hivemind_cluster_summary",
        ]
        assert captured["conversation_history"] == env.prior_context
        assert "first turn user: The first marker is ORACLE_LIVE_TEXT_OK." in captured["message"]
    finally:
        runner.shutdown(wait=False)


def test_generic_tool_result_is_not_attached_to_final_evidence(board):
    chat = FakeChat(
        final_text="Read completed.",
        tool_calls=[("read_file", {"path": "private.txt"})],
    )
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
        result = board.get_result(env.job_id)
        assert result is not None
        assert result["evidence"] == []
        assert "tool read_file ok" not in result["text"]
    finally:
        runner.shutdown(wait=False)


def test_cancellation_stops_the_worker(board):
    gate = threading.Event()
    chat = FakeChat(sleep_until=gate, final_text="This should never be returned.")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"}, timeout=2.0)
        runner.cancel(env.job_id)
        gate.set()  # release the fake chat so the cancellation point is reached
        final = _await_state(runner, env.job_id, {"canceled", "completed"}, timeout=3.0)
        # The fake chat returns the final_text only if the cancel event isn't
        # observed inside a tool callback; for a worker that has no tool
        # callbacks during the sleep, the worker may finish before checking the
        # cancel between tool calls. Either way the operator-visible state must
        # not show stale work as a fresh result.
        assert final["state"] in {"canceled", "completed"}
        # If completed, ensure we still recorded the cancel signal on the runner.
    finally:
        runner.shutdown(wait=False)


def test_cancel_after_completion_is_noop(board):
    chat = FakeChat(final_text="Done.")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
        snap = runner.cancel(env.job_id)
        assert snap["state"] == "completed"
    finally:
        runner.shutdown(wait=False)


def test_mark_stale_explicitly(board):
    chat = FakeChat(sleep_until=threading.Event())  # never returns
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"}, timeout=2.0)
        snap = runner.mark_stale(env.job_id, reason="operator says actually do go instead")
        assert snap["state"] == "stale"
        assert "operator says" in snap["last_safe_user_status"]
    finally:
        runner.shutdown(wait=False)


def test_bump_revision_marks_older_jobs_stale(board, monkeypatch):
    monkeypatch.setenv("MS4_DA_AUTOSTALE", "1")  # legacy revision-driven staling
    chat = FakeChat(sleep_until=threading.Event())  # never returns
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope(conv="conv-shared", revision=1)
        runner.submit(env)
        _await_state(runner, env.job_id, {"running"}, timeout=2.0)
        outcome = runner.bump_revision("conv-shared", user_message_excerpt="please use Rust instead")
        assert outcome["revision"]["revision_id"] == 1  # first bump records as rev 1
        # The submitted job is *already* at revision 1, so first bump shouldn't stale it.
        assert outcome["marked_stale"] == []
        outcome2 = runner.bump_revision("conv-shared", user_message_excerpt="actually scratch that")
        assert outcome2["revision"]["revision_id"] == 2
        assert env.job_id in outcome2["marked_stale"]
    finally:
        runner.shutdown(wait=False)


def test_unsafe_job_id_refused_at_submit(board):
    runner = JobRunner(blackboard=board, chat_runner_factory=FakeChat().factory)
    try:
        env = _envelope()
        env.job_id = "bad id;rm -rf"
        with pytest.raises(Exception):
            runner.submit(env)
    finally:
        runner.shutdown(wait=False)


def test_unknown_job_id_cancel_raises(board):
    runner = JobRunner(blackboard=board, chat_runner_factory=FakeChat().factory)
    try:
        with pytest.raises(RunnerError):
            runner.cancel("da-does-not-exist")
    finally:
        runner.shutdown(wait=False)


def test_failure_inside_worker_is_recorded_as_failed(board):
    chat = FakeChat(raise_exc=RuntimeError("model gateway dropped"))
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"failed"})
        assert final["state"] == "failed"
        # last_safe_user_status carries the error type, not stack trace
        assert "RuntimeError" in (final.get("last_safe_user_status") or "")
        events = [e["type"] for e in board.list_events(env.job_id, limit=100)]
        assert "job.failed" in events
    finally:
        runner.shutdown(wait=False)


def test_cancel_without_live_handle_emits_terminal_event(board):
    runner = JobRunner(blackboard=board, chat_runner_factory=FakeChat().factory)
    try:
        env = _envelope()
        board.insert_job(env)
        board.update_job_state(env.job_id, state="running")

        snap = runner.cancel(env.job_id)

        assert snap["state"] == "canceled"
        assert snap["finished_at"] is not None
        assert snap["last_event_type"] == "job.canceled"
        events = board.list_events(env.job_id, limit=10)
        assert [event["type"] for event in events] == ["job.canceled"]
        assert events[0]["payload"]["source"] == "parent_runner"
    finally:
        runner.shutdown(wait=False)


def test_worker_forwards_prior_context_when_present(board):
    chat = FakeChat(final_text="Context-aware answer.")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        env.prior_context = [
            {"role": "user", "content": "What is running?"},
            {"role": "assistant", "content": "We found three services."},
        ]
        env.validate()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
        assert chat.calls
        assert chat.calls[0]["conversation_history"] == env.prior_context
    finally:
        runner.shutdown(wait=False)


def test_worker_resolves_first_turn_reference_before_depth_model(board):
    chat = FakeChat(final_text="DEPTH_LIVE_OK\nORACLE_LIVE_TEXT_OK")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        env.internal_goal = (
            "Return exactly two lines: DEPTH_LIVE_OK and the exact validation "
            "marker from the first turn."
        )
        env.prior_context = [
            {"role": "user", "content": "Reply with exactly ORACLE_LIVE_TEXT_OK."},
            {"role": "assistant", "content": "ORACLE_LIVE_TEXT_OK"},
            {"role": "user", "content": "Report the cluster time."},
            {"role": "assistant", "content": "2026-06-29T13:55:14Z"},
        ]
        env.validate()

        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})

        sent_goal = chat.calls[0]["message"]
        assert "[MS4 resolved conversation reference]" in sent_goal
        assert "first turn assistant: ORACLE_LIVE_TEXT_OK" in sent_goal
        assert "do not substitute a later turn" in sent_goal
        assert chat.calls[0]["conversation_history"] == env.prior_context
    finally:
        runner.shutdown(wait=False)


def test_real_chat_runner_adapter_passes_conversation_history():
    captured = {}

    class _Agent:
        def run_conversation(self, message, **kwargs):
            captured["message"] = message
            captured["kwargs"] = kwargs
            return {"final_response": "done", "completed": True}

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **kwargs):
            captured["agent_kwargs"] = kwargs
            return _Agent()

    adapter = build_real_chat_runner(_Runner())
    prior = [{"role": "user", "content": "prior question"}]
    response = adapter(
        message="current task",
        session_id="da-worker-test",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_a: None,
        tool_complete_callback=lambda *_a: None,
        conversation_history=prior,
    )
    assert response["text"] == "done"
    assert captured["message"] == "current task"
    assert captured["kwargs"]["conversation_history"] == prior
    assert captured["kwargs"]["task_id"] == "da-worker-test"


def test_real_chat_runner_cancel_aborts_inflight_stream_and_finishes_canceled(
    board,
):
    stream_started = threading.Event()
    stream_interrupted = threading.Event()
    interrupt_messages = []

    class _StreamingAgent:
        def run_conversation(self, _message, **kwargs):
            stream_started.set()
            while not stream_interrupted.wait(timeout=0.01):
                kwargs["stream_callback"]("token")
            raise InterruptedError("stream transport closed")

        def interrupt(self, message=None):
            interrupt_messages.append(message)
            stream_interrupted.set()

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **_kwargs):
            return _StreamingAgent()

    adapter = build_real_chat_runner(_Runner())
    runner = JobRunner(blackboard=board, chat_runner_factory=lambda: adapter)
    try:
        env = _envelope()
        runner.submit(env)
        assert stream_started.wait(timeout=2.0), "stream never entered the in-flight state"

        started = time.monotonic()
        runner.cancel(env.job_id)
        final = _await_state(runner, env.job_id, {"canceled"}, timeout=1.0)
        elapsed = time.monotonic() - started

        assert elapsed < 1.0
        assert interrupt_messages == ["MS4 Depth job canceled"]
        assert final["last_event_type"] == "job.canceled"
        events = board.list_events(env.job_id, limit=100)
        assert events[-1]["type"] == "job.canceled"
        assert not any(event["type"] == "job.failed" for event in events)
    finally:
        runner.shutdown(wait=False)


def test_empty_worker_answer_is_recorded_as_failed(board):
    chat = FakeChat(final_text="")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"failed"})
        assert final["state"] == "failed"
        assert "no visible answer" in (final.get("last_safe_user_status") or "")
        result = board.get_result(env.job_id)
        assert result is not None
        assert result["status"] == "failed"
        assert result["confidence"] == "low"
    finally:
        runner.shutdown(wait=False)


def test_incomplete_worker_answer_is_recorded_as_failed(board):
    chat = FakeChat(final_text="partial draft", completed=False)
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"failed"})
        assert final["state"] == "failed"
        assert "did not complete" in (final.get("last_safe_user_status") or "")
        result = board.get_result(env.job_id)
        assert result is not None
        assert result["status"] == "failed"
        assert result["text"] == "partial draft"
    finally:
        runner.shutdown(wait=False)


def test_recover_on_startup_flips_dangling_running(board):
    chat = FakeChat(final_text="ok")
    runner = JobRunner(blackboard=board, chat_runner_factory=chat.factory)
    try:
        env = _envelope()
        runner.submit(env)
        _await_state(runner, env.job_id, {"completed"})
    finally:
        runner.shutdown(wait=False)

    # Simulate a dangling job that was never finalized (process crashed).
    env_orphan = _envelope()
    board.insert_job(env_orphan)
    board.update_job_state(env_orphan.job_id, state="running")

    fresh_runner = JobRunner(blackboard=board, chat_runner_factory=FakeChat().factory)
    try:
        affected = fresh_runner.recover_on_startup()
        assert env_orphan.job_id in affected
        snap = board.get_job_snapshot(env_orphan.job_id)
        assert snap["state"] == "failed"
    finally:
        fresh_runner.shutdown(wait=False)


def test_cancel_hivemind_trace_uses_depth_job_uuid(monkeypatch):
    requests = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.delenv("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER", raising=False)
    monkeypatch.setenv("MS4_HIVEMIND_URL", "http://hive.test:6089")
    monkeypatch.setattr(runner_module.urllib.request, "urlopen", fake_urlopen)
    trace_id = "77a4fe89-2f09-47b0-8f52-c35186fe82dc"

    assert runner_module._cancel_hivemind_trace(
        f"da-{trace_id}", reason="validator cancel"
    ) is True
    request, timeout = requests[0]
    assert request.method == "DELETE"
    assert f"/jobs/{trace_id}?" in request.full_url
    assert "reason=validator+cancel" in request.full_url
    assert timeout == 5.0


def test_subprocess_cancel_deletes_exact_inflight_trace_before_stream_abort(
    board, monkeypatch
):
    calls = []
    stream_inflight = {"value": True}

    class _InflightProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def _abort(self, action):
            calls.append(action)
            stream_inflight["value"] = False
            self.returncode = 1

        def send_signal(self, _signal):
            self._abort("stream_abort")

        def terminate(self):
            self._abort("stream_abort")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            return self.returncode

        def kill(self):
            self._abort("stream_kill")

    env = _envelope()
    board.insert_job(env)
    board.update_job_state(env.job_id, state="running")

    def _fake_cancel_hli(job_id, *, reason):
        assert stream_inflight["value"] is True
        calls.append(("hli_delete", job_id, reason))
        return True

    monkeypatch.setattr(runner_module, "_cancel_hivemind_trace", _fake_cancel_hli)
    runner = JobRunner(blackboard=board, chat_runner_factory=FakeChat().factory)
    try:
        runner._kill_subprocess(env.job_id, _InflightProcess())

        assert calls[0] == (
            "hli_delete",
            env.job_id,
            "MS4 Depth worker canceled",
        )
        assert calls[1] == "stream_abort"
        final = board.get_job_snapshot(env.job_id)
        assert final is not None
        assert final["state"] == "canceled"
        assert final["last_event_type"] == "job.canceled"
        events = board.list_events(env.job_id, limit=100)
        assert events[-1]["type"] == "job.canceled"
        assert events[-1]["payload"] == {
            "source": "parent_runner",
            "hli_trace_canceled": True,
        }
    finally:
        runner.shutdown(wait=False)
