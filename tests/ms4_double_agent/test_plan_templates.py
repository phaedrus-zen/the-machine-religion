"""Depth-job plan template tests (gamestream_benchmark).

Hermetic — a FakeTools object stands in for the typed HiveMind
wrappers, so neither HiveMind nor the menta_game_session orchestrator
is contacted. Mirrors the suite style of test_runner.py (thread
backend + explicit await loops, no sleeps for correctness).
"""

from __future__ import annotations

import json
import time

import pytest

from machine_spirit_4.double_agent import (
    Blackboard,
    JobEnvelope,
    JobRunner,
    safety,
)
from machine_spirit_4.double_agent import plan_templates
from machine_spirit_4.double_agent.plan_templates import (
    GAMESTREAM_LOBE_TYPE,
    MUTATIONS_ENV,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    PlanTemplateError,
    build_gamestream_benchmark_plan,
    build_gamestream_chat_runner,
    execute_gamestream_benchmark_plan,
    parse_goal_params,
    render_result_text,
)


EXPECTED_STEP_IDS = [
    "gpu.discover",
    "vm.gpu_p.provision",
    "benchmark.plan",
    "benchmark.run",
    "benchmark.results",
    "stream.moonlight",
]


class FakeTools:
    """Stands in for machine_spirit_4.gateway.hivemind_tools. Records
    every call; per-method canned responses / exceptions."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.raise_on: dict[str, Exception] = {}
        self.responses: dict[str, object] = {
            "list_capable_gpus": {"gpus": [{"uuid": "GPU-abc", "host": "desktop-1"}]},
            "start_benchmark": {"job_id": "gs-job-1", "plan": {"job_id": "gs-job-1"}},
            "run": {"final_state": "COMPLETE", "dry_run": True},
            "get_results": {
                "benchmark_summary": {"avg_fps": 101.5, "valid_count": 1},
                "benchmark_report": "all good",
            },
            "moonlight": {"ok": True, "verified": True},
            "create_vm": {"create": {"ok": True}, "deploy": {"ok": True}},
        }

    def _record(self, name: str, kwargs: dict):
        self.calls.append((name, kwargs))
        if name in self.raise_on:
            raise self.raise_on[name]

    def psykyo_game_session_list_capable_gpus(self, url, *, host_filter=None):
        self._record("list_capable_gpus", {"url": url, "host_filter": host_filter})
        return self.responses["list_capable_gpus"]

    def psykyo_game_session_start_benchmark(self, url, *, game, **kwargs):
        self._record("start_benchmark", {"url": url, "game": game, **kwargs})
        return self.responses["start_benchmark"]

    def game_session_run(self, url, *, job_id):
        self._record("run", {"url": url, "job_id": job_id})
        return self.responses["run"]

    def psykyo_game_session_get_results(self, url, *, job_id):
        self._record("get_results", {"url": url, "job_id": job_id})
        return self.responses["get_results"]

    def moonlight_stream(self, url, *, host, **kwargs):
        self._record("moonlight", {"url": url, "host": host, **kwargs})
        return self.responses["moonlight"]

    def create_game_stream_vm(self, url, *, name, confirm):
        self._record("create_vm", {"url": url, "name": name, "confirm": confirm})
        return self.responses["create_vm"]

    def called(self, name: str) -> list[dict]:
        return [kw for n, kw in self.calls if n == name]


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def test_plan_defaults_to_dry_run_and_flagship_chain():
    plan = build_gamestream_benchmark_plan()
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["template"] == GAMESTREAM_LOBE_TYPE
    assert plan["dry_run"] is True
    step_ids = [s["step_id"] for s in plan["steps"]]
    assert step_ids == EXPECTED_STEP_IDS
    by_id = {s["step_id"]: s for s in plan["steps"]}
    # GPU-P partial-GPU lane: the prebuilt template id is load-bearing.
    vm_step = by_id["vm.gpu_p.provision"]
    assert vm_step["kind"] == "mutating"
    assert vm_step["tool"] == "hivemind.vm.create_prebuilt@v1"
    assert vm_step["args"]["vm_type"] == "windows_game_stream_prebuilt"
    # Benchmark lane: psykyo game_session family.
    assert by_id["benchmark.plan"]["tool"] == (
        "hivemind.psykyo.game_session.start_benchmark@v1"
    )
    assert by_id["benchmark.plan"]["args"]["mode"] == "benchmark"
    assert by_id["benchmark.run"]["tool"] == "hivemind.game_session.run@v1"
    assert by_id["benchmark.results"]["tool"] == (
        "hivemind.psykyo.game_session.get_results@v1"
    )
    # Stream-back lane.
    moon = by_id["stream.moonlight"]
    assert moon["kind"] == "mutating"
    assert moon["tool"] == "hivemind.moonlight.stream@v1"


def test_plan_threads_pins_and_clamps_runs():
    plan = build_gamestream_benchmark_plan(
        game="cp2077",
        host="desktop-1",
        gpu_uuid="GPU-abc",
        benchmark_runs=999,  # clamped to 50
        quality="low",
    )
    by_id = {s["step_id"]: s for s in plan["steps"]}
    bench = by_id["benchmark.plan"]["args"]
    assert bench["game"] == "cp2077"
    assert bench["host"] == "desktop-1"
    assert bench["gpu_uuid"] == "GPU-abc"
    assert bench["benchmark_runs"] == 50
    assert bench["quality"] == "low"
    assert by_id["gpu.discover"]["args"]["host_filter"] == "desktop-1"
    # Explicit host flows to the moonlight leg too.
    assert by_id["stream.moonlight"]["args"]["host"] == "desktop-1"


def test_plan_requires_game_and_vm_name():
    with pytest.raises(PlanTemplateError):
        build_gamestream_benchmark_plan(game="  ")
    with pytest.raises(PlanTemplateError):
        build_gamestream_benchmark_plan(vm_name="")


def test_lobe_type_is_in_safety_allowlist():
    assert GAMESTREAM_LOBE_TYPE in safety.BACKGROUND_LOBE_TYPES
    assert safety.is_safe_background_lobe_type(GAMESTREAM_LOBE_TYPE)


# ---------------------------------------------------------------------------
# Goal-string parsing
# ---------------------------------------------------------------------------


def test_parse_goal_params_bare_json():
    params = parse_goal_params(
        json.dumps({"game": "cp2077", "benchmark_runs": 3, "requires_stream": False})
    )
    assert params == {"game": "cp2077", "benchmark_runs": 3, "requires_stream": False}


def test_parse_goal_params_embedded_json_and_unknown_keys():
    goal = 'Benchmark the new card: {"game": "cyberpunk-2077", "gpu_uuid": "GPU-xyz", "rm_rf": true} thanks'
    params = parse_goal_params(goal)
    assert params == {"game": "cyberpunk-2077", "gpu_uuid": "GPU-xyz"}


def test_parse_goal_params_garbage_degrades_to_defaults():
    assert parse_goal_params("just do the gamestream demo") == {}
    assert parse_goal_params("") == {}
    assert parse_goal_params(None) == {}
    # A JSON list is not a parameter object.
    assert parse_goal_params('["game"]') == {}


# ---------------------------------------------------------------------------
# Dry-run execution
# ---------------------------------------------------------------------------


def test_dry_run_executes_only_safe_steps_and_records_would_call(monkeypatch):
    monkeypatch.delenv(MUTATIONS_ENV, raising=False)
    fake = FakeTools()
    plan = build_gamestream_benchmark_plan()
    result = execute_gamestream_benchmark_plan(
        plan, hivemind_url="http://hive", tools=fake
    )
    assert result["schema"] == RESULT_SCHEMA
    assert result["dry_run"] is True
    called = [n for n, _ in fake.calls]
    assert called == ["list_capable_gpus", "start_benchmark", "run", "get_results"]
    # Mutating steps were never executed...
    assert fake.called("create_vm") == []
    assert fake.called("moonlight") == []
    # ...but are honestly recorded as would_call evidence.
    by_id = {s["step_id"]: s for s in result["steps"]}
    assert by_id["vm.gpu_p.provision"]["status"] == "skipped_dry_run"
    assert by_id["vm.gpu_p.provision"]["would_call"] == "hivemind.vm.create_prebuilt@v1"
    assert by_id["stream.moonlight"]["status"] == "skipped_dry_run"
    # job_id threaded from benchmark.plan into run/results.
    assert result["benchmark_job_id"] == "gs-job-1"
    assert fake.called("run")[0]["job_id"] == "gs-job-1"
    assert fake.called("get_results")[0]["job_id"] == "gs-job-1"
    assert result["final_state"] == "COMPLETE"
    assert result["benchmark_summary"]["avg_fps"] == 101.5
    assert result["errors"] == []


def test_dry_run_emits_tool_events_for_every_step(monkeypatch):
    monkeypatch.delenv(MUTATIONS_ENV, raising=False)
    fake = FakeTools()
    starts: list[tuple[str, str]] = []
    completes: list[tuple[str, str]] = []
    plan = build_gamestream_benchmark_plan()
    execute_gamestream_benchmark_plan(
        plan,
        hivemind_url="http://hive",
        tools=fake,
        tool_start_callback=lambda sid, tool, args: starts.append((sid, tool)),
        tool_complete_callback=lambda sid, tool, args, res: completes.append((sid, tool)),
    )
    # Every step (including dry-run-skipped mutations) narrates start +
    # complete so the UI timeline shows the full chain.
    assert [sid for sid, _ in starts] == EXPECTED_STEP_IDS
    assert [sid for sid, _ in completes] == EXPECTED_STEP_IDS


def test_callback_exceptions_propagate_for_cancellation(monkeypatch):
    monkeypatch.delenv(MUTATIONS_ENV, raising=False)

    class Canceled(RuntimeError):
        pass

    fake = FakeTools()
    seen: list[str] = []

    def _start(sid, tool, args):
        seen.append(sid)
        if sid == "benchmark.plan":
            raise Canceled()

    plan = build_gamestream_benchmark_plan()
    with pytest.raises(Canceled):
        execute_gamestream_benchmark_plan(
            plan, hivemind_url="http://hive", tools=fake,
            tool_start_callback=_start,
        )
    # The chain stopped at the cancellation point.
    assert seen[-1] == "benchmark.plan"
    assert fake.called("start_benchmark") == []


def test_benchmark_plan_failure_aborts_chain(monkeypatch):
    monkeypatch.delenv(MUTATIONS_ENV, raising=False)
    fake = FakeTools()
    fake.raise_on["start_benchmark"] = RuntimeError("orchestrator down")
    plan = build_gamestream_benchmark_plan()
    with pytest.raises(PlanTemplateError, match="benchmark.plan failed"):
        execute_gamestream_benchmark_plan(
            plan, hivemind_url="http://hive", tools=fake
        )
    # Nothing past the dead step executed.
    assert fake.called("run") == []
    assert fake.called("get_results") == []


def test_missing_job_id_skips_dependent_steps(monkeypatch):
    monkeypatch.delenv(MUTATIONS_ENV, raising=False)
    fake = FakeTools()
    fake.responses["start_benchmark"] = {"plan": {}}  # no job_id anywhere
    plan = build_gamestream_benchmark_plan()
    result = execute_gamestream_benchmark_plan(
        plan, hivemind_url="http://hive", tools=fake
    )
    by_id = {s["step_id"]: s for s in result["steps"]}
    assert by_id["benchmark.run"]["status"] == "skipped_dependency"
    assert by_id["benchmark.results"]["status"] == "skipped_dependency"
    assert fake.called("run") == []
    assert result["benchmark_job_id"] is None


def test_read_only_step_failure_is_fail_soft(monkeypatch):
    monkeypatch.delenv(MUTATIONS_ENV, raising=False)
    fake = FakeTools()
    fake.raise_on["list_capable_gpus"] = RuntimeError("syncdb cold")
    plan = build_gamestream_benchmark_plan()
    result = execute_gamestream_benchmark_plan(
        plan, hivemind_url="http://hive", tools=fake
    )
    by_id = {s["step_id"]: s for s in result["steps"]}
    assert by_id["gpu.discover"]["status"] == "error"
    # The benchmark chain still ran.
    assert result["benchmark_job_id"] == "gs-job-1"
    assert any("gpu.discover" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Live-mode gating
# ---------------------------------------------------------------------------


def test_live_mode_refused_without_env_gate(monkeypatch):
    monkeypatch.delenv(MUTATIONS_ENV, raising=False)
    fake = FakeTools()
    plan = build_gamestream_benchmark_plan(dry_run=False)
    with pytest.raises(PlanTemplateError, match=MUTATIONS_ENV):
        execute_gamestream_benchmark_plan(
            plan, hivemind_url="http://hive", tools=fake
        )
    assert fake.calls == []  # refused before any tool call


def test_live_mode_with_env_gate_executes_mutations(monkeypatch):
    monkeypatch.setenv(MUTATIONS_ENV, "1")
    fake = FakeTools()
    plan = build_gamestream_benchmark_plan(dry_run=False, host="desktop-1")
    result = execute_gamestream_benchmark_plan(
        plan, hivemind_url="http://hive", tools=fake
    )
    assert result["dry_run"] is False
    assert fake.called("create_vm")[0]["confirm"] is True
    assert fake.called("create_vm")[0]["name"] == plan["vm_name"]
    assert fake.called("moonlight")[0]["host"] == "desktop-1"
    by_id = {s["step_id"]: s for s in result["steps"]}
    assert by_id["vm.gpu_p.provision"]["status"] == "ok"
    assert by_id["stream.moonlight"]["status"] == "ok"


def test_live_mode_unresolved_moonlight_host_fails_loudly(monkeypatch):
    monkeypatch.setenv(MUTATIONS_ENV, "1")
    fake = FakeTools()
    plan = build_gamestream_benchmark_plan(dry_run=False)  # no host pin
    with pytest.raises(PlanTemplateError, match="moonlight host unresolved"):
        execute_gamestream_benchmark_plan(
            plan, hivemind_url="http://hive", tools=fake
        )
    assert fake.called("moonlight") == []


# ---------------------------------------------------------------------------
# Worker integration (thread backend, mirrors test_runner.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def board(tmp_path):
    return Blackboard(tmp_path / "double_agent.sqlite3")


def _await_state(runner: JobRunner, job_id: str, target_states: set[str], timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = runner.get(job_id)
        if snap is not None and snap.get("state") in target_states:
            return snap
        time.sleep(0.02)
    snap = runner.get(job_id) or {}
    raise AssertionError(
        f"job did not reach {target_states} in {timeout}s; last={snap.get('state')}"
    )


def test_gamestream_job_end_to_end_thread_backend(board, monkeypatch):
    monkeypatch.delenv(MUTATIONS_ENV, raising=False)
    fake = FakeTools()
    runner = JobRunner(
        blackboard=board,
        chat_runner_factory=lambda: build_gamestream_chat_runner(
            hivemind_url="http://hive", tools=fake
        ),
    )
    try:
        env = JobEnvelope(
            job_id=safety.new_job_id(),
            parent_conversation_id="conv-gs",
            conversation_revision_id=1,
            background_lobe_type=GAMESTREAM_LOBE_TYPE,
            user_visible_goal="Gamestream benchmark demo (dry-run).",
            internal_goal=json.dumps({"game": "cyberpunk-2077", "benchmark_runs": 2}),
        )
        env.validate()
        runner.submit(env)
        final = _await_state(runner, env.job_id, {"completed"})
        assert final["state"] == "completed"
        result = board.get_result(env.job_id)
        assert result is not None
        assert result["status"] == "success"
        assert "DRY-RUN" in result["text"]
        assert "gs-job-1" in result["text"]
        # The goal's JSON params reached the orchestrator call.
        assert fake.called("start_benchmark")[0]["benchmark_runs"] == 2
        # Step events flowed through the standard allowlisted machinery.
        events = board.list_events(env.job_id, limit=100)
        types = [e["type"] for e in events]
        assert types[0] == "job.started"
        assert types[-1] == "job.completed"
        assert "job.tool.call.started" in types
        assert "job.tool.call.completed" in types
        for t in types:
            assert t in safety.EVENT_TYPES, t
        # Mutating dry-run steps narrate their would_call in the events.
        tool_events = [
            e for e in events if e["type"] == "job.tool.call.started"
        ]
        narrated_tools = {e["payload"].get("tool") for e in tool_events}
        assert "hivemind.vm.create_prebuilt@v1" in narrated_tools
        assert "hivemind.moonlight.stream@v1" in narrated_tools
    finally:
        runner.shutdown(wait=False)


def test_render_result_text_mentions_mode_and_steps():
    text = render_result_text(
        {
            "dry_run": True,
            "game": "cyberpunk-2077",
            "vm_name": "ms4-gamestream-bench",
            "benchmark_job_id": "gs-job-9",
            "final_state": "COMPLETE",
            "benchmark_summary": {"avg_fps": 99.0},
            "steps": [
                {"step_id": "gpu.discover", "status": "ok", "tool": "t1"},
                {
                    "step_id": "stream.moonlight",
                    "status": "skipped_dry_run",
                    "would_call": "hivemind.moonlight.stream@v1",
                },
            ],
            "errors": [],
        }
    )
    assert "DRY-RUN" in text
    assert "gs-job-9" in text
    assert "COMPLETE" in text
    assert "would call hivemind.moonlight.stream@v1" in text


# ---------------------------------------------------------------------------
# Worker-entry routing (no subprocess spawn — direct function checks)
# ---------------------------------------------------------------------------


def test_worker_entry_routes_gamestream_lobe_to_template_runner(monkeypatch):
    from machine_spirit_4.double_agent import _worker_entry

    monkeypatch.delenv("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER", raising=False)
    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id="conv-gs",
        conversation_revision_id=1,
        background_lobe_type=GAMESTREAM_LOBE_TYPE,
        user_visible_goal="Gamestream demo.",
        internal_goal="{}",
    )
    env.validate()
    runner = _worker_entry._build_runner_or_die(env, blackboard=None)
    # The template runner is callable without Hermes ever importing.
    assert callable(runner)


def test_worker_entry_model_class_resolution(monkeypatch):
    from machine_spirit_4.double_agent import _worker_entry
    from machine_spirit_4.double_agent.depth_picker import DepthChoice

    monkeypatch.delenv("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER", raising=False)
    seen: dict = {}

    def fake_choose(*, hivemind_url, model_class=None, **kwargs):
        seen["model_class"] = model_class
        return DepthChoice(model_id="coder-model:7b", source="loaded")

    import machine_spirit_4.double_agent.depth_picker as dp

    monkeypatch.setattr(dp, "choose_depth_model", fake_choose)

    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id="conv-mc",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="Refactor the worker.",
        internal_goal="Refactor the worker.",
    )
    env.resource_request.model_class = "deep_coder"
    env.validate()
    _worker_entry._resolve_depth_model_for_envelope(env)
    assert seen["model_class"] == "deep_coder"
    assert env.resource_request.model_override == "coder-model:7b"


def test_worker_entry_model_class_respects_existing_override(monkeypatch):
    from machine_spirit_4.double_agent import _worker_entry

    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id="conv-mc",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="Refactor the worker.",
        internal_goal="Refactor the worker.",
    )
    env.resource_request.model_override = "pinned:latest"
    env.validate()
    _worker_entry._resolve_depth_model_for_envelope(env)
    assert env.resource_request.model_override == "pinned:latest"


def test_worker_entry_model_class_skipped_in_fake_runner_mode(monkeypatch):
    from machine_spirit_4.double_agent import _worker_entry

    monkeypatch.setenv(
        "MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER",
        "tests.ms4_double_agent._fake_chat_runners:quick_factory",
    )
    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id="conv-mc",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="Refactor the worker.",
        internal_goal="Refactor the worker.",
    )
    env.validate()
    _worker_entry._resolve_depth_model_for_envelope(env)
    assert env.resource_request.model_override is None
