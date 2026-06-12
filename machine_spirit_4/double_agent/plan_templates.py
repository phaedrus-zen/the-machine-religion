"""Depth-job plan templates — deterministic execution chains.

The Double Agent Depth Lobe has two execution lanes (see
``docs/double_agent/research_artifact_v2.md`` §3.3 and
``docs/ARCHITECTURE.md`` "Hermes runs in two places only"):

* **LLM-driven** (``background_lobe_type='deep_chat'`` etc.): the
  Hermes tool loop decides which tools to call. Quality depends on the
  depth model.
* **Deterministic templates** (this module): a pre-authored, ordered
  chain of HiveMind tool calls the worker walks step by step. No model
  in the loop for control flow; every step emits the same allowlisted
  ``job.tool.call.*`` events as the Hermes lane, so the Face Lobe / UI
  narration is identical.

First (flagship) template: **gamestream_benchmark** — the owner's
"create a Windows VM with a partial GPU, run a benchmark, stream it
back via moonlight" scenario:

1. ``hivemind.psykyo.game_session.list_capable_gpus@v1`` (read-only
   discovery)
2. ``hivemind.vm.create_prebuilt@v1`` with
   ``vm_type='windows_game_stream_prebuilt'`` — the GPU-P partial-GPU
   lane (``gateway.gpu_passthrough.GAME_STREAM_VM_TYPE``) — MUTATING
3. ``hivemind.psykyo.game_session.start_benchmark@v1`` (orchestrated
   plan; menta_game_session:6166 is dry-run server-side unless
   ``MENTA_GAME_SESSION_ALLOW_MUTATIONS=1``)
4. ``hivemind.game_session.run@v1`` (walk the state machine)
5. ``hivemind.psykyo.game_session.get_results@v1`` (benchmark_summary /
   benchmark_report)
6. ``hivemind.moonlight.stream@v1`` (stream-back leg) — MUTATING

Safety contract — DRY-RUN by default:

* Plans default to ``dry_run=True``. In dry-run the executor performs
  only ``read_only`` and ``orchestrated`` steps (the orchestrated ones
  are *additionally* dry-run server-side per the
  ``MENTA_GAME_SESSION_ALLOW_MUTATIONS`` gate) and records
  ``would_call`` evidence for ``mutating`` steps without executing
  them. This mirrors menta_game_session's own Phase-1 contract.
* A ``dry_run=False`` plan is REFUSED unless the MS4 process itself
  sees ``MENTA_GAME_SESSION_ALLOW_MUTATIONS=1`` — belt to the
  orchestrator's server-side suspenders. Flipping that env var is an
  operator decision, never a template default.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("ms4.double_agent.plan_templates")


GAMESTREAM_LOBE_TYPE = "gamestream_benchmark"
"""``JobEnvelope.background_lobe_type`` value that routes a Depth job
to this template instead of the Hermes loop (see
``_worker_entry._build_runner_or_die``)."""

MUTATIONS_ENV = "MENTA_GAME_SESSION_ALLOW_MUTATIONS"
"""Same env var menta_game_session (:6166) uses to gate real
mutations. MS4 mirrors it client-side: a live (non-dry-run) template
run is refused unless this is '1' in MS4's own environment."""

PLAN_SCHEMA = "Ms4GamestreamBenchmarkPlan.v1"
RESULT_SCHEMA = "Ms4GamestreamBenchmarkResult.v1"

GAME_STREAM_VM_TYPE = "windows_game_stream_prebuilt"
"""HiveMind ``vm.create_prebuilt`` template that provisions a Windows
VM with GPU-P (partial GPU) pre-wired. Kept in sync with
``gateway.gpu_passthrough.GAME_STREAM_VM_TYPE`` (duplicated here so the
double_agent package never imports the gateway package at module
import time)."""

STEP_KINDS = ("read_only", "orchestrated", "mutating")

DEFAULT_GAME = "cyberpunk-2077"
DEFAULT_VM_NAME = "ms4-gamestream-bench"
DEFAULT_STREAM_APP = "Desktop"
DEFAULT_VERIFY_SECONDS = 10


class PlanTemplateError(RuntimeError):
    """Raised when a plan cannot be built or is refused execution."""


def mutations_allowed() -> bool:
    """True when the operator has explicitly enabled live mutations."""
    return os.environ.get(MUTATIONS_ENV, "").strip() == "1"


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    step_id: str
    title: str
    kind: str  # one of STEP_KINDS
    tool: str  # canonical HiveMind tool id
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "kind": self.kind,
            "tool": self.tool,
            "args": dict(self.args),
        }


def build_gamestream_benchmark_plan(
    *,
    game: str = DEFAULT_GAME,
    vm_name: str = DEFAULT_VM_NAME,
    host: str | None = None,
    gpu_uuid: str | None = None,
    benchmark_runs: int = 1,
    quality: str | None = None,
    requires_stream: bool = True,
    stream_app: str = DEFAULT_STREAM_APP,
    verify_seconds: int = DEFAULT_VERIFY_SECONDS,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Build the flagship gamestream-benchmark plan
    (``Ms4GamestreamBenchmarkPlan.v1``).

    Pure function — no I/O, no env reads. ``dry_run`` is recorded on
    the plan; enforcement happens in
    :func:`execute_gamestream_benchmark_plan`.
    """
    if not game or not str(game).strip():
        raise PlanTemplateError("game is required")
    if not vm_name or not str(vm_name).strip():
        raise PlanTemplateError("vm_name is required")
    game = str(game).strip()
    vm_name = str(vm_name).strip()
    try:
        benchmark_runs = max(1, min(50, int(benchmark_runs)))
    except (TypeError, ValueError):
        benchmark_runs = 1

    bench_args: dict[str, Any] = {
        "game": game,
        "mode": "benchmark",
        "benchmark_runs": benchmark_runs,
        "requires_stream": bool(requires_stream),
        "client": "ms4-depth-lobe",
    }
    if quality:
        bench_args["quality"] = quality
    if gpu_uuid:
        bench_args["gpu_uuid"] = gpu_uuid
    if host:
        bench_args["host"] = host

    discover_args: dict[str, Any] = {}
    if host:
        discover_args["host_filter"] = host

    stream_args: dict[str, Any] = {
        # The Sunshine host is only known once the VM exists; the
        # executor substitutes the live value in live mode. The
        # placeholder keeps the dry-run evidence honest about what is
        # NOT yet known.
        "host": host or "<game-stream-vm-ip>",
        "app": stream_app,
        "verify_seconds": max(3, min(60, int(verify_seconds))),
    }

    steps = [
        PlanStep(
            step_id="gpu.discover",
            title="Discover gamestream-capable GPUs",
            kind="read_only",
            tool="hivemind.psykyo.game_session.list_capable_gpus@v1",
            args=discover_args,
        ),
        PlanStep(
            step_id="vm.gpu_p.provision",
            title="Create Windows game-stream VM with a partial GPU (GPU-P)",
            kind="mutating",
            tool="hivemind.vm.create_prebuilt@v1",
            args={"vm_type": GAME_STREAM_VM_TYPE, "name": vm_name},
        ),
        PlanStep(
            step_id="benchmark.plan",
            title=f"Plan {game} benchmark session ({benchmark_runs} run(s))",
            kind="orchestrated",
            tool="hivemind.psykyo.game_session.start_benchmark@v1",
            args=bench_args,
        ),
        PlanStep(
            step_id="benchmark.run",
            title="Walk the benchmark state machine to terminal",
            kind="orchestrated",
            tool="hivemind.game_session.run@v1",
            args={"job_id": "<benchmark.plan job_id>"},
        ),
        PlanStep(
            step_id="benchmark.results",
            title="Collect benchmark_summary / benchmark_report",
            kind="read_only",
            tool="hivemind.psykyo.game_session.get_results@v1",
            args={"job_id": "<benchmark.plan job_id>"},
        ),
        PlanStep(
            step_id="stream.moonlight",
            title="Stream the session back via Moonlight",
            kind="mutating",
            tool="hivemind.moonlight.stream@v1",
            args=stream_args,
        ),
    ]
    return {
        "schema": PLAN_SCHEMA,
        "template": GAMESTREAM_LOBE_TYPE,
        "dry_run": bool(dry_run),
        "game": game,
        "vm_name": vm_name,
        "steps": [s.to_dict() for s in steps],
    }


# ---------------------------------------------------------------------------
# Goal-string parameter parsing
# ---------------------------------------------------------------------------


_GOAL_PARAM_KEYS = {
    "game": str,
    "vm_name": str,
    "host": str,
    "gpu_uuid": str,
    "benchmark_runs": int,
    "quality": str,
    "requires_stream": bool,
    "stream_app": str,
    "verify_seconds": int,
    "dry_run": bool,
}


def parse_goal_params(goal: str | None) -> dict[str, Any]:
    """Extract template parameters from a job's ``internal_goal``.

    Accepts either a bare JSON object or free text containing one
    (``{"game": "cp2077", "benchmark_runs": 3}``). Unknown keys are
    ignored; values are coerced to the expected types. Anything
    unparsable degrades to the template defaults — a malformed goal
    must never crash the Depth job before it can report honestly.
    """
    params: dict[str, Any] = {}
    if not goal:
        return params
    text = str(goal).strip()
    candidate: Any = None
    try:
        candidate = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                candidate = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                candidate = None
    if not isinstance(candidate, dict):
        return params
    for key, typ in _GOAL_PARAM_KEYS.items():
        if key not in candidate:
            continue
        raw = candidate[key]
        try:
            if typ is bool:
                if isinstance(raw, bool):
                    params[key] = raw
                elif isinstance(raw, str):
                    params[key] = raw.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    params[key] = bool(raw)
            elif typ is int:
                params[key] = int(raw)
            else:
                value = str(raw).strip()
                if value:
                    params[key] = value
        except (TypeError, ValueError):
            continue
    return params


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------


def _default_tools():
    # Lazy import: double_agent must stay importable without the
    # gateway package (and vice versa) to avoid import cycles.
    from machine_spirit_4.gateway import hivemind_tools

    return hivemind_tools


def _excerpt(value: Any, limit: int = 2000) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    if len(rendered) > limit:
        rendered = rendered[: limit - 3] + "..."
    return rendered


def execute_gamestream_benchmark_plan(
    plan: dict[str, Any],
    *,
    hivemind_url: str,
    tools: Any | None = None,
    tool_start_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    tool_complete_callback: Callable[[str, str, dict[str, Any], Any], None] | None = None,
) -> dict[str, Any]:
    """Walk a ``Ms4GamestreamBenchmarkPlan.v1`` step by step.

    * ``tools`` defaults to :mod:`machine_spirit_4.gateway.hivemind_tools`;
      tests inject a fake with the same function names.
    * ``tool_start_callback(step_id, tool, args)`` /
      ``tool_complete_callback(step_id, tool, args, result)`` mirror the
      Hermes lifecycle callbacks so :class:`DoubleAgentWorker` can plug
      its own event emitters straight in. Exceptions raised by the
      callbacks (e.g. ``WorkerCanceled``) propagate — that is the
      cancellation path.
    * Fail-soft per stage like ``game_admin.plan_run_and_collect``,
      EXCEPT ``benchmark.plan`` (no job_id means the chain is dead —
      raise) and live-mode mutations (a failed flagship mutation must
      fail the job, not bury an error string).

    Returns ``Ms4GamestreamBenchmarkResult.v1``.
    """
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise PlanTemplateError(f"not a {PLAN_SCHEMA} plan")
    dry_run = bool(plan.get("dry_run", True))
    if not dry_run and not mutations_allowed():
        raise PlanTemplateError(
            f"live (dry_run=False) execution refused: {MUTATIONS_ENV} is not "
            "set to 1 in the MS4 environment. The menta_game_session "
            "orchestrator enforces the same gate server-side."
        )
    if tools is None:
        tools = _default_tools()

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "template": GAMESTREAM_LOBE_TYPE,
        "dry_run": dry_run,
        "game": plan.get("game"),
        "vm_name": plan.get("vm_name"),
        "benchmark_job_id": None,
        "final_state": None,
        "steps": [],
        "errors": [],
    }
    benchmark_job_id: str | None = None

    def _emit_start(step_id: str, tool: str, args: dict[str, Any]) -> None:
        if tool_start_callback is not None:
            tool_start_callback(step_id, tool, args)

    def _emit_complete(step_id: str, tool: str, args: dict[str, Any], outcome: Any) -> None:
        if tool_complete_callback is not None:
            tool_complete_callback(step_id, tool, args, outcome)

    for raw_step in plan.get("steps") or []:
        step_id = str(raw_step.get("step_id") or "")
        tool = str(raw_step.get("tool") or "")
        kind = str(raw_step.get("kind") or "read_only")
        args = dict(raw_step.get("args") or {})
        record: dict[str, Any] = {"step_id": step_id, "tool": tool, "kind": kind}

        # Resolve the benchmark job_id placeholder for dependent steps.
        needs_job_id = step_id in {"benchmark.run", "benchmark.results"}
        if needs_job_id:
            if benchmark_job_id is None:
                record["status"] = "skipped_dependency"
                record["detail"] = "no benchmark job_id from benchmark.plan"
                result["steps"].append(record)
                continue
            args["job_id"] = benchmark_job_id

        # Dry-run: mutating steps are recorded, never executed.
        if dry_run and kind == "mutating":
            record["status"] = "skipped_dry_run"
            record["would_call"] = tool
            record["args"] = args
            _emit_start(step_id, tool, args)
            _emit_complete(
                step_id, tool, args,
                {"skipped": True, "would_call": tool, "dry_run": True},
            )
            result["steps"].append(record)
            continue

        _emit_start(step_id, tool, args)
        try:
            outcome = _dispatch_step(
                tools, hivemind_url, step_id=step_id, tool=tool, args=args
            )
            record["status"] = "ok"
            record["result_excerpt"] = _excerpt(outcome)
        except PlanTemplateError:
            raise
        except Exception as exc:  # noqa: BLE001 — fail-soft per stage
            record["status"] = "error"
            record["detail"] = f"{type(exc).__name__}: {exc}"
            result["errors"].append(f"{step_id}: {record['detail']}")
            _emit_complete(step_id, tool, args, {"error": record["detail"]})
            result["steps"].append(record)
            if step_id == "benchmark.plan":
                # No job_id -> the rest of the chain is dead. Raise so
                # the worker lands the job in `failed` honestly.
                raise PlanTemplateError(
                    f"benchmark.plan failed; chain aborted: {record['detail']}"
                ) from exc
            if not dry_run and kind == "mutating":
                raise PlanTemplateError(
                    f"live mutation step {step_id} failed: {record['detail']}"
                ) from exc
            continue

        if step_id == "benchmark.plan" and isinstance(outcome, dict):
            job_id = outcome.get("job_id")
            if not job_id and isinstance(outcome.get("plan"), dict):
                job_id = outcome["plan"].get("job_id")
            if job_id:
                benchmark_job_id = str(job_id)
                result["benchmark_job_id"] = benchmark_job_id
        if step_id == "benchmark.run" and isinstance(outcome, dict):
            result["final_state"] = (
                outcome.get("final_state") or outcome.get("state") or None
            )
        if step_id == "benchmark.results" and isinstance(outcome, dict):
            summary = outcome.get("benchmark_summary")
            if summary is not None:
                result["benchmark_summary"] = summary

        _emit_complete(step_id, tool, args, outcome)
        result["steps"].append(record)

    return result


def _dispatch_step(
    tools: Any,
    hivemind_url: str,
    *,
    step_id: str,
    tool: str,
    args: dict[str, Any],
) -> Any:
    """Map a plan step to the typed wrapper that performs it."""
    if tool == "hivemind.psykyo.game_session.list_capable_gpus@v1":
        return tools.psykyo_game_session_list_capable_gpus(
            hivemind_url, host_filter=args.get("host_filter")
        )
    if tool == "hivemind.vm.create_prebuilt@v1":
        # Live mode only (dry-run short-circuits before dispatch).
        # gpu_passthrough.create_game_stream_vm is the sanctioned GPU-P
        # path (create + best-effort deploy, confirm-gated). An injected
        # tools object may supply its own create_game_stream_vm (tests).
        creator = getattr(tools, "create_game_stream_vm", None)
        if creator is None:
            from machine_spirit_4.gateway import gpu_passthrough

            creator = gpu_passthrough.create_game_stream_vm
        return creator(
            hivemind_url, name=str(args.get("name") or DEFAULT_VM_NAME), confirm=True
        )
    if tool == "hivemind.psykyo.game_session.start_benchmark@v1":
        kwargs = {k: v for k, v in args.items() if k not in {"game", "mode"}}
        return tools.psykyo_game_session_start_benchmark(
            hivemind_url, game=str(args.get("game") or DEFAULT_GAME), **kwargs
        )
    if tool == "hivemind.game_session.run@v1":
        return tools.game_session_run(hivemind_url, job_id=str(args["job_id"]))
    if tool == "hivemind.psykyo.game_session.get_results@v1":
        return tools.psykyo_game_session_get_results(
            hivemind_url, job_id=str(args["job_id"])
        )
    if tool == "hivemind.moonlight.stream@v1":
        host = str(args.get("host") or "")
        if not host or host.startswith("<"):
            # Live mode reached a stream step without a resolved host.
            # Refuse loudly instead of sending a placeholder to the
            # cluster — pass host=... in the job goal (VM-IP auto-
            # resolution from the create step is phase-2 work).
            raise PlanTemplateError(
                "moonlight host unresolved; pass 'host' in the job goal"
            )
        kwargs = {k: v for k, v in args.items() if k != "host"}
        return tools.moonlight_stream(hivemind_url, host=host, **kwargs)
    raise PlanTemplateError(f"step {step_id!r}: no dispatcher for tool {tool!r}")


# ---------------------------------------------------------------------------
# Result rendering + chat-runner adapter
# ---------------------------------------------------------------------------


def render_result_text(result: dict[str, Any]) -> str:
    """Human-readable summary of a template run — this becomes
    ``JobResult.text`` (full detail) and its first sentence becomes the
    user-safe summary, mirroring the Hermes lane."""
    dry_run = bool(result.get("dry_run", True))
    mode = "DRY-RUN" if dry_run else "LIVE"
    lines = [
        f"Gamestream benchmark template completed in {mode} mode for "
        f"{result.get('game')!s} (VM {result.get('vm_name')!s}).",
    ]
    if result.get("benchmark_job_id"):
        lines.append(f"Benchmark session job_id: {result['benchmark_job_id']}")
    if result.get("final_state"):
        lines.append(f"Orchestrator final state: {result['final_state']}")
    if result.get("benchmark_summary") is not None:
        lines.append(f"Benchmark summary: {_excerpt(result['benchmark_summary'], 500)}")
    for step in result.get("steps") or []:
        status = step.get("status")
        if status == "ok":
            lines.append(f"- {step.get('step_id')}: ok ({step.get('tool')})")
        elif status == "skipped_dry_run":
            lines.append(
                f"- {step.get('step_id')}: dry-run, would call {step.get('would_call')}"
            )
        elif status == "skipped_dependency":
            lines.append(
                f"- {step.get('step_id')}: skipped ({step.get('detail')})"
            )
        else:
            lines.append(f"- {step.get('step_id')}: ERROR {step.get('detail')}")
    if result.get("errors"):
        lines.append(f"Errors: {len(result['errors'])} (see steps above).")
    return "\n".join(lines)


def build_gamestream_chat_runner(
    *,
    hivemind_url: str | None = None,
    tools: Any | None = None,
) -> Callable[..., dict[str, Any]]:
    """Chat-runner-compatible adapter so :class:`DoubleAgentWorker`
    can execute the template through its existing lifecycle (events,
    cancellation, result persistence) with zero worker changes.

    Signature-compatible with ``build_real_chat_runner``'s ``_call``;
    the ``message`` is the job's ``internal_goal`` and may embed a JSON
    parameter object (see :func:`parse_goal_params`). ``model`` is
    accepted and ignored — there is no model in this lane.
    """

    def _call(
        *,
        message: str,
        session_id: str,
        model: str | None = None,
        stream_callback: Callable[[str], None] | None = None,
        tool_start_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
        tool_complete_callback: Callable[[str, str, dict[str, Any], Any], None] | None = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        url = hivemind_url or os.environ.get(
            "MS4_HIVEMIND_URL", "http://127.0.0.1:6089"
        )
        params = parse_goal_params(message)
        plan = build_gamestream_benchmark_plan(**params)
        if stream_callback is not None:
            # Liveness pulse only — never narrated (worker contract).
            stream_callback("gamestream-template")
        result = execute_gamestream_benchmark_plan(
            plan,
            hivemind_url=url,
            tools=tools,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
        )
        return {
            "text": render_result_text(result),
            "session_id": session_id,
            "model": model or "deterministic-template",
            "completed": True,
            "template_result": result,
        }

    return _call
