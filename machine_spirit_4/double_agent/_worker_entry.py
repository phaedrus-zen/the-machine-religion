"""Subprocess entrypoint for a Double Agent worker.

Launched by ``JobRunner._spawn_worker`` as a child Python process so
the parent gateway can issue a real OS-level ``terminate()`` when an
operator cancels a job. Without this, a blocking Hermes model call
would keep running until it returned naturally, no matter how many
threading.Event flags we flipped.

The subprocess:

  1. Reads the envelope from stdin as one JSON document (avoids
     command-line length / quoting hazards on Windows).
  2. Opens the same SQLite blackboard the parent uses (SQLite WAL
     handles cross-process writes).
  3. Builds the real Hermes-backed chat runner and runs
     :class:`DoubleAgentWorker.run` in-process.
  4. Exits 0 on success, non-zero on failure. The blackboard rows are
     the source of truth; the exit code is just for parent
     book-keeping.

We deliberately don't use ``multiprocessing`` here because Hermes /
Hermes' dependencies misbehave with ``fork`` on Linux and with
``spawn`` pickling on Windows. A plain script + subprocess argument is
the path that survives every platform we ship to.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import threading
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow the subprocess to be invoked without setting PYTHONPATH manually.
# `machine_spirit_4` is two parents up from this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from machine_spirit_4.double_agent import safety  # noqa: E402
from machine_spirit_4.double_agent.blackboard import Blackboard  # noqa: E402
from machine_spirit_4.double_agent.plan_templates import (  # noqa: E402
    GAMESTREAM_LOBE_TYPE,
    build_gamestream_chat_runner,
)
from machine_spirit_4.double_agent.schemas import JobEnvelope, JobEvent  # noqa: E402
from machine_spirit_4.double_agent.worker import (  # noqa: E402
    DoubleAgentWorker,
    build_real_chat_runner,
)


def _depth_hivemind_url() -> str:
    return (
        os.environ.get("MS4_HIVEMIND_URL")
        or os.environ.get("MS4_HIVEMIND_HLI_URL")
        or "http://127.0.0.1:6089"
    )


def _depth_ms3_url() -> str:
    return os.environ.get("MS4_MS3_URL", "http://127.0.0.1:9080")


def _depth_mcp_url(hivemind_url: str) -> str:
    from machine_spirit_4.gateway.hivemind_state import mcp_base_url

    base = mcp_base_url(hivemind_url).rstrip("/")
    if not base.endswith("/mcp"):
        base = f"{base}/mcp"
    return base


def _build_depth_preflight_urls() -> dict[str, str]:
    hivemind_url = _depth_hivemind_url().rstrip("/")
    return {
        "hivemind_url": hivemind_url,
        "hivemind_mcp_url": _depth_mcp_url(hivemind_url),
        "ms3_url": _depth_ms3_url().rstrip("/"),
    }


def _redact_env_sources() -> dict[str, bool]:
    return {
        "MS4_HIVEMIND_URL": bool(os.environ.get("MS4_HIVEMIND_URL")),
        "MS4_HIVEMIND_HLI_URL": bool(os.environ.get("MS4_HIVEMIND_HLI_URL")),
        "MS4_HIVEMIND_MCP_URL": bool(os.environ.get("MS4_HIVEMIND_MCP_URL")),
        "MS4_MS3_URL": bool(os.environ.get("MS4_MS3_URL")),
        "MS4_HIVEMIND_API_KEY": bool(os.environ.get("MS4_HIVEMIND_API_KEY")),
    }


def _http_probe(
    *,
    name: str,
    url: str,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "name": name,
        "url": url,
        "method": method,
        "ok": False,
        "reachable": False,
        "status": None,
        "error_type": None,
        "error": None,
    }
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers or {"Accept": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(4096)
            probe["reachable"] = True
            probe["status"] = resp.status
            probe["ok"] = 200 <= int(resp.status) < 400
            if raw:
                probe["body_prefix"] = raw.decode("utf-8", "replace")[:240]
    except urllib.error.HTTPError as exc:
        # 401/404/405 still proves the depth worker can reach the namespace.
        probe["reachable"] = True
        probe["status"] = exc.code
        probe["error_type"] = type(exc).__name__
        probe["error"] = str(exc)[:240]
        probe["ok"] = 400 <= int(exc.code) < 500
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        probe["error_type"] = type(exc).__name__
        probe["error"] = str(exc)[:240]
    return probe


def _is_loopback_url(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return False
    host = host.strip("[]").lower()
    return host in {"localhost", "::1"} or host.startswith("127.")


def _depth_preflight_diagnostics(
    *,
    urls: dict[str, str],
    probes: list[dict[str, Any]],
    os_name: str | None = None,
) -> list[dict[str, Any]]:
    worker_os_name = os.name if os_name is None else os_name
    loopback_urls = sorted(
        name for name, url in urls.items() if _is_loopback_url(str(url))
    )
    unreachable = sorted(
        str(probe.get("name"))
        for probe in probes
        if not bool(probe.get("reachable"))
    )
    if worker_os_name == "nt" or not loopback_urls or not unreachable:
        return []
    return [
        {
            "code": "posix_loopback_namespace_unreachable",
            "severity": "warning",
            "summary": (
                "This POSIX Depth worker is using loopback URLs that are "
                "unreachable from its namespace. 127.0.0.1 points at the "
                "worker namespace, not necessarily the Windows host."
            ),
            "loopback_urls": loopback_urls,
            "unreachable_probes": unreachable,
            "operator_action": (
                "Inject host-routable values for MS4_HIVEMIND_URL, "
                "MS4_HIVEMIND_MCP_URL, and MS4_MS3_URL, then rerun "
                "_worker_entry.py --preflight --json."
            ),
        }
    ]


def run_depth_worker_preflight(*, timeout: float = 3.0) -> dict[str, Any]:
    """Return a redacted reachability report from the worker namespace.

    This is intentionally separate from normal job execution. Operators can run
    ``_worker_entry.py --preflight --json`` with the same environment the depth
    lobe receives to prove whether that child process can see HiveMind/MS3/MCP.
    """
    urls = _build_depth_preflight_urls()
    cwd = Path.cwd()
    report: dict[str, Any] = {
        "schema": "Ms4DepthWorkerPreflight.v1",
        "ok": False,
        "runtime": {
            "cwd": str(cwd),
            "cwd_exists": cwd.exists(),
            "executable": sys.executable,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "os_name": os.name,
        },
        "env_configured": _redact_env_sources(),
        "urls": urls,
        "probes": [],
    }

    hli_health = f"{urls['hivemind_url']}/health"
    ms3_health = f"{urls['ms3_url']}/health"
    mcp_payload = json.dumps(
        {"jsonrpc": "2.0", "id": "ms4-depth-worker-preflight", "method": "tools/list"}
    ).encode("utf-8")
    mcp_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        from machine_spirit_4.gateway.hivemind_state import hivemind_auth_headers

        mcp_headers.update(hivemind_auth_headers())
    except Exception:
        pass

    report["probes"] = [
        _http_probe(name="hivemind_hli_health", url=hli_health, timeout=timeout),
        _http_probe(name="hivemind_mcp_tools_list", url=urls["hivemind_mcp_url"], method="POST", body=mcp_payload, headers=mcp_headers, timeout=timeout),
        _http_probe(name="ms3_health", url=ms3_health, timeout=timeout),
    ]
    report["diagnostics"] = _depth_preflight_diagnostics(
        urls=urls,
        probes=report["probes"],
        os_name=os.name,
    )
    report["ok"] = all(bool(probe.get("reachable")) for probe in report["probes"])
    return report


def _read_envelope_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("Double Agent worker subprocess: no envelope on stdin")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("Double Agent worker subprocess: envelope must be a JSON object")
    return payload


def _install_signal_handlers(cancel_event: threading.Event) -> None:
    """Translate signals into the in-process cancel event.

    On Windows, ``terminate()`` raises ``CTRL_BREAK_EVENT``; on POSIX,
    parent code sends ``SIGTERM``. Both flip the same in-process event
    that the worker checks at every natural checkpoint.
    """

    def _handler(_signum, _frame):
        cancel_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Some platforms (e.g. Windows) restrict which signals a
            # subprocess may handle. Best-effort.
            pass
    if os.name == "nt":
        try:
            signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def _build_runner_or_die(envelope: JobEnvelope, blackboard: Blackboard):
    """Construct the live Hermes runner the worker will drive.

    Failure here is recorded as a ``job.failed`` event on the
    blackboard before the process exits so the parent / UI can render
    a useful error instead of seeing the job hang in ``running``."""
    fake_spec = os.environ.get("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER")
    if fake_spec:
        return _load_fake_chat_runner(fake_spec)
    if envelope.background_lobe_type == GAMESTREAM_LOBE_TYPE:
        # Deterministic plan-template lane: no Hermes import, no model.
        # The template walks the GPU-P VM -> benchmark -> moonlight
        # chain (dry-run by default) and reports through the same
        # worker lifecycle callbacks as the Hermes lane.
        return build_gamestream_chat_runner()
    try:
        from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner

        from machine_spirit_4.scripts import runtime_common

        hermes_dir = str(runtime_common.hermes_dir())
        hivemind_url = _depth_hivemind_url()
        ms3_url = _depth_ms3_url()
        from machine_spirit_4.double_agent.depth_picker import depth_fallback_model

        default_model = depth_fallback_model()
        runner = Ms4HermesRunner(
            hermes_dir=hermes_dir,
            hivemind_url=hivemind_url,
            ms3_url=ms3_url,
            default_model=default_model,
        )
        return build_real_chat_runner(runner)
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            blackboard.insert_event(
                JobEvent.make(
                    job_id=envelope.job_id,
                    type="job.failed",
                    safe_user_status=f"Worker failed to construct Hermes runner: {type(exc).__name__}: {exc}",
                )
            )
            blackboard.insert_event(
                JobEvent.make(
                    job_id=envelope.job_id,
                    type="job.failed",
                    safe_user_status="operator detail attached",
                    visibility="operator_only",
                    payload={"traceback": tb[-4000:]},
                )
            )
            blackboard.update_job_state(
                envelope.job_id,
                state="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                last_safe_user_status=f"Worker failed to start: {type(exc).__name__}",
            )
        except Exception:
            pass
        raise SystemExit(2)


def _load_fake_chat_runner(spec: str):
    """Resolve a ``module:callable`` spec to a chat runner callable.

    Used only by the test suite (set via ``MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER``)
    so we can exercise the real subprocess + blackboard wiring without
    starting Hermes/HiveMind.
    """
    if ":" not in spec:
        raise SystemExit(
            f"MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER must be 'module:callable', got {spec!r}"
        )
    module_name, _, attr = spec.partition(":")
    import importlib

    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory()


def _resolve_depth_model_for_envelope(envelope: JobEnvelope) -> None:
    """Let ``resource_request.model_class`` drive the model pick for
    jobs that arrived WITHOUT an explicit ``model_override`` (e.g. REST
    submits from the web UI's deep-job dialog — the auto-router path
    always pins an override before submit).

    Phase-4 seed of the capability-lease routing described in
    ``docs/ARCHITECTURE.md`` ("phase 4 will route by
    resource_request.model_class"). Fail-soft: any picker failure
    leaves ``model_override`` unset so the worker falls back to the
    runner's default model exactly as before."""
    rr = envelope.resource_request
    if rr.model_override:
        return
    if envelope.background_lobe_type == GAMESTREAM_LOBE_TYPE:
        return  # deterministic template lane — no model in the loop
    if os.environ.get("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER"):
        return  # hermetic test mode: never touch the network
    try:
        from machine_spirit_4.double_agent.depth_picker import choose_depth_model

        choice = choose_depth_model(
            hivemind_url=_depth_hivemind_url(),
            model_class=rr.model_class,
        )
        rr.model_override = choice.model_id
    except Exception:  # noqa: BLE001 — picker must never kill the job
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Double Agent worker subprocess")
    parser.add_argument("--preflight", action="store_true", help="Run a redacted depth-worker namespace reachability preflight and exit.")
    parser.add_argument("--json", action="store_true", help="Emit JSON for --preflight.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-probe timeout for --preflight.")
    parser.add_argument("--db", help="Path to the Double Agent SQLite blackboard.")
    parser.add_argument("--job-id", help="Job id; used for sanity check against envelope.")
    args = parser.parse_args()

    if args.preflight:
        report = run_depth_worker_preflight(timeout=args.timeout)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            for probe in report["probes"]:
                status = probe.get("status")
                reachable = "reachable" if probe.get("reachable") else "unreachable"
                suffix = f" status={status}" if status is not None else ""
                error = f" error={probe.get('error')}" if probe.get("error") else ""
                print(f"{probe['name']}: {reachable}{suffix}{error}")
        return 0 if report["ok"] else 1

    if not args.db or not args.job_id:
        parser.error("--db and --job-id are required unless --preflight is used")

    if not safety.is_safe_job_id(args.job_id):
        raise SystemExit(f"Double Agent worker subprocess: refused unsafe job_id={args.job_id!r}")

    payload = _read_envelope_payload()
    envelope = JobEnvelope.from_dict(payload)
    if envelope.job_id != args.job_id:
        raise SystemExit(
            f"Double Agent worker subprocess: --job-id ({args.job_id}) does not match envelope ({envelope.job_id})"
        )

    db_path = Path(args.db).resolve()
    blackboard = Blackboard(db_path)
    cancel_event = threading.Event()
    _install_signal_handlers(cancel_event)
    _resolve_depth_model_for_envelope(envelope)
    chat_runner = _build_runner_or_die(envelope, blackboard)

    worker = DoubleAgentWorker(
        envelope,
        blackboard=blackboard,
        cancel_event=cancel_event,
        chat_runner=chat_runner,
    )
    result = worker.run()
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
