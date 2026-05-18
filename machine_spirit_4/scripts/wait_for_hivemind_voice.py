from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from runtime_common import ROOT, print_json, request_json, require_venv_python


@dataclass
class CheckResult:
    name: str
    ok: bool
    status: int
    detail: str


def endpoint_ok(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("status") == "healthy" or data.get("ok") is True:
        return True
    health = data.get("service_health")
    return isinstance(health, dict) and health.get("healthy") is True


def test_json_endpoint(name: str, url: str, predicate, timeout: int = 10) -> CheckResult:
    try:
        status, text, data = request_json("GET", url, timeout=timeout)
        return CheckResult(name=name, ok=200 <= status < 300 and bool(predicate(data)), status=status, detail=text[:280])
    except Exception as exc:
        return CheckResult(name=name, ok=False, status=0, detail=str(exc))


def readiness_snapshot(hivemind_url: str) -> list[CheckResult]:
    return [
        test_json_endpoint("hivemind_health", f"{hivemind_url}/v1/health", lambda j: isinstance(j, dict) and j.get("status") == "ok"),
        test_json_endpoint("hivemind_mcp_status", f"{hivemind_url}/v1/mcp/status", lambda j: isinstance(j, dict) and j.get("ok") is True),
        test_json_endpoint("asr_status", f"{hivemind_url}/provision/status/ASR", endpoint_ok),
        test_json_endpoint("tts_status", f"{hivemind_url}/provision/status/TTS", endpoint_ok),
        test_json_endpoint("tts_super_status", f"{hivemind_url}/provision/status/TTS_SUPER", endpoint_ok),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for HiveMind voice dependencies, then optionally run MS4 validation.")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--hivemind-url", default="http://127.0.0.1:6089")
    parser.add_argument("--ms3-url", default="http://127.0.0.1:9080")
    parser.add_argument("--skip-full-validation", action="store_true")
    args = parser.parse_args()

    hivemind_url = args.hivemind_url.rstrip("/")
    deadline = time.monotonic() + args.timeout_seconds
    print(f"Waiting for HiveMind voice readiness at {hivemind_url}", flush=True)
    print("Polling ASR, TTS, TTS_SUPER, and MCP", flush=True)

    while time.monotonic() < deadline:
        snapshot = readiness_snapshot(hivemind_url)
        print_json([asdict(row) for row in snapshot])
        if all(row.ok for row in snapshot):
            print("HiveMind voice dependencies are ready.", flush=True)
            if args.skip_full_validation:
                return 0
            env = {
                **os.environ.copy(),
                "MS4_HIVEMIND_BASE_URL": hivemind_url,
                "MS4_MS3_SIDECAR_URL": args.ms3_url.rstrip("/"),
            }
            python = require_venv_python()
            validator = ROOT / "machine_spirit_4" / "scripts" / "validate_ms4_runtime.py"
            return subprocess.call([str(python), str(validator)], cwd=str(ROOT), env=env)
        time.sleep(args.poll_seconds)

    print(f"Timed out waiting for HiveMind voice readiness after {args.timeout_seconds} seconds.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
