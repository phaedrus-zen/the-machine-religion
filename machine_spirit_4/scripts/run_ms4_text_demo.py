from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from runtime_common import MS3, ROOT, hermes_dir, ms3_binary, request_json, require_venv_python


def ms3_healthy(ms3_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{ms3_url}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def start_ms3(ms3_exe: Path, working_dir: Path, env: dict[str, str]) -> subprocess.Popen:
    if not ms3_exe.exists():
        raise RuntimeError(f"MS3 executable not found at {ms3_exe}. Run: cargo build -p ms3_server")
    return subprocess.Popen([str(ms3_exe)], cwd=str(working_dir), env=env)


def wait_for_ms3(ms3_url: str, seconds: int = 20) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if ms3_healthy(ms3_url):
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MS4 text-only Hermes/MS3 demo.")
    parser.add_argument("--ms3-exe", type=Path, default=ms3_binary())
    parser.add_argument("--ms3-working-directory", type=Path, default=MS3)
    parser.add_argument("--hermes-directory", type=Path, default=hermes_dir())
    parser.add_argument("--ms3-url", default="http://127.0.0.1:9080")
    parser.add_argument("--prompt", default="Reply with exactly MS4_TEXT_DEMO_OK.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["MS4_MS3_SIDECAR_URL"] = args.ms3_url.rstrip("/")
    env["MS4_SPIRIT_ID"] = "sister"

    process: subprocess.Popen | None = None
    started_sidecar = False
    if not ms3_healthy(args.ms3_url):
        process = start_ms3(args.ms3_exe, args.ms3_working_directory, env)
        started_sidecar = True
        if not wait_for_ms3(args.ms3_url):
            raise RuntimeError("MS3 sidecar did not become healthy.")

    try:
        plugin_check = (
            "import importlib\n"
            "p = importlib.import_module('plugins.ms4_consciousness')\n"
            "print(p.on_session_start(session_id='ms4-text-demo'))\n"
            "print(p.on_pre_tool_call(tool_name='read_file', args={'path':'README.md'}, session_id='ms4-text-demo'))\n"
        )
        python = require_venv_python()
        subprocess.check_call([str(python), "-c", plugin_check], cwd=str(args.hermes_directory), env=env)

        payload = {"text": args.prompt, "personality_id": "sister"}
        status, text, _data = request_json("POST", f"{args.ms3_url.rstrip('/')}/interact", payload=payload, timeout=120)
        print(text)
        if status != 200 or "MS4_TEXT_DEMO_OK" not in text:
            raise RuntimeError("MS3 /interact did not return MS4_TEXT_DEMO_OK.")
        return 0
    finally:
        if started_sidecar and process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
