from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from runtime_common import MS4, ROOT, is_port_listening, venv_python

# Idempotent bring-up + watchdog for MS3 + MS4 (gateway/MCP).
# Launched by the 'MS4-Spirit-AutoStart' scheduled task (At-Logon + 5-min repetition).
# Only launches services that are currently down; never kills anything.
HOST = "127.0.0.1"
PORTS = (9080, 9180, 9181)  # MS3 sidecar, MS4 gateway, MS4 MCP
HIVEMIND_HEALTH_URL = "http://127.0.0.1:6089/v1/health"
SUPERVISOR_LOG_DIR = MS4 / "logs" / "supervisor"


def log_path() -> Path:
    return SUPERVISOR_LOG_DIR / f"ms4_supervise_{time.strftime('%Y-%m-%d')}.log"


def write_log(message: str) -> None:
    with log_path().open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {message}\n")


def hivemind_health_status() -> str:
    try:
        with urllib.request.urlopen(HIVEMIND_HEALTH_URL, timeout=4) as response:
            return str(response.status)
    except Exception:
        return "unreachable"


def down_ports() -> list[int]:
    return [port for port in PORTS if not is_port_listening(HOST, port)]


def main() -> int:
    SUPERVISOR_LOG_DIR.mkdir(parents=True, exist_ok=True)

    python = venv_python()
    if not python.exists():
        write_log(f"ABORT: venv python missing at {python}")
        return 1

    down = down_ports()
    if not down:
        write_log("ok: {} all listening".format("/".join(str(port) for port in PORTS)))
        return 0

    write_log(
        "down=[{}] hivemind /v1/health={} -> start_ms4 --skip-validation".format(
            ",".join(str(port) for port in down), hivemind_health_status()
        )
    )
    start = MS4 / "scripts" / "start_ms4.py"
    with log_path().open("ab") as handle:
        code = subprocess.run(
            [str(python), str(start), "--skip-validation"],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode
    write_log(f"start_ms4 exit={code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
