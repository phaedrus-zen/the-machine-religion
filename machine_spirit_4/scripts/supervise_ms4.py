from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from runtime_common import MS4, ROOT, is_port_listening, venv_python

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from machine_spirit_4.retention import retention_schedule_seconds, run_retention_cycle

# Idempotent bring-up + watchdog for MS3 + MS4 (gateway/MCP).
# Launched in user context by the HiveMind Oracle Startup shortcut. Watch mode
# keeps the correct Hermes/WSL/desktop identity while checking every five
# minutes; one-shot mode remains available for validation and recovery.
# Only launches services that are currently down; never kills anything. The
# managed MS3 sidecar inherits start_ms4.py's loopback bind unless an operator
# explicitly sets MS3_HOST for a secured lab LAN.
HOST = "127.0.0.1"
PORTS = (9080, 9180, 9181)  # MS3 sidecar, MS4 gateway, MS4 MCP
HIVEMIND_HEALTH_URL = "http://127.0.0.1:6089/v1/health"
SUPERVISOR_LOG_DIR = MS4 / "logs" / "supervisor"
WATCH_MUTEX_NAME = "Local\\HiveMindMS4UserWatchdog"
_watch_mutex_handle: int | None = None


def log_path() -> Path:
    return SUPERVISOR_LOG_DIR / f"ms4_supervise_{time.strftime('%Y-%m-%d')}.log"


def write_log(message: str) -> None:
    with log_path().open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {message}\n")


def enforce_retention() -> dict:
    """Reuse the watchdog cadence; never expose file names or payloads in its log."""
    try:
        report = run_retention_cycle(
            root=ROOT,
            manifest_path=MS4 / "retention_manifest.json",
        )
    except Exception:
        report = {"ok": False, "reason": "cycle_failed", "actions": []}
    actions = list(report.get("actions") or [])
    failed = sum(1 for action in actions if action.get("success") is False)
    outcome = report.get("status") or report.get("reason") or "unknown"
    write_log(
        f"retention outcome={outcome} actions={len(actions)} "
        f"failures={failed} disk_pressure={bool(report.get('disk_pressure'))}"
    )
    return report


def hivemind_health_status() -> str:
    try:
        with urllib.request.urlopen(HIVEMIND_HEALTH_URL, timeout=4) as response:
            return str(response.status)
    except Exception:
        return "unreachable"


def down_ports() -> list[int]:
    return [port for port in PORTS if not is_port_listening(HOST, port)]


def open_oracle_ui_once() -> None:
    """Auto-open the Oracle voice front door once per boot when 9180 is up, so the operator never
    has to manually browse to localhost:9180 (the #1 'non-automatic' friction). Opt out with
    MS4_OPEN_UI=0. Keyed on boot time so the 5-min supervisor repetitions don't re-open it."""
    if os.environ.get("MS4_OPEN_UI", "1").strip() == "0":
        return
    url = "http://127.0.0.1:9180/"
    try:
        boot = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('o')"],
            text=True, timeout=10).strip()
    except Exception:
        boot = "unknown"
    marker = SUPERVISOR_LOG_DIR / "ms4_ui_opened.marker"
    try:
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == boot:
            return  # already opened this boot
    except Exception:
        pass
    opened = False
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    try:
        if edge.exists():
            creationflags = 0
            if os.name == "nt":
                # Keep the persistent Oracle window outside the watchdog's
                # process tree so it cannot hold a launcher open.
                creationflags = (
                    getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
                )
            subprocess.Popen(
                [str(edge), f"--app={url}"],
                creationflags=creationflags,
                close_fds=True,
            )
            opened = True
    except Exception as exc:
        write_log(f"Oracle UI Edge launch failed: {exc}")
        opened = False
    if not opened and os.name != "nt":
        try:
            import webbrowser

            opened = bool(webbrowser.open(url))
        except Exception:
            opened = False
    if opened:
        try:
            marker.write_text(boot, encoding="utf-8")
        except Exception:
            pass
        write_log(f"opened Oracle UI {url} (boot={boot})")


def supervise_once() -> int:
    SUPERVISOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    retention_report = enforce_retention()

    python = venv_python()
    if not python.exists():
        write_log(f"ABORT: venv python missing at {python}")
        return 1

    down = down_ports()
    if retention_report.get("disk_pressure") and down:
        write_log(
            f"ABORT: retention disk pressure blocks launch of {len(down)} missing service(s)"
        )
        return 1

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
    if code == 0 and not down:
        write_log("ok: {} all listening".format("/".join(str(port) for port in PORTS)))
        open_oracle_ui_once()  # auto-open the Oracle front door once per boot (MS4_OPEN_UI=0 to disable)
    return code


def acquire_watch_mutex() -> bool:
    """Allow one persistent user-session watchdog per Windows logon."""
    global _watch_mutex_handle
    if os.name != "nt":
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, WATCH_MUTEX_NAME)
    if not handle:
        return False
    if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        return False
    _watch_mutex_handle = int(handle)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Start or watch the local MS3/MS4 runtime.")
    parser.add_argument("--watch", action="store_true", help="Keep checking the runtime until logoff.")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.environ.get("MS4_WATCH_INTERVAL_SECS", "300")),
    )
    args = parser.parse_args()

    if not args.watch:
        return supervise_once()
    if not acquire_watch_mutex():
        return 0

    retention_interval = retention_schedule_seconds(MS4 / "retention_manifest.json")
    interval = max(30, min(args.interval_seconds, retention_interval))
    SUPERVISOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_log(f"user watchdog started interval_seconds={interval}")
    while True:
        try:
            code = supervise_once()
            if code != 0:
                write_log(f"managed iteration failed code={code}")
        except Exception as exc:
            write_log(f"watchdog iteration failed: {exc}")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            write_log("user watchdog stopped")
            return 0


if __name__ == "__main__":
    sys.exit(main())
