from __future__ import annotations

import argparse
import subprocess
import sys

from runtime_common import (
    LOG_DIR,
    MS3,
    ROOT,
    is_port_listening,
    launch_process,
    ms3_binary,
    ms4_env,
    require_venv_python,
    wait_http,
)


def ensure_ms3_binary() -> None:
    if ms3_binary().exists():
        return
    subprocess.check_call(["cargo", "build", "-p", "ms3_server"], cwd=str(MS3))


def main() -> int:
    parser = argparse.ArgumentParser(description="Start MS3, MS4 Gateway, and MS4 MCP with cross-platform Python.")
    parser.add_argument("--gateway-port", type=int, default=9180)
    parser.add_argument("--mcp-port", type=int, default=9181)
    parser.add_argument("--ms3-port", type=int, default=9080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--ms3-host",
        default=None,
        help="MS3 bind address (default: MS3_HOST or 127.0.0.1; use 0.0.0.0 only for a secured lab LAN)",
    )
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    python = require_venv_python()
    env = ms4_env(ms3_port=args.ms3_port, ms3_host=args.ms3_host)
    ms3_bind_host = env["MS3_HOST"]
    ms3_probe_host = "127.0.0.1" if ms3_bind_host == "0.0.0.0" else ms3_bind_host
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ensure_ms3_binary()

    if not is_port_listening(ms3_probe_host, args.ms3_port):
        launch_process(
            [str(ms3_binary())],
            cwd=MS3,
            env=env,
            stdout_path=LOG_DIR / "ms3_stdout.log",
            stderr_path=LOG_DIR / "ms3_stderr.log",
        )

    if not wait_http(
        f"http://{ms3_probe_host}:{args.ms3_port}/health",
        expected_json={"status": "alive", "service": "Machine Spirit 3"},
    ):
        raise RuntimeError(f"MS3 did not become healthy on port {args.ms3_port}")

    if not is_port_listening(args.host, args.gateway_port):
        launch_process(
            [
                str(python),
                "-m",
                "machine_spirit_4.gateway.server",
            ],
            cwd=ROOT,
            env=ms4_env(
                gateway_port=args.gateway_port,
                gateway_host=args.host,
                ms3_port=args.ms3_port,
                ms3_host=args.ms3_host,
            ),
            stdout_path=LOG_DIR / "ms4_gateway_stdout.log",
            stderr_path=LOG_DIR / "ms4_gateway_stderr.log",
        )

    if not wait_http(
        f"http://{args.host}:{args.gateway_port}/api/v1/ms4_gateway/status",
        expected_json={"service": "ms4-gateway", "status": "ready", "endpoint": "/chat"},
    ):
        raise RuntimeError(f"MS4 gateway did not become healthy on port {args.gateway_port}")

    if not is_port_listening(args.host, args.mcp_port):
        launch_process(
            [
                str(python),
                "-m",
                "machine_spirit_4.mcp.server",
            ],
            cwd=ROOT,
            env=ms4_env(
                mcp_port=args.mcp_port,
                mcp_host=args.host,
                ms3_port=args.ms3_port,
                ms3_host=args.ms3_host,
            ),
            stdout_path=LOG_DIR / "ms4_mcp_stdout.log",
            stderr_path=LOG_DIR / "ms4_mcp_stderr.log",
        )

    if not wait_http(
        f"http://{args.host}:{args.mcp_port}/api/v1/ms4_mcp/status",
        expected_json={"service": "ms4-mcp-server", "status": "ready", "endpoint": "/mcp"},
    ):
        raise RuntimeError(f"MS4 MCP did not become healthy on port {args.mcp_port}")

    if not args.skip_validation:
        validation_env = ms4_env(ms3_port=args.ms3_port, ms3_host=args.ms3_host)
        validation_env["MS4_GATEWAY_URL"] = f"http://{args.host}:{args.gateway_port}"
        validation_env["MS4_MCP_URL"] = f"http://{args.host}:{args.mcp_port}/mcp"
        for script in ("validate_ms4_fusion.py", "validate_ms4_mcp.py"):
            code = subprocess.call([str(python), str(ROOT / "machine_spirit_4" / "scripts" / script)], cwd=str(ROOT), env=validation_env)
            if code != 0:
                return code

    print("MS4 runtime ready:")
    print(f"  MS3:          http://{ms3_probe_host}:{args.ms3_port}/ (bind {ms3_bind_host})")
    print(f"  MS4 Gateway:  http://{args.host}:{args.gateway_port}/")
    print(f"  MS4 MCP:      http://{args.host}:{args.mcp_port}/mcp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
