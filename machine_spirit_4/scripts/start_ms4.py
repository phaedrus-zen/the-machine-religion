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
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    python = require_venv_python()
    env = ms4_env(ms3_port=args.ms3_port)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ensure_ms3_binary()

    if not is_port_listening(args.host, args.ms3_port):
        launch_process(
            [str(ms3_binary())],
            cwd=MS3,
            env=env,
            stdout_path=LOG_DIR / "ms3_stdout.log",
            stderr_path=LOG_DIR / "ms3_stderr.log",
        )

    if not wait_http(f"http://{args.host}:{args.ms3_port}/health"):
        raise RuntimeError(f"MS3 did not become healthy on port {args.ms3_port}")

    if not is_port_listening(args.host, args.gateway_port):
        launch_process(
            [
                str(python),
                str(ROOT / "machine_spirit_4" / "scripts" / "run_ms4_gateway.py"),
                "--port",
                str(args.gateway_port),
                "--host",
                args.host,
            ],
            cwd=ROOT,
            env=ms4_env(gateway_port=args.gateway_port, gateway_host=args.host, ms3_port=args.ms3_port),
            stdout_path=LOG_DIR / "ms4_gateway_stdout.log",
            stderr_path=LOG_DIR / "ms4_gateway_stderr.log",
        )

    if not wait_http(f"http://{args.host}:{args.gateway_port}/healthcheck/basic"):
        raise RuntimeError(f"MS4 gateway did not become healthy on port {args.gateway_port}")

    if not is_port_listening(args.host, args.mcp_port):
        launch_process(
            [
                str(python),
                str(ROOT / "machine_spirit_4" / "scripts" / "run_ms4_mcp.py"),
                "--port",
                str(args.mcp_port),
                "--host",
                args.host,
            ],
            cwd=ROOT,
            env=ms4_env(mcp_port=args.mcp_port, mcp_host=args.host, ms3_port=args.ms3_port),
            stdout_path=LOG_DIR / "ms4_mcp_stdout.log",
            stderr_path=LOG_DIR / "ms4_mcp_stderr.log",
        )

    if not wait_http(f"http://{args.host}:{args.mcp_port}/healthcheck/basic"):
        raise RuntimeError(f"MS4 MCP did not become healthy on port {args.mcp_port}")

    if not args.skip_validation:
        validation_env = ms4_env(ms3_port=args.ms3_port)
        validation_env["MS4_GATEWAY_URL"] = f"http://{args.host}:{args.gateway_port}"
        validation_env["MS4_MCP_URL"] = f"http://{args.host}:{args.mcp_port}/mcp"
        for script in ("validate_ms4_fusion.py", "validate_ms4_mcp.py"):
            code = subprocess.call([str(python), str(ROOT / "machine_spirit_4" / "scripts" / script)], cwd=str(ROOT), env=validation_env)
            if code != 0:
                return code

    print("MS4 runtime ready:")
    print(f"  MS3:          http://{args.host}:{args.ms3_port}/")
    print(f"  MS4 Gateway:  http://{args.host}:{args.gateway_port}/")
    print(f"  MS4 MCP:      http://{args.host}:{args.mcp_port}/mcp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
