from __future__ import annotations

import argparse
import subprocess
import sys

from runtime_common import ROOT, ms4_env, require_venv_python


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MS4 MCP server using the contained Python runtime.")
    parser.add_argument("--port", type=int, default=9181)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    env = ms4_env(mcp_port=args.port, mcp_host=args.host)
    python = require_venv_python()
    return subprocess.call([str(python), "-m", "machine_spirit_4.mcp.server"], cwd=str(ROOT), env=env)


if __name__ == "__main__":
    sys.exit(main())
