from __future__ import annotations

import argparse
import subprocess
import sys

from runtime_common import ROOT, ms4_env, require_venv_python


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MS4 Gateway using the contained Python runtime.")
    parser.add_argument("--port", type=int, default=9180)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    env = ms4_env(gateway_port=args.port, gateway_host=args.host)
    python = require_venv_python()
    return subprocess.call([str(python), "-m", "machine_spirit_4.gateway.server"], cwd=str(ROOT), env=env)


if __name__ == "__main__":
    sys.exit(main())
