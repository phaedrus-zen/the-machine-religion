from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Machine Spirit 3 with cross-platform Python.")
    parser.add_argument("--hivemind-url", default=os.environ.get("HIVEMIND_GATEWAY_URL", "http://localhost:6089"))
    parser.add_argument(
        "--host",
        default=os.environ.get("MS3_HOST", "127.0.0.1"),
        help="MS3 bind address (default: loopback; use 0.0.0.0 only for an explicitly secured lab LAN)",
    )
    parser.add_argument("--rust-log", default=os.environ.get("RUST_LOG", "info"))
    parser.add_argument("--debug", action="store_true", help="Run cargo without --release")
    args = parser.parse_args()

    env = os.environ.copy()
    env["RUST_LOG"] = args.rust_log
    env["HIVEMIND_GATEWAY_URL"] = args.hivemind_url
    env["MS3_HOST"] = args.host

    print("=" * 39)
    print("  Machine Spirit 3")
    print("  The soul lives with the scripture.")
    print("=" * 39)
    print()

    cmd = ["cargo", "run"]
    if not args.debug:
        cmd.append("--release")
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    sys.exit(main())
