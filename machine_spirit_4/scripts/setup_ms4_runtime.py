from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path

from runtime_common import MS4, VENV, hermes_dir, playwright_browsers_path, venv_python

RUNTIME = MS4 / "runtime"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd or MS4))


def create_venv() -> None:
    if not venv_python().exists():
        builder = venv.EnvBuilder(with_pip=True, clear=False)
        builder.create(VENV)


def install_requirements(include_optional: bool) -> None:
    req = MS4 / ("requirements-full-hermes.txt" if include_optional else "requirements.txt")
    run([str(venv_python()), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(venv_python()), "-m", "pip", "install", "-r", str(req)])


def install_hermes_editable(include_optional: bool, source: Path) -> None:
    if not include_optional:
        return
    if not source.exists():
        print(f"Skipping Hermes editable install; directory not found: {source}", flush=True)
        return
    run([str(venv_python()), "-m", "pip", "install", "-e", str(source)])


def install_playwright_browsers(include_optional: bool) -> None:
    if not include_optional:
        return
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(playwright_browsers_path())
    subprocess.call([str(venv_python()), "-m", "playwright", "install", "chromium"], env=env)


def write_manifest(include_optional: bool, source: Path) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "Ms4RuntimeManifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": str(venv_python()),
        "venv": str(VENV),
        "include_optional": include_optional,
        "requirements": str(MS4 / ("requirements-full-hermes.txt" if include_optional else "requirements.txt")),
        "hermes_path": str(source),
        "playwright_browsers_path": str(playwright_browsers_path()),
    }
    (RUNTIME / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true", help="Install only core MS4 dependencies")
    parser.add_argument("--skip-playwright-browsers", action="store_true")
    parser.add_argument("--hermes-dir", type=Path, default=hermes_dir(), help="Hermes source directory for editable install")
    args = parser.parse_args()

    include_optional = not args.core_only
    create_venv()
    install_requirements(include_optional)
    install_hermes_editable(include_optional, args.hermes_dir)
    if not args.skip_playwright_browsers:
        install_playwright_browsers(include_optional)
    write_manifest(include_optional, args.hermes_dir)
    print(f"MS4 runtime ready: {venv_python()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
