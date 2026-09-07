import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).with_name("oracle_voice_input_session.mjs")


def test_oracle_voice_input_session_virtual_time_contracts() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the Oracle voice input session tests")

    completed = subprocess.run(
        [node, str(HARNESS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Node contract harness failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, "Node contract harness emitted no result"
    summary = json.loads(lines[-1])
    assert summary == {"ok": True, "tests": 49}
