import json
import subprocess
import sys
from pathlib import Path


def test_hivemind_live_contracts():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "validate_hivemind_contracts.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    failed = [row for row in payload if not row["ok"]]
    assert failed == []
