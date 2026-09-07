from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runtime_common import ROOT, exit_code, print_json, request_json, require_venv_python


@dataclass
class CheckResult:
    name: str
    url: str
    status: int
    ok: bool
    excerpt: str


CHECKS: list[dict[str, Any]] = [
    {"name": "hivemind_health", "method": "GET", "url": "http://127.0.0.1:6089/v1/health", "contains": '"status":"ok"'},
    {"name": "hivemind_models", "method": "GET", "url": "http://127.0.0.1:6089/v1/models", "contains": '"data"', "timeout": 30},
    {"name": "hivemind_mcp_status", "method": "GET", "url": "http://127.0.0.1:6089/v1/mcp/status", "contains": '"ok":true'},
    {"name": "hivemind_resources", "method": "GET", "url": "http://127.0.0.1:6089/v1/resources/status", "contains": '"catalog"'},
    {"name": "app_registry_health", "method": "GET", "url": "http://127.0.0.1:6110/health", "contains": '"ok":true'},
    {"name": "carrier_sync_status", "method": "GET", "url": "http://127.0.0.1:6130/api/v1/carrier_sync/status", "contains": '"node_id"'},
    {"name": "ms3_health", "method": "GET", "url": "http://127.0.0.1:9080/health", "contains": '"service":"Machine Spirit 3"'},
    {
        "name": "ms3_identity_verify",
        "method": "POST",
        "url": "http://127.0.0.1:9080/identity/verify",
        "contains": '"identity_confirmed":true',
        "payload": {"spirit_id": "sister", "allow_initialize": False},
    },
    {
        "name": "ms3_ethics_evaluate",
        "method": "POST",
        "url": "http://127.0.0.1:9080/ethics/evaluate",
        "contains": '"decision":"allow"',
        "payload": {
            "schema": "ActionIntent.v1",
            "spirit_id": "sister",
            "action_id": "ms4-validation-read",
            "proposed_by": "hermes",
            "action_type": "tool",
            "description": "Read a local status file for validation",
            "inputs_used": ["validation"],
            "risk_class": "low",
            "requires_safety_clearance": False,
            "payload": {"tool_name": "read_file"},
        },
    },
]


def compact_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except json.JSONDecodeError:
        return text


def run_check(check: dict[str, Any]) -> CheckResult:
    try:
        status, text, _data = request_json(
            check["method"],
            check["url"],
            payload=check.get("payload"),
            timeout=check.get("timeout", 8),
        )
        compact = compact_json(text)
        return CheckResult(
            name=check["name"],
            url=check["url"],
            status=status,
            ok=status == 200 and check["contains"] in compact,
            excerpt=compact[:240],
        )
    except Exception as exc:
        return CheckResult(check["name"], check["url"], 0, False, str(exc))


def main() -> int:
    results = [asdict(run_check(check)) for check in CHECKS]
    print_json(results)
    if not all(row["ok"] for row in results):
        return 1

    chat_voice_script = ROOT / "machine_spirit_4" / "scripts" / "validate_ms4_chat_voice.py"
    if chat_voice_script.exists():
        python = require_venv_python()
        return subprocess.call([str(python), str(chat_voice_script)], cwd=str(ROOT))
    return exit_code(True)


if __name__ == "__main__":
    sys.exit(main())
