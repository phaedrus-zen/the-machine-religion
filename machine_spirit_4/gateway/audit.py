from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_PATH = Path(__file__).resolve().parents[1] / "logs" / "ms4_audit.jsonl"


def audit_path_from_env() -> Path:
    return Path(os.environ.get("MS4_AUDIT_LOG", str(DEFAULT_AUDIT_PATH)))


def append_event(event_type: str, data: dict[str, Any], *, audit_path: Path | None = None) -> dict[str, Any]:
    path = audit_path or audit_path_from_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **data,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return event


def read_events(*, limit: int = 100, audit_path: Path | None = None) -> list[dict[str, Any]]:
    path = audit_path or audit_path_from_env()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines[-max(limit, 1):]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event_type": "corrupt_audit_line", "raw": line})
    return events
