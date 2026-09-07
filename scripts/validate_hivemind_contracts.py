from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass


@dataclass
class Check:
    name: str
    url: str
    status: int
    ok: bool
    excerpt: str


CHECKS = [
    ("hivemind_health", "http://127.0.0.1:6089/v1/health", '"status":"ok"', 8),
    ("hivemind_models", "http://127.0.0.1:6089/v1/models", '"data"', 30),
    ("hivemind_mcp_status", "http://127.0.0.1:6089/v1/mcp/status", '"ok":true', 8),
    ("hivemind_resources", "http://127.0.0.1:6089/v1/resources/status", '"catalog"', 8),
    ("app_registry_health", "http://127.0.0.1:6110/health", '"ok":true', 8),
    ("carrier_sync_status", "http://127.0.0.1:6130/api/v1/carrier_sync/status", '"node_id"', 8),
]


def compact_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except json.JSONDecodeError:
        return text


def run_check(name: str, url: str, contains: str, timeout: int) -> Check:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            compact = compact_json(body)
            return Check(name, url, response.status, response.status == 200 and contains in compact, compact[:240])
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return Check(name, url, exc.code, False, compact_json(body)[:240])
    except Exception as exc:
        return Check(name, url, 0, False, str(exc))


def main() -> int:
    results = [asdict(run_check(*check)) for check in CHECKS]
    print(json.dumps(results, indent=2))
    return 0 if all(row["ok"] for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
