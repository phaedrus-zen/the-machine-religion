from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from machine_spirit_4.mcp.manifest_policy import (  # noqa: E402
    ManifestPolicyError,
    load_and_validate_manifest,
)
from machine_spirit_4.mcp.tools import build_tool_registry  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the offline MS4 MCP manifest policy.")
    parser.add_argument("--manifest", type=Path, help="Optional manifest path for validation.")
    args = parser.parse_args(argv)
    registry = build_tool_registry()
    try:
        manifest, by_name = load_and_validate_manifest(
            path=args.manifest,
            registry_names=registry,
            registry_runtime_actions=(
                name for name, tool in registry.items() if tool.is_runtime_action
            ),
        )
    except ManifestPolicyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    runtime_actions = [
        tool for tool in by_name.values() if tool.get("kind") == "runtime_action"
    ]
    print(
        json.dumps(
            {
                "schema": "Ms4McpManifestValidation.v1",
                "valid": True,
                "manifest_schema": manifest.get("schema"),
                "tools": len(by_name),
                "runtime_actions": len(runtime_actions),
                "missing_policy_fields": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
