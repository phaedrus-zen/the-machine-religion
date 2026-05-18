from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from machine_spirit_4.deps_status import dependency_status


def main() -> int:
    # Reports playwright, pyautogui, mss, desktop_control, Hermes, MS3, and contained-venv status.
    print(json.dumps(dependency_status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
