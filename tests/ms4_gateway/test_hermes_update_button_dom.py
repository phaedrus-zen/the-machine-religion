"""Execute the real Hermes update button script in a dependency-free DOM."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "machine_spirit_4" / "web" / "index.html"
HARNESS = Path(__file__).with_name("hermes_update_button_dom.mjs")


def _run_harness(mode: str) -> dict:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is required for Hermes update button DOM tests")
    completed = subprocess.run(
        [node, str(HARNESS), str(HTML), mode],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Hermes update button harness failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def test_cold_button_hydrates_then_posts_signed_target_and_renders_success():
    result = _run_harness("success")

    assert result["coldDisabled"] is True
    assert result["filled"] == {"disabled": False, "text": "Update Hermes"}
    assert result["postCalls"] == [
        {
            "path": "/api/v1/hermes/update",
            "method": "POST",
            "body": {"target_version": "0.20.0"},
        }
    ]
    assert result["getCalls"] == 2
    assert result["bannerVisible"] is True
    assert result["bannerSuccess"] is True
    assert result["bannerError"] is False
    assert result["headline"] == "Hermes updated to 0.20.0"


def test_preflight_rejection_survives_a_later_version_refresh():
    result = _run_harness("preflight-400")

    assert len(result["postCalls"]) == 1
    assert result["getCalls"] == 2
    assert result["bannerVisible"] is True
    assert result["bannerError"] is True
    assert result["headline"] == "Hermes update could not be started"
    assert result["subline"] == "disk preflight failed: insufficient free space"


def test_blocked_unsigned_release_never_posts():
    result = _run_harness("blocked")

    assert result["filled"] == {"disabled": True, "text": "No signed update"}
    assert result["postCalls"] == []
    assert result["getCalls"] == 1
    assert result["bannerVisible"] is True
    assert result["bannerError"] is True


def test_unknown_release_state_disables_instead_of_dead_click():
    result = _run_harness("unknown")

    assert result["filled"] == {"disabled": True, "text": "Update unavailable"}
    assert result["postCalls"] == []
    assert result["getCalls"] == 1


def test_allow_unsigned_policy_enables_button_and_posts_newest_unsigned_release():
    """Live shape 2026-09-07: 0.20.0 installed, newest v2026.8.31 unsigned,
    HERMES_RELEASE_SIGNATURE_POLICY=allow_unsigned -> enabled Update button."""
    result = _run_harness("allow-unsigned")

    assert result["coldDisabled"] is True
    assert result["filled"] == {"disabled": False, "text": "Update Hermes"}
    assert result["postCalls"] == [
        {
            "path": "/api/v1/hermes/update",
            "method": "POST",
            "body": {"target_version": "0.21.0"},
        }
    ]
    assert result["getCalls"] == 2
    assert result["bannerVisible"] is True
    assert result["bannerError"] is False
    assert result["headline"] == "Hermes update available: 0.20.0 → 0.21.0"
    assert result["subline"] == (
        "newest release v2026.8.31 is unsigned; policy allows "
        "(HERMES_RELEASE_SIGNATURE_POLICY=allow_unsigned)"
    )
    assert result["buttonDisabled"] is False
