"""Directory-scoped fixtures for the ``ms4_hermes_admin`` suite.

The F4 supply-chain provenance gate (``installer._verify_release_provenance``)
is a MANDATORY, fail-closed step inside ``run_update_job`` that runs after the
release tag/objects are fetched but BEFORE the first working-tree mutation or
install. The pre-existing F2/F3/F6/F8 suites (``test_installer.py``,
``test_installer_r1.py``, ``test_reviewer_adversarial.py``) drive
``run_update_job`` with mocked pip/validate and UNSIGNED throwaway git tags, so
the real gate would (correctly) refuse before the scenario those suites assert
is ever reached.

To keep those suites BYTE-IDENTICAL and green -- exactly as they already inject
``installer._pip_install_editable`` / ``installer._validate`` -- this autouse
fixture neutralizes the provenance seam to a no-op for every test that is NOT
explicitly marked ``real_provenance``. The real gate (accept + every refusal
direction, the true installer wiring, and verification-before-mutation ordering)
is proven against real SSH signatures in ``test_provenance_r1.py``, whose tests
carry the ``real_provenance`` marker and therefore run the UNPATCHED gate.

This mirrors the accepted-core testing philosophy: an unrelated, already-proven
subsystem is stubbed so a suite can isolate the behavior it owns. It does not
weaken F4 -- it scopes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from machine_spirit_4.hermes_admin import installer
from machine_spirit_4.scripts.runtime_common import WINDOWS_HERMES_GIT_BASH_PATH


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_provenance: run the real F4 provenance gate for this test "
        "(do NOT neutralize installer._verify_release_provenance).",
    )
    config.addinivalue_line(
        "markers",
        "real_git_bash_preflight: run the real Windows Git Bash identity gate "
        "with deterministic test fakes.",
    )


@pytest.fixture(autouse=True)
def _neutralize_provenance_gate(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    """Default: replace the F4 seam with a no-op so the pre-F4 suites stay green.

    Opt a test into the real gate with ``@pytest.mark.real_provenance``.
    """
    if installer.sys.platform == "win32":
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", WINDOWS_HERMES_GIT_BASH_PATH)
    if request.node.get_closest_marker("real_provenance") is not None:
        return
    monkeypatch.setattr(installer, "_verify_release_origin_preflight", lambda **_kwargs: "origin")
    monkeypatch.setattr(installer, "_verify_release_provenance", lambda **_kwargs: None)


@pytest.fixture(autouse=True)
def _neutralize_git_bash_preflight(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep unrelated updater tests deterministic and off the host binary."""
    if request.node.get_closest_marker("real_git_bash_preflight") is not None:
        return
    monkeypatch.setattr(installer, "_require_valid_git_bash_candidate", lambda: None)


# --------------------------------------------------------------------------- #
# F4 parent-git-escape repair (R2): basetemp isolation guard.
#
# The shipped bug turned a failed nested ``git init`` (e.g. a Windows MAX_PATH
# overflow inside a deep pytest tmp dir) into a ``git -C child`` that walked UP
# and committed the enclosing checkout as "seed" by "F4 Test". Defense-in-depth
# at the SESSION root: refuse to run this suite at all when pytest's basetemp is
# itself inside a git repository (so a walk-up would have a real parent to find)
# or is long enough to risk the MAX_PATH overflow that triggers the failed init.
# --------------------------------------------------------------------------- #
_MAX_BASETEMP_LEN = 150  # conservative Windows MAX_PATH budget for nested tmp dirs


def _enclosing_repo(start: Path):
    """Return the nearest ancestor (incl. ``start``) containing ``.git``, or None."""
    resolved = Path(start).resolve()
    for candidate in [resolved, *resolved.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _basetemp_rejection_reason(root: Path):
    """Why this basetemp is unsafe for the suite, or ``None`` if it is safe.

    Pure and deterministic so the R2 isolation-regression suite can assert both
    the rejection directions and the no-false-positive (safe root) direction
    without spawning pytest.
    """
    root = Path(root)
    enclosing = _enclosing_repo(root)
    if enclosing is not None:
        return (
            f"basetemp {root} is inside git repo {enclosing}: a failed nested "
            f"'git init' could walk up and mutate that parent checkout"
        )
    if len(str(root)) > _MAX_BASETEMP_LEN:
        return (
            f"basetemp {root} is {len(str(root))} chars (> {_MAX_BASETEMP_LEN}); "
            f"Windows MAX_PATH risk can make a nested 'git init' fail"
        )
    return None


@pytest.fixture(autouse=True)
def _guard_basetemp_isolated(tmp_path_factory: pytest.TempPathFactory):
    """Abort fail-closed if the session temp root is unsafe (see notes above)."""
    reason = _basetemp_rejection_reason(Path(tmp_path_factory.getbasetemp()))
    if reason is not None:
        pytest.fail(reason)
