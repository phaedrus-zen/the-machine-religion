"""F4 parent-git-escape repair (R2) -- direction-sensitive regression lock.

Background
----------
The shipped F4 provenance/installer git fixtures created throwaway repos with a
nested ``git init`` whose failure was swallowed (``check=False``). On Windows a
deep pytest ``tmp_path`` could overflow MAX_PATH, so ``git init`` failed, the
child directory had no valid ``.git``, and the very next ``git -C <child>``
config/add/commit *walked UP* and committed the enclosing dirty checkout as
"seed" by "F4 Test" -- an accidental parent-repo mutation.

This module locks the fail-closed repair. It imports the LIVE (post-fix) shipped
helpers -- ``_g`` / ``_init`` / ``_assert_isolated_repo`` / ``_commit_repo`` from
``test_provenance_r1``, ``_init_repo`` / ``_assert_isolated_repo`` from
``test_installer``, and the basetemp guard from ``conftest`` -- and drives them
against DISPOSABLE "sentinel parent" repos created under ``tmp_path``. Every
enumerated unsafe direction must be REJECTED before any parent mutation, and the
genuinely-isolated happy path must still be ACCEPTED (no over-broad guard).

Safety
------
Every git-mutating attempt targets a sentinel parent that this module creates
under ``tmp_path`` (which ``conftest._guard_basetemp_isolated`` proves is not
inside any real checkout). No real repository is ever the walk-up target. A
module-scoped tripwire additionally asserts that the checkout SHIPPING this test
(the enclosing repo of ``__file__``) has an unchanged HEAD across the whole run.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git not on PATH")

# TMR checkout root (…/TMR/tests/ms4_hermes_admin/this_file.py -> parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_HERE = Path(__file__).resolve().parent


def _load_sibling(mod_name: str, filename: str):
    """Load a sibling test/conftest module by PATH under a distinct name.

    Distinct name avoids clobbering pytest's own import of the same file; loading
    by path is import-mode independent so we always exercise the LIVE helpers.
    """
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _load_sibling("_f4r2_shipped_provenance", "test_provenance_r1.py")
ti = _load_sibling("_f4r2_shipped_installer", "test_installer.py")
cf = _load_sibling("_f4r2_conftest", "conftest.py")


# --------------------------------------------------------------------------- #
# Raw git helpers (independent of the code under test).
# --------------------------------------------------------------------------- #
def _run(cwd, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [GIT, "-C", str(cwd), *args],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False,
    )


def _sentinel_parent(tmp_path: Path, name: str = "parent") -> Path:
    """A throwaway DIRTY 'operator checkout' living only under tmp_path."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    assert _run(root, "init", "-q").returncode == 0
    _run(root, "config", "user.name", "Sentinel")
    _run(root, "config", "user.email", "sentinel@example.invalid")
    _run(root, "config", "commit.gpgsign", "false")
    (root / "keep.txt").write_text("baseline\n", encoding="utf-8")
    _run(root, "add", "-A")
    _run(root, "commit", "-q", "-m", "sentinel-baseline")
    # make it DIRTY, like the canonical repo was at incident time
    (root / "dirty_tracked.txt").write_text("uncommitted-important-work\n", encoding="utf-8")
    return root


def _parent_state(root: Path) -> dict:
    """Commit identity of ``root`` -- the exact escape invariant.

    A parent-git escape manifests as a NEW COMMIT / moved ref (HEAD, count,
    subject, author change). Untracked scratch files this module writes *inside*
    a sentinel parent are deliberately excluded: they are our own noise, not a
    git mutation, so they must not be part of the "was the parent mutated" gate.
    """
    def g(*a):
        return _run(root, *a).stdout.strip()

    return {
        "head": g("rev-parse", "HEAD"),
        "count": g("rev-list", "--count", "HEAD"),
        "last_subject": g("log", "-1", "--format=%s"),
        "last_author": g("log", "-1", "--format=%an <%ae>"),
    }


def _deep_overflow_dir(base: Path, fill: str) -> Path:
    """A directory whose ``<dir>\\repo`` path is ~250 chars (Windows MAX_PATH)."""
    pad = max(60, 250 - len(str(base)) - len("\\repo") - 1)
    return base / (fill * pad)


# --------------------------------------------------------------------------- #
# Tripwire: the checkout that SHIPS this test must never gain a commit.
# --------------------------------------------------------------------------- #
def _repo_head(repo: Path) -> str:
    p = _run(repo, "rev-parse", "HEAD")
    return p.stdout.strip() if p.returncode == 0 else f"<err:{p.returncode}>"


@pytest.fixture(scope="module", autouse=True)
def _own_checkout_tripwire():
    before = _repo_head(_REPO_ROOT)
    yield
    after = _repo_head(_REPO_ROOT)
    assert after == before, (
        f"own checkout HEAD changed during the regression run "
        f"({_REPO_ROOT}): {before} -> {after} -- a parent-git escape may have fired"
    )


# =========================================================================== #
# Direction 1: swallowed init failure -> _init now RAISES (never swallows).
# =========================================================================== #
def test_swallowed_init_failure_raises_deterministic(tmp_path):
    """Any nonzero ``git init`` aborts fail-closed before config/add/commit."""
    parent = _sentinel_parent(tmp_path, "d1")
    before = _parent_state(parent)
    blocker = parent / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")  # git init cannot mkdir under a file
    with pytest.raises(AssertionError, match="git init failed"):
        tp._init(blocker / "repo")
    assert _parent_state(parent) == before  # sentinel parent untouched


def test_swallowed_init_failure_maxpath(tmp_path):
    """The exact incident trigger: a MAX_PATH overflow must RAISE, not swallow."""
    parent = _sentinel_parent(tmp_path, "d2")
    before = _parent_state(parent)
    deep = _deep_overflow_dir(parent, "d")
    deep.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [GIT, "init", "-q", str(deep / "probe")],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False,
    )
    if probe.returncode == 0:
        pytest.skip("host does not overflow MAX_PATH here (long paths enabled)")
    with pytest.raises(AssertionError, match="git init failed"):
        tp._init(deep / "repo")
    assert _parent_state(parent) == before


# =========================================================================== #
# Direction 2: absent .git child + config/add/commit after failed init
#              -> the C1b escape. _g now REJECTS before mutating the parent.
# =========================================================================== #
def test_absent_git_child_config_rejected(tmp_path):
    parent = _sentinel_parent(tmp_path, "d3")
    before = _parent_state(parent)
    child = parent / "child_no_own_git"
    child.mkdir(parents=True, exist_ok=True)  # exists, NO .git -> git -C walks up
    with pytest.raises(AssertionError, match="refusing git write"):
        tp._g(child, "config", "user.name", "F4 Test")
    assert _parent_state(parent) == before


def test_absent_git_child_add_and_commit_rejected(tmp_path):
    """The shipped escape (add/commit walk-up) is now blocked; parent unchanged."""
    parent = _sentinel_parent(tmp_path, "d4")
    before = _parent_state(parent)
    child = parent / "child_no_own_git"
    child.mkdir(parents=True, exist_ok=True)
    (child / "f.txt").write_text("hello\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="refusing git write"):
        tp._g(child, "add", "-A")
    with pytest.raises(AssertionError, match="refusing git write"):
        tp._g(child, "commit", "-q", "-m", "seed")
    after = _parent_state(parent)
    assert after == before
    assert after["last_subject"] == "sentinel-baseline"  # never became "seed"
    assert after["last_author"] == "Sentinel <sentinel@example.invalid>"


# =========================================================================== #
# Direction 3: partial/empty .git child -> C1d. _g still REJECTS.
# =========================================================================== #
def test_partial_git_child_mutation_rejected(tmp_path):
    parent = _sentinel_parent(tmp_path, "d5")
    before = _parent_state(parent)
    child = parent / "child_partial_git"
    (child / ".git").mkdir(parents=True, exist_ok=True)  # init made .git then overflowed
    with pytest.raises(AssertionError, match="refusing git write"):
        tp._g(child, "commit", "-q", "-m", "seed")
    assert _parent_state(parent) == before


# =========================================================================== #
# Direction 4: wrong child top-level / parent discovery -> _assert rejects.
# =========================================================================== #
def test_assert_isolated_rejects_walkup_child(tmp_path):
    parent = _sentinel_parent(tmp_path, "d6")
    child = parent / "child_no_git"
    child.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AssertionError, match="refusing git write"):
        tp._assert_isolated_repo(child)


def test_parent_git_discovery_readonly_documented(tmp_path):
    """Document the hazard: a no-.git child DOES resolve to the parent top-level;
    the guard converts that discovery into a hard refusal (read-only, no mutation).
    """
    parent = _sentinel_parent(tmp_path, "d7")
    before = _parent_state(parent)
    child = parent / "sub" / "deeper"
    child.mkdir(parents=True, exist_ok=True)
    show_top = _run(child, "rev-parse", "--show-toplevel").stdout.strip()
    assert Path(show_top).resolve() == parent.resolve()  # walk-up hazard is real
    with pytest.raises(AssertionError, match="refusing git write"):
        tp._assert_isolated_repo(child)
    assert _parent_state(parent) == before


# =========================================================================== #
# Direction 5: temp root under a parent checkout / overlong root -> basetemp guard.
# =========================================================================== #
def test_basetemp_guard_rejects_root_inside_repo(tmp_path):
    parent = _sentinel_parent(tmp_path, "d8")
    inside = parent / "a" / "b" / "bt"
    inside.mkdir(parents=True, exist_ok=True)
    assert cf._enclosing_repo(inside) == parent.resolve()
    reason = cf._basetemp_rejection_reason(inside)
    assert reason is not None and "inside git repo" in reason


def test_basetemp_guard_rejects_overlong_root(tmp_path):
    long_root = tmp_path / ("z" * 200)  # > 150-char budget, not inside a repo
    reason = cf._basetemp_rejection_reason(long_root)
    assert reason is not None and "chars" in reason


def test_basetemp_guard_accepts_safe_root(tmp_path):
    """No over-broad rejection: a short root outside any repo is accepted."""
    safe = tmp_path / "safe_bt"
    safe.mkdir(parents=True, exist_ok=True)
    assert cf._enclosing_repo(safe) is None
    assert cf._basetemp_rejection_reason(safe) is None


# =========================================================================== #
# Direction 6: sibling installer path (test_installer._init_repo / _assert).
# =========================================================================== #
def test_sibling_installer_init_failcloses(tmp_path):
    """test_installer._init_repo aborts before any add/commit on a failed init."""
    parent = _sentinel_parent(tmp_path, "d9")
    before = _parent_state(parent)
    blocker = parent / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    raised = None
    try:
        ti._init_repo(blocker / "repo")  # _git(None,'init',...) check=True -> fails
    except BaseException as exc:  # pytest.fail raises BaseException (Failed)
        raised = exc
    assert raised is not None
    assert _parent_state(parent) == before


def test_sibling_installer_assert_rejects_walkup_child(tmp_path):
    parent = _sentinel_parent(tmp_path, "d10")
    child = parent / "child_no_git"
    child.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AssertionError, match="refusing git write"):
        ti._assert_isolated_repo(child)


# =========================================================================== #
# Direction 7: isolated happy path -> ACCEPTED (guard is not over-broad).
# =========================================================================== #
def test_isolated_child_happy_path_accepted(tmp_path):
    isolated = tmp_path / "isolated"
    tp._init(isolated)  # asserts isolation internally; must not raise
    tp._assert_isolated_repo(isolated)  # explicit positive control
    (isolated / "f.txt").write_text("hello\n", encoding="utf-8")
    tp._g(isolated, "add", "-A")  # mutating gate passes on a real isolated repo
    tp._g(isolated, "commit", "-q", "-m", "seed")
    head = tp._g(isolated, "rev-parse", "HEAD").stdout.strip()
    assert len(head) == 40


def test_commit_repo_helper_happy_path(tmp_path):
    repo, commit = tp._commit_repo(tmp_path, name="happy")
    assert (repo / ".git").is_dir()
    assert len(commit) == 40
    assert tp._g(repo, "rev-parse", "HEAD").stdout.strip() == commit


def test_installer_isolated_fixtures_happy_path(tmp_path):
    """The sibling guard does not break the installer's real-git fixtures."""
    remote, commit = ti._make_release_remote(tmp_path, "v9.9.9")
    assert len(commit) == 40
    target, baseline = ti._make_target_checkout(tmp_path, remote)
    assert len(baseline) == 40
    assert (target / ".git").is_dir()
