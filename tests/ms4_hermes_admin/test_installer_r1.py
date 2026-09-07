"""Hermes cross-platform updater R1 behaviors (F2 clean-check + F3 rollback).

Everything here is hermetic and self-contained:

* Real, throwaway Git repositories (created under ``tmp_path``) exercise the
  actual ``git`` executable for every lifecycle/rollback assertion, so option
  parsing, dirty-worktree gating, commit pinning, branch-vs-detached restore,
  and untracked-file handling are proven against real git -- never a fake.
* Only ``pip`` and the Python validation subprocess are injected (there is no
  real ``hermes-agent`` wheel to install in CI), and they are injected as
  argv-free no-ops / controllable failures. Git is never faked.
* No network: ``versioning`` release lookups are monkeypatched to fixed values
  and a local bare "remote" provides the release tags.

The companion suite ``test_installer.py`` keeps the pre-existing 49 tests; this
file adds the F2/F3-specific coverage the audit called out (including the
managed-subtree ignore behavior that replaced the old self-blocking
``untracked-plugin`` refusal case).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from machine_spirit_4.hermes_admin import installer, state, versioning


# --------------------------------------------------------------------------- #
# Real-git fixtures (self-contained; no dependency on test_installer.py).
# --------------------------------------------------------------------------- #
def _git(cwd: Path | None, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        pytest.fail(f"git fixture failed: {result.args!r}\nstdout={result.stdout}\nstderr={result.stderr}")
    return result


def _init_repo(path: Path, *, bare: bool = False) -> None:
    args = ["init"]
    if bare:
        args.append("--bare")
    args.append(str(path))
    _git(None, *args)
    if not bare:
        _git(path, "config", "user.name", "Hermes R1 Test")
        _git(path, "config", "user.email", "hermes-r1@example.invalid")
        _git(path, "config", "commit.gpgsign", "false")


def _make_multi_release_remote(tmp_path: Path, tags: list[str]) -> tuple[Path, dict[str, str]]:
    """Bare "remote" carrying ``tags`` on sequential commits."""
    remote = tmp_path / "release remote.git"
    seed = tmp_path / "release seed"
    _init_repo(remote, bare=True)
    _init_repo(seed)
    _git(seed, "remote", "add", "origin", str(remote))
    commits: dict[str, str] = {}
    for idx, tag in enumerate(tags):
        (seed / f"release_{idx}.txt").write_text(f"release {idx}\n", encoding="utf-8")
        _git(seed, "add", "-A")
        _git(seed, "commit", "-m", f"release {idx}")
        commit = _git(seed, "rev-parse", "HEAD").stdout.strip()
        _git(seed, "update-ref", f"refs/tags/{tag}", commit)
        _git(seed, "push", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
        commits[tag] = commit
    return remote, commits


def _make_target_checkout(tmp_path: Path, remote: Path, *, name: str = "target checkout") -> tuple[Path, str]:
    target = tmp_path / name
    _init_repo(target)
    (target / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(target, "add", "tracked.txt")
    _git(target, "commit", "-m", "baseline")
    baseline = _git(target, "rev-parse", "HEAD").stdout.strip()
    _git(target, "remote", "add", "origin", str(remote))
    return target, baseline


def _make_plugin_src(tmp_path: Path, *, name: str = "plugin_src") -> Path:
    src = tmp_path / name
    (src / "sub").mkdir(parents=True)
    (src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "sub" / "nested.txt").write_text("nested\n", encoding="utf-8")
    return src


def _managed_dst(directory: Path) -> Path:
    return directory / "plugins" / "ms4_consciousness"


def _status(target: Path) -> str:
    return _git(target, "status", "--porcelain=v1", "--untracked-files=all").stdout


def _current_branch(target: Path) -> str | None:
    res = _git(target, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    return res.stdout.strip() if res.returncode == 0 else None


def _begin(tmp_path: Path) -> None:
    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")


def _use_editable_versioning(
    monkeypatch,
    directory: Path,
    tag_by_version: dict[str, str],
    *,
    latest: str,
    vetted_origin: str,
) -> None:
    versioning._clear_caches_for_test()
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: vetted_origin,
    )
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {"mode": "editable", "version": "0.13.0", "directory": directory},
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        lambda force_refresh=False: versioning.CachedLatest(
            version=latest,
            published_at="2026-05-16T00:00:00Z",
            tag_name=tag_by_version.get(latest, f"v{latest}"),
            html_url="https://example/release",
            fetched_at_unix=0.0,
        ),
    )
    monkeypatch.setattr(installer.versioning, "recent_releases", lambda force_refresh=False: [])
    monkeypatch.setattr(
        installer.versioning,
        "resolve_git_tag",
        lambda target: tag_by_version.get(target.lstrip("v"), f"v{target.lstrip('v')}"),
    )


def _mock_pip_and_validate(monkeypatch) -> None:
    """pip + validation are the only injected boundaries; git stays real."""
    monkeypatch.setattr(installer, "_pip_install_editable", lambda *a, **k: None)
    monkeypatch.setattr(installer, "_reinstall_editable", lambda *a, **k: None)
    monkeypatch.setattr(
        installer,
        "_validate",
        lambda python, *, expected_version, editable_directory=None, expected_commit=None: expected_version,
    )


# --------------------------------------------------------------------------- #
# F2 -- clean-check path containment (pure-unit, deterministic).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [
        "plugins/ms4_consciousness/x.py",
        "plugins/ms4_consciousness/a/b/c.txt",
        "plugins/ms4_consciousness/.keep",
    ],
)
def test_is_within_managed_plugin_accepts_strict_children(path):
    assert installer._is_within_managed_plugin(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "",  # empty
        "plugins",  # parent only
        "plugins/ms4_consciousness",  # bare dir, no child
        "plugins/ms4_consciousness_backup/x",  # path-prefix sibling
        "plugins/ms4_consciousnessX/y",  # prefix collision
        "plugins/other/x",  # unrelated sibling
        "Plugins/ms4_consciousness/x",  # case variant (parent)
        "plugins/MS4_Consciousness/x",  # case variant (name)
        "plugins/ms4_consciousness/../evil",  # traversal out
        "plugins/ms4_consciousness/./x",  # dot segment
        "plugins/ms4_consciousness//x",  # empty segment
        "plugins\\ms4_consciousness\\x",  # backslash (never a git path)
        "../plugins/ms4_consciousness/x",  # escape prefix
        "ms4_consciousness/x",  # missing parent
    ],
)
def test_is_within_managed_plugin_rejects_everything_else(path):
    assert installer._is_within_managed_plugin(path) is False


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (" M tracked.py", ["tracked.py"]),
        ("?? new file.txt", ["new file.txt"]),
        ("MM plugins/ms4_consciousness/x", ["plugins/ms4_consciousness/x"]),
        ("R  old.py -> new.py", ["old.py", "new.py"]),
        ("C  a.py -> b.py", ["a.py", "b.py"]),
    ],
)
def test_status_paths_parses_known_shapes(line, expected):
    assert installer._status_paths(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "",  # too short
        "M",  # too short
        ' M "quoted.py"',  # C-quoted path -> fail closed
        'R  "q" -> plain',  # quoted origin -> fail closed
        " M ",  # empty path field
    ],
)
def test_status_paths_fails_closed_on_ambiguous_lines(line):
    assert installer._status_paths(line) is None


# --------------------------------------------------------------------------- #
# F2 -- clean-check against a REAL git checkout.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "relpath",
    [
        "operator.txt",  # unrelated untracked, repo root
        "docs/notes.md",  # unrelated untracked, nested
        "plugins/__init__.py",  # inside plugins/ but NOT the managed dir
        "plugins/ms4_consciousness_backup/keep.txt",  # path-prefix sibling
        "plugins/ms4_consciousness.bak/relic.txt",  # backup relic
        "plugins/other/x.txt",  # unrelated sibling under plugins/
        ".ms4-upgrade-backup-2026-05-18/operator.txt",  # unexpected backup content
        ".ms4-upgrade-backup-2026-05-18-extra/ms4_consciousness/client.py",
    ],
)
def test_clean_check_refuses_untracked_outside_managed_subtree(tmp_path, relpath):
    _begin(tmp_path)
    remote, _c = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, _baseline = _make_target_checkout(tmp_path, remote)
    victim = target / relpath
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(installer.HermesUpgradeError, match="outside the MS4-managed"):
        installer._ensure_clean_checkout(target)

    assert victim.read_text(encoding="utf-8") == "operator-owned\n"


def test_clean_check_refuses_tracked_modification(tmp_path):
    _begin(tmp_path)
    remote, _c = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, _baseline = _make_target_checkout(tmp_path, remote)
    backup = target / ".ms4-upgrade-backup-2026-05-18" / "ms4_consciousness"
    backup.mkdir(parents=True)
    (backup / "client.py").write_text("managed plugin backup\n", encoding="utf-8")
    (target / "tracked.txt").write_text("operator edit\n", encoding="utf-8")

    with pytest.raises(
        installer.HermesUpgradeError, match="tracked or untracked changes"
    ) as caught:
        installer._ensure_clean_checkout(target)

    assert (target / "tracked.txt").read_text(encoding="utf-8") == "operator edit\n"
    assert "tracked.txt" in str(caught.value)
    assert ".ms4-upgrade-backup" not in str(caught.value)


def test_clean_check_refuses_tracked_edit_under_backup_named_tree(tmp_path):
    _begin(tmp_path)
    remote, _c = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, _baseline = _make_target_checkout(tmp_path, remote)
    tracked = (
        target
        / ".ms4-upgrade-backup-2026-05-18"
        / "ms4_consciousness"
        / "client.py"
    )
    tracked.parent.mkdir(parents=True)
    tracked.write_text("committed\n", encoding="utf-8")
    _git(target, "add", tracked.relative_to(target).as_posix())
    _git(target, "commit", "-m", "track reserved-looking path")
    tracked.write_text("operator edit\n", encoding="utf-8")

    with pytest.raises(installer.HermesUpgradeError) as caught:
        installer._ensure_clean_checkout(target)

    assert tracked.relative_to(target).as_posix() in str(caught.value)
    assert tracked.read_text(encoding="utf-8") == "operator edit\n"


def test_clean_check_ignores_untracked_files_inside_managed_subtree(tmp_path):
    _begin(tmp_path)
    remote, _c = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, _baseline = _make_target_checkout(tmp_path, remote)
    managed = _managed_dst(target)
    (managed / "sub").mkdir(parents=True)
    (managed / "__init__.py").write_text("", encoding="utf-8")
    (managed / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    (managed / "sub" / "nested.txt").write_text("nested\n", encoding="utf-8")
    # Sanity: git really does report these untracked paths.
    assert "plugins/ms4_consciousness/" in _status(target)

    # No raise: the managed subtree is the one and only ignored region.
    installer._ensure_clean_checkout(target)


def test_clean_check_ignores_legacy_ms4_upgrade_backup(tmp_path):
    _begin(tmp_path)
    remote, _c = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, _baseline = _make_target_checkout(tmp_path, remote)
    backup = target / ".ms4-upgrade-backup-2026-05-18"
    plugin_backup = backup / "ms4_consciousness"
    plugin_backup.mkdir(parents=True)
    (plugin_backup / "client.py").write_text("managed plugin backup\n", encoding="utf-8")
    (backup / "test_ms4_consciousness_plugin.py").write_text(
        "managed test backup\n", encoding="utf-8"
    )
    assert ".ms4-upgrade-backup-2026-05-18/" in _status(target)

    installer._ensure_clean_checkout(target)


def test_clean_check_mixed_refuses_when_any_path_is_outside(tmp_path):
    _begin(tmp_path)
    remote, _c = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, _baseline = _make_target_checkout(tmp_path, remote)
    managed = _managed_dst(target)
    managed.mkdir(parents=True)
    (managed / "ok.txt").write_text("managed\n", encoding="utf-8")
    outside = target / "operator.txt"
    outside.write_text("operator\n", encoding="utf-8")

    with pytest.raises(installer.HermesUpgradeError, match="outside the MS4-managed"):
        installer._ensure_clean_checkout(target)

    assert outside.read_text(encoding="utf-8") == "operator\n"


# --------------------------------------------------------------------------- #
# F2 -- symlink / junction escape of the managed subtree.
# --------------------------------------------------------------------------- #
def _make_dir_link(link: Path, target: Path) -> str:
    """Create a directory symlink, or (Windows, no privilege) a junction.

    Returns the link kind. Raises RuntimeError if neither is possible so the
    caller can skip honestly rather than fake platform evidence.
    """
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        # Junction via the cmd builtin ``mklink /J`` (no admin required). This
        # is test-fixture scaffolding only; product code never shells out.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and link.exists():
            return "junction"
    raise RuntimeError("cannot create a directory symlink/junction on this host/privilege level")


def test_managed_containment_accepts_a_real_contained_directory(tmp_path):
    directory = tmp_path / "checkout"
    managed = _managed_dst(directory)
    managed.mkdir(parents=True)
    (managed / "plugin.yaml").write_text("x\n", encoding="utf-8")
    assert installer._managed_plugin_is_safely_contained(directory) is True


def test_managed_containment_rejects_symlink_or_junction_escape(tmp_path):
    directory = tmp_path / "checkout"
    (directory / "plugins").mkdir(parents=True)
    outside = tmp_path / "outside_target"
    (outside / "evil.txt").parent.mkdir(parents=True, exist_ok=True)
    (outside / "evil.txt").write_text("outside\n", encoding="utf-8")
    link = _managed_dst(directory)
    try:
        kind = _make_dir_link(link, outside)
    except RuntimeError as exc:  # pragma: no cover - privilege dependent
        pytest.skip(str(exc))

    assert installer._managed_plugin_is_safely_contained(directory) is False, f"link kind={kind}"


def test_managed_backup_containment_rejects_symlink_or_junction_escape(tmp_path):
    directory = tmp_path / "checkout"
    directory.mkdir()
    outside = tmp_path / "outside_backup"
    plugin = outside / "ms4_consciousness"
    plugin.mkdir(parents=True)
    (plugin / "client.py").write_text("outside\n", encoding="utf-8")
    link = directory / ".ms4-upgrade-backup-2026-05-18"
    try:
        kind = _make_dir_link(link, outside)
    except RuntimeError as exc:  # pragma: no cover - privilege dependent
        pytest.skip(str(exc))

    assert installer._managed_backup_path_is_safely_contained(
        directory, ".ms4-upgrade-backup-2026-05-18/ms4_consciousness/client.py"
    ) is False, f"link kind={kind}"


def test_clean_check_refuses_when_managed_subtree_escapes_containment(tmp_path, monkeypatch):
    """Wiring proof (deterministic): when every dirty path claims to be inside
    the managed subtree but containment cannot be proven, fail closed on all.
    """
    _begin(tmp_path)
    directory = tmp_path / "checkout"
    directory.mkdir()

    def fake_run(cmd, *, cwd=None, timeout=30):
        if "for-each-ref" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        assert "status" in cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout="?? plugins/ms4_consciousness/x.txt\n M plugins/ms4_consciousness/y.txt\n", stderr=""
        )

    monkeypatch.setattr(installer, "_run", fake_run)
    monkeypatch.setattr(installer, "_managed_plugin_is_safely_contained", lambda d: False)

    with pytest.raises(installer.HermesUpgradeError, match="outside the MS4-managed"):
        installer._ensure_clean_checkout(directory)


# --------------------------------------------------------------------------- #
# F3 -- success path + repeat update (real git).
# --------------------------------------------------------------------------- #
def test_editable_success_retains_new_commit_and_plugin(tmp_path, monkeypatch):
    remote, commits = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, baseline = _make_target_checkout(tmp_path, remote)
    plugin_src = _make_plugin_src(tmp_path)
    _use_editable_versioning(monkeypatch, target, {"0.14.0": "v0.14.0"}, latest="0.14.0", vetted_origin=str(remote))
    _mock_pip_and_validate(monkeypatch)
    _begin(tmp_path)

    result = installer.run_update_job(
        target_version="0.14.0", plugin_src=plugin_src, venv_python=tmp_path / "py", vetted_origin=str(remote)
    )

    assert result["to_version"] == "0.14.0"
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == commits["v0.14.0"]
    assert _git(target, "rev-parse", "HEAD").stdout.strip() != baseline
    managed = _managed_dst(target)
    assert (managed / "plugin.yaml").read_text(encoding="utf-8") == "name: ms4_consciousness\n"
    assert (managed / "sub" / "nested.txt").read_text(encoding="utf-8") == "nested\n"
    assert state.last_update().status == "success"


def test_repeat_update_passes_clean_check_after_managed_plugin_stamp(tmp_path, monkeypatch):
    remote, commits = _make_multi_release_remote(tmp_path, ["v0.14.0", "v0.15.0"])
    ms4_root = tmp_path / "ms4"
    target, _baseline = _make_target_checkout(
        tmp_path,
        remote,
        name=Path("ms4") / "runtime" / "hermes-managed",
    )
    plugin_src = _make_plugin_src(tmp_path)
    _mock_pip_and_validate(monkeypatch)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_validate_managed_install",
        lambda _python, *, expected_version, **_kwargs: expected_version,
    )

    # First update stamps the (untracked) plugin in the exact managed checkout.
    _use_editable_versioning(monkeypatch, target, {"0.14.0": "v0.14.0"}, latest="0.14.0", vetted_origin=str(remote))
    _begin(tmp_path)
    first = installer.run_update_job(target_version="0.14.0", plugin_src=plugin_src, venv_python=tmp_path / "py", vetted_origin=str(remote))
    assert first["to_version"] == "0.14.0"
    assert (_managed_dst(target) / "plugin.yaml").exists()
    assert "plugins/ms4_consciousness/" in _status(target)  # untracked, present

    # Managed-path allowances apply here, so repeat update must not self-block.
    _use_editable_versioning(monkeypatch, target, {"0.15.0": "v0.15.0"}, latest="0.15.0", vetted_origin=str(remote))
    _begin(tmp_path)
    second = installer.run_update_job(target_version="0.15.0", plugin_src=plugin_src, venv_python=tmp_path / "py", vetted_origin=str(remote))

    assert second["to_version"] == "0.15.0"
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == commits["v0.15.0"]
    assert (_managed_dst(target) / "plugin.yaml").exists()
    assert state.last_update().status == "success"


# --------------------------------------------------------------------------- #
# F3 -- rollback restores exact prior state on each failure point (real git).
# --------------------------------------------------------------------------- #
def _prime_editable_failure(tmp_path, monkeypatch):
    remote, commits = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, baseline = _make_target_checkout(tmp_path, remote)
    plugin_src = _make_plugin_src(tmp_path)
    vetted_origin = str(remote)
    _use_editable_versioning(monkeypatch, target, {"0.14.0": "v0.14.0"}, latest="0.14.0", vetted_origin=vetted_origin)
    _mock_pip_and_validate(monkeypatch)
    _begin(tmp_path)
    return target, baseline, commits, plugin_src, vetted_origin


def _inject_checkout_failure(monkeypatch):
    real = installer._git_detach_checkout

    def failing(directory, commit):
        real(directory, commit)  # real mutation happens...
        raise installer.HermesUpgradeError("simulated checkout failure")

    monkeypatch.setattr(installer, "_git_detach_checkout", failing)


def _inject_stamp_failure(monkeypatch):
    real = installer._restamp_managed_plugin
    seen = {"n": 0}

    def flaky(plugin_src, directory):
        seen["n"] += 1
        if seen["n"] == 1:
            raise installer.HermesUpgradeError("simulated stamp failure")
        return real(plugin_src, directory)  # rollback re-stamp is real

    monkeypatch.setattr(installer, "_restamp_managed_plugin", flaky)


def _inject_pip_failure(monkeypatch):
    def boom(*_a, **_k):
        raise installer.HermesUpgradeError("simulated pip failure")

    monkeypatch.setattr(installer, "_pip_install_editable", boom)


def _inject_validation_failure(monkeypatch):
    def boom(*_a, **_k):
        raise installer.HermesUpgradeError("simulated validation failure")

    monkeypatch.setattr(installer, "_validate", boom)


@pytest.mark.parametrize(
    ("injector", "needle"),
    [
        (_inject_checkout_failure, "simulated checkout failure"),
        (_inject_stamp_failure, "simulated stamp failure"),
        (_inject_pip_failure, "simulated pip failure"),
        (_inject_validation_failure, "simulated validation failure"),
    ],
    ids=["checkout", "stamp", "pip", "validation"],
)
def test_editable_failure_rolls_back_to_prior_branch_state(tmp_path, monkeypatch, injector, needle):
    target, baseline, _commits, plugin_src, vetted_origin = _prime_editable_failure(tmp_path, monkeypatch)
    assert _current_branch(target) is not None  # start on a branch
    injector(monkeypatch)

    with pytest.raises(installer.HermesUpgradeError) as caught:
        installer.run_update_job(target_version="0.14.0", plugin_src=plugin_src, venv_python=tmp_path / "py", vetted_origin=vetted_origin)

    message = str(caught.value)
    assert needle in message
    assert "rolled back to the prior state" in message
    # Exact prior HEAD + checkout MODE restored.
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == baseline
    assert _current_branch(target) is not None  # back on the branch, not detached
    # Managed plugin re-stamped from source-of-truth.
    assert (_managed_dst(target) / "plugin.yaml").read_text(encoding="utf-8") == "name: ms4_consciousness\n"
    final = state.last_update()
    assert final.status == "failed"
    assert needle in (final.error or "")
    assert any(entry["phase"] == "rolled_back" for entry in final.progress)


def test_editable_failure_rolls_back_to_prior_detached_state(tmp_path, monkeypatch):
    target, baseline, _commits, plugin_src, vetted_origin = _prime_editable_failure(tmp_path, monkeypatch)
    _git(target, "checkout", "--detach", baseline)  # pre-state: DETACHED
    assert _current_branch(target) is None
    _inject_validation_failure(monkeypatch)

    with pytest.raises(installer.HermesUpgradeError, match="rolled back to the prior state"):
        installer.run_update_job(target_version="0.14.0", plugin_src=plugin_src, venv_python=tmp_path / "py", vetted_origin=vetted_origin)

    assert _git(target, "rev-parse", "HEAD").stdout.strip() == baseline
    assert _current_branch(target) is None  # restored AS detached, not forced onto a branch


def test_rollback_failure_is_fatal_and_preserves_both_errors(tmp_path, monkeypatch):
    target, _baseline, _commits, plugin_src, vetted_origin = _prime_editable_failure(tmp_path, monkeypatch)
    _inject_pip_failure(monkeypatch)

    def restore_boom(_pre):
        raise installer.HermesUpgradeError("simulated rollback restore failure")

    monkeypatch.setattr(installer, "_restore_editable_commit", restore_boom)

    with pytest.raises(installer.HermesUpgradeError) as caught:
        installer.run_update_job(target_version="0.14.0", plugin_src=plugin_src, venv_python=tmp_path / "py", vetted_origin=vetted_origin)

    message = str(caught.value)
    assert "manual recovery required" in message
    assert "simulated pip failure" in message  # primary
    assert "simulated rollback restore failure" in message  # rollback
    final = state.last_update()
    assert final.status == "failed"
    assert any(entry["phase"] == "rollback_failed" for entry in final.progress)


# --------------------------------------------------------------------------- #
# F3 -- destructive-command tripwire + prestate capture.
# --------------------------------------------------------------------------- #
def test_rollback_emits_no_destructive_git_or_deletion_commands(tmp_path, monkeypatch):
    """The full failure+rollback lifecycle must never emit ``git reset --hard``,
    ``git clean``, ``git restore``, or ``git rm``, and the only recursive
    filesystem delete must target the managed subtree. When the whole-commit
    rollback checkout already restores the prior tree (this scenario) no
    path-level ``git checkout <prior> -- <path>`` is emitted either; that
    scoped repair exists only for paths the forward checkout clobbered and is
    proven in ``test_product_update_gate.py`` (ignore-case collision).
    """
    target, baseline, _commits, plugin_src, vetted_origin = _prime_editable_failure(tmp_path, monkeypatch)

    recorded: list[list[str]] = []
    real_run = subprocess.run

    def spy_run(cmd, **kwargs):
        recorded.append([str(c) for c in cmd])
        return real_run(cmd, **kwargs)

    rmtree_targets: list[str] = []
    real_rmtree = shutil.rmtree

    def spy_rmtree(path, *a, **k):
        rmtree_targets.append(str(path))
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(installer.subprocess, "run", spy_run)
    monkeypatch.setattr(installer.shutil, "rmtree", spy_rmtree)
    _inject_validation_failure(monkeypatch)

    with pytest.raises(installer.HermesUpgradeError, match="rolled back to the prior state"):
        installer.run_update_job(target_version="0.14.0", plugin_src=plugin_src, venv_python=tmp_path / "py", vetted_origin=vetted_origin)

    assert recorded, "expected at least one git command"
    git_cmds = [c for c in recorded if Path(c[0]).name.lower() in {"git", "git.exe"}]
    assert git_cmds, "expected real git commands during the lifecycle"
    for cmd in git_cmds:
        assert "reset" not in cmd, cmd
        assert "--hard" not in cmd, cmd
        assert "clean" not in cmd, cmd
        assert "restore" not in cmd, cmd
        assert "rm" not in cmd, cmd
        if "checkout" in cmd:
            after = cmd[cmd.index("checkout") + 1:]
            assert "--" not in after, cmd  # never `git checkout -- <path>`
    # Recursive delete is confined to the managed subtree, nothing else.
    for tgt in rmtree_targets:
        normalized = tgt.replace("\\", "/")
        assert normalized.endswith("plugins/ms4_consciousness"), tgt
    # And the prior state really was restored.
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == baseline


def test_capture_prestate_reports_branch_then_detached(tmp_path, monkeypatch):
    remote, _commits = _make_multi_release_remote(tmp_path, ["v0.14.0"])
    target, baseline = _make_target_checkout(tmp_path, remote)
    _begin(tmp_path)

    on_branch = installer._capture_editable_prestate(target)
    assert on_branch.mode == "branch"
    assert on_branch.branch is not None
    assert on_branch.commit == baseline.lower()

    _git(target, "checkout", "--detach", baseline)
    detached = installer._capture_editable_prestate(target)
    assert detached.mode == "detached"
    assert detached.branch is None
    assert detached.commit == baseline.lower()
