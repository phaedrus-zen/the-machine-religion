"""Installer phase, integrity, and rollback tests.

Most tests mock subprocess. Adversarial checkout tests use isolated
temporary Git repositories so option parsing, dirty-worktree gates,
and commit pinning are proven against the real executable without
touching an operator checkout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from machine_spirit_4.hermes_admin import installer, state, versioning
from machine_spirit_4.scripts import runtime_common


class FakeProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_subprocess(
    monkeypatch,
    *,
    fail_on: str | None = None,
    installed_version: str = "0.14.0",
):
    calls: list[list[str]] = []

    def fake_run(
        cmd,
        cwd=None,
        stdin=None,
        capture_output=True,
        text=True,
        timeout=None,
        check=False,
        env=None,
    ):
        calls.append(list(cmd))
        if fail_on and any(fail_on in str(c) for c in cmd):
            return FakeProcess(stderr=f"simulated failure: {fail_on}", returncode=1)
        if "for-each-ref" in cmd:
            return FakeProcess()
        if "status" in cmd and "--porcelain=v1" in cmd:
            return FakeProcess()
        if (
            "config" in cmd
            and "--null" in cmd
            and "--get-all" in cmd
            and "remote.origin.url" in cmd
        ):
            return FakeProcess(stdout=b"origin\0")
        if "ls-remote" in cmd and "--get-url" in cmd:
            return FakeProcess(stdout=f"{cmd[-1]}\n")
        if "rev-parse" in cmd:
            return FakeProcess(stdout=f"{'a' * 40}\n")
        if cmd[0].endswith(("python.exe", "python")) and "-c" in cmd:
            return FakeProcess(stdout=f"{installed_version}\n")
        return FakeProcess(stdout="ok\n")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    return calls


def _install_fake_versioning(monkeypatch, *, mode_value: str, directory: Path | None, tag_name: str = "v0.14.0"):
    versioning._clear_caches_for_test()
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {"mode": mode_value, "version": "0.13.0", "directory": directory},
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        lambda force_refresh=False: versioning.CachedLatest(
            version="0.14.0",
            published_at="2026-05-16T00:00:00Z",
            tag_name=tag_name,
            html_url="https://example/release",
            fetched_at_unix=0.0,
        ),
    )
    monkeypatch.setattr(installer.versioning, "recent_releases", lambda force_refresh=False: [])
    monkeypatch.setattr(installer.versioning, "resolve_git_tag", lambda target: tag_name)


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
        pytest.fail(
            f"real git fixture failed: {result.args!r}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def _assert_isolated_repo(path: Path, *, bare: bool = False) -> None:
    """Sibling defense-in-depth for the F4 parent-git-escape repair (R2).

    Mirrors ``test_provenance_r1._assert_isolated_repo``. Even though this file's
    ``_init_repo`` already fails closed on a nonzero ``git init`` (``_git`` uses
    ``check=True``), assert that the freshly-created repo is genuinely its OWN
    isolated git dir before any ``git -C <path>`` write -- so an init that
    "succeeds" into a non-isolated/partial state can never let a subsequent
    mutation escape to an enclosing parent checkout.
    """
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--absolute-git-dir"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"refusing git write: no isolated repo at {path} "
            f"(rev-parse failed rc={proc.returncode}: {proc.stderr.strip()!r})"
        )
    expected = os.path.normcase(os.path.realpath(
        str(path) if bare else os.path.join(str(path), ".git")))
    got = os.path.normcase(os.path.realpath(proc.stdout.strip()))
    if got != expected:
        raise AssertionError(
            f"refusing git write: {path} resolves to git dir {got!r}, not the "
            f"isolated child {expected!r} (would escape to a parent repo)"
        )


def _init_repo(path: Path, *, bare: bool = False) -> None:
    args = ["init"]
    if bare:
        args.append("--bare")
    args.append(str(path))
    _git(None, *args)
    # F4 parent-git-escape repair (R2): prove the freshly-created repo is its own
    # isolated git dir before any 'git -C <path>' write can walk up to a parent.
    _assert_isolated_repo(path, bare=bare)
    if not bare:
        _git(path, "config", "user.name", "Hermes Updater Test")
        _git(path, "config", "user.email", "hermes-updater@example.invalid")


def _make_release_remote(tmp_path: Path, tag: str) -> tuple[Path, str]:
    remote = tmp_path / "release remote.git"
    seed = tmp_path / "release seed"
    _init_repo(remote, bare=True)
    _init_repo(seed)
    (seed / "release.txt").write_text("release\n", encoding="utf-8")
    _git(seed, "add", "release.txt")
    _git(seed, "commit", "-m", "release")
    commit = _git(seed, "rev-parse", "HEAD").stdout.strip()
    tag_ref = f"refs/tags/{tag}"
    _git(seed, "update-ref", tag_ref, commit)
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", f"{tag_ref}:{tag_ref}")
    return remote, commit


def _make_target_checkout(tmp_path: Path, remote: Path) -> tuple[Path, str]:
    target = tmp_path / "target checkout"
    _init_repo(target)
    (target / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(target, "add", "tracked.txt")
    _git(target, "commit", "-m", "baseline")
    baseline = _git(target, "rev-parse", "HEAD").stdout.strip()
    _git(target, "remote", "add", "origin", str(remote))
    return target, baseline


def _install_fake_distribution(root: Path, version: str) -> None:
    dist_info = root / f"hermes_agent-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: hermes-agent\nVersion: {version}\n",
        encoding="utf-8",
    )
    plugin = root / "plugins" / "ms4_consciousness"
    plugin.mkdir(parents=True)
    (root / "plugins" / "__init__.py").write_text("", encoding="utf-8")
    (plugin / "__init__.py").write_text("", encoding="utf-8")


def test_editable_upgrade_runs_all_phases(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    hermes_dir = tmp_path / "hermes-agent"
    (hermes_dir / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    _install_fake_versioning(monkeypatch, mode_value="editable", directory=hermes_dir, tag_name="v2026.5.16")
    calls = _install_fake_subprocess(monkeypatch)

    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    result = installer.run_update_job(
        target_version="0.14.0",
        plugin_src=plugin_src,
        venv_python=venv_python,
        vetted_origin="origin",
    )

    final = state.last_update()
    assert final is not None
    assert final.status == "success"
    assert result["to_version"] == "0.14.0"
    phases = [entry["phase"] for entry in final.progress]
    for required in ("preflight", "fetching_release", "fetching_remote", "checking_out", "syncing_plugin", "pip_installing", "validating", "done"):
        assert required in phases, f"missing phase {required} in {phases}"
    plugin_dst = hermes_dir / "plugins" / "ms4_consciousness"
    assert (plugin_dst / "plugin.yaml").read_text(encoding="utf-8") == "name: ms4_consciousness\n"
    flat_cmds = [" ".join(c) for c in calls]
    assert any("git" in c and "fetch" in c for c in flat_cmds)
    assert any("checkout" in c and "--detach" in c and "a" * 40 in c for c in flat_cmds), flat_cmds
    assert any("pip" in c and "install" in c and "-e" in c for c in flat_cmds)
    fetch_cmd = next(c for c in calls if "fetch" in c)
    assert "--tags" not in fetch_cmd
    assert "--prune" not in fetch_cmd
    assert fetch_cmd[-1] == "refs/tags/v2026.5.16:refs/tags/v2026.5.16"
    assert any("--no-input" in c and "-e" in c for c in calls)


def test_pypi_upgrade_invokes_wheel_install(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    calls = _install_fake_subprocess(monkeypatch)

    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="pypi")
    result = installer.run_update_job(
        target_version="0.14.0",
        plugin_src=plugin_src,
        venv_python=venv_python,
    )

    final = state.last_update()
    assert final is not None
    assert final.status == "success"
    assert result["to_version"] == "0.14.0"
    assert any("hermes-agent==0.14.0" in " ".join(c) for c in calls)
    assert any("--no-input" in c for c in calls)
    assert not any("git" in c[0] for c in calls)


@pytest.mark.parametrize(
    ("platform_name", "root", "expected"),
    [
        (
            "win32",
            PureWindowsPath("C:/HiveMind/machine_spirit_4"),
            PureWindowsPath("C:/HiveMind/machine_spirit_4/.venv/Scripts/python.exe"),
        ),
        (
            "linux",
            PurePosixPath("/opt/HiveMind/machine_spirit_4"),
            PurePosixPath("/opt/HiveMind/machine_spirit_4/.venv/bin/python"),
        ),
        (
            "darwin",
            PurePosixPath("/Applications/HiveMind/machine_spirit_4"),
            PurePosixPath("/Applications/HiveMind/machine_spirit_4/.venv/bin/python"),
        ),
    ],
)
def test_default_venv_python_is_platform_native(platform_name, root, expected):
    assert installer._default_venv_python(root, platform_name=platform_name) == expected


@pytest.mark.parametrize(
    "directory",
    [
        PureWindowsPath("C:/Users/operator/Hermes Agent"),
        PurePosixPath("/opt/hermes-agent"),
        PurePosixPath("/Applications/Hermes Agent"),
    ],
    ids=["windows", "linux", "macos"],
)
def test_git_fetch_command_is_targeted_and_shell_free(directory):
    command = installer._git_fetch_command(directory, "v2026.7.1")
    assert command == [
        "git",
        "-C",
        str(directory),
        "-c",
        "credential.interactive=never",
        "-c",
        "maintenance.auto=false",
        "fetch",
        "--no-tags",
        "origin",
        "refs/tags/v2026.7.1:refs/tags/v2026.7.1",
    ]
    assert "--tags" not in command
    assert "--prune" not in command


def test_git_subprocess_is_noninteractive_and_status_command_is_truthful(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.16.0", to_version="0.18.0", install_mode="editable")
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return FakeProcess(stdout="ok\n")

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-tree"))
    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    cmd = ["git", "-C", r"C:\Hermes Checkout", "fetch"]
    installer._run(cmd, timeout=7)

    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert captured["env"]["GCM_INTERACTIVE"] == "Never"
    assert captured["env"]["GIT_OPTIONAL_LOCKS"] == "0"
    assert "GIT_DIR" not in captured["env"]
    assert "GIT_WORK_TREE" not in captured["env"]
    assert state.last_update().progress[-2]["note"] == f"$ {installer._display_command(cmd)}"
    assert installer._display_command(cmd, platform_name="win32") == (
        'git -C "C:\\Hermes Checkout" fetch'
    )
    posix_cmd = ["git", "-C", "/Applications/Hermes Agent", "fetch"]
    expected_posix = "git -C '/Applications/Hermes Agent' fetch"
    assert installer._display_command(posix_cmd, platform_name="linux") == expected_posix
    assert installer._display_command(posix_cmd, platform_name="darwin") == expected_posix


def test_windows_git_subprocess_forces_trusted_longpaths_config(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "alias.status")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "!echo attacker-controlled")
    monkeypatch.setenv("GIT_CONFIG_KEY_7", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_7", "attacker-hooks")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'credential.helper=attacker'")

    monkeypatch.setattr(installer.sys, "platform", "win32")
    windows_env = installer._command_env(["git", "status"])
    assert {
        key: value for key, value in windows_env.items() if key.startswith("GIT_CONFIG")
    } == {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.longpaths",
        "GIT_CONFIG_VALUE_0": "true",
    }

    monkeypatch.setattr(installer.sys, "platform", "linux")
    posix_env = installer._command_env(["git", "status"])
    assert not {key for key in posix_env if key.startswith("GIT_CONFIG")}


def test_provenance_git_runner_strips_repository_selection_env(
    tmp_path,
    monkeypatch,
):
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-tree"))
    monkeypatch.setattr(installer.provenance.subprocess, "run", fake_run)

    installer.provenance._default_git_runner(["git", "status"])

    assert "GIT_DIR" not in captured["env"]
    assert "GIT_WORK_TREE" not in captured["env"]
    assert captured["env"]["GIT_OPTIONAL_LOCKS"] == "0"


def test_timeout_error_preserves_the_exact_platform_command(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.16.0", to_version="0.18.0", install_mode="editable")
    cmd = ["git", "-C", r"C:\Hermes Checkout", "fetch"]

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=7)

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    with pytest.raises(installer.HermesUpgradeError) as caught:
        installer._run(cmd, timeout=7)

    message = str(caught.value)
    # The command is still reproduced verbatim with platform-exact quoting --
    # that is what this test has always guarded.
    assert installer._display_command(cmd) in message
    assert '"C:\\Hermes Checkout"' in message
    assert "after 7s" in message
    # R4: a timeout is now a distinct, self-describing outcome rather than an
    # error string indistinguishable from a command that actually failed.
    assert isinstance(caught.value, installer.HermesUpgradeTimeoutError)
    assert "TIMED OUT (did not fail)" in message


@pytest.mark.parametrize(
    "tag",
    [
        "--detach",
        "-B",
        ".v2026.7.1",
        "v2026..7.1",
        "v2026.7.1.",
        "v2026.7.1.lock",
        "v2026.7.1.LOCK",
    ],
)
def test_invalid_git_tag_is_rejected_before_command_construction(tag, tmp_path):
    with pytest.raises(installer.HermesUpgradeError, match="unsafe resolved git tag"):
        installer._git_fetch_command(tmp_path, tag)


def test_option_shaped_tag_is_rejected_before_fetch_with_real_git(tmp_path):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, _release_commit = _make_release_remote(tmp_path, "--detach")
    target, baseline = _make_target_checkout(tmp_path, remote)
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")

    with pytest.raises(installer.HermesUpgradeError, match="unsafe resolved git tag"):
        installer._git_fetch_checkout(target, "--detach")

    assert _git(target, "rev-parse", "HEAD").stdout.strip() == baseline
    assert _git(
        target,
        "show-ref",
        "--verify",
        "refs/tags/--detach",
        check=False,
    ).returncode != 0


def test_option_shaped_resolved_tag_starts_no_mutating_fake_command(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    hermes_dir = tmp_path / "hermes-agent"
    (hermes_dir / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    _install_fake_versioning(monkeypatch, mode_value="editable", directory=hermes_dir, tag_name="--detach")
    calls = _install_fake_subprocess(monkeypatch)
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")

    with pytest.raises(installer.HermesUpgradeError, match="unsafe resolved git tag"):
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=tmp_path / "python",
            vetted_origin="origin",
        )

    assert not any("fetch" in cmd or "checkout" in cmd or "pip" in cmd for cmd in calls)


def test_valid_release_checkout_is_commit_pinned_with_real_git(tmp_path):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, release_commit = _make_release_remote(tmp_path, "v2026.7.1")
    target, baseline = _make_target_checkout(tmp_path, remote)
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")

    fetched_commit = installer._git_fetch_checkout(target, "v2026.7.1")

    assert fetched_commit == release_commit
    assert fetched_commit != baseline
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == release_commit
    assert _git(target, "symbolic-ref", "-q", "HEAD", check=False).returncode != 0


def test_validation_rejects_real_installed_version_mismatch(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    _install_fake_distribution(tmp_path, "0.13.9")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="pypi")

    with pytest.raises(installer.HermesUpgradeError, match="installed version 0.13.9.*expected 0.14.0"):
        installer._validate(Path(sys.executable), expected_version="0.14.0")


def test_update_job_rejects_fake_installed_version_mismatch(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    _install_fake_subprocess(monkeypatch, installed_version="0.13.9")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="pypi")

    with pytest.raises(installer.HermesUpgradeError, match="installed version 0.13.9.*expected 0.14.0"):
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=tmp_path / "python",
        )

    assert state.last_update().status == "failed"


def test_validation_rejects_real_git_head_mismatch(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, release_commit = _make_release_remote(tmp_path, "v2026.7.1")
    target, baseline = _make_target_checkout(tmp_path, remote)
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    fetched_commit = installer._git_fetch_checkout(target, "v2026.7.1")
    assert fetched_commit == release_commit
    _git(target, "checkout", "--detach", baseline)
    _install_fake_distribution(tmp_path, "0.14.0")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    with pytest.raises(installer.HermesUpgradeError, match="HEAD .*does not match fetched release commit"):
        installer._validate(
            Path(sys.executable),
            expected_version="0.14.0",
            editable_directory=target,
            expected_commit=release_commit,
        )


def test_update_job_rejects_fake_git_head_mismatch(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    hermes_dir = tmp_path / "hermes-agent"
    (hermes_dir / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    expected_commit = "a" * 40
    actual_commit = "b" * 40
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if "status" in cmd:
            return FakeProcess()
        if (
            "config" in cmd
            and "--null" in cmd
            and "--get-all" in cmd
            and "remote.origin.url" in cmd
        ):
            return FakeProcess(stdout=b"origin\0")
        if "ls-remote" in cmd and "--get-url" in cmd:
            return FakeProcess(stdout=f"{cmd[-1]}\n")
        if "rev-parse" in cmd:
            if any(str(arg).startswith("refs/tags/") for arg in cmd):
                return FakeProcess(stdout=f"{expected_commit}\n")
            return FakeProcess(stdout=f"{actual_commit}\n")
        if cmd[0].endswith(("python.exe", "python")) and "-c" in cmd:
            return FakeProcess(stdout="0.14.0\n")
        return FakeProcess()

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    _install_fake_versioning(monkeypatch, mode_value="editable", directory=hermes_dir, tag_name="v2026.7.1")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")

    with pytest.raises(installer.HermesUpgradeError, match="HEAD .*does not match fetched release commit"):
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=tmp_path / "python",
            vetted_origin="origin",
        )

    assert any("rev-parse" in cmd and any("refs/tags/" in str(arg) for arg in cmd) for cmd in calls)
    assert state.last_update().status == "failed"


def test_fake_dirty_checkout_fails_before_mutating_command(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    hermes_dir = tmp_path / "hermes-agent"
    (hermes_dir / ".git").mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if "for-each-ref" in cmd:
            return FakeProcess()
        if "status" in cmd:
            return FakeProcess(stdout=" M tracked.py\n?? operator.txt\n")
        pytest.fail(f"mutating command ran after dirty status: {cmd!r}")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")

    with pytest.raises(installer.HermesUpgradeError, match="tracked or untracked changes") as caught:
        installer._git_fetch_checkout(hermes_dir, "v2026.7.1")

    assert "tracked.py" in str(caught.value)
    assert "operator.txt" in str(caught.value)
    assert len(calls) == 2


def test_external_checkout_gate_treats_managed_named_paths_as_dirty(tmp_path):
    external = tmp_path / "external"
    _init_repo(external)
    (external / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(external, "add", "tracked.txt")
    _git(external, "commit", "-m", "baseline")
    plugin = external / "plugins" / "ms4_consciousness"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")

    with pytest.raises(installer.HermesDirtyCheckoutError):
        installer._ensure_pristine_external_checkout(external)


def test_sync_plugin_refuses_existing_destination_without_deleting_operator_files(tmp_path):
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    plugin_dst = tmp_path / "hermes-agent" / "plugins" / "ms4_consciousness"
    plugin_dst.mkdir(parents=True)
    operator_file = plugin_dst / "operator.txt"
    operator_file.write_bytes(b"operator-owned\r\n")

    with pytest.raises(installer.HermesUpgradeError, match="refusing recursive replacement"):
        installer._sync_plugin(plugin_src, plugin_dst)

    assert operator_file.read_bytes() == b"operator-owned\r\n"
    assert not (plugin_dst / "plugin.yaml").exists()


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked-outside"])
def test_dirty_checkout_preserves_operator_files_before_real_git_mutation(
    dirty_kind,
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, _release_commit = _make_release_remote(tmp_path, "v2026.7.1")
    ms4_root = tmp_path / "ms4"
    target = ms4_root / "runtime" / "hermes-managed"
    _init_repo(target)
    (target / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(target, "add", "tracked.txt")
    _git(target, "commit", "-m", "baseline")
    baseline = _git(target, "rev-parse", "HEAD").stdout.strip()
    _git(target, "remote", "add", "origin", str(remote))
    if dirty_kind == "tracked":
        operator_file = target / "tracked.txt"
    else:
        # F2 migration: the pre-F2 "untracked-plugin" case asserted refusal for
        # an untracked file INSIDE plugins/ms4_consciousness/. F2 deliberately
        # overturns that -- the MS4-managed, MS4-generated plugin subtree is now
        # ignored by the clean-check so repeat updates don't self-block (the
        # ignore + re-stamp behavior is proven in test_installer_r1.py). The
        # protective intent (an UNRELATED untracked file must still fail closed)
        # is preserved here by placing the file OUTSIDE the managed subtree.
        operator_file = target / "operator.txt"
    operator_file.write_text(f"operator-{dirty_kind}\n", encoding="utf-8")
    original_bytes = operator_file.read_bytes()
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    _install_fake_versioning(monkeypatch, mode_value="editable", directory=target, tag_name="v2026.7.1")
    # A dirty *managed* checkout still fails closed. Dirty external development
    # checkouts migrate to the isolated managed target (covered below).
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: str(remote),
    )
    monkeypatch.setattr(installer, "_pip_install_editable", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(installer, "_validate", lambda *_args, **_kwargs: "0.14.0")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")

    with pytest.raises(installer.HermesUpgradeError, match="tracked or untracked changes"):
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=tmp_path / "python",
            vetted_origin=str(remote),
        )

    assert operator_file.read_bytes() == original_bytes
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == baseline
    assert _git(
        target,
        "show-ref",
        "--verify",
        "refs/tags/v2026.7.1",
        check=False,
    ).returncode != 0
    assert state.last_update().status == "failed"


def test_refuses_unsafe_target_version(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "python"
    venv_python.write_text("")

    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    calls = _install_fake_subprocess(monkeypatch)

    state.start_job(from_version="0.13.0", to_version=None, install_mode="pypi")
    with pytest.raises(installer.HermesUpgradeError):
        installer.run_update_job(
            target_version="0.14.0; rm -rf /",
            plugin_src=plugin_src,
            venv_python=venv_python,
        )
    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert "unsafe target_version" in (final.error or "")
    assert not any("install" in " ".join(c) for c in calls if c)


def test_pip_failure_finalizes_with_error(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "python"
    venv_python.write_text("")

    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    _install_fake_subprocess(monkeypatch, fail_on="hermes-agent==")

    state.start_job(from_version="0.13.0", to_version=None, install_mode="pypi")
    with pytest.raises(installer.HermesUpgradeError):
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=venv_python,
        )
    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert "simulated failure" in (final.error or "")


def test_trigger_update_is_idempotent_when_already_running(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "python"
    venv_python.write_text("")
    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)

    existing = state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="pypi",
    )
    installer._set_local_update_job(existing.job_id)
    held = installer._WholeJobLock.acquire(installer._update_lock_path())
    try:
        snap = installer.trigger_update(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=venv_python,
        )
    finally:
        held.release()
        installer._clear_local_update_job(existing.job_id)
    assert snap.status == "running"
    assert state.update_in_progress() is True


def test_trigger_retry_current_editable_persists_terminal_noop_without_product_effects(
    tmp_path,
    monkeypatch,
):
    current = "0.18.2"
    state_path = tmp_path / "state" / "snap.json"
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "HEAD").write_text(
        "ref: refs/heads/main\n",
        encoding="utf-8",
    )
    plugin_src = tmp_path / "plugin"
    plugin_src.mkdir()
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir()
    venv_python.write_bytes(b"venv sentinel\n")

    state._reset_for_tests(state_path)
    seeded = state.start_job(
        from_version=current,
        to_version=current,
        install_mode="editable",
    )
    state.finalize_job(error="prior terminal failure")
    prior_finished_at = seeded.finished_at
    lock_path = installer._update_lock_path()
    lock_path.write_bytes(b"\0")
    before_state = state_path.read_bytes()
    before_lock = lock_path.read_bytes()
    assert installer._LOCAL_UPDATE_JOB_ID is None

    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": current,
            "directory": checkout,
        },
    )

    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"Retry current/current reached forbidden effect: {name}")

        return fail

    monkeypatch.setattr(
        installer,
        "_require_valid_git_bash_candidate",
        forbidden("external validation"),
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        forbidden("release lookup"),
    )
    monkeypatch.setattr(
        installer.versioning,
        "recent_releases",
        forbidden("recent release lookup"),
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        forbidden("origin"),
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_provenance",
        forbidden("provenance"),
    )
    monkeypatch.setattr(
        installer,
        "_git_fetch_resolve",
        forbidden("git fetch"),
    )
    monkeypatch.setattr(
        installer,
        "_git_detach_checkout",
        forbidden("git checkout"),
    )
    monkeypatch.setattr(
        installer,
        "_restamp_managed_plugin",
        forbidden("plugin restamp"),
    )
    monkeypatch.setattr(
        installer,
        "_pip_install_editable",
        forbidden("pip install"),
    )
    monkeypatch.setattr(installer, "_validate", forbidden("validation"))
    monkeypatch.setattr(
        installer,
        "_validate_managed_install",
        forbidden("managed validation"),
    )
    monkeypatch.setattr(
        installer,
        "_publish_active_checkout",
        forbidden("marker publication"),
    )
    monkeypatch.setattr(
        installer.state,
        "fail_interrupted_job",
        forbidden("interrupted-job mutation"),
    )
    monkeypatch.setattr(installer.state, "start_job", forbidden("start_job"))
    monkeypatch.setattr(installer.state, "set_phase", forbidden("set_phase"))
    monkeypatch.setattr(
        installer.state,
        "append_progress",
        forbidden("append_progress"),
    )
    monkeypatch.setattr(
        installer.state,
        "finalize_job",
        forbidden("finalize_job"),
    )
    monkeypatch.setattr(
        installer,
        "_set_local_update_job",
        forbidden("local job mutation"),
    )
    monkeypatch.setattr(
        installer.threading,
        "Thread",
        forbidden("worker creation"),
    )

    result = installer.trigger_update(
        target_version=None,
        request_user="retry-user",
        plugin_src=plugin_src,
        venv_python=venv_python,
    )

    assert result.status == "success"
    assert result.phase == "done"
    assert result.job_id == seeded.job_id
    assert result.started_at == seeded.started_at
    assert result.from_version == current
    assert result.to_version == current
    assert json.loads(state_path.read_text(encoding="utf-8")) == result.to_dict()
    assert result.error is None
    assert result.progress == []
    assert result.request_user == "retry-user"
    assert result.finished_at is not None
    assert result.finished_at != prior_finished_at
    assert state_path.read_bytes() != before_state
    assert lock_path.read_bytes() == before_lock
    assert installer._LOCAL_UPDATE_JOB_ID is None
    in_memory = state.last_update(refresh=False)
    assert in_memory == result
    force_refreshed = state.last_update(refresh=True)
    assert force_refreshed == result

    state._reset_for_tests(state_path)
    state.initialize_state(state_path)
    rehydrated = state.last_update(refresh=False)
    assert rehydrated == result


def test_cross_process_update_lock_blocks_releases_and_survives_stale_exit(tmp_path):
    lock_path = tmp_path / "runtime" / "hermes_update.lock"
    repo_root = Path(__file__).resolve().parents[2]
    holder_code = """
import sys
from pathlib import Path
from machine_spirit_4.hermes_admin.installer import _WholeJobLock

lock = _WholeJobLock.acquire(Path(sys.argv[1]))
print("LOCKED", flush=True)
sys.stdin.read(1)
lock.release()
"""
    probe_code = """
import sys
from pathlib import Path
from machine_spirit_4.hermes_admin.installer import (
    HermesUpdateLockedError,
    _WholeJobLock,
)

try:
    lock = _WholeJobLock.acquire(Path(sys.argv[1]))
except HermesUpdateLockedError:
    print("BLOCKED")
else:
    lock.release()
    print("ACQUIRED")
"""
    stale_code = """
import os
import sys
from pathlib import Path
from machine_spirit_4.hermes_admin.installer import _WholeJobLock

_lock = _WholeJobLock.acquire(Path(sys.argv[1]))
print("LOCKED", flush=True)
os._exit(0)
"""

    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(lock_path)],
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        holder_line = holder.stdout.readline().strip()
        holder_error = holder.stderr.read() if not holder_line and holder.stderr else ""
        assert holder_line == "LOCKED", holder_error
        blocked = subprocess.run(
            [sys.executable, "-c", probe_code, str(lock_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert blocked.returncode == 0, blocked.stderr
        assert blocked.stdout.strip() == "BLOCKED"
    finally:
        if holder.poll() is None and holder.stdin is not None:
            try:
                holder.stdin.write("x")
                holder.stdin.flush()
            except OSError:
                pass
        holder.wait(timeout=10)

    released = subprocess.run(
        [sys.executable, "-c", probe_code, str(lock_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert released.returncode == 0, released.stderr
    assert released.stdout.strip() == "ACQUIRED"

    stale = subprocess.run(
        [sys.executable, "-c", stale_code, str(lock_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert stale.returncode == 0, stale.stderr
    assert stale.stdout.strip() == "LOCKED"

    after_exit = subprocess.run(
        [sys.executable, "-c", probe_code, str(lock_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert after_exit.returncode == 0, after_exit.stderr
    assert after_exit.stdout.strip() == "ACQUIRED"


def test_run_update_job_holds_lock_for_entire_locked_body(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    lock_path = installer._update_lock_path()
    observed = {"blocked_inside": False}

    def locked_body(**_kwargs):
        with pytest.raises(installer.HermesUpdateLockedError):
            installer._WholeJobLock.acquire(lock_path)
        observed["blocked_inside"] = True
        return {"to_version": "0.14.0"}

    monkeypatch.setattr(installer, "_run_update_job_locked", locked_body)

    assert installer.run_update_job(target_version="0.14.0") == {
        "to_version": "0.14.0"
    }
    assert observed["blocked_inside"] is True
    installer._WholeJobLock.acquire(lock_path).release()


def test_update_lock_open_failure_is_normalized(tmp_path):
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("x", encoding="utf-8")

    with pytest.raises(
        installer.HermesUpgradeError,
        match="Could not acquire Hermes update lock",
    ):
        installer._WholeJobLock.acquire(not_a_directory / "hermes_update.lock")


def test_trigger_update_lock_contention_does_not_start_or_replace_state(
    tmp_path,
):
    state._reset_for_tests(tmp_path / "snap.json")
    held = installer._WholeJobLock.acquire(installer._update_lock_path())
    try:
        with pytest.raises(installer.HermesUpdateLockedError):
            installer.trigger_update(target_version="0.14.0")
    finally:
        held.release()

    assert state.last_update() is None


@pytest.mark.parametrize("failure_stage", ["construct", "start"])
def test_trigger_worker_startup_failure_finalizes_before_unlock(
    failure_stage,
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    events: list[str] = []
    real_finalize = state.finalize_job
    real_release = installer._WholeJobLock.release

    def record_finalize(**kwargs):
        events.append("finalize")
        return real_finalize(**kwargs)

    def record_release(self):
        events.append("release")
        return real_release(self)

    class StartFailureThread:
        def __init__(self, **_kwargs):
            if failure_stage == "construct":
                raise RuntimeError("simulated thread construction failure")

        def start(self):
            raise RuntimeError("simulated thread start failure")

    monkeypatch.setattr(state, "finalize_job", record_finalize)
    monkeypatch.setattr(installer._WholeJobLock, "release", record_release)
    monkeypatch.setattr(installer.threading, "Thread", StartFailureThread)

    with pytest.raises(
        installer.HermesUpgradeError,
        match="could not start Hermes update worker",
    ):
        installer.trigger_update(target_version="0.14.0")

    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert events[:2] == ["finalize", "release"]
    installer._WholeJobLock.acquire(installer._update_lock_path()).release()


def test_trigger_contention_preserves_live_durable_job_without_stale_return(
    tmp_path,
):
    state_path = tmp_path / "runtime" / "snap.json"
    repo_root = Path(__file__).resolve().parents[2]
    holder_code = """
import sys
from pathlib import Path
from machine_spirit_4.hermes_admin import installer, state

state._reset_for_tests(Path(sys.argv[1]))
lock = installer._WholeJobLock.acquire(installer._update_lock_path())
snap = state.start_job(
    from_version="0.13.0",
    to_version="0.14.0",
    install_mode="editable",
)
print(snap.job_id, flush=True)
sys.stdin.read(1)
lock.release()
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(state_path)],
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        job_id = holder.stdout.readline().strip()
        holder_error = holder.stderr.read() if not job_id and holder.stderr else ""
        assert job_id, holder_error

        state._reset_for_tests(state_path)
        with pytest.raises(installer.HermesUpdateLockedError):
            installer.trigger_update(target_version="0.14.0")
        observed = state.last_update()
        assert observed is not None
        assert observed.job_id == job_id
        assert observed.status == "running"
        durable = json.loads(state_path.read_text(encoding="utf-8"))
        assert durable["job_id"] == job_id
        assert durable["status"] == "running"
    finally:
        if holder.poll() is None and holder.stdin is not None:
            holder.stdin.write("x")
            holder.stdin.flush()
        holder.wait(timeout=10)


def test_missing_install_mode_fails_fast(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "python"
    venv_python.write_text("")

    monkeypatch.setattr(installer.versioning, "install_mode", lambda: {"mode": "missing", "version": None, "directory": None})

    state.start_job(from_version=None, to_version="0.14.0", install_mode="missing")
    with pytest.raises(installer.HermesUpgradeError):
        installer.run_update_job(target_version="0.14.0", plugin_src=plugin_src, venv_python=venv_python)
    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert "not installed" in (final.error or "")


def test_no_powershell_or_bash_only_python_subprocess(tmp_path, monkeypatch):
    """Cross-platform contract: installer never invokes shell scripts directly.

    Mirrors the cross-platform rule baked into the TMR repo — pip and
    git are cross-platform CLIs; PowerShell / Bash scripts are not.
    """
    seen: list[list[str]] = []

    def fake_run(
        cmd,
        cwd=None,
        stdin=None,
        capture_output=True,
        text=True,
        timeout=None,
        check=False,
        env=None,
    ):
        seen.append(list(cmd))
        return FakeProcess(stdout="0.14.0\n")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="pypi")
    installer.run_update_job(target_version="0.14.0")

    flat = [" ".join(c).lower() for c in seen]
    forbidden = ("powershell", "pwsh.exe", "/bin/bash -c", "cmd.exe", "winget")
    assert not any(any(token in line for token in forbidden) for line in flat), flat


def test_dirty_editable_update_uses_managed_checkout_without_touching_operator_tree(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, release_commit = _make_release_remote(tmp_path, "v2026.5.16")
    release_seed = tmp_path / "release seed"
    (release_seed / "newer.txt").write_text("newer\n", encoding="utf-8")
    _git(release_seed, "add", "newer.txt")
    _git(release_seed, "commit", "-m", "newer release")
    newer_commit = _git(release_seed, "rev-parse", "HEAD").stdout.strip()
    _git(release_seed, "update-ref", "refs/tags/v2026.6.1", newer_commit)
    _git(
        release_seed,
        "push",
        "origin",
        "refs/tags/v2026.6.1:refs/tags/v2026.6.1",
    )
    operator_checkout, operator_head = _make_target_checkout(tmp_path, remote)
    operator_file = operator_checkout / "tracked.txt"
    operator_file.write_bytes(b"operator-owned\r\nwindows-path-work\r\n")
    operator_bytes = operator_file.read_bytes()
    operator_status = _git(
        operator_checkout, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout

    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text(
        "name: ms4_consciousness\n", encoding="utf-8"
    )
    installed_from: list[Path] = []
    provenance_calls: list[dict[str, Any]] = []
    fetch_origins: list[str] = []

    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=operator_checkout,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        lambda force_refresh=False: versioning.CachedLatest(
            version="0.15.0",
            published_at="2026-06-01T00:00:00Z",
            tag_name="v2026.6.1",
            html_url="https://example/newer",
            fetched_at_unix=0.0,
        ),
    )
    monkeypatch.setattr(
        installer.versioning,
        "resolve_git_tag",
        lambda target: {
            "0.14.0": "v2026.5.16",
            "0.15.0": "v2026.6.1",
        }[target],
    )
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: str(remote),
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_provenance",
        lambda **kwargs: provenance_calls.append(dict(kwargs)),
    )
    real_fetch_resolve = installer._git_fetch_resolve

    def record_fetch_origin(directory, target_tag, *, remote_url="origin"):
        fetch_origins.append(remote_url)
        return real_fetch_resolve(
            directory,
            target_tag,
            remote_url=remote_url,
        )

    monkeypatch.setattr(installer, "_git_fetch_resolve", record_fetch_origin)
    monkeypatch.setattr(
        installer,
        "_pip_install_editable",
        lambda _python, directory: installed_from.append(Path(directory)),
    )
    managed_validation_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        installer,
        "_validate_managed_install",
        lambda _python, **kwargs: (
            managed_validation_calls.append(dict(kwargs)) or "0.14.0"
        ),
    )
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    result = installer.run_update_job(
        target_version="0.14.0",
        plugin_src=plugin_src,
        venv_python=tmp_path / "python",
        vetted_origin=str(remote),
    )

    assert result == {"to_version": "0.14.0"}
    assert operator_file.read_bytes() == operator_bytes
    assert _git(operator_checkout, "rev-parse", "HEAD").stdout.strip() == operator_head
    assert (
        _git(
            operator_checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout
        == operator_status
    )
    assert _git(managed_checkout, "rev-parse", "HEAD").stdout.strip() == release_commit
    assert release_commit != newer_commit
    assert (
        _git(
            managed_checkout,
            "show-ref",
            "--verify",
            "refs/tags/v2026.6.1",
            check=False,
        ).returncode
        != 0
    )
    assert _git(managed_checkout, "remote", "get-url", "origin").stdout.strip() == str(
        remote
    )
    assert provenance_calls == [
        {
            "mode": "editable",
            "target": "0.14.0",
            "tag": "v2026.5.16",
            "directory": managed_checkout,
            "expected_commit": release_commit,
            "current_version": "0.13.0",
        }
    ]
    assert fetch_origins == [str(remote)]
    assert installed_from == [managed_checkout]
    assert managed_validation_calls == [
        {
            "expected_version": "0.14.0",
            "editable_directory": managed_checkout,
            "expected_commit": release_commit,
        }
    ]
    assert (managed_checkout / "plugins" / "ms4_consciousness" / "plugin.yaml").exists()
    marker = ms4_root / "runtime" / "hermes_active_checkout.json"
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_payload == {
        "schema": "Ms4HermesActiveCheckout.v1",
        "directory": str(managed_checkout),
        "version": "0.14.0",
        "commit": release_commit,
    }
    monkeypatch.setattr(runtime_common, "MS4", ms4_root)
    monkeypatch.delenv("MS4_HERMES_DIR", raising=False)
    assert runtime_common.hermes_dir() == managed_checkout
    assert state.last_update().status == "success"


def test_dirty_external_same_version_migrates_without_touching_source(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, release_commit = _make_release_remote(tmp_path, "v2026.5.16")
    operator_checkout, operator_head = _make_target_checkout(tmp_path, remote)
    operator_file = operator_checkout / "tracked.txt"
    operator_file.write_bytes(b"operator-pin-work\r\n")
    before = operator_file.read_bytes()
    before_status = _git(
        operator_checkout, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout
    ms4_root = tmp_path / "ms4"
    managed = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed)
    _git(managed, "remote", "add", "origin", str(remote))
    _git(
        managed,
        "fetch",
        "origin",
        "refs/tags/v2026.5.16:refs/tags/v2026.5.16",
    )
    _git(managed, "checkout", "--detach", release_commit)
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text(
        "name: ms4_consciousness\n",
        encoding="utf-8",
    )
    installed_from: list[Path] = []
    validated_from: list[Path] = []

    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=operator_checkout,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.14.0",
            "directory": operator_checkout,
        },
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: str(remote),
    )
    monkeypatch.setattr(
        installer,
        "_pip_install_editable",
        lambda _python, directory: installed_from.append(Path(directory)),
    )
    monkeypatch.setattr(
        installer,
        "_validate_managed_install",
        lambda _python, **kwargs: (
            pytest.fail("managed validation ran before managed pip activation")
            if installed_from != [managed]
            else (
                validated_from.append(Path(kwargs["editable_directory"]))
                or "0.14.0"
            )
        ),
    )
    state.start_job(
        from_version="0.14.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    result = installer.run_update_job(
        target_version="0.14.0",
        plugin_src=plugin_src,
        venv_python=tmp_path / "python",
        vetted_origin=str(remote),
    )

    assert result == {"to_version": "0.14.0"}
    assert installed_from == [managed]
    assert validated_from == [managed]
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == release_commit
    assert operator_file.read_bytes() == before
    assert _git(operator_checkout, "rev-parse", "HEAD").stdout.strip() == operator_head
    assert (
        _git(
            operator_checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout
        == before_status
    )
    final = state.last_update()
    assert final.status == "success"


@pytest.mark.parametrize("matching_head", [True, False], ids=["noop", "correct"])
def test_managed_same_version_requires_provenance_and_corrects_mismatch(
    matching_head,
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, release_commit = _make_release_remote(tmp_path, "v2026.5.16")
    ms4_root = tmp_path / "ms4"
    managed = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed)
    (managed / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(managed, "add", "tracked.txt")
    _git(managed, "commit", "-m", "baseline")
    baseline = _git(managed, "rev-parse", "HEAD").stdout.strip()
    _git(managed, "remote", "add", "origin", str(remote))
    if matching_head:
        _git(
            managed,
            "fetch",
            "origin",
            "refs/tags/v2026.5.16:refs/tags/v2026.5.16",
        )
        _git(managed, "checkout", "--detach", release_commit)
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text(
        "name: ms4_consciousness\n",
        encoding="utf-8",
    )
    provenance_calls: list[dict[str, Any]] = []
    install_calls: list[Path] = []
    validation_calls: list[Path] = []
    fetch_origins: list[str] = []

    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=managed,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.14.0",
            "directory": managed,
        },
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: str(remote),
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_provenance",
        lambda **kwargs: provenance_calls.append(dict(kwargs)),
    )
    real_fetch_resolve = installer._git_fetch_resolve

    def record_fetch(directory, target_tag, *, remote_url="origin"):
        fetch_origins.append(remote_url)
        return real_fetch_resolve(
            directory,
            target_tag,
            remote_url=remote_url,
        )

    monkeypatch.setattr(installer, "_git_fetch_resolve", record_fetch)
    monkeypatch.setattr(
        installer,
        "_pip_install_editable",
        lambda _python, directory: install_calls.append(Path(directory)),
    )
    monkeypatch.setattr(
        installer,
        "_validate_managed_install",
        lambda _python, **kwargs: (
            validation_calls.append(Path(kwargs["editable_directory"]))
            or "0.14.0"
        ),
    )
    state.start_job(
        from_version="0.14.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    result = installer.run_update_job(
        target_version="0.14.0",
        plugin_src=plugin_src,
        venv_python=tmp_path / "python",
        vetted_origin=str(remote),
    )

    assert result == {"to_version": "0.14.0"}
    assert fetch_origins == [str(remote)]
    assert provenance_calls == [
        {
            "mode": "editable",
            "target": "0.14.0",
            "tag": "v2026.5.16",
            "directory": managed,
            "expected_commit": release_commit,
            "current_version": "0.14.0",
        }
    ]
    assert validation_calls == [managed]
    assert install_calls == ([] if matching_head else [managed])
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == release_commit
    if not matching_head:
        assert release_commit != baseline


def test_managed_same_version_marker_failure_does_not_install(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, release_commit = _make_release_remote(tmp_path, "v2026.5.16")
    ms4_root = tmp_path / "ms4"
    managed = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed)
    _git(managed, "remote", "add", "origin", str(remote))
    _git(
        managed,
        "fetch",
        "origin",
        "refs/tags/v2026.5.16:refs/tags/v2026.5.16",
    )
    _git(managed, "checkout", "--detach", release_commit)
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text(
        "name: ms4_consciousness\n",
        encoding="utf-8",
    )
    install_calls: list[Path] = []

    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=managed,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.14.0",
            "directory": managed,
        },
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: str(remote),
    )
    monkeypatch.setattr(
        installer,
        "_validate_managed_install",
        lambda *_args, **_kwargs: "0.14.0",
    )
    monkeypatch.setattr(
        installer,
        "_publish_active_checkout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            installer.HermesUpgradeError("simulated marker I/O failure")
        ),
    )
    monkeypatch.setattr(
        installer,
        "_pip_install_editable",
        lambda _python, directory: install_calls.append(Path(directory)),
    )
    monkeypatch.setattr(installer, "_reinstall_editable", lambda *_args: None)
    state.start_job(
        from_version="0.14.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(
        installer.HermesUpgradeError,
        match="simulated marker I/O failure",
    ):
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=tmp_path / "python",
            vetted_origin=str(remote),
        )

    assert install_calls == []
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == release_commit


def test_managed_validation_failure_never_publishes_or_reinstalls_dirty_source(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, _release_commit = _make_release_remote(tmp_path, "v2026.5.16")
    operator_checkout, operator_head = _make_target_checkout(tmp_path, remote)
    operator_file = operator_checkout / "tracked.txt"
    operator_file.write_bytes(b"operator-rollback-work\r\n")
    before = operator_file.read_bytes()
    before_status = _git(
        operator_checkout, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text(
        "name: ms4_consciousness\n", encoding="utf-8"
    )
    reinstall_calls: list[Path] = []

    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=operator_checkout,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(installer, "MS4_ROOT", tmp_path / "ms4")
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: str(remote),
    )
    monkeypatch.setattr(
        installer,
        "_pip_install_editable",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        installer,
        "_validate_managed_install",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            installer.HermesUpgradeError("simulated managed validation failure")
        ),
    )
    monkeypatch.setattr(
        installer,
        "_reinstall_editable",
        lambda _python, directory: reinstall_calls.append(Path(directory)),
    )
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(
        installer.HermesUpgradeError,
        match="manual recovery required.*operator checkout was not touched",
    ):
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=tmp_path / "python",
            vetted_origin=str(remote),
        )

    assert reinstall_calls == []
    assert not (
        tmp_path / "ms4" / "runtime" / "hermes_active_checkout.json"
    ).exists()
    assert operator_file.read_bytes() == before
    assert _git(operator_checkout, "rev-parse", "HEAD").stdout.strip() == operator_head
    assert (
        _git(
            operator_checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout
        == before_status
    )
    final = state.last_update()
    assert final.status == "failed"
    assert any(entry["phase"] == "rollback_failed" for entry in final.progress)


def test_managed_migration_state_write_failure_remains_manually_classified(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "operator-checkout"
    source.mkdir()
    primary = installer.HermesUpgradeError("simulated managed validation failure")
    monkeypatch.setattr(
        state,
        "set_phase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            state.StatePersistenceError("simulated rollback state failure")
        ),
    )

    with pytest.raises(installer.HermesUpgradeError) as caught:
        installer._rollback_managed_migration(
            source,
            primary,
            install_attempted=True,
        )

    message = str(caught.value)
    assert "manual recovery required" in message
    assert "operator checkout was not touched" in message
    assert "rollback state failure" in message


@pytest.mark.parametrize("failure_stage", ["marker", "finalize"])
def test_managed_post_install_failure_rolls_back_in_place_update(
    failure_stage,
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    remote, release_commit = _make_release_remote(tmp_path, "v2026.5.16")
    ms4_root = tmp_path / "ms4"
    managed = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed)
    (managed / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(managed, "add", "tracked.txt")
    _git(managed, "commit", "-m", "baseline")
    baseline = _git(managed, "rev-parse", "HEAD").stdout.strip()
    _git(managed, "remote", "add", "origin", str(remote))
    marker = ms4_root / "runtime" / "hermes_active_checkout.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "Ms4HermesActiveCheckout.v1",
                "directory": str(managed),
                "version": "0.13.0",
                "commit": baseline,
            }
        ),
        encoding="utf-8",
    )
    marker_before = marker.read_bytes()
    plugin_src = tmp_path / "plugin-src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text(
        "name: ms4_consciousness\n",
        encoding="utf-8",
    )

    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=managed,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: str(remote),
    )
    monkeypatch.setattr(installer, "_pip_install_editable", lambda *_args: None)
    monkeypatch.setattr(installer, "_reinstall_editable", lambda *_args: None)
    monkeypatch.setattr(
        installer,
        "_validate_managed_install",
        lambda *_args, **_kwargs: "0.14.0",
    )
    failure_text = f"simulated {failure_stage} persistence failure"
    if failure_stage == "marker":
        monkeypatch.setattr(
            installer,
            "_publish_active_checkout",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(failure_text)
            ),
        )
    else:
        real_finalize = state.finalize_job

        def fail_success_finalize(*, error=None, to_version=None):
            if error is None:
                raise state.StatePersistenceError(failure_text)
            return real_finalize(error=error, to_version=to_version)

        monkeypatch.setattr(state, "finalize_job", fail_success_finalize)
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(installer.HermesUpgradeError) as caught:
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=tmp_path / "python",
            vetted_origin=str(remote),
        )

    message = str(caught.value).lower()
    assert "rollback" in message or "rolled back" in message
    assert failure_text in str(caught.value)
    assert release_commit != baseline
    assert _git(managed, "rev-parse", "HEAD").stdout.strip() == baseline
    assert marker.read_bytes() == marker_before
    monkeypatch.setattr(runtime_common, "MS4", ms4_root)
    monkeypatch.delenv("MS4_HERMES_DIR", raising=False)
    assert runtime_common.hermes_dir() == managed
    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert any(entry["phase"] == "rolled_back" for entry in final.progress)


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        (
            PureWindowsPath("C:/HiveMind/machine_spirit_4"),
            PureWindowsPath(
                "C:/HiveMind/machine_spirit_4/runtime/hermes-managed"
            ),
        ),
        (
            PurePosixPath("/opt/HiveMind/machine_spirit_4"),
            PurePosixPath("/opt/HiveMind/machine_spirit_4/runtime/hermes-managed"),
        ),
        (
            PurePosixPath("/Applications/HiveMind/machine_spirit_4"),
            PurePosixPath(
                "/Applications/HiveMind/machine_spirit_4/runtime/hermes-managed"
            ),
        ),
    ],
    ids=["windows", "linux", "macos"],
)
def test_default_managed_checkout_is_platform_native(root, expected):
    assert installer._default_managed_checkout(root) == expected


@pytest.mark.parametrize(
    "candidate_rel",
    [
        "../hermes-managed",
        "runtime/hermes-managed-backup",
        "runtime/other/hermes-managed",
    ],
)
def test_managed_checkout_rejects_paths_outside_exact_runtime_target(
    candidate_rel,
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    ms4_root.mkdir()
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(
        installer.HermesUpgradeError,
        match="must be exactly",
    ):
        installer._assert_managed_checkout_path(ms4_root / candidate_rel)


def test_managed_checkout_rejects_external_gitdir_worktree(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _init_repo(source)
    (source / "tracked.txt").write_text("source\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "source")
    _git(source, "remote", "add", "origin", str(source))

    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    managed_checkout.parent.mkdir(parents=True)
    _git(source, "worktree", "add", "-b", "managed-test", str(managed_checkout))
    assert (managed_checkout / ".git").is_file()
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    assert installer._is_exact_managed_checkout(managed_checkout) is False
    with pytest.raises(
        installer.HermesUpgradeError,
        match="contained .git directory",
    ):
        installer._prepare_managed_checkout(managed_checkout, str(source))


@pytest.mark.parametrize(
    "component",
    ["ms4-root", "runtime", "managed", "git-dir", "common-dir", "config"],
)
def test_exact_managed_checkout_rejects_reported_reparse_component(
    component,
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(
        managed_checkout,
        origin=_VETTED_HERMES_ORIGIN,
    )
    _git(managed_checkout, "commit", "--allow-empty", "-m", "managed")
    targets = {
        "ms4-root": ms4_root,
        "runtime": managed_checkout.parent,
        "managed": managed_checkout,
        "git-dir": managed_checkout / ".git",
        "common-dir": managed_checkout / ".git",
        "config": managed_checkout / ".git" / "config",
    }
    reported_reparse = targets[component]
    before = _tree_bytes(managed_checkout)
    real_check = installer._path_is_reparse_point
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_path_is_reparse_point",
        lambda path: Path(path) == reported_reparse or real_check(path),
    )

    assert installer._is_exact_managed_checkout(managed_checkout) is False
    assert _tree_bytes(managed_checkout) == before


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction behavior")
def test_managed_checkout_rejects_real_runtime_junction(
    tmp_path,
    monkeypatch,
):
    redirected_runtime = tmp_path / "redirected-runtime"
    redirected_checkout = redirected_runtime / "hermes-managed"
    _init_unborn_managed_checkout(
        redirected_checkout,
        origin=_VETTED_HERMES_ORIGIN,
    )
    redirected_before = _tree_bytes(redirected_checkout)
    ms4_root = tmp_path / "ms4"
    ms4_root.mkdir()
    runtime_junction = ms4_root / "runtime"
    created = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(runtime_junction),
            str(redirected_runtime),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"could not create temporary junction: {created.stderr!r}")
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    try:
        with pytest.raises(installer.HermesUpgradeError):
            installer._assert_managed_checkout_path(
                runtime_junction / "hermes-managed"
            )
    finally:
        os.rmdir(runtime_junction)

    assert redirected_runtime.is_dir()
    assert _tree_bytes(redirected_checkout) == redirected_before


def test_managed_checkout_initializes_new_directory_with_origin(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    source_origin = str(tmp_path / "release.git")
    assert not managed_checkout.exists()
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    installer._prepare_managed_checkout(managed_checkout, source_origin)

    assert (
        _git(managed_checkout, "remote", "get-url", "origin").stdout.strip()
        == source_origin
    )
    assert _git(
        managed_checkout,
        "rev-parse",
        "--verify",
        "HEAD",
        check=False,
    ).returncode != 0


def test_new_checkout_global_rewrite_refuses_before_parent_creation(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "uncreated-ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    global_config = tmp_path / "malicious-global.gitconfig"
    _git(
        None,
        "config",
        "--file",
        str(global_config),
        "url.https://evil.invalid/mirror/.insteadOf",
        "https://github.com/",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config.resolve()))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    assert not ms4_root.exists()

    with pytest.raises(installer.HermesUpgradeError, match="rewrit"):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert not ms4_root.exists()
    assert not managed_checkout.exists()


_LIVE_STALE_MANAGED_ORIGIN = (
    r"C:\temp\ms4-hermes-red-noop-20260711-1837"
    r"\test_pin_current_version_is_no0\release remote.git"
)
_VETTED_HERMES_ORIGIN = "https://github.com/NousResearch/hermes-agent.git"


def _tree_bytes(root: Path) -> dict[str, tuple[str, bytes | str | None]]:
    """Snapshot names, file bytes, and links without following directory links."""
    snapshot: dict[str, tuple[str, bytes | str | None]] = {}

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                snapshot[relative] = ("symlink", os.readlink(child))
            elif child.is_dir():
                snapshot[relative] = ("directory", None)
                visit(child)
            elif child.is_file():
                snapshot[relative] = ("file", child.read_bytes())
            else:
                snapshot[relative] = ("other", None)

    visit(root)
    return snapshot


def _init_unborn_managed_checkout(
    managed_checkout: Path,
    *,
    origin: str = _LIVE_STALE_MANAGED_ORIGIN,
) -> None:
    _init_repo(managed_checkout)
    _git(managed_checkout, "remote", "add", "origin", origin)
    head = _git(
        managed_checkout,
        "rev-parse",
        "--verify",
        "--quiet",
        "HEAD^{commit}",
        check=False,
    )
    assert (head.returncode, head.stdout, head.stderr) == (1, "", "")
    assert {entry.name for entry in managed_checkout.iterdir()} == {".git"}


@pytest.mark.parametrize(
    "origin",
    [
        "malformed\r\norigin",
        "malformed\norigin",
        "malformed\rorigin",
        "malformed\torigin",
        "malformed\x01origin",
        "malformed\x7forigin",
        " leading-origin",
        "trailing-origin ",
    ],
    ids=[
        "crlf",
        "lf",
        "cr",
        "tab",
        "control",
        "delete",
        "leading-space",
        "trailing-space",
    ],
)
def test_malformed_origin_value_refuses_without_mutation(
    origin,
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout, origin=origin)
    before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(installer.HermesUpgradeError):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before


def test_nul_in_local_origin_config_refuses_without_mutation(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed_checkout)
    config = managed_checkout / ".git" / "config"
    config.write_bytes(
        config.read_bytes()
        + b'[remote "origin"]\n\turl = malformed\x00origin\n'
    )
    before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(installer.HermesUpgradeError):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before


@pytest.mark.parametrize(
    "origin_bytes",
    [
        b"",
        b"malformed\xfforigin",
    ],
    ids=["empty", "invalid-utf8"],
)
def test_invalid_byte_origin_refuses_without_mutation(
    origin_bytes,
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed_checkout)
    config = managed_checkout / ".git" / "config"
    config.write_bytes(
        config.read_bytes()
        + b'[remote "origin"]\n\turl = '
        + origin_bytes
        + b"\n"
    )
    before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(installer.HermesUpgradeError):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode"),
    [
        (b"value", b"", 0),
        (b"\0", b"", 0),
        (b"value\0\0", b"", 0),
        (b"value\0", b"unexpected stderr", 0),
        (b"value\0", b"", 128),
    ],
    ids=[
        "missing-terminator",
        "empty-value",
        "empty-second-value",
        "stderr",
        "nonzero",
    ],
)
def test_config_values_rejects_malformed_nul_framing(
    stdout,
    stderr,
    returncode,
):
    with pytest.raises(installer.HermesUpgradeError):
        installer._config_values(
            FakeProcess(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
            ),
            "test origin",
        )


@pytest.mark.parametrize(
    "failure_kind",
    ["timeout", "file-not-found", "os-error"],
)
def test_run_capture_bytes_normalizes_spawn_failures_without_mutation(
    failure_kind,
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    before = _tree_bytes(probe_root)
    command = ["git", "config", "--null", "--get-all", "remote.origin.url"]

    def fail_spawn(*_args, **_kwargs):
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(command, 7)
        if failure_kind == "file-not-found":
            raise FileNotFoundError("simulated missing git")
        raise OSError("simulated spawn failure")

    monkeypatch.setattr(installer.subprocess, "run", fail_spawn)

    with pytest.raises(installer.HermesUpgradeError):
        installer._run_capture_bytes(command, cwd=probe_root, timeout=7)

    assert _tree_bytes(probe_root) == before


def test_existing_managed_checkout_missing_origin_is_not_recovered(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed_checkout)
    before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(installer.HermesUpgradeError):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before
    assert (
        _git(
            managed_checkout,
            "config",
            "--local",
            "--get-all",
            "remote.origin.url",
            check=False,
        ).returncode
        == 1
    )


def test_existing_managed_checkout_ambiguous_origin_query_does_not_mutate(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed_checkout)
    before = _tree_bytes(managed_checkout)
    real_capture = installer._run_capture_bytes

    def fail_local_origin_query(cmd, **kwargs):
        if (
            "config" in cmd
            and "--get-all" in cmd
            and "remote.origin.url" in cmd
        ):
            return FakeProcess(
                stderr=b"fatal: simulated local config query failure",
                returncode=128,
            )
        return real_capture(cmd, **kwargs)

    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(installer, "_run_capture_bytes", fail_local_origin_query)

    with pytest.raises(installer.HermesUpgradeError):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before


def test_existing_managed_checkout_duplicate_origins_refuse_before_set_url(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    _git(
        managed_checkout,
        "config",
        "--local",
        "--add",
        "remote.origin.url",
        "https://second.invalid/hermes.git",
    )
    before = _tree_bytes(managed_checkout)
    commands: list[list[str]] = []
    real_run = installer._run

    def record_run(cmd, **kwargs):
        commands.append(list(cmd))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(installer, "_run", record_run)

    with pytest.raises(installer.HermesUpgradeError):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before
    assert not any("remote" in cmd and "set-url" in cmd for cmd in commands)


def test_existing_managed_checkout_external_config_include_is_refused(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed_checkout)
    external_config = tmp_path / "operator-config"
    _git(
        None,
        "config",
        "--file",
        str(external_config),
        "remote.origin.url",
        _LIVE_STALE_MANAGED_ORIGIN,
    )
    _git(
        managed_checkout,
        "config",
        "--local",
        "include.path",
        str(external_config.resolve()),
    )
    managed_before = _tree_bytes(managed_checkout)
    external_before = external_config.read_bytes()
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(installer.HermesUpgradeError):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == managed_before
    assert external_config.read_bytes() == external_before


def test_worktree_config_external_include_is_refused_without_mutation(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    git_dir = managed_checkout / ".git"
    external_config = tmp_path / "operator-worktree-config"
    _git(
        None,
        "config",
        "--file",
        str(external_config),
        "fixture.external",
        "present",
    )
    _git(
        managed_checkout,
        "config",
        "--local",
        "extensions.worktreeConfig",
        "true",
    )
    _git(
        managed_checkout,
        "config",
        "--worktree",
        "include.path",
        str(external_config.resolve()),
    )
    assert (
        "operator-worktree-config"
        in _git(
            managed_checkout,
            "config",
            "--show-origin",
            "--get",
            "fixture.external",
        ).stdout
    )
    main_before = (git_dir / "config").read_bytes()
    worktree_before = (git_dir / "config.worktree").read_bytes()
    external_before = external_config.read_bytes()
    tree_before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(installer.HermesUpgradeError, match="worktree"):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert (git_dir / "config").read_bytes() == main_before
    assert (git_dir / "config.worktree").read_bytes() == worktree_before
    assert external_config.read_bytes() == external_before
    assert _tree_bytes(managed_checkout) == tree_before


@pytest.mark.parametrize(
    "topology",
    ["extension-only", "config-worktree-only"],
)
def test_worktree_config_topology_is_always_refused(
    topology,
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    git_dir = managed_checkout / ".git"
    if topology == "extension-only":
        _git(
            managed_checkout,
            "config",
            "--local",
            "extensions.worktreeConfig",
            "true",
        )
    else:
        _git(
            None,
            "config",
            "--file",
            str(git_dir / "config.worktree"),
            "fixture.present",
            "true",
        )
    before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(installer.HermesUpgradeError, match="worktree"):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before


def test_stale_origin_repair_uses_raw_local_url_not_rewritten_display(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    raw_origin = "gh:NousResearch/hermes-agent.git"
    _init_unborn_managed_checkout(managed_checkout, origin=raw_origin)
    _git(
        managed_checkout,
        "config",
        "--local",
        "url.https://github.com/.insteadOf",
        "gh:",
    )
    assert (
        _git(managed_checkout, "remote", "get-url", "origin").stdout.strip()
        == _VETTED_HERMES_ORIGIN
    )
    observed_commands: list[list[str]] = []
    real_run = installer._run

    def record_run(cmd, **kwargs):
        observed_commands.append(list(cmd))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(installer, "_run", record_run)

    installer._prepare_managed_checkout(
        managed_checkout,
        _VETTED_HERMES_ORIGIN,
    )

    assert (
        _git(
            managed_checkout,
            "config",
            "--local",
            "--get-all",
            "remote.origin.url",
        ).stdout.splitlines()
        == [_VETTED_HERMES_ORIGIN]
    )
    assert not any("fetch" in command for command in observed_commands)
    raw_consumers = [
        command for command in observed_commands if raw_origin in command
    ]
    assert len(raw_consumers) == 1
    assert "--fixed-value" in raw_consumers[0]
    assert "--replace-all" in raw_consumers[0]


def test_vetted_source_url_rewrite_refuses_before_stale_origin_mutation(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    evil_base = "https://evil.invalid/mirror/"
    _git(
        managed_checkout,
        "config",
        "--local",
        f"url.{evil_base}.insteadOf",
        "https://github.com/",
    )
    assert (
        _git(
            managed_checkout,
            "ls-remote",
            "--get-url",
            _VETTED_HERMES_ORIGIN,
        ).stdout.strip()
        == "https://evil.invalid/mirror/NousResearch/hermes-agent.git"
    )
    before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    with pytest.raises(installer.HermesUpgradeError, match="rewrit"):
        installer._prepare_managed_checkout(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before


def test_correct_managed_origin_rewrite_refuses_before_fetch(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(
        managed_checkout,
        origin=_VETTED_HERMES_ORIGIN,
    )
    _git(managed_checkout, "commit", "--allow-empty", "-m", "managed")
    _git(
        managed_checkout,
        "config",
        "--local",
        "url.https://evil.invalid/mirror/.insteadOf",
        "https://github.com/",
    )
    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=managed_checkout,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: _VETTED_HERMES_ORIGIN,
    )
    monkeypatch.setattr(
        installer,
        "_git_fetch_resolve",
        lambda *_args, **_kwargs: pytest.fail("fetch reached after URL rewrite"),
    )
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(installer.HermesUpgradeError, match="rewrit"):
        installer.run_update_job(target_version="0.14.0")


@pytest.mark.parametrize(
    "origin_state",
    ["duplicate", "raw-rewritten"],
)
def test_normal_managed_update_validates_raw_origin_before_fetch(
    origin_state,
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    raw_origin = (
        "gh:NousResearch/hermes-agent.git"
        if origin_state == "raw-rewritten"
        else _VETTED_HERMES_ORIGIN
    )
    _init_unborn_managed_checkout(managed_checkout, origin=raw_origin)
    _git(managed_checkout, "commit", "--allow-empty", "-m", "managed")
    if origin_state == "duplicate":
        _git(
            managed_checkout,
            "config",
            "--add",
            "remote.origin.url",
            "https://second.invalid/hermes.git",
        )
    else:
        _git(
            managed_checkout,
            "config",
            "url.https://github.com/.insteadOf",
            "gh:",
        )
        assert (
            _git(managed_checkout, "remote", "get-url", "origin").stdout.strip()
            == _VETTED_HERMES_ORIGIN
        )
    before = _tree_bytes(managed_checkout)
    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=managed_checkout,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: _VETTED_HERMES_ORIGIN,
    )
    monkeypatch.setattr(
        installer,
        "_git_fetch_resolve",
        lambda *_args, **_kwargs: pytest.fail(
            "fetch reached with unsafe raw managed origin"
        ),
    )
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(installer.HermesUpgradeError):
        installer.run_update_job(
            target_version="0.14.0",
            vetted_origin=_VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(managed_checkout) == before


@pytest.mark.parametrize(
    "origin_variant",
    [
        "HTTPS://GITHUB.COM/NousResearch/hermes-agent",
        "https://github.com/NousResearch/hermes-agent",
        "https://github.com/NousResearch/hermes-agent/",
        " https://github.com/NousResearch/hermes-agent.git ",
    ],
    ids=["case", "missing-dot-git", "trailing-slash", "whitespace"],
)
def test_public_update_without_vetted_origin_requires_exact_canonical_source(
    origin_variant,
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    external_checkout = tmp_path / "external-checkout"
    _init_repo(external_checkout)
    _git(external_checkout, "commit", "--allow-empty", "-m", "external")
    _git(external_checkout, "remote", "add", "origin", origin_variant)
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=external_checkout,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: origin_variant,
    )
    monkeypatch.setattr(
        installer,
        "_git_fetch_resolve",
        lambda *_args, **_kwargs: pytest.fail(
            "public fetch reached with noncanonical source origin"
        ),
    )
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(installer.HermesUpgradeError, match="canonical"):
        installer.run_update_job(target_version="0.14.0")

    assert not managed_checkout.exists()


def test_public_update_requires_current_origin_equal_explicit_vetted_value(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    external_checkout = tmp_path / "external-checkout"
    _init_repo(external_checkout)
    _git(external_checkout, "commit", "--allow-empty", "-m", "external")
    actual_origin = str(tmp_path / "actual-source.git")
    vetted_origin = str(tmp_path / "vetted-source.git")
    _git(external_checkout, "remote", "add", "origin", actual_origin)
    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=external_checkout,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: actual_origin,
    )
    monkeypatch.setattr(
        installer,
        "_git_fetch_resolve",
        lambda *_args, **_kwargs: pytest.fail(
            "fetch reached after exact vetted-origin mismatch"
        ),
    )
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(installer.HermesUpgradeError, match="exact vetted"):
        installer.run_update_job(
            target_version="0.14.0",
            vetted_origin=vetted_origin,
        )


def test_trigger_update_rejects_noncanonical_origin_before_worker_start(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    external_checkout = tmp_path / "external-checkout"
    _init_repo(external_checkout)
    noncanonical = "https://github.com/NousResearch/hermes-agent"
    _git(external_checkout, "remote", "add", "origin", noncanonical)
    _install_fake_versioning(
        monkeypatch,
        mode_value="editable",
        directory=external_checkout,
        tag_name="v2026.5.16",
    )
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: noncanonical,
    )
    monkeypatch.setattr(
        installer.threading,
        "Thread",
        lambda **_kwargs: pytest.fail("worker constructed for noncanonical origin"),
    )

    with pytest.raises(installer.HermesUpgradeError, match="canonical"):
        installer.trigger_update(target_version="0.14.0")

    assert state.last_update() is None


def test_explicit_url_fetch_helper_rejects_rewrite_before_fetch(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    checkout = tmp_path / "checkout"
    _init_repo(checkout)
    _git(
        checkout,
        "config",
        "url.https://evil.invalid/mirror/.insteadOf",
        "https://github.com/",
    )
    before = _tree_bytes(checkout)
    real_run = installer._run

    def block_fetch(cmd, **kwargs):
        if "fetch" in cmd:
            pytest.fail("fetch reached after explicit URL rewrite")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(installer, "_run", block_fetch)
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(installer.HermesUpgradeError, match="rewrit"):
        installer._git_fetch_resolve(
            checkout,
            "v2026.5.16",
            remote_url=_VETTED_HERMES_ORIGIN,
        )

    assert _tree_bytes(checkout) == before


def test_composed_fetch_checkout_rejects_origin_rewrite_before_fetch(
    tmp_path,
    monkeypatch,
):
    state._reset_for_tests(tmp_path / "snap.json")
    checkout = tmp_path / "checkout"
    _init_repo(checkout)
    _git(checkout, "remote", "add", "origin", _VETTED_HERMES_ORIGIN)
    _git(
        checkout,
        "config",
        "url.https://evil.invalid/mirror/.insteadOf",
        "https://github.com/",
    )
    before = _tree_bytes(checkout)
    real_run = installer._run

    def block_fetch(cmd, **kwargs):
        if "fetch" in cmd:
            pytest.fail("fetch reached through composed helper after URL rewrite")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(installer, "_run", block_fetch)
    state.start_job(
        from_version="0.13.0",
        to_version="0.14.0",
        install_mode="editable",
    )

    with pytest.raises(installer.HermesUpgradeError, match="rewrit"):
        installer._git_fetch_checkout(checkout, "v2026.5.16")

    assert _tree_bytes(checkout) == before


def test_managed_checkout_repairs_exact_live_stale_origin_only_when_empty_unborn(
    tmp_path,
    monkeypatch,
):
    """Hermetically reproduce the live stale-origin/unborn managed checkout."""
    actual_managed = Path(installer._default_managed_checkout(installer.MS4_ROOT))
    ms4_root = tmp_path / "hermetic-ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    assert managed_checkout == Path(installer._default_managed_checkout(ms4_root))
    assert os.path.normcase(os.path.abspath(managed_checkout)) != os.path.normcase(
        os.path.abspath(actual_managed)
    )
    _init_unborn_managed_checkout(managed_checkout)
    _git(
        managed_checkout,
        "config",
        "--local",
        "remote.origin.pushurl",
        "https://push.example.invalid/hermes.git",
    )
    _git(
        managed_checkout,
        "remote",
        "add",
        "mirror",
        "https://mirror.example.invalid/hermes.git",
    )
    assert (
        _git(managed_checkout, "remote", "get-url", "origin").stdout.strip()
        == _LIVE_STALE_MANAGED_ORIGIN
    )
    before_tree = _tree_bytes(managed_checkout)
    before_config = (managed_checkout / ".git" / "config").read_bytes()
    before_entries = _git(
        managed_checkout, "config", "--local", "--list"
    ).stdout.splitlines()
    before_head_ref = _git(
        managed_checkout, "symbolic-ref", "--quiet", "HEAD"
    ).stdout

    observed_commands: list[list[str]] = []
    real_run = installer._run
    real_run_capture = installer._run_capture
    real_run_capture_bytes = installer._run_capture_bytes

    def guard_actual_runtime(real_call):
        def guarded(cmd, **kwargs):
            command = list(cmd)
            observed_commands.append(command)
            if command and Path(command[0]).name.casefold() in {"git", "git.exe"}:
                if "-C" in command:
                    command_root = Path(command[command.index("-C") + 1])
                    assert command_root == managed_checkout
                    assert os.path.normcase(
                        os.path.abspath(command_root)
                    ) != os.path.normcase(os.path.abspath(actual_managed))
            return real_call(cmd, **kwargs)

        return guarded

    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(installer, "_run", guard_actual_runtime(real_run))
    monkeypatch.setattr(
        installer,
        "_run_capture",
        guard_actual_runtime(real_run_capture),
    )
    monkeypatch.setattr(
        installer,
        "_run_capture_bytes",
        guard_actual_runtime(real_run_capture_bytes),
    )

    installer._prepare_managed_checkout(
        managed_checkout,
        _VETTED_HERMES_ORIGIN,
    )

    assert (
        _git(managed_checkout, "remote", "get-url", "origin").stdout.strip()
        == _VETTED_HERMES_ORIGIN
    )
    after_tree = _tree_bytes(managed_checkout)
    after_config = (managed_checkout / ".git" / "config").read_bytes()
    assert {
        key: value for key, value in after_tree.items() if key != ".git/config"
    } == {
        key: value for key, value in before_tree.items() if key != ".git/config"
    }
    changed_config_lines = [
        (before, after)
        for before, after in zip(
            before_config.splitlines(keepends=True),
            after_config.splitlines(keepends=True),
            strict=True,
        )
        if before != after
    ]
    assert len(changed_config_lines) == 1
    old_url_line, new_url_line = changed_config_lines[0]
    assert old_url_line.lstrip().startswith(b"url = ")
    assert new_url_line.strip() == (
        b"url = " + _VETTED_HERMES_ORIGIN.encode("utf-8")
    )
    after_entries = _git(
        managed_checkout, "config", "--local", "--list"
    ).stdout.splitlines()
    changed_entries = [
        (before, after)
        for before, after in zip(before_entries, after_entries, strict=True)
        if before != after
    ]
    assert changed_entries == [
        (
            f"remote.origin.url={_LIVE_STALE_MANAGED_ORIGIN}",
            f"remote.origin.url={_VETTED_HERMES_ORIGIN}",
        )
    ]
    assert [
        command
        for command in observed_commands
        if "config" in command
        and "--replace-all" in command
        and "remote.origin.url" in command
    ] == [
        [
            "git",
            "config",
            "--file",
            str(managed_checkout / ".git" / "config"),
            "--fixed-value",
            "--replace-all",
            "remote.origin.url",
            _VETTED_HERMES_ORIGIN,
            _LIVE_STALE_MANAGED_ORIGIN,
        ]
    ]
    head = _git(
        managed_checkout,
        "rev-parse",
        "--verify",
        "--quiet",
        "HEAD^{commit}",
        check=False,
    )
    assert (head.returncode, head.stdout, head.stderr) == (1, "", "")
    assert (
        _git(managed_checkout, "symbolic-ref", "--quiet", "HEAD").stdout
        == before_head_ref
    )
    assert _git(
        managed_checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout == ""
    assert {entry.name for entry in managed_checkout.iterdir()} == {".git"}
    assert all(str(actual_managed) not in command for command in observed_commands)


def test_managed_origin_repair_tolerates_its_own_runtime_state_churn(
    tmp_path,
    monkeypatch,
):
    """The live update-state file is a sibling of the managed checkout."""
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    assert {entry.name for entry in managed_checkout.iterdir()} == {".git"}

    state_path = managed_checkout.parent / "hermes_update_state.json"
    state._reset_for_tests(state_path)
    state.start_job(
        from_version="0.16.0",
        to_version="0.18.2",
        install_mode="editable",
    )
    state_before = state_path.read_bytes()
    checkout_before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    installer._prepare_managed_checkout(
        managed_checkout,
        _VETTED_HERMES_ORIGIN,
    )

    assert state_path.read_bytes() != state_before
    assert (
        _git(managed_checkout, "remote", "get-url", "origin").stdout.strip()
        == _VETTED_HERMES_ORIGIN
    )
    checkout_after = _tree_bytes(managed_checkout)
    assert {
        key: value for key, value in checkout_after.items() if key != ".git/config"
    } == {
        key: value for key, value in checkout_before.items() if key != ".git/config"
    }
    head = _git(
        managed_checkout,
        "rev-parse",
        "--verify",
        "--quiet",
        "HEAD^{commit}",
        check=False,
    )
    assert (head.returncode, head.stdout, head.stderr) == (1, "", "")
    assert _git(
        managed_checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout == ""
    assert {entry.name for entry in managed_checkout.iterdir()} == {".git"}


def test_origin_repair_refuses_wrong_path_without_mutation(tmp_path, monkeypatch):
    ms4_root = tmp_path / "ms4"
    wrong_checkout = ms4_root / "runtime" / "not-hermes-managed"
    _init_unborn_managed_checkout(wrong_checkout)
    before = _tree_bytes(wrong_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    assert (
        installer._repair_empty_unborn_managed_origin(
            wrong_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    assert _tree_bytes(wrong_checkout) == before
    assert (
        _git(wrong_checkout, "remote", "get-url", "origin").stdout.strip()
        == _LIVE_STALE_MANAGED_ORIGIN
    )


def test_origin_repair_refuses_managed_path_symlink_without_mutation(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    redirected = tmp_path / "redirected-managed"
    _init_unborn_managed_checkout(redirected)
    redirected_before = _tree_bytes(redirected)
    managed_checkout.parent.mkdir(parents=True)
    try:
        managed_checkout.symlink_to(redirected, target_is_directory=True)
    except OSError:
        _init_unborn_managed_checkout(managed_checkout)
        managed_before = _tree_bytes(managed_checkout)
        real_islink = installer.os.path.islink
        monkeypatch.setattr(
            installer.os.path,
            "islink",
            lambda path: Path(path) == managed_checkout or real_islink(path),
        )
        expected_link = None
    else:
        managed_before = None
        expected_link = os.readlink(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    assert (
        installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    if expected_link is None:
        assert _tree_bytes(managed_checkout) == managed_before
    else:
        assert managed_checkout.is_symlink()
        assert os.readlink(managed_checkout) == expected_link
        assert _tree_bytes(redirected) == redirected_before


def test_origin_repair_refuses_windows_runtime_junction_reparse_report(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    before = _tree_bytes(managed_checkout)
    real_reparse_check = installer._path_is_reparse_point
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_path_is_reparse_point",
        lambda path: Path(path) == managed_checkout.parent
        or real_reparse_check(path),
    )

    assert (
        installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    assert _tree_bytes(managed_checkout) == before


def test_origin_repair_refuses_git_file_external_gitdir_without_mutation(
    tmp_path,
    monkeypatch,
):
    external = tmp_path / "external"
    _init_unborn_managed_checkout(external)
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    managed_checkout.mkdir(parents=True)
    (managed_checkout / ".git").write_text(
        f"gitdir: {external / '.git'}\n",
        encoding="utf-8",
    )
    managed_before = _tree_bytes(managed_checkout)
    external_before = _tree_bytes(external)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    assert (
        installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    assert _tree_bytes(managed_checkout) == managed_before
    assert _tree_bytes(external) == external_before


def test_origin_repair_refuses_external_git_common_dir_without_mutation(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    git_dir = managed_checkout / ".git"
    external_common = tmp_path / "operator-common.git"
    shutil.copytree(git_dir, external_common)
    (git_dir / "commondir").write_text(
        f"{external_common.resolve()}\n",
        encoding="utf-8",
    )
    assert (
        Path(
            _git(
                managed_checkout,
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ).stdout.strip()
        ).resolve()
        == external_common.resolve()
    )
    managed_before = _tree_bytes(managed_checkout)
    external_before = _tree_bytes(external_common)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    assert (
        installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    assert _tree_bytes(managed_checkout) == managed_before
    assert _tree_bytes(external_common) == external_before


def test_origin_repair_refuses_git_directory_symlink_or_reparse_without_mutation(
    tmp_path,
    monkeypatch,
):
    external = tmp_path / "external"
    _init_unborn_managed_checkout(external)
    external_before = _tree_bytes(external)
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    managed_checkout.mkdir(parents=True)
    git_dir = managed_checkout / ".git"
    try:
        git_dir.symlink_to(external / ".git", target_is_directory=True)
    except OSError:
        _init_repo(managed_checkout)
        _git(
            managed_checkout,
            "remote",
            "add",
            "origin",
            _LIVE_STALE_MANAGED_ORIGIN,
        )
        managed_before = _tree_bytes(managed_checkout)
        real_reparse_check = installer._path_is_reparse_point
        monkeypatch.setattr(
            installer,
            "_path_is_reparse_point",
            lambda path: Path(path) == git_dir or real_reparse_check(path),
        )
        expected_link = None
    else:
        managed_before = None
        expected_link = os.readlink(git_dir)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    assert (
        installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    if expected_link is None:
        assert _tree_bytes(managed_checkout) == managed_before
    else:
        assert git_dir.is_symlink()
        assert os.readlink(git_dir) == expected_link
        assert _tree_bytes(external) == external_before


def test_origin_repair_refuses_valid_head_without_mutation(tmp_path, monkeypatch):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    _git(managed_checkout, "commit", "--allow-empty", "-m", "valid HEAD")
    before = _tree_bytes(managed_checkout)
    before_head = _git(managed_checkout, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    assert (
        installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    assert _tree_bytes(managed_checkout) == before
    assert _git(managed_checkout, "rev-parse", "HEAD").stdout.strip() == before_head


@pytest.mark.parametrize(
    ("entry_name", "is_directory"),
    [
        ("operator.txt", False),
        (".hidden-operator", False),
        (".hidden-operator-dir", True),
    ],
    ids=["visible-file", "hidden-file", "hidden-directory"],
)
def test_origin_repair_refuses_any_non_git_root_content_without_mutation(
    entry_name,
    is_directory,
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    operator_entry = managed_checkout / entry_name
    if is_directory:
        operator_entry.mkdir()
    else:
        operator_entry.write_bytes(b"operator-owned\r\n")
    before = _tree_bytes(managed_checkout)
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)

    assert (
        installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    assert _tree_bytes(managed_checkout) == before


@pytest.mark.parametrize("failure_mode", ["generic-error", "timeout"])
def test_origin_repair_never_treats_ambiguous_head_failure_as_unborn(
    failure_mode,
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    before = _tree_bytes(managed_checkout)
    real_capture = installer._run_capture

    def fail_head_query(cmd, **kwargs):
        if "rev-parse" in cmd and any(
            str(argument).startswith("HEAD") for argument in cmd
        ):
            if failure_mode == "timeout":
                raise installer.HermesUpgradeError("simulated HEAD query timeout")
            return FakeProcess(
                stderr="fatal: simulated malformed repository",
                returncode=128,
            )
        return real_capture(cmd, **kwargs)

    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(installer, "_run_capture", fail_head_query)

    try:
        repaired = installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
    except installer.HermesUpgradeError:
        repaired = False

    assert repaired is False
    assert _tree_bytes(managed_checkout) == before
    assert (
        _git(managed_checkout, "remote", "get-url", "origin").stdout.strip()
        == _LIVE_STALE_MANAGED_ORIGIN
    )


def test_origin_repair_refuses_detected_guard_race_before_mutation(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    managed_checkout = ms4_root / "runtime" / "hermes-managed"
    _init_unborn_managed_checkout(managed_checkout)
    config = managed_checkout / ".git" / "config"
    config_before = config.read_bytes()
    raced_file = managed_checkout / ".raced-operator-file"
    real_only_git = installer._managed_checkout_has_only_git_entry
    calls = 0

    def inject_race(directory):
        nonlocal calls
        calls += 1
        result = real_only_git(directory)
        if calls == 1:
            raced_file.write_bytes(b"created by simulated concurrent operator\r\n")
        return result

    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(
        installer,
        "_managed_checkout_has_only_git_entry",
        inject_race,
    )

    assert (
        installer._repair_empty_unborn_managed_origin(
            managed_checkout,
            _VETTED_HERMES_ORIGIN,
        )
        is False
    )

    assert calls >= 2
    assert raced_file.read_bytes() == b"created by simulated concurrent operator\r\n"
    assert config.read_bytes() == config_before
    assert (
        _git(managed_checkout, "remote", "get-url", "origin").stdout.strip()
        == _LIVE_STALE_MANAGED_ORIGIN
    )


def test_runtime_marker_rejects_checkout_outside_ms4_runtime(tmp_path):
    ms4_root = tmp_path / "ms4"
    marker = ms4_root / "runtime" / "hermes_active_checkout.json"
    marker.parent.mkdir(parents=True)
    outside = tmp_path / "operator-checkout"
    (outside / ".git").mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema": "Ms4HermesActiveCheckout.v1",
                "directory": str(outside),
                "version": "0.14.0",
                "commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )

    assert runtime_common._active_checkout_from_marker(ms4_root) is None


def test_valid_marker_overrides_inherited_external_restart_env(tmp_path, monkeypatch):
    ms4_root = tmp_path / "ms4"
    managed = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed)
    (managed / "tracked.txt").write_text("managed\n", encoding="utf-8")
    _git(managed, "add", "tracked.txt")
    _git(managed, "commit", "-m", "managed")
    commit = _git(managed, "rev-parse", "HEAD").stdout.strip()
    marker = ms4_root / "runtime" / "hermes_active_checkout.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "Ms4HermesActiveCheckout.v1",
                "directory": str(managed),
                "version": "0.14.0",
                "commit": commit,
            }
        ),
        encoding="utf-8",
    )
    external = tmp_path / "operator-checkout"
    _init_repo(external)
    (external / "tracked.txt").write_text("external\n", encoding="utf-8")
    _git(external, "add", "tracked.txt")
    _git(external, "commit", "-m", "external")
    monkeypatch.setattr(runtime_common, "MS4", ms4_root)
    monkeypatch.setenv("MS4_HERMES_DIR", str(external))
    monkeypatch.setenv("GIT_DIR", str(external / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(external))

    assert runtime_common.hermes_dir() == managed
    assert runtime_common.ms4_env()["MS4_HERMES_DIR"] == str(managed)


def test_explicit_restart_env_is_preserved_without_valid_marker(
    tmp_path,
    monkeypatch,
):
    ms4_root = tmp_path / "ms4"
    external = tmp_path / "operator-checkout"
    monkeypatch.setattr(runtime_common, "MS4", ms4_root)
    monkeypatch.setenv("MS4_HERMES_DIR", str(external))

    assert runtime_common.hermes_dir() == external
    assert runtime_common.ms4_env()["MS4_HERMES_DIR"] == str(external)


@pytest.mark.parametrize(
    ("platform_name", "environment", "expected"),
    [
        (
            "win32",
            {},
            PureWindowsPath("C:/Users/operator/Documents/hermes-agent"),
        ),
        (
            "darwin",
            {},
            PurePosixPath("/Users/operator/Library/Application Support/hermes-agent"),
        ),
        (
            "linux",
            {},
            PurePosixPath("/home/operator/.local/share/hermes-agent"),
        ),
        (
            "linux",
            {"XDG_DATA_HOME": "/xdg"},
            PurePosixPath("/xdg/hermes-agent"),
        ),
    ],
)
def test_runtime_default_hermes_dir_matches_platform_contract(
    platform_name,
    environment,
    expected,
):
    home = (
        PureWindowsPath("C:/Users/operator")
        if platform_name == "win32"
        else PurePosixPath(
            "/Users/operator" if platform_name == "darwin" else "/home/operator"
        )
    )
    assert (
        runtime_common._default_hermes_dir(platform_name, environment, home)
        == expected
    )


@pytest.mark.parametrize(
    ("version", "commit_value"),
    [
        ("0.14.0", "b" * 40),
        ("unsafe version", "head"),
    ],
)
def test_runtime_marker_rejects_version_or_head_mismatch(
    version,
    commit_value,
    tmp_path,
):
    ms4_root = tmp_path / "ms4"
    managed = ms4_root / "runtime" / "hermes-managed"
    _init_repo(managed)
    (managed / "tracked.txt").write_text("managed\n", encoding="utf-8")
    _git(managed, "add", "tracked.txt")
    _git(managed, "commit", "-m", "managed")
    head = _git(managed, "rev-parse", "HEAD").stdout.strip()
    marker = ms4_root / "runtime" / "hermes_active_checkout.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "Ms4HermesActiveCheckout.v1",
                "directory": str(managed),
                "version": version,
                "commit": head if commit_value == "head" else commit_value,
            }
        ),
        encoding="utf-8",
    )

    assert runtime_common._active_checkout_from_marker(ms4_root) is None


def test_realpath_alias_is_not_exact_managed_checkout(tmp_path, monkeypatch):
    ms4_root = tmp_path / "ms4"
    exact = ms4_root / "runtime" / "hermes-managed"
    alias = tmp_path / "alias-to-managed"
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    monkeypatch.setattr(installer, "_normalized_real_path", lambda _path: "same")

    assert installer._same_path(alias, exact) is True
    assert installer._is_exact_managed_checkout(alias) is False


@pytest.mark.parametrize(
    ("fault", "error_match"),
    [
        ("direct_url", "direct_url"),
        ("plugin", "plugin_origin.*outside"),
        ("none", None),
    ],
)
def test_managed_install_validation_requires_contained_metadata_and_imports(
    fault,
    error_match,
    tmp_path,
    monkeypatch,
):
    managed = tmp_path / "ms4" / "runtime" / "hermes-managed"
    outside = tmp_path / "operator-checkout"
    (managed / ".git").mkdir(parents=True)
    monkeypatch.setattr(installer, "MS4_ROOT", tmp_path / "ms4")
    monkeypatch.setattr(
        installer,
        "_is_exact_managed_checkout",
        lambda directory: Path(directory) == managed,
    )
    monkeypatch.setattr(installer, "_validate", lambda *_args, **_kwargs: "0.14.0")
    direct_url_root = outside if fault == "direct_url" else managed
    plugin_root = outside if fault == "plugin" else managed
    payload = {
        "distributions": [
            {
                "version": "0.14.0",
                "direct_url": {
                    "url": direct_url_root.as_uri(),
                    "dir_info": {"editable": True},
                },
            }
        ],
        "plugin_origin": str(
            plugin_root / "plugins" / "ms4_consciousness" / "__init__.py"
        ),
        "run_agent_origin": str(managed / "run_agent.py"),
    }
    monkeypatch.setattr(
        installer,
        "_run",
        lambda *_args, **_kwargs: FakeProcess(stdout=json.dumps(payload)),
    )

    if error_match is not None:
        with pytest.raises(installer.HermesUpgradeError, match=error_match):
            installer._validate_managed_install(
                tmp_path / "python",
                expected_version="0.14.0",
                editable_directory=managed,
                expected_commit="a" * 40,
            )
    else:
        assert (
            installer._validate_managed_install(
                tmp_path / "python",
                expected_version="0.14.0",
                editable_directory=managed,
                expected_commit="a" * 40,
            )
            == "0.14.0"
        )
