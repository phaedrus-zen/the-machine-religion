"""Offline product gate for the Hermes ``Update`` transaction.

The four outcomes here share a disposable packaged-release repository under
``tmp_path``.  The directory-level basetemp guard rejects any parent Git root,
and ``_init_repo`` additionally proves each fixture repository is isolated
before a ``git -C`` write.  Release discovery and package installation are
test-local fakes; checkout, trigger/worker ownership, durable state, and
rollback use the production implementation.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from machine_spirit_4.hermes_admin import installer, state, versioning


PRIOR_VERSION = "1.0.0"
TARGET_VERSION = "2.0.0"
TARGET_TAG = "v2.0.0"


@dataclass(frozen=True)
class _ProductFixture:
    remote: Path
    checkout: Path
    plugin_src: Path
    installed_version: Path
    prior_commit: str
    target_commit: str
    published: versioning.CachedLatest


def _git(cwd: Path | None, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git fixture failed: {result.args!r}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def _assert_isolated_repo(path: Path, *, bare: bool) -> None:
    actual = Path(_git(None, "-C", str(path), "rev-parse", "--absolute-git-dir"))
    expected = path if bare else path / ".git"
    assert os.path.normcase(os.path.realpath(actual)) == os.path.normcase(
        os.path.realpath(expected)
    )


def _init_repo(path: Path, *, bare: bool = False) -> None:
    args = ["init", "-q"]
    if bare:
        args.append("--bare")
    _git(None, *args, str(path))
    _assert_isolated_repo(path, bare=bare)
    if not bare:
        _git(path, "config", "user.name", "Hermes Product Gate")
        _git(path, "config", "user.email", "hermes-product-gate@example.invalid")
        _git(path, "config", "commit.gpgsign", "false")
        _git(path, "config", "tag.gpgSign", "false")


def _make_product_fixture(
    tmp_path: Path,
    *,
    tag_message: str = "fake signed release",
    prior_files: dict[str, bytes] | None = None,
    target_mutation: Callable[[Path], None] | None = None,
    checkout: Path | None = None,
    checkout_config: dict[str, str] | None = None,
) -> _ProductFixture:
    """Disposable packaged release: ``prior`` (installed) -> ``TARGET_TAG``.

    ``prior_files`` land in the prior commit byte-for-byte (the seed pins
    ``core.autocrlf=false`` so CRLF payloads survive ``git add`` on a Windows
    host whose system config sets ``autocrlf=true``). ``target_mutation`` runs
    inside the seed just before the target commit, so a scenario can rewrite,
    delete, or -- via plumbing -- add a case-colliding path outside the
    managed paths. ``checkout_config`` is applied to the installed clone only
    (never the global config).
    """
    remote = tmp_path / "release-remote.git"
    seed = tmp_path / "release-seed"
    if checkout is None:
        checkout = tmp_path / "installed-editable"
    _init_repo(seed)
    _git(seed, "config", "core.autocrlf", "false")

    package_version = seed / "package-version.txt"
    package_version.write_bytes(f"{PRIOR_VERSION}\n".encode())
    _git(seed, "add", "package-version.txt")
    for relpath, payload in (prior_files or {}).items():
        target = seed.joinpath(*relpath.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        _git(seed, "add", relpath)
    _git(seed, "commit", "-q", "-m", "packaged Hermes 1.0.0")
    prior_commit = _git(seed, "rev-parse", "HEAD")
    _git(seed, "branch", "prior", prior_commit)

    package_version.write_bytes(f"{TARGET_VERSION}\n".encode())
    _git(seed, "add", "package-version.txt")
    if target_mutation is not None:
        target_mutation(seed)
    _git(seed, "commit", "-q", "-m", "packaged Hermes 2.0.0")
    target_commit = _git(seed, "rev-parse", "HEAD")
    _git(seed, "tag", "-a", TARGET_TAG, "-m", tag_message, target_commit)
    _git(seed, "branch", "-M", "main")
    _git(None, "clone", "-q", "--bare", str(seed), str(remote))
    _assert_isolated_repo(remote, bare=True)
    _git(None, "-C", str(remote), "symbolic-ref", "HEAD", "refs/heads/main")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    _git(None, "clone", "-q", "--no-tags", "--branch", "prior", str(remote), str(checkout))
    _assert_isolated_repo(checkout, bare=False)
    for key, value in (checkout_config or {}).items():
        _git(checkout, "config", key, value)

    plugin_src = tmp_path / "ms4_consciousness"
    plugin_src.mkdir()
    (plugin_src / "__init__.py").write_bytes(b"")
    (plugin_src / "plugin.yaml").write_bytes(b"name: ms4_consciousness\n")
    installed_version = tmp_path / "installed-version.txt"
    installed_version.write_bytes(f"{PRIOR_VERSION}\n".encode())

    return _ProductFixture(
        remote=remote,
        checkout=checkout,
        plugin_src=plugin_src,
        installed_version=installed_version,
        prior_commit=prior_commit,
        target_commit=target_commit,
        published=versioning.CachedLatest(
            version=TARGET_VERSION,
            published_at="2026-09-06T00:00:00Z",
            tag_name=TARGET_TAG,
            html_url="https://example.invalid/hermes/releases/v2.0.0",
            fetched_at_unix=0.0,
        ),
    )


def _wire_offline_release(
    monkeypatch: pytest.MonkeyPatch,
    fixture: _ProductFixture,
    *,
    signed: bool = True,
    policy: str = versioning.RELEASE_SIGNATURE_POLICY_REQUIRE_SIGNED,
    newest_is_target: bool = False,
) -> None:
    # The four original outcomes prove the signed-only gate; pin it explicitly
    # now that the shipped default is ``allow_unsigned``.
    monkeypatch.setenv(versioning.RELEASE_SIGNATURE_POLICY_ENV, policy)
    versioning._clear_caches_for_test()

    def install_mode() -> dict[str, object]:
        return {
            "mode": "editable",
            "version": fixture.installed_version.read_text(encoding="utf-8").strip(),
            "directory": fixture.checkout,
        }

    newest_publication = {
        "tag_name": "v3.0.0",
        "name": "Hermes Agent v3.0.0",
        "published_at": "2026-09-07T00:00:00Z",
        "prerelease": False,
        "html_url": "https://example.invalid/hermes/releases/v3.0.0",
    }
    target_release = {
        "tag_name": TARGET_TAG,
        "name": f"Hermes Agent v{TARGET_VERSION}",
        "published_at": fixture.published.published_at,
        "prerelease": False,
        "html_url": fixture.published.html_url,
    }

    def fake_release_get(url: str, timeout: int = 8):
        if url == versioning.LATEST_RELEASE_URL:
            return target_release if newest_is_target else newest_publication
        if url == versioning.RECENT_RELEASES_URL:
            return [target_release] if newest_is_target else [newest_publication, target_release]
        raise AssertionError(f"unexpected release-network request: {url}")

    monkeypatch.setattr(installer, "CANONICAL_HERMES_SOURCE_ORIGIN", str(fixture.remote))
    monkeypatch.setattr(
        installer,
        "_verify_release_origin_preflight",
        lambda **_kwargs: str(fixture.remote),
    )
    monkeypatch.setattr(installer, "_preflight_disk_space", lambda *_a, **_k: None)
    monkeypatch.setattr(installer.versioning, "install_mode", install_mode)
    monkeypatch.setattr(
        installer.versioning,
        "_http_get_json",
        fake_release_get,
    )
    monkeypatch.setattr(
        installer.versioning,
        "official_tag_signature_state",
        lambda tag, force_refresh=False: (
            "signed" if signed and tag == TARGET_TAG else "unsigned"
        ),
    )


def _await_terminal(job_id: str, *, timeout: float = 10.0) -> state.UpdateJobSnapshot:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = state.last_update(refresh=True)
        if snapshot is not None and snapshot.job_id == job_id and snapshot.status != "running":
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"Hermes update job {job_id} did not reach a terminal state")


def _packaged_version_bytes(directory: Path) -> bytes:
    version = (directory / "package-version.txt").read_text(encoding="utf-8").strip()
    return f"{version}\n".encode()


def _forbid_effect(name: str):
    def forbidden(*_args, **_kwargs):
        raise AssertionError(f"{name} reached after product-gate refusal")

    return forbidden


def test_newer_signed_editable_release_reaches_terminal_success_and_new_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_product_fixture(tmp_path)
    _wire_offline_release(monkeypatch, fixture)
    state._reset_for_tests(tmp_path / "state" / "update.json")
    provenance_calls: list[dict[str, object]] = []
    install_heads: list[str] = []
    fake_python = tmp_path / "fake-python"
    real_run = installer._run

    def verify_release(**kwargs) -> None:
        provenance_calls.append(kwargs)

    def run(cmd, *, cwd=None, timeout=600):
        if Path(cmd[0]) == fake_python and cmd[1:4] == ["-m", "pip", "install"]:
            directory = Path(cmd[-1])
            install_heads.append(installer._git_head(directory))
            fixture.installed_version.write_bytes(_packaged_version_bytes(directory))
            return subprocess.CompletedProcess(cmd, 0, stdout="installed\n", stderr="")
        if Path(cmd[0]) == fake_python and cmd[1:2] == ["-c"]:
            installed = fixture.installed_version.read_text(encoding="utf-8").strip()
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"{installed}\n",
                stderr="",
            )
        return real_run(cmd, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(installer, "_verify_release_provenance", verify_release)
    monkeypatch.setattr(installer, "_run", run)

    accepted = installer.trigger_update(
        plugin_src=fixture.plugin_src,
        venv_python=fake_python,
        request_user="product-gate",
    )
    final = _await_terminal(accepted.job_id)

    assert final.status == "success"
    assert final.phase == "done"
    assert final.from_version == PRIOR_VERSION
    assert final.to_version == TARGET_VERSION
    assert fixture.installed_version.read_bytes() == f"{TARGET_VERSION}\n".encode()
    assert _git(fixture.checkout, "rev-parse", "HEAD") == fixture.target_commit
    assert install_heads == [fixture.target_commit]
    assert len(provenance_calls) == 1
    assert provenance_calls[0]["target"] == TARGET_VERSION
    assert provenance_calls[0]["tag"] == TARGET_TAG
    assert provenance_calls[0]["expected_commit"] == fixture.target_commit


def test_no_newer_signed_release_refuses_before_job_or_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_product_fixture(tmp_path)
    _wire_offline_release(monkeypatch, fixture, signed=False)
    state._reset_for_tests(tmp_path / "state" / "update.json")
    before_head = _git(fixture.checkout, "rev-parse", "HEAD")
    before_installed = fixture.installed_version.read_bytes()

    monkeypatch.setattr(installer.state, "start_job", _forbid_effect("job start"))
    monkeypatch.setattr(installer, "_pip_install_editable", _forbid_effect("install"))
    monkeypatch.setattr(installer, "_reinstall_editable", _forbid_effect("rollback install"))

    with pytest.raises(
        installer.HermesUpgradeError,
        match="No newer signed Hermes release is available",
    ):
        installer.trigger_update(
            plugin_src=fixture.plugin_src,
            venv_python=tmp_path / "fake-python",
        )

    assert state.last_update(refresh=True) is None
    assert _git(fixture.checkout, "rev-parse", "HEAD") == before_head
    assert fixture.installed_version.read_bytes() == before_installed


def test_preflight_failure_refuses_before_job_or_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_product_fixture(tmp_path)
    _wire_offline_release(monkeypatch, fixture)
    state._reset_for_tests(tmp_path / "state" / "update.json")
    before_head = _git(fixture.checkout, "rev-parse", "HEAD")
    before_installed = fixture.installed_version.read_bytes()

    def fail_preflight(*_args, **_kwargs) -> None:
        raise installer.HermesInsufficientDiskError("simulated product preflight failure")

    monkeypatch.setattr(installer, "_preflight_disk_space", fail_preflight)
    monkeypatch.setattr(
        installer.versioning,
        "latest_signed_version",
        _forbid_effect("signed release resolution"),
    )
    monkeypatch.setattr(installer.state, "start_job", _forbid_effect("job start"))
    monkeypatch.setattr(installer, "_pip_install_editable", _forbid_effect("install"))
    monkeypatch.setattr(installer, "_reinstall_editable", _forbid_effect("rollback install"))

    with pytest.raises(
        installer.HermesInsufficientDiskError,
        match="simulated product preflight failure",
    ):
        installer.trigger_update(
            plugin_src=fixture.plugin_src,
            venv_python=tmp_path / "fake-python",
        )

    assert state.last_update(refresh=True) is None
    assert _git(fixture.checkout, "rev-parse", "HEAD") == before_head
    assert fixture.installed_version.read_bytes() == before_installed


def test_mid_install_failure_rolls_back_prior_head_and_installed_version_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_product_fixture(tmp_path)
    _wire_offline_release(monkeypatch, fixture)
    state._reset_for_tests(tmp_path / "state" / "update.json")
    before_head = _git(fixture.checkout, "rev-parse", "HEAD")
    before_installed = fixture.installed_version.read_bytes()
    rollback_heads: list[str] = []
    fake_python = tmp_path / "fake-python"
    real_run = installer._run
    install_attempts = 0

    def run(cmd, *, cwd=None, timeout=600):
        nonlocal install_attempts
        if Path(cmd[0]) == fake_python and cmd[1:4] == ["-m", "pip", "install"]:
            install_attempts += 1
            directory = Path(cmd[-1])
            head = installer._git_head(directory)
            fixture.installed_version.write_bytes(_packaged_version_bytes(directory))
            if install_attempts == 1:
                assert head == fixture.target_commit
                raise RuntimeError("simulated mid-install failure")
            rollback_heads.append(head)
            return subprocess.CompletedProcess(cmd, 0, stdout="restored\n", stderr="")
        if Path(cmd[0]) == fake_python and cmd[1:2] == ["-c"]:
            raise AssertionError("validation reached after simulated install failure")
        return real_run(cmd, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(installer, "_run", run)

    accepted = installer.trigger_update(
        plugin_src=fixture.plugin_src,
        venv_python=fake_python,
    )
    final = _await_terminal(accepted.job_id)

    assert final.status == "failed"
    assert final.phase == "error"
    assert "simulated mid-install failure" in (final.error or "")
    assert "rolled back to the prior state" in (final.error or "")
    assert before_head == fixture.prior_commit
    assert _git(fixture.checkout, "rev-parse", "HEAD") == before_head
    assert fixture.installed_version.read_bytes() == before_installed
    assert rollback_heads == [fixture.prior_commit]
    phases = [entry["phase"] for entry in final.progress]
    assert "rolling_back" in phases
    assert "rolled_back" in phases


@pytest.mark.real_provenance
def test_unsigned_newest_under_allow_unsigned_completes_and_reports_new_installed_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operator decision 2026-09-07: the newest published release is UNSIGNED and
    ``HERMES_RELEASE_SIGNATURE_POLICY=allow_unsigned`` (the default) installs it.

    ``real_provenance`` keeps the installer's verification slot un-stubbed, so
    the signature-independent bindings (tag -> fetched commit, upstream
    tag-object version/tag/commit, no replace refs) run for real against the
    fixture repository; only the signer check is skipped, and that is logged.
    """
    fixture = _make_product_fixture(
        tmp_path,
        tag_message=f"Hermes Agent v{TARGET_VERSION} ({TARGET_VERSION})",
    )
    _wire_offline_release(
        monkeypatch,
        fixture,
        signed=False,
        policy=versioning.RELEASE_SIGNATURE_POLICY_ALLOW_UNSIGNED,
        newest_is_target=True,
    )
    for key in list(os.environ):
        if key.startswith("MS4_HERMES_PROVENANCE_"):
            monkeypatch.delenv(key, raising=False)
    state._reset_for_tests(tmp_path / "state" / "update.json")
    install_heads: list[str] = []
    fake_python = tmp_path / "fake-python"
    real_run = installer._run

    def run(cmd, *, cwd=None, timeout=600):
        if Path(cmd[0]) == fake_python and cmd[1:4] == ["-m", "pip", "install"]:
            directory = Path(cmd[-1])
            install_heads.append(installer._git_head(directory))
            fixture.installed_version.write_bytes(_packaged_version_bytes(directory))
            return subprocess.CompletedProcess(cmd, 0, stdout="installed\n", stderr="")
        if Path(cmd[0]) == fake_python and cmd[1:2] == ["-c"]:
            installed = fixture.installed_version.read_text(encoding="utf-8").strip()
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{installed}\n", stderr="")
        return real_run(cmd, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(installer, "_run", run)
    caplog.set_level(logging.WARNING, logger="machine_spirit_4.hermes_admin.installer")

    before = versioning.version_info(force_refresh_latest=True)
    assert before["release_signature_policy"] == "allow_unsigned"
    assert before["latest"] == TARGET_VERSION
    assert before["latest_signature_state"] == "unsigned"
    assert before["update_available"] is True
    assert before["operator_state"] == "update_available"

    accepted = installer.trigger_update(
        plugin_src=fixture.plugin_src,
        venv_python=fake_python,
        request_user="product-gate",
    )
    assert accepted.status == "running"
    assert accepted.to_version == TARGET_VERSION
    final = _await_terminal(accepted.job_id)

    assert final.status == "success"
    assert final.phase == "done"
    assert final.from_version == PRIOR_VERSION
    assert final.to_version == TARGET_VERSION
    assert fixture.installed_version.read_bytes() == f"{TARGET_VERSION}\n".encode()
    assert _git(fixture.checkout, "rev-parse", "HEAD") == fixture.target_commit
    assert install_heads == [fixture.target_commit]
    notes = [entry.get("note") or "" for entry in final.progress]
    warning = f"signature policy allow_unsigned: installing unsigned tag {TARGET_TAG}"
    assert any(warning in note for note in notes)
    assert any(
        f"bound to commit {fixture.target_commit} and version {TARGET_VERSION}" in note
        for note in notes
    )
    assert "verifying_provenance" in [entry["phase"] for entry in final.progress]
    assert any(
        record.levelno == logging.WARNING and warning in record.getMessage()
        for record in caplog.records
    )

    after = versioning.version_info()
    assert after["current"] == TARGET_VERSION
    assert after["release_signature_policy"] == "allow_unsigned"
    assert after["update_available"] is False
    assert after["operator_state"] == "current"
    assert after["last_update"]["status"] == "success"


def test_unsigned_newest_under_require_signed_still_refuses_before_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same unsigned newest release; strict policy keeps today's refusal."""
    fixture = _make_product_fixture(tmp_path)
    _wire_offline_release(
        monkeypatch,
        fixture,
        signed=False,
        policy=versioning.RELEASE_SIGNATURE_POLICY_REQUIRE_SIGNED,
        newest_is_target=True,
    )
    state._reset_for_tests(tmp_path / "state" / "update.json")
    before_head = _git(fixture.checkout, "rev-parse", "HEAD")
    monkeypatch.setattr(installer.state, "start_job", _forbid_effect("job start"))
    monkeypatch.setattr(installer, "_pip_install_editable", _forbid_effect("install"))

    info = versioning.version_info(force_refresh_latest=True)
    assert info["release_signature_policy"] == "require_signed"
    assert info["update_available"] is False
    assert info["operator_state"] == "blocked"
    assert info["update_blocked_reason"] == "official_tag_unsigned"
    with pytest.raises(
        installer.HermesUpgradeError,
        match="No newer signed Hermes release is available",
    ):
        installer.trigger_update(
            plugin_src=fixture.plugin_src,
            venv_python=tmp_path / "fake-python",
        )
    assert state.last_update(refresh=True) is None
    assert _git(fixture.checkout, "rev-parse", "HEAD") == before_head


# --------------------------------------------------------------------------- #
# Live Windows reproduction (jobs b892a866 / 91fe4bd0, 2026-09-07).
#
# hermes-agent v2026.8.31 adds ``contributors/emails/agent@Agents-Mac-mini.local``
# next to the existing ``agent@agents-Mac-mini.local``. On an ignore-case
# filesystem (``core.ignorecase=true``, every Windows clone) both index entries
# share ONE on-disk file: the forward checkout writes the new entry over the
# old one, ``status`` reports the untouched entry as `` M``, the old updater
# mistook its own checkout for an operator edit and rolled back, and the
# whole-commit rollback checkout unlinked the shared file (`` D``). The blobs
# themselves are LF-only; the line-ending hypothesis is refuted by the
# ``autocrlf=true`` CRLF scenario below, which completes on the OLD gate too.
# --------------------------------------------------------------------------- #
COLLIDING_LOWER = "contributors/emails/agent@agents-Mac-mini.local"
COLLIDING_UPPER = "contributors/emails/agent@Agents-Mac-mini.local"
LIVE_CHECKOUT_CONFIG = {"core.autocrlf": "true", "core.ignorecase": "true"}
REFUSAL_TEXT = (
    "Hermes checkout has tracked or untracked changes outside the MS4-managed "
    "updater paths"
)


def _filesystem_folds_case(root: Path) -> bool:
    probe = root / "CaseProbe.tmp"
    probe.write_bytes(b"")
    try:
        return (root / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


def _add_case_colliding_contributor(seed: Path) -> None:
    """Add ``COLLIDING_UPPER`` through the index only (plumbing), exactly as the
    upstream release carries it; on an ignore-case host a worktree write would
    silently overwrite ``COLLIDING_LOWER`` instead of adding a second entry."""
    blob_source = seed.parent / "colliding-blob.txt"
    blob_source.write_bytes(b"skip-agent\n")
    blob = _git(seed, "hash-object", "-w", str(blob_source))
    _git(seed, "update-index", "--add", "--cacheinfo", f"100644,{blob},{COLLIDING_UPPER}")
    assert sorted(_git(seed, "ls-files", "--", "contributors/").splitlines()) == sorted(
        [COLLIDING_LOWER, COLLIDING_UPPER]
    )


def _status(checkout: Path) -> str:
    """Raw porcelain status (NOT stripped: the leading status column matters)."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "core.quotePath=false",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _status_outside_managed(checkout: Path) -> list[str]:
    """Status lines other than the (always untracked) re-stamped managed plugin."""
    return [
        line
        for line in _status(checkout).splitlines()
        if line.strip() and not line[3:].startswith(installer.MANAGED_PLUGIN_RELPOSIX + "/")
    ]


def _worktree_matches(checkout: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--quiet", commit, "--"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _notes(snapshot: state.UpdateJobSnapshot) -> list[str]:
    return [entry.get("note") or "" for entry in snapshot.progress]


def _install_runner(
    fixture: _ProductFixture,
    fake_python: Path,
    *,
    fail_first_install: bool = False,
):
    """pip/validate fakes; git stays real. Returns (runner, install_heads)."""
    real_run = installer._run
    install_heads: list[str] = []

    def run(cmd, *, cwd=None, timeout=600):
        if Path(cmd[0]) == fake_python and cmd[1:4] == ["-m", "pip", "install"]:
            directory = Path(cmd[-1])
            install_heads.append(installer._git_head(directory))
            fixture.installed_version.write_bytes(_packaged_version_bytes(directory))
            if fail_first_install and len(install_heads) == 1:
                raise RuntimeError("simulated mid-install failure")
            return subprocess.CompletedProcess(cmd, 0, stdout="installed\n", stderr="")
        if Path(cmd[0]) == fake_python and cmd[1:2] == ["-c"]:
            installed = fixture.installed_version.read_text(encoding="utf-8").strip()
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{installed}\n", stderr="")
        return real_run(cmd, cwd=cwd, timeout=timeout)

    return run, install_heads


def _run_product_update(
    fixture: _ProductFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail_first_install: bool = False,
) -> tuple[state.UpdateJobSnapshot, list[str]]:
    _wire_offline_release(monkeypatch, fixture)
    state._reset_for_tests(tmp_path / "state" / "update.json")
    fake_python = tmp_path / "fake-python"
    run, install_heads = _install_runner(
        fixture, fake_python, fail_first_install=fail_first_install
    )
    monkeypatch.setattr(installer, "_run", run)
    accepted = installer.trigger_update(
        plugin_src=fixture.plugin_src,
        venv_python=fake_python,
        request_user="product-gate",
    )
    return _await_terminal(accepted.job_id), install_heads


def _emulate_pre_fix_updater(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two repairs, switched off: no updater-produced waivers, no path-level
    tree restore. Everything else (baseline plumbing included) is production."""
    monkeypatch.setattr(installer, "_explain_updater_status_entry", lambda *_a, **_k: None)
    monkeypatch.setattr(
        installer,
        "_restore_prior_tree",
        lambda _pre, _target: installer._TreeRestoreReport((), (), (), ()),
    )


def _colliding_fixture(tmp_path: Path) -> _ProductFixture:
    if not _filesystem_folds_case(tmp_path):
        pytest.skip("ignore-case filename collision needs a case-insensitive filesystem")
    return _make_product_fixture(
        tmp_path,
        prior_files={COLLIDING_LOWER: b"momomojo\n"},
        target_mutation=_add_case_colliding_contributor,
        checkout_config=LIVE_CHECKOUT_CONFIG,
    )


def test_case_colliding_release_reproduces_live_refusal_and_lost_file_without_the_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _colliding_fixture(tmp_path)
    _emulate_pre_fix_updater(monkeypatch)
    assert _status_outside_managed(fixture.checkout) == []

    final, install_heads = _run_product_update(fixture, monkeypatch, tmp_path)

    assert final.status == "failed"
    assert REFUSAL_TEXT in (final.error or "")
    assert f" M {COLLIDING_LOWER}" in (final.error or "")
    assert "rolled back to the prior state" in (final.error or "")
    assert install_heads == [fixture.prior_commit]  # only the rollback reinstall
    assert _git(fixture.checkout, "rev-parse", "HEAD") == fixture.prior_commit
    # The live symptom: the rollback checkout unlinked the shared file.
    assert f" D {COLLIDING_LOWER}" in _status(fixture.checkout)
    assert not fixture.checkout.joinpath(*COLLIDING_LOWER.split("/")).exists()


def test_case_colliding_release_completes_and_reports_new_installed_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _colliding_fixture(tmp_path)

    final, install_heads = _run_product_update(fixture, monkeypatch, tmp_path)

    assert final.status == "success", final.error
    assert final.phase == "done"
    assert final.to_version == TARGET_VERSION
    assert fixture.installed_version.read_bytes() == f"{TARGET_VERSION}\n".encode()
    assert _git(fixture.checkout, "rev-parse", "HEAD") == fixture.target_commit
    assert install_heads == [fixture.target_commit]
    waiver = (
        f"status entry is the updater's own, not operator dirt:  M {COLLIDING_LOWER} "
        f"(case-insensitive filename collision with {COLLIDING_UPPER})"
    )
    assert waiver in _notes(final)
    # The collision is permanent on this filesystem; the NEXT update's
    # pre-mutation gate must not refuse it either (it is not operator dirt).
    state.start_job(from_version=TARGET_VERSION, to_version="3.0.0", install_mode="editable")
    assert f" M {COLLIDING_LOWER}" in installer._ensure_clean_checkout(fixture.checkout)


def test_mid_install_failure_after_case_colliding_checkout_restores_full_prior_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _colliding_fixture(tmp_path)

    final, install_heads = _run_product_update(
        fixture, monkeypatch, tmp_path, fail_first_install=True
    )

    assert final.status == "failed"
    assert "simulated mid-install failure" in (final.error or "")
    assert "rolled back to the prior state" in (final.error or "")
    assert install_heads == [fixture.target_commit, fixture.prior_commit]
    assert _git(fixture.checkout, "rev-parse", "HEAD") == fixture.prior_commit
    restored = fixture.checkout.joinpath(*COLLIDING_LOWER.split("/"))
    assert restored.read_bytes().replace(b"\r\n", b"\n") == b"momomojo\n"
    assert _status_outside_managed(fixture.checkout) == []
    assert _worktree_matches(fixture.checkout, fixture.prior_commit)
    notes = _notes(final)
    assert any(
        note.startswith("rollback restored 1 path(s) from") and COLLIDING_LOWER in note
        for note in notes
    )
    assert any(
        "rollback tree verified: working tree matches" in note
        and "(restored 1, removed 0)" in note
        for note in notes
    )
    phases = [entry["phase"] for entry in final.progress]
    assert "rolled_back" in phases
    assert "rollback_failed" not in phases


def test_release_rewriting_file_to_crlf_completes_under_autocrlf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refutes the line-ending hypothesis for the live failure: a release that
    turns an LF file into CRLF is stable under ``core.autocrlf=true`` (modern git
    never re-normalizes a blob that already carries CR), so it never produced
    the refusal -- even with the repairs switched off it completes. Either way
    the update must land the CRLF bytes and leave a clean status."""
    notes_path = "docs/NOTES.txt"

    def rewrite_to_crlf(seed: Path) -> None:
        (seed / "docs" / "NOTES.txt").write_bytes(b"line one\r\nline two\r\n")
        _git(seed, "add", notes_path)

    fixture = _make_product_fixture(
        tmp_path,
        prior_files={notes_path: b"line one\nline two\n"},
        target_mutation=rewrite_to_crlf,
        checkout_config=LIVE_CHECKOUT_CONFIG,
    )
    _emulate_pre_fix_updater(monkeypatch)

    final, install_heads = _run_product_update(fixture, monkeypatch, tmp_path)

    assert final.status == "success", final.error
    assert final.to_version == TARGET_VERSION
    assert install_heads == [fixture.target_commit]
    assert (fixture.checkout / "docs" / "NOTES.txt").read_bytes() == b"line one\r\nline two\r\n"
    assert _status_outside_managed(fixture.checkout) == []


def test_release_deleting_file_outside_managed_paths_rolls_back_full_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retired = "contributors/emails/retired@example.invalid"

    def delete_contributor(seed: Path) -> None:
        _git(seed, "rm", "-q", "--", retired)

    fixture = _make_product_fixture(
        tmp_path,
        prior_files={retired: b"retired\n"},
        target_mutation=delete_contributor,
        checkout_config=LIVE_CHECKOUT_CONFIG,
    )

    final, install_heads = _run_product_update(
        fixture, monkeypatch, tmp_path, fail_first_install=True
    )

    assert final.status == "failed"
    assert "rolled back to the prior state" in (final.error or "")
    assert install_heads == [fixture.target_commit, fixture.prior_commit]
    assert _git(fixture.checkout, "rev-parse", "HEAD") == fixture.prior_commit
    assert fixture.checkout.joinpath(*retired.split("/")).read_bytes().replace(
        b"\r\n", b"\n"
    ) == b"retired\n"
    assert _status_outside_managed(fixture.checkout) == []
    assert _worktree_matches(fixture.checkout, fixture.prior_commit)
    assert any(
        "rollback tree verified: working tree matches" in note
        and "(restored 0, removed 0)" in note
        for note in _notes(final)
    )


def test_operator_edit_outside_managed_paths_is_still_refused_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The managed checkout topology of the live box: an operator edit to a
    tracked file outside the managed paths is refused before fetch/checkout."""
    notes_path = "docs/NOTES.txt"
    ms4_root = tmp_path / "ms4"
    managed = ms4_root / "runtime" / "hermes-managed"
    monkeypatch.setattr(installer, "MS4_ROOT", ms4_root)
    fixture = _make_product_fixture(
        tmp_path,
        prior_files={notes_path: b"line one\n"},
        checkout=managed,
        checkout_config=LIVE_CHECKOUT_CONFIG,
    )
    edited = fixture.checkout / "docs" / "NOTES.txt"
    edited.write_bytes(b"operator edit\n")
    before_installed = fixture.installed_version.read_bytes()

    final, install_heads = _run_product_update(fixture, monkeypatch, tmp_path)

    assert final.status == "failed"
    assert REFUSAL_TEXT in (final.error or "")
    assert f" M {notes_path}" in (final.error or "")
    assert "rolled back" not in (final.error or "")
    assert install_heads == []
    assert _git(fixture.checkout, "rev-parse", "HEAD") == fixture.prior_commit
    assert edited.read_bytes() == b"operator edit\n"
    assert fixture.installed_version.read_bytes() == before_installed
    assert not any("checkout --detach" in note or " fetch " in note for note in _notes(final))


def test_line_ending_only_difference_is_not_operator_dirt(
    tmp_path: Path,
) -> None:
    """A tracked file whose only difference is CR at EOL (an editor saved it
    with CRLF on an ``autocrlf=false`` clone) is not dirt; a content edit is."""
    notes_path = "docs/NOTES.txt"
    fixture = _make_product_fixture(
        tmp_path,
        prior_files={notes_path: b"line one\nline two\n"},
        checkout_config={"core.autocrlf": "false"},
    )
    state._reset_for_tests(tmp_path / "state" / "update.json")
    state.start_job(from_version=PRIOR_VERSION, to_version=TARGET_VERSION, install_mode="editable")
    notes = fixture.checkout / "docs" / "NOTES.txt"

    notes.write_bytes(b"line one\r\nline two\r\n")
    assert f" M {notes_path}" in _status(fixture.checkout)  # git does report it
    assert f" M {notes_path}" in installer._ensure_clean_checkout(fixture.checkout)
    assert notes.read_bytes() == b"line one\r\nline two\r\n"  # never rewritten

    notes.write_bytes(b"line one\r\nline 2\r\n")
    with pytest.raises(installer.HermesDirtyCheckoutError, match=REFUSAL_TEXT) as caught:
        installer._ensure_clean_checkout(fixture.checkout)
    assert notes_path in str(caught.value)
