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
) -> _ProductFixture:
    remote = tmp_path / "release-remote.git"
    seed = tmp_path / "release-seed"
    checkout = tmp_path / "installed-editable"
    _init_repo(seed)

    package_version = seed / "package-version.txt"
    package_version.write_bytes(f"{PRIOR_VERSION}\n".encode())
    _git(seed, "add", "package-version.txt")
    _git(seed, "commit", "-q", "-m", "packaged Hermes 1.0.0")
    prior_commit = _git(seed, "rev-parse", "HEAD")
    _git(seed, "branch", "prior", prior_commit)

    package_version.write_bytes(f"{TARGET_VERSION}\n".encode())
    _git(seed, "add", "package-version.txt")
    _git(seed, "commit", "-q", "-m", "packaged Hermes 2.0.0")
    target_commit = _git(seed, "rev-parse", "HEAD")
    _git(seed, "tag", "-a", TARGET_TAG, "-m", tag_message, target_commit)
    _git(seed, "branch", "-M", "main")
    _git(None, "clone", "-q", "--bare", str(seed), str(remote))
    _assert_isolated_repo(remote, bare=True)
    _git(None, "-C", str(remote), "symbolic-ref", "HEAD", "refs/heads/main")
    _git(None, "clone", "-q", "--no-tags", "--branch", "prior", str(remote), str(checkout))
    _assert_isolated_repo(checkout, bare=False)

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
