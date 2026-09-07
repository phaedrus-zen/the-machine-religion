"""Fail-first durable Hermes current/latest/state reconciliation.

Live 9180 shape (2026-08-26): current 0.20.0, latest 0.19.0 (PyPI fallback),
install_mode editable, last_update failed targeting 0.20.1. Installed >=
latest must become a clean current/newer no-op that supersedes failure
presentation without deleting the audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from machine_spirit_4.hermes_admin import installer, state, versioning


def _failed_job(
    tmp_path: Path,
    *,
    from_version: str = "0.20.0",
    to_version: str = "0.20.1",
    error: str = "F4 provenance verification failed: tag 'v2026.8.13' did not verify",
):
    path = tmp_path / "hermes_update_state.json"
    state._reset_for_tests(path)
    snap = state.start_job(
        from_version=from_version,
        to_version=to_version,
        install_mode="editable",
        request_user="ms4-gateway",
    )
    state.append_progress("audit breadcrumb: origin verified")
    state.finalize_job(error=error)
    return path, snap


def _seed_cached_latest(version: str, *, fetched_at_unix: float = 0.0) -> None:
    """Local cache only. ``fetched_at_unix=0`` is expired so TTL cannot hide HTTP."""
    versioning._clear_caches_for_test()
    with versioning._CACHE.lock:
        versioning._CACHE.latest = versioning.CachedLatest(
            version=version,
            published_at="",
            tag_name=f"v{version}",
            html_url=f"https://pypi.org/project/hermes-agent/{version}/",
            fetched_at_unix=fetched_at_unix,
        )


def _forbid_http_get_json(http_calls: list[str], *, label: str):
    """Monkeypatch target: record URL then fail. Do not raise inside the real
    ``_http_get_json`` (it swallows ``Exception``); replace the function."""

    def forbidden(url, timeout=8):
        http_calls.append(url)
        pytest.fail(f"{label} called _http_get_json: {url}")

    return forbidden


def _patch_install_and_latest(monkeypatch, *, current: str | None, latest: str | None):
    monkeypatch.setattr(
        versioning,
        "install_mode",
        lambda: {
            "mode": "editable" if current else "missing",
            "version": current,
            "directory": None,
            "direct_url": None,
        },
    )
    if latest is None:
        monkeypatch.setattr(versioning, "latest_version", lambda force_refresh=False: None)
        return

    cached = versioning.CachedLatest(
        version=latest,
        published_at="",
        tag_name=f"v{latest}",
        html_url=f"https://pypi.org/project/hermes-agent/{latest}/",
        fetched_at_unix=0.0,
    )
    monkeypatch.setattr(
        versioning,
        "latest_version",
        lambda force_refresh=False: cached,
    )
    monkeypatch.setattr(
        versioning,
        "official_tag_signature_state",
        lambda *_args, **_kwargs: "unknown",
    )


def test_installed_relation_orders_semver_and_fails_closed():
    assert versioning.installed_relation("0.20.0", "0.19.0") == "newer"
    assert versioning.installed_relation("0.20.0", "0.20.0") == "current"
    assert versioning.installed_relation("0.19.0", "0.20.0") == "older"
    assert versioning.installed_relation("not-a-version", "0.20.0") == "unknown"
    assert versioning.installed_relation("0.20.0", None) == "unknown"
    assert versioning.installed_relation(None, "0.20.0") == "unknown"
    assert versioning.installed_relation("0.20.0", "latest") == "unknown"


def test_version_info_newer_than_latest_supersedes_stale_failure_keeps_audit(
    tmp_path, monkeypatch
):
    path, seeded = _failed_job(tmp_path)
    _patch_install_and_latest(monkeypatch, current="0.20.0", latest="0.19.0")
    monkeypatch.setattr(versioning, "git_describe", lambda *_args, **_kwargs: "v2026.8.3")

    info = versioning.version_info(force_refresh_latest=True)
    last = info["last_update"]

    assert info["current"] == "0.20.0"
    assert info["latest"] == "0.19.0"
    assert info["installed_relation"] == "newer"
    assert info["operator_state"] == "newer"
    assert info["update_available"] is False
    assert last is not None
    assert last["status"] == "success"
    assert last["phase"] == "done"
    assert last["error"] is None
    assert last["job_id"] == seeded.job_id
    assert last["to_version"] == "0.20.0"
    progress_notes = [entry.get("note") or "" for entry in last["progress"]]
    assert any("audit breadcrumb: origin verified" in note for note in progress_notes)
    assert any("did not verify" in note for note in progress_notes)
    assert last.get("superseded_failure")
    assert last["superseded_failure"]["status"] == "failed"
    assert last["superseded_failure"]["to_version"] == "0.20.1"
    durable = json.loads(path.read_text(encoding="utf-8"))
    assert durable["status"] == "success"
    assert durable["job_id"] == seeded.job_id
    assert any("did not verify" in (e.get("note") or "") for e in durable["progress"])


def test_version_info_current_equals_latest_supersedes_stale_failure(
    tmp_path, monkeypatch
):
    _failed_job(tmp_path, from_version="0.20.0", to_version="0.20.0")
    _patch_install_and_latest(monkeypatch, current="0.20.0", latest="0.20.0")
    monkeypatch.setattr(
        versioning,
        "official_tag_signature_state",
        lambda *_args, **_kwargs: "signed",
    )
    info = versioning.version_info(force_refresh_latest=True)
    assert info["installed_relation"] == "current"
    assert info["operator_state"] == "current"
    assert info["last_update"]["status"] == "success"
    assert info["last_update"]["progress"], "audit trail must remain"


def test_version_info_older_than_latest_keeps_failed_actionable(tmp_path, monkeypatch):
    path, seeded = _failed_job(tmp_path)
    _patch_install_and_latest(monkeypatch, current="0.19.0", latest="0.20.1")
    monkeypatch.setattr(
        versioning,
        "official_tag_signature_state",
        lambda *_args, **_kwargs: "signed",
    )
    info = versioning.version_info(force_refresh_latest=True)
    assert info["installed_relation"] == "older"
    assert info["update_available"] is True
    assert info["operator_state"] == "failed"
    last = info["last_update"]
    assert last["status"] == "failed"
    assert last["job_id"] == seeded.job_id
    assert last["error"]
    durable = json.loads(path.read_text(encoding="utf-8"))
    assert durable["status"] == "failed"


def test_compute_operator_state_unsigned_block_outranks_stale_failed():
    assert (
        versioning.compute_operator_state(
            update_in_progress=False,
            last_status="failed",
            relation="older",
            update_available_flag=False,
            blocked_reason="official_tag_unsigned",
        )
        == "blocked"
    )
    assert (
        versioning.compute_operator_state(
            update_in_progress=False,
            last_status="failed",
            relation="older",
            update_available_flag=True,
            blocked_reason=None,
        )
        == "failed"
    )
    assert (
        versioning.compute_operator_state(
            update_in_progress=False,
            last_status="failed",
            relation="newer",
            update_available_flag=False,
            blocked_reason="official_tag_unsigned",
        )
        == "newer"
    )


def test_version_info_older_unsigned_latest_blocks_primary_keeps_failed_audit(
    tmp_path, monkeypatch
):
    """Live shape: installed 0.20.0, latest 0.20.5 unsigned, prior failed job.

    Blocked unsigned is the operator headline. Durable failed history is
    audit only. Signed-fallback discovery finds no candidate; no reconcile to
    success and no job starts.
    """
    monkeypatch.setenv(versioning.RELEASE_SIGNATURE_POLICY_ENV, "require_signed")
    versioning._clear_caches_for_test()
    path, seeded = _failed_job(tmp_path)
    _patch_install_and_latest(monkeypatch, current="0.20.0", latest="0.20.5")
    monkeypatch.setattr(
        versioning,
        "official_tag_signature_state",
        lambda *_args, **_kwargs: "unsigned",
    )
    monkeypatch.setattr(versioning, "git_describe", lambda *_args, **_kwargs: None)
    http_calls: list[str] = []

    def no_signed_releases(url, timeout=8):
        http_calls.append(url)
        assert url == versioning.RECENT_RELEASES_URL
        return []

    monkeypatch.setattr(versioning, "_http_get_json", no_signed_releases)
    monkeypatch.setattr(installer, "trigger_update", lambda *a, **k: pytest.fail("trigger_update"))
    monkeypatch.setattr(
        state,
        "reconcile_stale_failure_for_current_or_newer",
        lambda **_kwargs: pytest.fail("must not supersede failed audit when older+unsigned"),
    )

    info = versioning.version_info(force_refresh_latest=False)
    last = info["last_update"]

    assert http_calls == [versioning.RECENT_RELEASES_URL]
    assert info["current"] == "0.20.0"
    assert info["latest"] == "0.20.5"
    assert info["installed_relation"] == "older"
    assert info["latest_signature_state"] == "unsigned"
    assert info["update_available"] is False
    assert info["update_blocked_reason"] == "official_tag_unsigned"
    assert info["operator_state"] == "blocked"
    assert last is not None
    assert last["status"] == "failed"
    assert last["job_id"] == seeded.job_id
    assert last["to_version"] == "0.20.1"
    assert last["error"]
    assert last.get("superseded_failure") in (None, {})
    durable = json.loads(path.read_text(encoding="utf-8"))
    assert durable["status"] == "failed"
    assert durable["job_id"] == seeded.job_id
    assert any("did not verify" in (e.get("note") or "") for e in durable["progress"])


def test_version_info_malformed_or_missing_versions_fail_closed(tmp_path, monkeypatch):
    _failed_job(tmp_path)
    _patch_install_and_latest(monkeypatch, current="not-a-version", latest="0.19.0")
    info = versioning.version_info(force_refresh_latest=True)
    assert info["installed_relation"] == "unknown"
    assert info["operator_state"] == "failed"
    assert info["last_update"]["status"] == "failed"

    _failed_job(tmp_path)
    _patch_install_and_latest(monkeypatch, current="0.20.0", latest=None)
    info = versioning.version_info(force_refresh_latest=True)
    assert info["installed_relation"] == "unknown"
    assert info["last_update"]["status"] == "failed"


def test_reconcile_does_not_touch_running_or_success_jobs(tmp_path, monkeypatch):
    path = tmp_path / "hermes_update_state.json"
    state._reset_for_tests(path)
    running = state.start_job(
        from_version="0.20.0", to_version="0.20.1", install_mode="editable"
    )
    _patch_install_and_latest(monkeypatch, current="0.20.0", latest="0.19.0")
    info = versioning.version_info(force_refresh_latest=True)
    assert info["last_update"]["status"] == "running"
    assert info["last_update"]["job_id"] == running.job_id
    assert info["operator_state"] == "running"

    state.finalize_job(to_version="0.20.0")
    info = versioning.version_info(force_refresh_latest=True)
    assert info["last_update"]["status"] == "success"
    assert info["last_update"]["to_version"] == "0.20.0"


def test_verified_success_supersedes_prior_failure_on_finalize(tmp_path):
    _failed_job(tmp_path)
    state.start_job(
        from_version="0.20.0", to_version="0.20.2", install_mode="editable"
    )
    state.finalize_job(to_version="0.20.2")
    final = state.last_update()
    assert final is not None
    assert final.status == "success"
    assert final.to_version == "0.20.2"


def test_trigger_update_current_newer_is_noop_without_product_effects(
    tmp_path, monkeypatch
):
    path, seeded = _failed_job(tmp_path)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin"
    plugin_src.mkdir()
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir()
    venv_python.write_bytes(b"venv\n")

    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": checkout,
        },
    )
    _seed_cached_latest("0.19.0")
    http_calls: list[str] = []
    monkeypatch.setattr(
        versioning, "_http_get_json", _forbid_http_get_json(http_calls, label="current/newer no-op")
    )

    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"current/newer no-op reached forbidden effect: {name}")

        return fail

    monkeypatch.setattr(installer, "_require_valid_git_bash_candidate", forbidden("git-bash"))
    monkeypatch.setattr(installer, "_verify_release_origin_preflight", forbidden("origin"))
    monkeypatch.setattr(installer, "_verify_release_provenance", forbidden("provenance"))
    monkeypatch.setattr(installer, "_git_fetch_resolve", forbidden("git fetch"))
    monkeypatch.setattr(installer, "_git_detach_checkout", forbidden("checkout"))
    monkeypatch.setattr(installer, "_pip_install_editable", forbidden("pip"))
    monkeypatch.setattr(installer, "_validate", forbidden("validate"))
    monkeypatch.setattr(installer.threading, "Thread", forbidden("worker"))
    monkeypatch.setattr(installer.state, "start_job", forbidden("start_job"))

    result = installer.trigger_update(
        target_version=None,
        request_user="ui-update-button",
        plugin_src=plugin_src,
        venv_python=venv_python,
    )
    assert result.status == "success"
    assert result.job_id == seeded.job_id
    assert result.to_version == "0.20.0"
    assert http_calls == []
    durable = json.loads(path.read_text(encoding="utf-8"))
    assert durable["status"] == "success"
    assert any("did not verify" in (e.get("note") or "") for e in durable["progress"])


@pytest.mark.parametrize(
    ("cached_latest", "relation"),
    [("0.20.0", "current"), ("0.19.0", "newer")],
    ids=["current", "installed-newer"],
)
def test_trigger_cached_current_and_newer_noop_makes_zero_http_calls(
    tmp_path, monkeypatch, cached_latest, relation
):
    """Expired local cache is enough to prove no-op; _http_get_json must not run."""
    path, seeded = _failed_job(tmp_path)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin"
    plugin_src.mkdir()
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir()
    venv_python.write_bytes(b"venv\n")
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": checkout,
        },
    )
    _seed_cached_latest(cached_latest)
    http_calls: list[str] = []
    monkeypatch.setattr(
        versioning,
        "_http_get_json",
        _forbid_http_get_json(http_calls, label=f"{relation} no-op"),
    )

    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"{relation} no-op reached forbidden effect: {name}")

        return fail

    monkeypatch.setattr(installer, "_require_valid_git_bash_candidate", forbidden("git-bash"))
    monkeypatch.setattr(installer.threading, "Thread", forbidden("worker"))
    monkeypatch.setattr(installer.state, "start_job", forbidden("start_job"))
    monkeypatch.setattr(installer, "_git_fetch_resolve", forbidden("git fetch"))

    result = installer.trigger_update(
        target_version=None,
        plugin_src=plugin_src,
        venv_python=venv_python,
    )
    assert result.status == "success"
    assert result.job_id == seeded.job_id
    assert result.to_version == "0.20.0"
    assert http_calls == []
    durable = json.loads(path.read_text(encoding="utf-8"))
    assert durable["status"] == "success"
    assert any(relation in (e.get("note") or "") for e in durable["progress"])


def test_trigger_update_explicit_current_target_is_noop_without_product_effects(
    tmp_path, monkeypatch
):
    """Pin dialog default: POST {target_version: installed} while latest is newer unsigned.

    Live 9180: installed 0.20.0, latest signed option 0.20.0, discovered latest
    0.20.6 unsigned, last_update failed targeting 0.20.1. The target=None
    current/newer branch must not be required; an explicit equal target is
    itself a durable zero-effect no-op.
    """
    path, seeded = _failed_job(tmp_path)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin"
    plugin_src.mkdir()
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir()
    venv_python.write_bytes(b"venv\n")
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": checkout,
        },
    )
    _seed_cached_latest("0.20.6")
    http_calls: list[str] = []
    monkeypatch.setattr(
        versioning,
        "_http_get_json",
        _forbid_http_get_json(http_calls, label="explicit-current pin no-op"),
    )

    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"explicit-current pin no-op reached forbidden effect: {name}")

        return fail

    monkeypatch.setattr(installer, "_require_valid_git_bash_candidate", forbidden("git-bash"))
    monkeypatch.setattr(installer, "_verify_release_origin_preflight", forbidden("origin"))
    monkeypatch.setattr(installer, "_verify_release_provenance", forbidden("provenance"))
    monkeypatch.setattr(installer, "_git_fetch_resolve", forbidden("git fetch"))
    monkeypatch.setattr(installer, "_git_detach_checkout", forbidden("checkout"))
    monkeypatch.setattr(installer, "_pip_install_editable", forbidden("pip"))
    monkeypatch.setattr(installer, "_validate", forbidden("validate"))
    monkeypatch.setattr(installer.threading, "Thread", forbidden("worker"))
    monkeypatch.setattr(installer.state, "start_job", forbidden("start_job"))

    result = installer.trigger_update(
        target_version="0.20.0",
        request_user="ui-pin-dialog",
        plugin_src=plugin_src,
        venv_python=venv_python,
    )
    assert result.status == "success"
    assert result.job_id == seeded.job_id
    assert result.to_version == "0.20.0"
    assert result.error is None
    assert http_calls == []
    durable = json.loads(path.read_text(encoding="utf-8"))
    assert durable["status"] == "success"
    assert durable["job_id"] == seeded.job_id
    assert any("did not verify" in (e.get("note") or "") for e in durable["progress"])
    assert durable.get("superseded_failure")
    assert durable["superseded_failure"]["status"] == "failed"
    assert durable["superseded_failure"]["to_version"] == "0.20.1"

    again = installer.trigger_update(
        target_version="0.20.0",
        request_user="ui-pin-dialog",
        plugin_src=plugin_src,
        venv_python=venv_python,
    )
    assert again.status == "success"
    assert again.job_id == seeded.job_id
    assert http_calls == []
    durable_again = json.loads(path.read_text(encoding="utf-8"))
    assert durable_again["status"] == "success"
    assert durable_again.get("superseded_failure")["to_version"] == "0.20.1"


def test_trigger_update_explicit_current_target_without_prior_state_is_durable_noop(
    tmp_path, monkeypatch
):
    path = tmp_path / "hermes_update_state.json"
    state._reset_for_tests(path)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": checkout,
        },
    )

    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"no-state equal-current pin reached forbidden effect: {name}")

        return fail

    monkeypatch.setattr(installer, "_require_valid_git_bash_candidate", forbidden("git-bash"))
    monkeypatch.setattr(installer, "_verify_release_origin_preflight", forbidden("origin"))
    monkeypatch.setattr(installer, "_verify_release_provenance", forbidden("provenance"))
    monkeypatch.setattr(installer, "_git_fetch_resolve", forbidden("git fetch"))
    monkeypatch.setattr(installer, "_git_detach_checkout", forbidden("checkout"))
    monkeypatch.setattr(installer, "_pip_install_editable", forbidden("pip"))
    monkeypatch.setattr(installer, "_validate", forbidden("validate"))
    monkeypatch.setattr(installer.threading, "Thread", forbidden("worker"))

    result = installer.trigger_update(
        target_version="0.20.0",
        request_user="ui-pin-dialog",
    )
    assert result.status == "success"
    assert result.from_version == "0.20.0"
    assert result.to_version == "0.20.0"
    assert result.error is None
    durable = json.loads(path.read_text(encoding="utf-8"))
    assert durable["status"] == "success"
    assert durable["from_version"] == durable["to_version"] == "0.20.0"
    assert durable["request_user"] == "ui-pin-dialog"
    assert any("skipped Git/install" in (entry.get("note") or "") for entry in durable["progress"])


def test_trigger_unknown_cached_latest_validates_git_bash_before_http(
    tmp_path, monkeypatch
):
    """Empty local latest evidence fails closed: Git-Bash before any HTTP."""
    _failed_job(tmp_path)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin"
    plugin_src.mkdir()
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir()
    venv_python.write_bytes(b"venv\n")
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": checkout,
        },
    )
    versioning._clear_caches_for_test()
    reached: list[str] = []
    http_calls: list[str] = []

    def git_bash():
        reached.append("git-bash")
        raise installer.HermesUpgradeError("blocked next operation: git-bash")

    def forbidden_http(url, timeout=8):
        reached.append("http")
        http_calls.append(url)
        pytest.fail(f"empty-cache path called _http_get_json before Git-Bash: {url}")

    monkeypatch.setattr(installer, "_require_valid_git_bash_candidate", git_bash)
    monkeypatch.setattr(versioning, "_http_get_json", forbidden_http)

    with pytest.raises(installer.HermesUpgradeError, match="git-bash"):
        installer.trigger_update(
            target_version=None,
            plugin_src=plugin_src,
            venv_python=venv_python,
        )
    assert reached == ["git-bash"]
    assert http_calls == []


def test_trigger_update_refuses_downgrade_target(tmp_path, monkeypatch):
    _failed_job(tmp_path)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin"
    plugin_src.mkdir()
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir()
    venv_python.write_bytes(b"venv\n")
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": checkout,
        },
    )
    with pytest.raises(installer.HermesUpgradeError, match="downgrade"):
        installer.trigger_update(
            target_version="0.19.0",
            plugin_src=plugin_src,
            venv_python=venv_python,
        )
    assert state.last_update().status == "failed"


def test_initialize_state_reconciles_on_startup(tmp_path, monkeypatch):
    path, seeded = _failed_job(tmp_path)
    monkeypatch.setattr(
        versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": None,
            "direct_url": None,
        },
    )
    monkeypatch.setattr(
        versioning,
        "latest_version",
        lambda force_refresh=False: versioning.CachedLatest(
            version="0.19.0",
            published_at="",
            tag_name="v0.19.0",
            html_url="https://pypi.org/project/hermes-agent/0.19.0/",
            fetched_at_unix=0.0,
        ),
    )
    state.initialize_state(path)
    versioning.reconcile_durable_terminal_state()
    final = state.last_update()
    assert final is not None
    assert final.status == "success"
    assert final.job_id == seeded.job_id
    assert any("did not verify" in (e.get("note") or "") for e in final.progress)
