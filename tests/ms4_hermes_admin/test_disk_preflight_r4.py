"""R4: free-space preflight, timeout classification, and process-tree bounding.

Regression cover for the 2026-06-30 managed-update failure (job
``1c178865-59ff-464e-9360-ec98da220b3d``, 0.16.0 -> 0.17.0), which surfaced only
as::

    Command timed out after 600s: git -C ...\\hermes-agent fetch --tags --prune origin

Forensics on the operator checkout showed the fetch was NOT hung: it wrote
``.git/objects/pack/tmp_pack_NOKclV`` (78.6 MB) across 630 s -- ~128 KB/s --
against a volume with 249 MB free, and was killed mid-write by the wall-clock
budget. A second orphaned ``tmp_pack`` (47.5 MB) from an earlier attempt the
same morning was still sitting there consuming the little space that remained.
The job also ran 635.6 s against a 600 s budget, because ``subprocess.run``
kills only the direct child while ``git``'s grandchildren keep the captured
pipes open.

So this file proves three things the updater previously did not do:

1. measure free space and refuse *fast*, with the actual numbers and a remedy,
   rather than committing to a fetch that cannot fit;
2. report a timeout as a TIMEOUT (a distinct exception type) rather than as an
   indistinguishable command failure;
3. actually bound the wall clock, by killing the whole process tree and
   draining under a second short bound.

Everything is hermetic: real git for the repository fixtures, no network, and
free space is injected rather than depending on the host's actual volume.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from machine_spirit_4.hermes_admin import installer, state, versioning


MIB = 1024**2
GIB = 1024**3

#: The exact free-space figure on C: when job 1c178865 timed out.
OBSERVED_FAILURE_FREE_BYTES = 249 * MIB
#: The exact hermes-agent packed size at the time of the failure.
OBSERVED_REPO_PACK_BYTES = 525 * MIB


# --------------------------------------------------------------------------- #
# Fixtures (real git; mirrors test_installer_r1.py conventions).
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
        pytest.fail(f"git fixture failed: {result.args!r}\n{result.stdout}\n{result.stderr}")
    return result


def _init_repo(path: Path, *, bare: bool = False) -> None:
    args = ["init"]
    if bare:
        args.append("--bare")
    args.append(str(path))
    _git(None, *args)
    if not bare:
        _git(path, "config", "user.name", "Hermes R4 Test")
        _git(path, "config", "user.email", "hermes-r4@example.invalid")
        _git(path, "config", "commit.gpgsign", "false")


def _make_remote_with_tag(tmp_path: Path, tag: str) -> tuple[Path, str]:
    remote = tmp_path / "release remote.git"
    seed = tmp_path / "release seed"
    _init_repo(remote, bare=True)
    _init_repo(seed)
    _git(seed, "remote", "add", "origin", str(remote))
    (seed / "release.txt").write_text("release\n", encoding="utf-8")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "release")
    commit = _git(seed, "rev-parse", "HEAD").stdout.strip()
    _git(seed, "update-ref", f"refs/tags/{tag}", commit)
    _git(seed, "push", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
    return remote, commit


def _make_target_checkout(tmp_path: Path, remote: Path) -> Path:
    target = tmp_path / "target checkout"
    _init_repo(target)
    (target / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(target, "add", "tracked.txt")
    _git(target, "commit", "-m", "baseline")
    _git(target, "remote", "add", "origin", str(remote))
    return target


def _make_plugin_src(tmp_path: Path) -> Path:
    src = tmp_path / "plugin_src"
    src.mkdir(parents=True)
    (src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")
    return src


def _use_editable_versioning(monkeypatch, directory: Path, *, vetted_origin: str) -> None:
    versioning._clear_caches_for_test()
    monkeypatch.setattr(
        installer, "_verify_release_origin_preflight", lambda **_kwargs: vetted_origin
    )
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {"mode": "editable", "version": "0.16.0", "directory": directory},
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        lambda force_refresh=False: versioning.CachedLatest(
            version="0.17.0",
            published_at="2026-06-30T00:00:00Z",
            tag_name="v0.17.0",
            html_url="https://example/release",
            fetched_at_unix=0.0,
        ),
    )
    monkeypatch.setattr(installer.versioning, "recent_releases", lambda force_refresh=False: [])
    monkeypatch.setattr(installer.versioning, "resolve_git_tag", lambda target: "v0.17.0")


def _mock_pip_and_validate(monkeypatch) -> None:
    monkeypatch.setattr(installer, "_pip_install_editable", lambda *a, **k: None)
    monkeypatch.setattr(installer, "_reinstall_editable", lambda *a, **k: None)
    monkeypatch.setattr(
        installer,
        "_validate",
        lambda python, *, expected_version, editable_directory=None, expected_commit=None, **_k: expected_version,
    )


def _seed_pack_dir(directory: Path, *, packs: dict[str, int], tmp_packs: dict[str, int]) -> None:
    pack_dir = directory / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    for name, size in {**packs, **tmp_packs}.items():
        (pack_dir / name).write_bytes(b"\0" * size)


# --------------------------------------------------------------------------- #
# The free-space budget itself (pure units).
# --------------------------------------------------------------------------- #
def test_required_free_bytes_never_drops_below_the_absolute_floor():
    assert installer._required_free_bytes(0, environ={}) == installer.ABSOLUTE_MIN_FREE_BYTES
    assert installer._required_free_bytes(10 * MIB, environ={}) == installer.ABSOLUTE_MIN_FREE_BYTES


def test_required_free_bytes_scales_with_repository_pack_size():
    """A 525 MiB repo (hermes-agent) must demand more than a trivial one."""
    huge = installer._required_free_bytes(8 * GIB, environ={})
    assert huge == 4 * GIB
    assert huge > installer._required_free_bytes(OBSERVED_REPO_PACK_BYTES, environ={})


def test_observed_failure_conditions_are_refused_by_the_budget():
    """249 MB free against a 525 MiB repo is below requirement -- by construction."""
    required = installer._required_free_bytes(OBSERVED_REPO_PACK_BYTES, environ={})
    assert OBSERVED_FAILURE_FREE_BYTES < required


def test_fresh_empty_checkout_is_budgeted_for_the_whole_history():
    """The F2 migration fetches a full history into an EMPTY managed checkout.

    Its own pack bytes are 0, so without the source-checkout reference the
    guard would demand only the floor while ~2x the repository actually
    arrives (download pack + indexed pack side by side).
    """
    repo = 1024 * MIB
    fresh = installer._required_free_bytes(0, reference_pack_bytes=repo, environ={})
    incremental = installer._required_free_bytes(repo, environ={})

    assert fresh == 2 * repo
    assert fresh > incremental
    # Without the reference it would collapse to the floor -- the bug this guards.
    assert installer._required_free_bytes(0, environ={}) == installer.ABSOLUTE_MIN_FREE_BYTES


def test_fresh_regime_still_respects_the_floor_for_a_tiny_source():
    assert (
        installer._required_free_bytes(0, reference_pack_bytes=1024, environ={})
        == installer.ABSOLUTE_MIN_FREE_BYTES
    )


def test_override_wins_over_both_regimes():
    environ = {installer.MIN_FREE_BYTES_ENV: str(7 * MIB)}
    assert (
        installer._required_free_bytes(0, reference_pack_bytes=99 * GIB, environ=environ)
        == 7 * MIB
    )


@pytest.mark.parametrize("raw", ["", "   "])
def test_min_free_override_absent_is_none(raw):
    assert installer._configured_min_free_bytes({installer.MIN_FREE_BYTES_ENV: raw}) is None


def test_min_free_override_is_honored():
    environ = {installer.MIN_FREE_BYTES_ENV: str(64 * MIB)}
    assert installer._required_free_bytes(8 * GIB, environ=environ) == 64 * MIB


@pytest.mark.parametrize("raw", ["lots", "1GB", "0x10", "12.5", "-1"])
def test_min_free_override_fails_closed_on_junk(raw):
    """A typo must not silently disable the guard."""
    with pytest.raises(installer.HermesUpgradeError):
        installer._configured_min_free_bytes({installer.MIN_FREE_BYTES_ENV: raw})


# --------------------------------------------------------------------------- #
# Pack accounting, including the orphaned partial fetches the bug leaves behind.
# --------------------------------------------------------------------------- #
def test_pack_dir_usage_separates_real_packs_from_partial_fetch_orphans(tmp_path):
    repo = tmp_path / "repo"
    _seed_pack_dir(
        repo,
        packs={"pack-aaa.pack": 4096, "pack-bbb.pack": 2048},
        tmp_packs={"tmp_pack_NOKclV": 512, "tmp_pack_aGWHbW": 256},
    )
    (repo / ".git" / "objects" / "pack" / "pack-aaa.idx").write_bytes(b"\0" * 99)

    pack_bytes, orphan_bytes, orphans = installer._pack_dir_usage(repo)

    assert pack_bytes == 4096 + 2048  # .idx is not counted as pack payload
    assert orphan_bytes == 512 + 256
    assert orphans == ["tmp_pack_NOKclV", "tmp_pack_aGWHbW"]


def test_pack_dir_usage_is_quiet_on_a_non_repository(tmp_path):
    assert installer._pack_dir_usage(tmp_path / "nope") == (0, 0, [])


# --------------------------------------------------------------------------- #
# The preflight decision.
# --------------------------------------------------------------------------- #
def test_preflight_passes_and_reports_when_space_is_ample(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "_free_bytes", lambda _p: 400 * GIB)
    measured = installer._preflight_disk_space(tmp_path, environ={}, record=False)
    assert measured["free_bytes"] == 400 * GIB
    assert measured["required_bytes"] == installer.ABSOLUTE_MIN_FREE_BYTES


def test_preflight_refuses_the_exact_observed_failure_conditions(tmp_path, monkeypatch):
    repo = tmp_path / "hermes-agent"
    _seed_pack_dir(repo, packs={"pack-aaa.pack": 1024}, tmp_packs={})
    monkeypatch.setattr(installer, "_free_bytes", lambda _p: OBSERVED_FAILURE_FREE_BYTES)
    monkeypatch.setattr(
        installer, "_pack_dir_usage", lambda _d: (OBSERVED_REPO_PACK_BYTES, 0, [])
    )

    with pytest.raises(installer.HermesInsufficientDiskError) as exc_info:
        installer._preflight_disk_space(repo, environ={}, record=False)

    message = str(exc_info.value)
    # Actionable: says what it saw, what it needs, and the override knob.
    assert "249.0 MiB free" in message
    assert "1.00 GiB required" in message
    assert installer.MIN_FREE_BYTES_ENV in message
    assert str(repo) in message


def test_preflight_refusal_points_at_reclaimable_partial_fetch_garbage(tmp_path, monkeypatch):
    """The orphans a previous timeout left behind are the operator's easiest win."""
    repo = tmp_path / "hermes-agent"
    monkeypatch.setattr(installer, "_free_bytes", lambda _p: 200 * MIB)
    monkeypatch.setattr(
        installer,
        "_pack_dir_usage",
        lambda _d: (OBSERVED_REPO_PACK_BYTES, 126 * MIB, ["tmp_pack_aGWHbW", "tmp_pack_NOKclV"]),
    )

    with pytest.raises(installer.HermesInsufficientDiskError) as exc_info:
        installer._preflight_disk_space(repo, environ={}, record=False)

    message = str(exc_info.value)
    assert "gc --prune=now" in message
    assert "tmp_pack_NOKclV" in message
    assert "126.0 MiB" in message


def test_preflight_surfaces_an_unmeasurable_volume_rather_than_assuming_ok(tmp_path, monkeypatch):
    def boom(_path):
        raise OSError(5, "device not ready")

    monkeypatch.setattr(installer, "_free_bytes", boom)
    with pytest.raises(installer.HermesUpgradeError, match="could not measure free space"):
        installer._preflight_disk_space(tmp_path, environ={}, record=False)


# --------------------------------------------------------------------------- #
# End-to-end: the update job must refuse BEFORE the fetch, and finish fast.
# --------------------------------------------------------------------------- #
def test_update_job_refuses_before_any_fetch_when_the_volume_is_starved(
    tmp_path, monkeypatch
):
    """The regression: 249 MB free must fail in milliseconds, not after 600 s.

    Pre-fix the job would proceed into ``_git_fetch_resolve`` and only surface
    an opaque "Command timed out after 600s" ten minutes later.
    """
    remote, _commit = _make_remote_with_tag(tmp_path, "v0.17.0")
    target = _make_target_checkout(tmp_path, remote)
    plugin_src = _make_plugin_src(tmp_path)
    _use_editable_versioning(monkeypatch, target, vetted_origin=str(remote))
    _mock_pip_and_validate(monkeypatch)

    fetches: list[object] = []
    monkeypatch.setattr(
        installer,
        "_git_fetch_resolve",
        lambda *a, **k: fetches.append(a) or "0" * 40,
    )
    monkeypatch.setattr(installer, "_free_bytes", lambda _p: OBSERVED_FAILURE_FREE_BYTES)
    monkeypatch.setattr(
        installer, "_pack_dir_usage", lambda _d: (OBSERVED_REPO_PACK_BYTES, 0, [])
    )

    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.16.0", to_version="0.17.0", install_mode="editable")

    started = time.monotonic()
    with pytest.raises(installer.HermesInsufficientDiskError):
        installer.run_update_job(
            target_version="0.17.0",
            plugin_src=plugin_src,
            venv_python=tmp_path / "py",
            vetted_origin=str(remote),
        )
    elapsed = time.monotonic() - started

    assert fetches == [], "disk preflight must gate the network step, not follow it"
    assert elapsed < 30, f"preflight must fail fast, took {elapsed:.1f}s"

    # And the job is terminal + durable, not wedged in 'running'.
    snapshot = state.last_update(refresh=True)
    assert snapshot is not None
    assert snapshot.status == "failed"
    assert "insufficient free space" in (snapshot.error or "")
    assert installer._local_update_snapshot() is None


def test_update_job_still_succeeds_when_space_is_ample(tmp_path, monkeypatch):
    """No-false-positive direction: the guard must not block a healthy update."""
    remote, commit = _make_remote_with_tag(tmp_path, "v0.17.0")
    target = _make_target_checkout(tmp_path, remote)
    plugin_src = _make_plugin_src(tmp_path)
    _use_editable_versioning(monkeypatch, target, vetted_origin=str(remote))
    _mock_pip_and_validate(monkeypatch)
    monkeypatch.setattr(installer, "_free_bytes", lambda _p: 400 * GIB)

    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.16.0", to_version="0.17.0", install_mode="editable")

    result = installer.run_update_job(
        target_version="0.17.0",
        plugin_src=plugin_src,
        venv_python=tmp_path / "py",
        vetted_origin=str(remote),
    )

    assert result["to_version"] == "0.17.0"
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == commit
    assert state.last_update().status == "success"


# --------------------------------------------------------------------------- #
# 'Timed out' is not 'failed'.
# --------------------------------------------------------------------------- #
def test_timeout_raises_a_distinct_type_and_says_it_timed_out(tmp_path, monkeypatch):
    def fake_bounded(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, 600)

    monkeypatch.setattr(installer, "_run_bounded", fake_bounded)
    monkeypatch.setattr(installer, "_free_bytes", lambda _p: 400 * GIB)
    state._reset_for_tests(tmp_path / "snap.json")

    with pytest.raises(installer.HermesUpgradeTimeoutError) as exc_info:
        installer._run(["git", "-C", str(tmp_path), "fetch", "origin"], timeout=600)

    message = str(exc_info.value)
    assert "TIMED OUT (did not fail)" in message
    assert "Re-running the update is safe" in message
    # Still a HermesUpgradeError, so every existing handler keeps working.
    assert isinstance(exc_info.value, installer.HermesUpgradeError)


def test_timeout_on_a_starved_volume_names_disk_starvation_as_the_cause(
    tmp_path, monkeypatch
):
    def fake_bounded(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, 600)

    monkeypatch.setattr(installer, "_run_bounded", fake_bounded)
    monkeypatch.setattr(installer, "_free_bytes", lambda _p: OBSERVED_FAILURE_FREE_BYTES)
    monkeypatch.setattr(
        installer, "_pack_dir_usage", lambda _d: (OBSERVED_REPO_PACK_BYTES, 0, [])
    )
    state._reset_for_tests(tmp_path / "snap.json")

    with pytest.raises(installer.HermesUpgradeTimeoutError) as exc_info:
        installer._run(["git", "-C", str(tmp_path), "fetch", "origin"], timeout=600)

    message = str(exc_info.value)
    assert "starved for disk, not hung" in message
    assert "249.0 MiB" in message


def test_git_directory_hint_recovers_the_dash_c_operand():
    assert installer._git_directory_hint(
        ["git", "-C", r"C:\repo", "fetch", "origin"]
    ) == Path(r"C:\repo")
    assert installer._git_directory_hint(["git", "status"]) is None
    assert installer._git_directory_hint(["pip", "-C", "x", "install"]) is None
    # Trailing "-C" with no operand must not raise.
    assert installer._git_directory_hint(["git", "fetch", "-C"]) is None


# --------------------------------------------------------------------------- #
# The wall-clock budget must actually bound the call.
# --------------------------------------------------------------------------- #
_GRANDCHILD_SLEEP_SECONDS = 45
_TREE_KILL_BUDGET_SECONDS = 25


def _write_pipe_holding_script(tmp_path: Path) -> Path:
    """A child that leaves a grandchild holding the inherited stdout/stderr pipes."""
    script = tmp_path / "hold_pipes.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep("
        f"{_GRANDCHILD_SLEEP_SECONDS})'])\n"
        f"time.sleep({_GRANDCHILD_SLEEP_SECONDS})\n",
        encoding="utf-8",
    )
    return script


def test_run_bounded_timeout_is_not_defeated_by_pipe_holding_grandchildren(tmp_path):
    """Job 1c178865 overran its 600 s budget by ~35 s for exactly this reason.

    ``subprocess.run(timeout=...)`` kills only the direct child, then blocks in
    ``communicate()`` until every inheritor of the captured pipes exits. With a
    45 s grandchild that is a 45 s overrun; the tree kill must cap it.
    """
    script = _write_pipe_holding_script(tmp_path)

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        installer._run_bounded([sys.executable, str(script)], timeout=3)
    elapsed = time.monotonic() - started

    assert elapsed < _TREE_KILL_BUDGET_SECONDS, (
        f"timeout did not bound the call: {elapsed:.1f}s elapsed for a 3s budget "
        f"against a {_GRANDCHILD_SLEEP_SECONDS}s grandchild"
    )


def test_run_converts_the_bounded_timeout_and_stays_within_budget(tmp_path):
    script = _write_pipe_holding_script(tmp_path)
    state._reset_for_tests(tmp_path / "snap.json")

    started = time.monotonic()
    with pytest.raises(installer.HermesUpgradeTimeoutError):
        installer._run([sys.executable, str(script)], timeout=3)
    elapsed = time.monotonic() - started

    assert elapsed < _TREE_KILL_BUDGET_SECONDS


def test_run_bounded_returns_normal_results_unchanged(tmp_path):
    result = installer._run_bounded(
        [sys.executable, "-c", "import sys; print('ok'); sys.exit(3)"],
        timeout=60,
    )
    assert result.returncode == 3
    assert result.stdout.strip() == "ok"
