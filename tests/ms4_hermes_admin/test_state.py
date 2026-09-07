"""Persistent update-job snapshot tests."""

from __future__ import annotations

import json

import pytest

from machine_spirit_4.hermes_admin import state


def test_full_job_lifecycle(tmp_path):
    state._reset_for_tests(tmp_path / "snap.json")
    assert state.last_update() is None
    assert state.update_in_progress() is False

    snap = state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable", request_user="op")
    assert snap.status == "running"
    assert snap.phase == "queued"
    assert state.update_in_progress() is True

    state.set_phase("checking_out", note="checkout v0.14.0")
    state.append_progress("git checkout v0.14.0 OK")
    state.set_phase("pip_installing")
    state.finalize_job(to_version="0.14.0")

    final = state.last_update()
    assert final is not None
    assert final.status == "success"
    assert final.phase == "done"
    assert final.finished_at is not None
    assert final.to_version == "0.14.0"
    assert any(entry.get("note") == "git checkout v0.14.0 OK" for entry in final.progress)
    assert state.update_in_progress() is False


def test_finalize_job_with_error_marks_failed(tmp_path):
    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    state.set_phase("checking_out")
    state.finalize_job(error="git fetch failed: dns lookup")

    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert final.phase == "error"
    assert final.error == "git fetch failed: dns lookup"
    assert any(entry.get("note") == "git fetch failed: dns lookup" for entry in final.progress)


def test_persisted_snapshot_round_trips(tmp_path):
    path = tmp_path / "snap.json"
    state._reset_for_tests(path)
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="pypi")
    state.set_phase("pip_installing")
    state.finalize_job(to_version="0.14.0")

    state._reset_for_tests(path)
    state.initialize_state(path)
    final = state.last_update()
    assert final is not None
    assert final.status == "success"
    assert final.to_version == "0.14.0"


def test_hydrate_preserves_running_until_lock_owner_is_classified(tmp_path):
    path = tmp_path / "snap.json"
    payload = {
        "job_id": "abc",
        "started_at": "2026-05-18T00:00:00+00:00",
        "finished_at": None,
        "status": "running",
        "phase": "pip_installing",
        "from_version": "0.13.0",
        "to_version": "0.14.0",
        "install_mode": "editable",
        "progress": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    state._reset_for_tests(path)
    state.initialize_state(path)
    final = state.last_update()
    assert final is not None
    assert final.status == "running"
    assert final.phase == "pip_installing"

    state.fail_interrupted_job()
    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert final.phase == "error"
    assert "restarted while update was running" in (final.error or "")
    assert final.finished_at is not None


def test_last_update_refreshes_changed_durable_snapshot(tmp_path):
    path = tmp_path / "snap.json"
    first = {
        "job_id": "first",
        "started_at": "2026-05-18T00:00:00+00:00",
        "finished_at": None,
        "status": "running",
        "phase": "pip_installing",
        "from_version": "0.13.0",
        "to_version": "0.14.0",
        "install_mode": "editable",
        "progress": [],
    }
    second = {
        **first,
        "job_id": "second",
        "finished_at": "2026-05-18T00:01:00+00:00",
        "status": "success",
        "phase": "done",
    }
    path.write_text(json.dumps(first), encoding="utf-8")
    state._reset_for_tests(path)
    state.initialize_state(path)
    path.write_text(json.dumps(second), encoding="utf-8")

    refreshed = state.last_update()
    assert refreshed is not None
    assert refreshed.job_id == "second"
    assert refreshed.status == "success"


def test_start_job_persistence_failure_is_visible_and_not_published(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "snap.json"
    state._reset_for_tests(path)
    monkeypatch.setattr(
        state.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("simulated durable replace failure")
        ),
    )

    with pytest.raises(
        state.StatePersistenceError,
        match="simulated durable replace failure",
    ):
        state.start_job(
            from_version="0.13.0",
            to_version="0.14.0",
            install_mode="editable",
        )

    assert state.last_update(refresh=False) is None
    assert not path.exists()


def test_verifying_origin_is_a_documented_phase():
    assert "verifying_origin" in state.PHASES
