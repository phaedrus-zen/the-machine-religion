"""Persistent update-job snapshot tests.

Mirrors ``ollama_admin::load_persisted_update_state`` recovery
semantics: a snapshot stuck in ``running`` from a previous crash
must flip to ``failed`` on hydrate so the UI never claims an
update is in flight forever.
"""

from __future__ import annotations

import json

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


def test_hydrate_flips_stuck_running_snapshot_to_failed(tmp_path):
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
    assert final.status == "failed"
    assert final.phase == "error"
    assert "restarted while update was running" in (final.error or "")
    assert final.finished_at is not None
