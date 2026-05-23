"""Persistent Hermes update job snapshot.

Mirrors HiveMind ``ollama_admin::UPDATE_JOB`` plus the on-disk
``deps/ollama_update_state.json`` persistence so the dashboard can
show ``"last update finished N min ago"`` even after a gateway
restart, and so a gateway that crashed mid-update flips its dangling
``running`` snapshot to ``failed`` on next startup instead of
claiming an update is in flight forever.

The snapshot file lives under ``machine_spirit_4/runtime/`` so it
ships with the contained MS4 runtime, never inside the Hermes
checkout itself.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MS4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = MS4_ROOT / "runtime" / "hermes_update_state.json"

PHASES = (
    "queued",
    "preflight",
    "fetching_release",
    "fetching_remote",
    "checking_out",
    "syncing_plugin",
    "pip_installing",
    "validating",
    "done",
    "error",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UpdateJobSnapshot:
    job_id: str
    started_at: str
    finished_at: str | None
    status: str  # "running" | "success" | "failed"
    phase: str
    from_version: str | None
    to_version: str | None
    install_mode: str | None
    error: str | None = None
    progress: list[dict[str, Any]] = field(default_factory=list)
    request_user: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UpdateJobSnapshot":
        return cls(
            job_id=str(payload.get("job_id") or ""),
            started_at=str(payload.get("started_at") or ""),
            finished_at=payload.get("finished_at"),
            status=str(payload.get("status") or "running"),
            phase=str(payload.get("phase") or "queued"),
            from_version=payload.get("from_version"),
            to_version=payload.get("to_version"),
            install_mode=payload.get("install_mode"),
            error=payload.get("error"),
            progress=list(payload.get("progress") or []),
            request_user=payload.get("request_user"),
        )


class _StateStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.path = DEFAULT_STATE_PATH
        self.snapshot: UpdateJobSnapshot | None = None
        self.hydrated = False

    def set_path(self, path: Path) -> None:
        with self.lock:
            self.path = path
            self.hydrated = False

    def hydrate(self) -> None:
        with self.lock:
            if self.hydrated:
                return
            self.hydrated = True
            path = self.path
            if not path.exists():
                return
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return
            try:
                snap = UpdateJobSnapshot.from_dict(payload if isinstance(payload, dict) else {})
            except Exception:
                return
            if snap.status == "running":
                snap.status = "failed"
                snap.phase = "error"
                snap.error = (
                    "Gateway restarted while update was running; install may have completed but "
                    "the bookkeeping was lost. Re-check version and re-trigger if needed."
                )
                snap.finished_at = _now_iso()
            self.snapshot = snap

    def persist_locked(self) -> None:
        snap = self.snapshot
        if snap is None:
            return
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="hermes_update_", suffix=".json", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(snap.to_dict(), handle, indent=2)
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


_STORE = _StateStore()


def initialize_state(path: Path | None = None) -> None:
    """Hydrate the on-disk snapshot at gateway/MCP startup."""
    if path is not None:
        _STORE.set_path(path)
    _STORE.hydrate()


def last_update() -> UpdateJobSnapshot | None:
    _STORE.hydrate()
    with _STORE.lock:
        return _STORE.snapshot


def update_in_progress() -> bool:
    snap = last_update()
    return bool(snap and snap.status == "running")


def start_job(
    *,
    from_version: str | None,
    to_version: str | None,
    install_mode: str | None,
    request_user: str | None = None,
) -> UpdateJobSnapshot:
    _STORE.hydrate()
    snap = UpdateJobSnapshot(
        job_id=str(uuid.uuid4()),
        started_at=_now_iso(),
        finished_at=None,
        status="running",
        phase="queued",
        from_version=from_version,
        to_version=to_version,
        install_mode=install_mode,
        error=None,
        progress=[{"phase": "queued", "ts": _now_iso(), "note": "Update job accepted"}],
        request_user=request_user,
    )
    with _STORE.lock:
        _STORE.snapshot = snap
        _STORE.persist_locked()
    return snap


def set_phase(phase: str, *, note: str | None = None) -> None:
    with _STORE.lock:
        snap = _STORE.snapshot
        if snap is None:
            return
        snap.phase = phase
        entry: dict[str, Any] = {"phase": phase, "ts": _now_iso()}
        if note:
            entry["note"] = note
        snap.progress.append(entry)
        _STORE.persist_locked()


def append_progress(note: str) -> None:
    with _STORE.lock:
        snap = _STORE.snapshot
        if snap is None:
            return
        snap.progress.append({"phase": snap.phase, "ts": _now_iso(), "note": note})
        _STORE.persist_locked()


def finalize_job(*, error: str | None = None, to_version: str | None = None) -> None:
    with _STORE.lock:
        snap = _STORE.snapshot
        if snap is None:
            return
        snap.finished_at = _now_iso()
        if to_version:
            snap.to_version = to_version
        if error:
            snap.status = "failed"
            snap.phase = "error"
            snap.error = error
            snap.progress.append({"phase": "error", "ts": snap.finished_at, "note": error})
        else:
            snap.status = "success"
            snap.phase = "done"
            snap.progress.append({"phase": "done", "ts": snap.finished_at})
        _STORE.persist_locked()


def _reset_for_tests(path: Path | None = None) -> None:
    """Test-only helper: drop in-memory state and optionally rebind path."""
    with _STORE.lock:
        _STORE.snapshot = None
        _STORE.hydrated = False
        if path is not None:
            _STORE.path = path
