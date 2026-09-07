"""Persistent Hermes update job snapshot.

Mirrors HiveMind ``ollama_admin::UPDATE_JOB`` plus the on-disk
``deps/ollama_update_state.json`` persistence so the dashboard can
show ``"last update finished N min ago"`` even after a gateway
restart, and so a gateway that crashed mid-update flips its dangling
``running`` snapshot visible across processes. The OS-backed updater
lock decides whether that owner is still live before crash recovery.

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
    "verifying_origin",
    "fetching_remote",
    # F4 supply-chain provenance gate (editable + pypi). Runs after the release
    # tag/objects are fetched but BEFORE the first working-tree mutation or
    # install, so a provenance refusal transitions straight to ``error`` with
    # nothing to roll back.
    "verifying_provenance",
    "checking_out",
    "syncing_plugin",
    "pip_installing",
    "validating",
    # F3 editable-rollback phases. ``rolling_back`` -> ``rolled_back`` on a
    # successful restore of the pre-update commit/install/managed-plugin;
    # ``rolling_back`` -> ``rollback_failed`` when the restore itself fails
    # (fatal, manual recovery required). ``set_phase`` accepts any string;
    # these are enumerated for documentation and UI labeling only.
    "rolling_back",
    "rolled_back",
    "rollback_failed",
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
    superseded_failure: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UpdateJobSnapshot":
        superseded = payload.get("superseded_failure")
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
            superseded_failure=superseded if isinstance(superseded, dict) else None,
        )


class StatePersistenceError(RuntimeError):
    """A durable update-state write failed."""


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

    def hydrate(self, *, force: bool = False) -> None:
        with self.lock:
            if self.hydrated and not force:
                return
            self.hydrated = True
            path = self.path
            if not path.exists():
                self.snapshot = None
                return
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return
            try:
                snap = UpdateJobSnapshot.from_dict(payload if isinstance(payload, dict) else {})
            except Exception:
                return
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
        except Exception as exc:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise StatePersistenceError(
                f"could not persist Hermes update state: {exc}"
            ) from exc


_STORE = _StateStore()


def initialize_state(path: Path | None = None) -> None:
    """Hydrate the on-disk snapshot at gateway/MCP startup."""
    if path is not None:
        _STORE.set_path(path)
    _STORE.hydrate(force=True)


def last_update(*, refresh: bool = True) -> UpdateJobSnapshot | None:
    _STORE.hydrate(force=refresh)
    with _STORE.lock:
        return _STORE.snapshot


def update_in_progress(*, refresh: bool = True) -> bool:
    snap = last_update(refresh=refresh)
    return bool(snap and snap.status == "running")


def fail_interrupted_job() -> UpdateJobSnapshot | None:
    """Mark a durable running snapshot failed after lock ownership is proven absent."""
    _STORE.hydrate(force=True)
    with _STORE.lock:
        snap = _STORE.snapshot
        if snap is None or snap.status != "running":
            return snap
        snap.status = "failed"
        snap.phase = "error"
        snap.error = (
            "Gateway restarted while update was running; install may have completed but "
            "the bookkeeping was lost. Re-check version and re-trigger if needed."
        )
        snap.finished_at = _now_iso()
        snap.progress.append(
            {"phase": "error", "ts": snap.finished_at, "note": snap.error}
        )
        _STORE.persist_locked()
        return snap


def start_job(
    *,
    from_version: str | None,
    to_version: str | None,
    install_mode: str | None,
    request_user: str | None = None,
) -> UpdateJobSnapshot:
    _STORE.hydrate(force=True)
    with _STORE.lock:
        existing = _STORE.snapshot
        if existing is not None and existing.status == "running":
            return existing
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
        previous = _STORE.snapshot
        _STORE.snapshot = snap
        try:
            _STORE.persist_locked()
        except Exception:
            _STORE.snapshot = previous
            raise
    return snap


def reconcile_stale_failure_for_current_or_newer(
    *,
    current_version: str,
    latest_version: str,
    relation: str,
    request_user: str | None = None,
) -> UpdateJobSnapshot | None:
    """Supersede a durable failed job when installed is current or newer.

    Presentation becomes a verified no-op success. The original failure
    stays in ``progress`` and ``superseded_failure`` so audit is not deleted.
    Running jobs and unknown/older relations are left untouched (fail closed).
    """
    if relation not in {"current", "newer"}:
        return last_update(refresh=True)
    _STORE.hydrate(force=True)
    with _STORE.lock:
        previous = _STORE.snapshot
        if previous is None or previous.status != "failed":
            return previous
        superseded = {
            "status": previous.status,
            "phase": previous.phase,
            "error": previous.error,
            "from_version": previous.from_version,
            "to_version": previous.to_version,
            "finished_at": previous.finished_at,
        }
        original_error = previous.error
        original_to = previous.to_version
        previous.status = "success"
        previous.phase = "done"
        previous.error = None
        previous.from_version = current_version
        previous.to_version = current_version
        previous.finished_at = _now_iso()
        previous.superseded_failure = superseded
        if request_user:
            previous.request_user = request_user
        previous.progress.append(
            {
                "phase": "done",
                "ts": previous.finished_at,
                "note": (
                    f"Verified {relation} no-op: installed {current_version} >= "
                    f"latest {latest_version}; superseded stale failure targeting "
                    f"{original_to} without deleting audit"
                    + (f" ({original_error})" if original_error else "")
                ),
            }
        )
        _STORE.persist_locked()
        return previous


def publish_current_noop_success(
    *,
    expected_job_id: str,
    current_version: str,
    request_user: str | None = None,
) -> UpdateJobSnapshot:
    """Durably replace one exact failed editable job with a clean no-op success.

    The caller must hold the updater's whole-job lock for this compare-and-swap.
    """
    _STORE.hydrate(force=True)
    with _STORE.lock:
        previous = _STORE.snapshot
        if (
            previous is None
            or previous.job_id != expected_job_id
            or previous.status != "failed"
            or previous.install_mode != "editable"
            or previous.to_version != current_version
        ):
            raise RuntimeError(
                "refusing terminal no-op publication because the expected "
                "failed editable Hermes job changed"
            )
        snap = UpdateJobSnapshot(
            job_id=previous.job_id,
            started_at=previous.started_at,
            finished_at=_now_iso(),
            status="success",
            phase="done",
            from_version=current_version,
            to_version=current_version,
            install_mode="editable",
            error=None,
            progress=[],
            request_user=request_user,
        )
        _STORE.snapshot = snap
        try:
            _STORE.persist_locked()
        except Exception:
            _STORE.snapshot = previous
            raise
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
