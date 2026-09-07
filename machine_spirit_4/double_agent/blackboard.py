"""SQLite-backed blackboard for Double Agent jobs, events, results and
conversation revisions.

Design notes:

* SQLite chosen because (a) it's stdlib, no new dependency; (b) it
  survives gateway restart for the hydration-and-recovery contract;
  (c) it gives the multi-job indexed query we need (list active by
  conversation, filter by state, etc.) without a server process.
* All writes go through this module — the runner, the worker, and the
  gateway never touch SQL directly. This is the single place where
  schemas → SQL row → JSON round-trip is implemented, so the safety
  guards in :mod:`schemas` and :mod:`safety` are the only validation
  surface.
* Connections are short-lived per call (no pooling). SQLite's WAL
  journal mode is enabled so a long-running SSE reader can iterate the
  events table concurrently with the worker thread appending to it.
* Recovery semantics mirror ``hermes_admin.state``: on
  :func:`hydrate_recover`, any job left in ``running`` or ``queued`` is
  flipped to ``failed`` with an explicit recovery message, because the
  in-process worker thread that owned it is gone.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import safety
from .schemas import (
    ConversationRevision,
    JobEnvelope,
    JobEvent,
    JobResult,
    SchemaError,
)


MS4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = MS4_ROOT / "runtime" / "double_agent.sqlite3"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id                     TEXT PRIMARY KEY,
    parent_conversation_id     TEXT NOT NULL,
    conversation_revision_id   INTEGER NOT NULL,
    state                      TEXT NOT NULL,
    background_lobe_type       TEXT NOT NULL,
    user_visible_goal          TEXT NOT NULL,
    priority                   TEXT NOT NULL,
    latency_class              TEXT NOT NULL,
    created_at                 TEXT NOT NULL,
    updated_at                 TEXT NOT NULL,
    finished_at                TEXT,
    last_safe_user_status      TEXT,
    envelope_json              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS jobs_by_conversation
    ON jobs (parent_conversation_id, conversation_revision_id);
CREATE INDEX IF NOT EXISTS jobs_by_state
    ON jobs (state, updated_at);

CREATE TABLE IF NOT EXISTS events (
    event_id   TEXT PRIMARY KEY,
    job_id     TEXT NOT NULL,
    type       TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    visibility TEXT NOT NULL,
    event_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_job
    ON events (job_id, timestamp);

CREATE TABLE IF NOT EXISTS results (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_revisions (
    conversation_id     TEXT NOT NULL,
    revision_id         INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    user_message_excerpt TEXT NOT NULL,
    turn_id              TEXT,
    PRIMARY KEY (conversation_id, revision_id)
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


class Blackboard:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self._init_lock = threading.Lock()
        self._initialized = False

    def _ensure_init(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA_SQL)
                revision_columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(conversation_revisions)")
                }
                if "turn_id" not in revision_columns:
                    conn.execute(
                        "ALTER TABLE conversation_revisions ADD COLUMN turn_id TEXT"
                    )
                conn.commit()
            self._initialized = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            str(self.path),
            timeout=15,
            isolation_level=None,  # autocommit; we control transactions explicitly
            check_same_thread=False,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def insert_job(self, envelope: JobEnvelope) -> None:
        envelope.validate()
        self._ensure_init()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, parent_conversation_id, conversation_revision_id,
                    state, background_lobe_type, user_visible_goal,
                    priority, latency_class, created_at, updated_at,
                    finished_at, last_safe_user_status, envelope_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    envelope.job_id,
                    envelope.parent_conversation_id,
                    envelope.conversation_revision_id,
                    envelope.state,
                    envelope.background_lobe_type,
                    envelope.user_visible_goal,
                    envelope.priority,
                    envelope.latency_class,
                    envelope.created_at,
                    envelope.updated_at,
                    envelope.finished_at,
                    envelope.user_visible_goal,  # bootstrap safe_user_status until first event lands
                    json.dumps(envelope.to_dict(), ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_job(self, job_id: str) -> JobEnvelope | None:
        if not safety.is_safe_job_id(job_id):
            return None
        self._ensure_init()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT envelope_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            return JobEnvelope.from_dict(json.loads(row["envelope_json"]))

    def get_job_snapshot(self, job_id: str) -> dict[str, Any] | None:
        if not safety.is_safe_job_id(job_id):
            return None
        self._ensure_init()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT j.envelope_json, j.last_safe_user_status, j.state,
                       j.updated_at, j.finished_at,
                       (SELECT timestamp FROM events e WHERE e.job_id = j.job_id
                        ORDER BY timestamp DESC LIMIT 1) AS last_event_at,
                       (SELECT type FROM events e WHERE e.job_id = j.job_id
                        ORDER BY timestamp DESC LIMIT 1) AS last_event_type
                FROM jobs j WHERE j.job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        envelope = json.loads(row["envelope_json"])
        envelope["state"] = row["state"]
        envelope["updated_at"] = row["updated_at"]
        envelope["finished_at"] = row["finished_at"]
        # Re-compute is_stale relative to the conversation's current revision.
        current_rev = self.current_revision(envelope["parent_conversation_id"])
        envelope["is_stale"] = (
            envelope["state"] == "stale"
            or (current_rev is not None and envelope["conversation_revision_id"] < current_rev)
        )
        envelope["last_safe_user_status"] = row["last_safe_user_status"]
        envelope["last_event_at"] = row["last_event_at"]
        envelope["last_event_type"] = row["last_event_type"]
        return envelope

    def list_jobs(
        self,
        *,
        conversation_id: str | None = None,
        states: tuple[str, ...] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self._ensure_init()
        clauses = []
        params: list[Any] = []
        if conversation_id is not None:
            if not safety.is_safe_conversation_id(conversation_id):
                return []
            clauses.append("parent_conversation_id = ?")
            params.append(conversation_id)
        if states:
            placeholders = ",".join("?" * len(states))
            clauses.append(f"state IN ({placeholders})")
            params.extend(states)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT job_id FROM jobs{where} ORDER BY updated_at DESC LIMIT ?",
                (*params, max(1, min(int(limit), 500))),
            ).fetchall()
        out = []
        for row in rows:
            snapshot = self.get_job_snapshot(row["job_id"])
            if snapshot is not None:
                out.append(snapshot)
        return out

    def update_job_state(
        self,
        job_id: str,
        *,
        state: str,
        finished_at: str | None = None,
        last_safe_user_status: str | None = None,
    ) -> None:
        if not safety.is_safe_job_id(job_id):
            raise SchemaError(f"unsafe job_id: {job_id!r}")
        if not safety.is_safe_state(state):
            raise SchemaError(f"unsafe state: {state!r}")
        self._ensure_init()
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                if last_safe_user_status is not None:
                    conn.execute(
                        "UPDATE jobs SET state = ?, updated_at = ?, finished_at = ?, "
                        "last_safe_user_status = ? WHERE job_id = ?",
                        (
                            state,
                            _now_iso(),
                            finished_at,
                            safety.coerce_safe_user_status(last_safe_user_status),
                            job_id,
                        ),
                    )
                else:
                    conn.execute(
                        "UPDATE jobs SET state = ?, updated_at = ?, finished_at = ? "
                        "WHERE job_id = ?",
                        (state, _now_iso(), finished_at, job_id),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def insert_event(self, event: JobEvent) -> None:
        event.validate()
        self._ensure_init()
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "INSERT INTO events (event_id, job_id, type, timestamp, visibility, event_json) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        event.event_id,
                        event.job_id,
                        event.type,
                        event.timestamp,
                        event.visibility,
                        json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True),
                    ),
                )
                if event.visibility == "user_safe" and event.safe_user_status:
                    conn.execute(
                        "UPDATE jobs SET last_safe_user_status = ?, updated_at = ? WHERE job_id = ?",
                        (event.safe_user_status, _now_iso(), event.job_id),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def list_events(
        self,
        job_id: str,
        *,
        after_event_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not safety.is_safe_job_id(job_id):
            return []
        self._ensure_init()
        with self._connect() as conn:
            if after_event_id is None:
                rows = conn.execute(
                    "SELECT event_json FROM events WHERE job_id = ? ORDER BY timestamp ASC LIMIT ?",
                    (job_id, max(1, min(int(limit), 1000))),
                ).fetchall()
            else:
                cutoff = conn.execute(
                    "SELECT timestamp FROM events WHERE event_id = ?",
                    (after_event_id,),
                ).fetchone()
                if cutoff is None:
                    rows = conn.execute(
                        "SELECT event_json FROM events WHERE job_id = ? ORDER BY timestamp ASC LIMIT ?",
                        (job_id, max(1, min(int(limit), 1000))),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT event_json FROM events WHERE job_id = ? AND timestamp > ? "
                        "ORDER BY timestamp ASC LIMIT ?",
                        (job_id, cutoff["timestamp"], max(1, min(int(limit), 1000))),
                    ).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def insert_result(self, result: JobResult) -> None:
        result.validate()
        self._ensure_init()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO results (job_id, status, finished_at, result_json) "
                "VALUES (?,?,?,?)",
                (
                    result.job_id,
                    result.status,
                    result.finished_at,
                    json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        if not safety.is_safe_job_id(job_id):
            return None
        self._ensure_init()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM results WHERE job_id = ?", (job_id,),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    # ------------------------------------------------------------------
    # Conversation revisions
    # ------------------------------------------------------------------

    def current_revision(self, conversation_id: str) -> int | None:
        if not safety.is_safe_conversation_id(conversation_id):
            return None
        self._ensure_init()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(revision_id) AS r FROM conversation_revisions WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["r"]) if row and row["r"] is not None else None

    def bump_revision(
        self,
        conversation_id: str,
        *,
        user_message_excerpt: str = "",
        turn_id: str | None = None,
    ) -> ConversationRevision:
        if not safety.is_safe_conversation_id(conversation_id):
            raise SchemaError(f"unsafe conversation_id: {conversation_id!r}")
        self._ensure_init()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT MAX(revision_id) AS r FROM conversation_revisions WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                next_rev = (int(row["r"]) + 1) if row and row["r"] is not None else 1
                revision = ConversationRevision(
                    conversation_id=conversation_id,
                    revision_id=next_rev,
                    user_message_excerpt=user_message_excerpt,
                    turn_id=turn_id,
                )
                revision.validate()
                conn.execute(
                    "INSERT INTO conversation_revisions "
                    "(conversation_id, revision_id, created_at, user_message_excerpt, turn_id) "
                    "VALUES (?,?,?,?,?)",
                    (
                        revision.conversation_id,
                        revision.revision_id,
                        revision.created_at,
                        revision.user_message_excerpt,
                        revision.turn_id,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return revision

    def mark_stale_jobs_for_revision(
        self,
        conversation_id: str,
        current_revision_id: int,
    ) -> list[str]:
        """Mark any job whose revision is older than ``current_revision_id``
        as ``stale``. Returns the list of affected job_ids."""
        if not safety.is_safe_conversation_id(conversation_id):
            return []
        self._ensure_init()
        affected: list[str] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT job_id FROM jobs WHERE parent_conversation_id = ? "
                    "AND conversation_revision_id < ? AND state IN ('queued','running')",
                    (conversation_id, int(current_revision_id)),
                ).fetchall()
                for row in rows:
                    affected.append(row["job_id"])
                    conn.execute(
                        "UPDATE jobs SET state = 'stale', updated_at = ?, "
                        "last_safe_user_status = ? WHERE job_id = ?",
                        (
                            _now_iso(),
                            "Marked stale: a newer user message changed the request.",
                            row["job_id"],
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return affected

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def hydrate_recover(self) -> list[str]:
        """Called at gateway startup. Any job left in ``queued``/``running``
        is flipped to ``failed`` (its in-process worker thread is gone).
        Returns the affected job ids."""
        self._ensure_init()
        affected: list[str] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT job_id FROM jobs WHERE state IN ('queued','running')"
                ).fetchall()
                for row in rows:
                    affected.append(row["job_id"])
                    conn.execute(
                        "UPDATE jobs SET state = 'failed', updated_at = ?, finished_at = ?, "
                        "last_safe_user_status = ? WHERE job_id = ?",
                        (
                            _now_iso(),
                            _now_iso(),
                            "Marked failed at gateway startup: worker did not survive restart.",
                            row["job_id"],
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return affected


# Module-level singleton used by the gateway/MCP/CLI surfaces. Tests
# construct their own Blackboard with a tmp_path-backed sqlite to avoid
# touching the production db file.
_DEFAULT_BLACKBOARD: Blackboard | None = None


def default_blackboard() -> Blackboard:
    global _DEFAULT_BLACKBOARD
    if _DEFAULT_BLACKBOARD is None:
        _DEFAULT_BLACKBOARD = Blackboard()
    return _DEFAULT_BLACKBOARD


def _reset_default_blackboard_for_tests(path: Path | None = None) -> Blackboard:
    """Test helper: rebind the module singleton."""
    global _DEFAULT_BLACKBOARD
    _DEFAULT_BLACKBOARD = Blackboard(path)
    return _DEFAULT_BLACKBOARD
