from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_PATH = Path(__file__).resolve().parents[1] / "logs" / "ms4_audit.jsonl"
VOICE_TURN_EVENT_TYPES = ("voice_turn_complete", "voice_turn_failed")
DEFAULT_MATCH_MAX_BYTES = 2_000_000
DEFAULT_TAIL_MAX_BYTES = 1_000_000
SCAN_BUDGET_EXHAUSTED = "scan_budget_exhausted"
SCAN_TIMEOUT = "scan_timeout"
GENERATION_CHANGED = "generation_changed"
SOURCE_INCOMPLETE_AUDIT_SCAN = "incomplete-audit-scan"
MAX_GENERATION_RETRIES = 3
GENERATION_SAMPLE_BYTES = 64 * 1024
GENERATION_MID_SAMPLE_BYTES = 4096

_LAST_GOOD_VOICE_TURNS: list[dict[str, Any]] = []
_LAST_GOOD_BY_SCOPE: dict[str, dict[str, Any]] = {}
_LAST_GOOD_LOCK = threading.Lock()
_SCAN_CHUNK_HOOK = None


def audit_path_from_env() -> Path:
    return Path(os.environ.get("MS4_AUDIT_LOG", str(DEFAULT_AUDIT_PATH)))


def voice_turns_path(audit_path: Path | None = None) -> Path:
    raw = os.environ.get("MS4_VOICE_TURNS_LOG")
    if raw:
        return Path(raw)
    return (audit_path or audit_path_from_env()).with_name("ms4_voice_turns.jsonl")


def append_event(event_type: str, data: dict[str, Any], *, audit_path: Path | None = None) -> dict[str, Any]:
    path = audit_path or audit_path_from_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **data,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    if event_type in VOICE_TURN_EVENT_TYPES:
        _append_voice_turn_ring(event, audit_path=path)
    return event


def _tail_bytes(path: Path, max_bytes: int) -> bytes:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if max_bytes > 0 and size > max_bytes:
            handle.seek(size - max_bytes)
            return handle.read()
        return handle.read()


def _decode_tailed_bytes(raw: bytes, *, truncated: bool) -> list[str]:
    text = raw.decode("utf-8", "replace")
    if truncated:
        newline = text.find("\n")
        if newline >= 0:
            text = text[newline + 1 :]
    return text.splitlines()


def _parse_event_line(line: str) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return {"event_type": "corrupt_audit_line", "raw": line}
    if isinstance(payload, dict):
        return payload
    return {"event_type": "corrupt_audit_line", "raw": line}


def read_events(
    *,
    limit: int = 100,
    audit_path: Path | None = None,
    max_bytes: int = DEFAULT_TAIL_MAX_BYTES,
) -> list[dict[str, Any]]:
    path = audit_path or audit_path_from_env()
    if not path.exists():
        return []
    size = path.stat().st_size
    raw = _tail_bytes(path, max(max_bytes, 4096))
    lines = _decode_tailed_bytes(raw, truncated=size > max(max_bytes, 4096))
    events = [_parse_event_line(line) for line in lines[-max(limit, 1) :]]
    return events


def _scan_result(
    *,
    events: list[dict[str, Any]],
    complete: bool,
    reached_start: bool,
    reason: str | None,
    bytes_scanned: int = 0,
    file_size: int = 0,
    generation: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    payload = {
        "events": events,
        "complete": complete,
        "reached_start": reached_start,
        "reason": reason,
        "bytes_scanned": bytes_scanned,
        "file_size": file_size,
    }
    if generation is not None:
        payload["generation"] = generation
    return payload


def _generation_key(path: Path, *, handle: Any | None = None) -> tuple[Any, ...]:
    try:
        st = os.fstat(handle.fileno()) if handle is not None else path.stat()
    except OSError:
        return ("missing", os.fspath(path))
    return (st.st_dev, st.st_ino)


def _generation_size(path: Path, *, handle: Any | None = None) -> int:
    try:
        if handle is not None:
            return int(os.fstat(handle.fileno()).st_size)
        return int(path.stat().st_size)
    except OSError:
        return -1


def _generation_mtime_ns(path: Path, *, handle: Any | None = None) -> int:
    try:
        st = os.fstat(handle.fileno()) if handle is not None else path.stat()
        return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    except OSError:
        return -1


def _read_path_slice(path: Path, offset: int, length: int) -> bytes | None:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
    except OSError:
        return None
    if len(data) != length:
        return None
    return data


def _read_handle_slice(handle: Any, offset: int, length: int) -> bytes | None:
    try:
        here = handle.tell()
        handle.seek(offset)
        data = handle.read(length)
        handle.seek(here)
    except OSError:
        return None
    if len(data) != length:
        return None
    return data


def _bounded_content_digest(
    path: Path, size: int, *, handle: Any | None = None
) -> str:
    """SHA-256 of size plus bounded head/mid/tail slices. Never reads the whole file."""
    hasher = hashlib.sha256()
    hasher.update(int(max(size, 0)).to_bytes(8, "little", signed=False))
    if size <= 0:
        return hasher.hexdigest()
    window = min(GENERATION_SAMPLE_BYTES, size)

    def take(offset: int, length: int) -> bytes | None:
        if handle is not None:
            return _read_handle_slice(handle, offset, length)
        return _read_path_slice(path, offset, length)

    head = take(0, window)
    if head is None:
        return "unreadable"
    hasher.update(head)
    if size > window:
        tail = take(size - window, window)
        if tail is None:
            return "unreadable"
        hasher.update(tail)
        if size > 2 * window:
            mid_len = min(GENERATION_MID_SAMPLE_BYTES, window)
            mid = take((size - mid_len) // 2, mid_len)
            if mid is None:
                return "unreadable"
            hasher.update(mid)
    return hasher.hexdigest()


def _file_generation_token(path: Path, *, handle: Any | None = None) -> tuple[Any, ...]:
    """Verified metadata+content generation for last-good cache identity.

    Equal-size and larger same-inode truncate+regrow change mtime and/or the
    bounded content digest even when ``st_dev``/``st_ino``/``st_size`` match.
    """
    if handle is None:
        try:
            path.stat()
        except OSError:
            return ("missing", os.fspath(path))
    identity = _generation_key(path, handle=handle)
    size = _generation_size(path, handle=handle)
    mtime_ns = _generation_mtime_ns(path, handle=handle)
    digest = _bounded_content_digest(path, max(size, 0), handle=handle)
    return (identity, size, mtime_ns, digest)


def _generation_unproven(
    path: Path,
    handle: Any,
    *,
    opened_identity: tuple[Any, ...],
    opened_size: int,
    opened_mtime_ns: int,
    consumed: list[tuple[int, bytes]],
) -> bool:
    """True when the opened generation can no longer be proven coherent.

    Same-inode truncate+regrow to an equal or larger size is a new generation
    even though ``st_size < opened_size`` never fires.
    """
    try:
        handle_id = _generation_key(path, handle=handle)
        path_id = _generation_key(path)
        handle_size = _generation_size(path, handle=handle)
        path_size = _generation_size(path)
        handle_mtime = _generation_mtime_ns(path, handle=handle)
        path_mtime = _generation_mtime_ns(path)
    except OSError:
        return True
    if handle_id != opened_identity or path_id != opened_identity:
        return True
    if handle_size != opened_size or path_size != opened_size:
        return True
    if handle_mtime != opened_mtime_ns or path_mtime != opened_mtime_ns:
        return True
    for offset, expected in consumed:
        from_path = _read_path_slice(path, offset, len(expected))
        from_handle = _read_handle_slice(handle, offset, len(expected))
        if from_path != expected or from_handle != expected:
            return True
    return False


def _scope_key(audit_path: Path) -> str:
    return str(audit_path.resolve()) + "\0" + str(voice_turns_path(audit_path).resolve())


def _clear_last_good(audit_path: Path) -> None:
    with _LAST_GOOD_LOCK:
        _LAST_GOOD_BY_SCOPE.pop(_scope_key(audit_path), None)


def _scope_generation(
    audit_path: Path, *, audit_token: tuple[Any, ...] | None = None
) -> tuple[Any, ...]:
    ring = voice_turns_path(audit_path)
    audit_gen = (
        audit_token if audit_token is not None else _file_generation_token(audit_path)
    )
    ring_gen = _file_generation_token(ring) if ring.exists() else ()
    return (audit_gen, ring_gen)


def _store_last_good(
    audit_path: Path,
    turns: list[dict[str, Any]],
    generation: tuple[Any, ...] | None = None,
) -> None:
    token = generation if generation is not None else _scope_generation(audit_path)
    with _LAST_GOOD_LOCK:
        _LAST_GOOD_BY_SCOPE[_scope_key(audit_path)] = {
            "turns": list(turns),
            "generation": token,
        }
        _LAST_GOOD_VOICE_TURNS[:] = list(turns)


def _load_last_good(audit_path: Path, limit: int) -> tuple[list[dict[str, Any]], bool]:
    with _LAST_GOOD_LOCK:
        key = _scope_key(audit_path)
        entry = _LAST_GOOD_BY_SCOPE.get(key)
        if not entry:
            return [], False
        current = _scope_generation(audit_path)
        mismatch = entry.get("generation") != current
        if mismatch:
            _LAST_GOOD_BY_SCOPE.pop(key, None)
            return [], True
        turns = list(entry.get("turns") or [])[:limit]
        return turns, False


def _scan_one_generation(
    *,
    path: Path,
    wanted: set[str],
    limit: int,
    max_bytes: int,
    deadline: float,
    chunk: int,
) -> dict[str, Any]:
    handle = path.open("rb")
    try:
        opened_identity = _generation_key(path, handle=handle)
        opened_size = _generation_size(path, handle=handle)
        opened_mtime_ns = _generation_mtime_ns(path, handle=handle)
        if opened_size <= 0:
            return {
                "drifted": False,
                "result": _scan_result(
                    events=[],
                    complete=True,
                    reached_start=True,
                    reason=None,
                    file_size=0,
                    generation=_file_generation_token(path, handle=handle),
                ),
            }
        budget = min(max(max_bytes, 0), opened_size)
        pos = opened_size
        buf = b""
        matched: list[dict[str, Any]] = []
        bytes_scanned = 0
        timeout_hit = False
        consumed: list[tuple[int, bytes]] = []

        def unproven() -> bool:
            return _generation_unproven(
                path,
                handle,
                opened_identity=opened_identity,
                opened_size=opened_size,
                opened_mtime_ns=opened_mtime_ns,
                consumed=consumed,
            )

        while pos > 0 and budget > 0 and len(matched) < limit:
            if time.monotonic() >= deadline:
                timeout_hit = True
                break
            if unproven():
                return {"drifted": True, "result": None}
            take = min(chunk, pos, budget)
            pos -= take
            budget -= take
            bytes_scanned += take
            handle.seek(pos)
            chunk_bytes = handle.read(take)
            if len(chunk_bytes) != take:
                return {"drifted": True, "result": None}
            consumed.append((pos, chunk_bytes))
            buf = chunk_bytes + buf
            hook = _SCAN_CHUNK_HOOK
            if hook is not None:
                hook(path, handle)
            if unproven():
                matched.clear()
                return {"drifted": True, "result": None}
            parts = buf.split(b"\n")
            buf = parts[0]
            for raw_line in reversed(parts[1:]):
                if not raw_line.strip():
                    continue
                event = _parse_event_line(raw_line.decode("utf-8", "replace"))
                if event.get("event_type") in wanted:
                    matched.append(event)
                    if len(matched) >= limit:
                        break
        if unproven():
            matched.clear()
            return {"drifted": True, "result": None}
        reached_start = pos == 0
        if buf.strip() and len(matched) < limit and reached_start:
            event = _parse_event_line(buf.decode("utf-8", "replace"))
            if event.get("event_type") in wanted:
                matched.append(event)
        if unproven():
            matched.clear()
            return {"drifted": True, "result": None}
        filled = len(matched) >= limit
        complete = filled or reached_start
        reason = None
        if not complete:
            reason = SCAN_TIMEOUT if timeout_hit else SCAN_BUDGET_EXHAUSTED
        return {
            "drifted": False,
            "result": _scan_result(
                events=matched,
                complete=complete,
                reached_start=reached_start,
                reason=reason,
                bytes_scanned=bytes_scanned,
                file_size=opened_size,
                generation=_file_generation_token(path, handle=handle),
            ),
        }
    finally:
        handle.close()


def read_matching_events(
    *,
    event_types: tuple[str, ...],
    limit: int = 25,
    audit_path: Path | None = None,
    max_bytes: int = DEFAULT_MATCH_MAX_BYTES,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    """Newest-first matching events via a bounded reverse byte scan.

    Never loads the whole file through ``Path.read_text``. Distinguishes a
    complete scan that reached the start of the file (or filled ``limit``)
    from an incomplete scan that exhausted ``max_bytes``, timed out, or
    lost its file generation to concurrent replace/truncation.
    """
    path = audit_path or audit_path_from_env()
    wanted = set(event_types)
    if not path.exists():
        return _scan_result(
            events=[], complete=True, reached_start=True, reason=None, file_size=0
        )
    size = path.stat().st_size
    if limit < 1:
        return _scan_result(
            events=[], complete=True, reached_start=True, reason=None, file_size=size
        )
    if timeout_s <= 0:
        return _scan_result(
            events=[],
            complete=False,
            reached_start=False,
            reason=SCAN_TIMEOUT,
            file_size=size,
        )
    if size == 0:
        return _scan_result(
            events=[], complete=True, reached_start=True, reason=None, file_size=0
        )
    deadline = time.monotonic() + max(timeout_s, 0.0)
    chunk = 64 * 1024
    attempts = 0
    max_attempts = 1 + MAX_GENERATION_RETRIES
    timed_out = False
    while attempts < max_attempts:
        attempts += 1
        if time.monotonic() >= deadline:
            timed_out = True
            break
        if not path.exists():
            return _scan_result(
                events=[], complete=True, reached_start=True, reason=None, file_size=0
            )
        outcome = _scan_one_generation(
            path=path,
            wanted=wanted,
            limit=limit,
            max_bytes=max_bytes,
            deadline=deadline,
            chunk=chunk,
        )
        if outcome["drifted"]:
            continue
        return outcome["result"]
    live_size = path.stat().st_size if path.exists() else 0
    return _scan_result(
        events=[],
        complete=False,
        reached_start=False,
        reason=SCAN_TIMEOUT if timed_out else GENERATION_CHANGED,
        file_size=live_size,
    )


def _append_voice_turn_ring(event: dict[str, Any], *, audit_path: Path) -> None:
    ring = voice_turns_path(audit_path)
    ring.parent.mkdir(parents=True, exist_ok=True)
    with ring.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        size = ring.stat().st_size
    except OSError:
        return
    # Keep the ring bounded (~200 recent turns, ~512 KiB).
    if size < 512_000:
        return
    raw = _tail_bytes(ring, 256_000)
    lines = _decode_tailed_bytes(raw, truncated=True)
    keep = lines[-200:]
    tmp = ring.with_suffix(ring.suffix + ".tmp")
    tmp.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    tmp.replace(ring)


def _read_voice_turn_ring(audit_path: Path, limit: int) -> list[dict[str, Any]]:
    ring = voice_turns_path(audit_path)
    if not ring.exists() or limit < 1:
        return []
    raw = _tail_bytes(ring, 256_000)
    lines = _decode_tailed_bytes(raw, truncated=ring.stat().st_size > 256_000)
    events = [_parse_event_line(line) for line in lines if line.strip()]
    wanted = [
        event
        for event in events
        if event.get("event_type") in VOICE_TURN_EVENT_TYPES
    ]
    wanted.reverse()
    return wanted[:limit]


def _voice_turns_payload(
    *,
    turns: list[dict[str, Any]],
    limit: int,
    source: str,
    error: str | None,
    complete: bool,
    stale: bool,
    audit_path: Path,
    generation: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "Ms4VoiceRecentTurns.v1",
        "turns": turns[:limit],
        "limit": limit,
        "source": source,
        "error": error,
        "complete": complete,
        "stale": stale,
    }
    if payload["turns"] and not stale:
        _store_last_good(audit_path, payload["turns"], generation=generation)
    return payload


def read_recent_voice_turns(
    *,
    limit: int = 25,
    audit_path: Path | None = None,
    max_bytes: int = DEFAULT_MATCH_MAX_BYTES,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    path = audit_path or audit_path_from_env()
    error: str | None = None
    source = "ring"
    stale = False
    complete = True
    turns = _read_voice_turn_ring(path, limit)
    if timeout_s <= 0 and len(turns) < limit:
        last_good, gen_mismatch = _load_last_good(path, limit)
        if last_good and not gen_mismatch:
            return _voice_turns_payload(
                turns=last_good,
                limit=limit,
                source="last-good",
                error="voice recent-turns timed out; returning last-good",
                complete=False,
                stale=True,
                audit_path=path,
            )
        return _voice_turns_payload(
            turns=[],
            limit=limit,
            source="error",
            error="voice recent-turns timed out before a matching event",
            complete=False,
            stale=False,
            audit_path=path,
        )
    scan: dict[str, Any] = {
        "events": [],
        "complete": True,
        "reason": None,
        "reached_start": True,
    }
    scanned_history = False
    if len(turns) < limit:
        scanned_history = True
        try:
            scan = read_matching_events(
                event_types=VOICE_TURN_EVENT_TYPES,
                limit=limit,
                audit_path=path,
                max_bytes=max_bytes,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - last-good/explicit-error
            error = str(exc)
            scan = {
                "events": [],
                "complete": False,
                "reason": "scan_error",
                "reached_start": False,
            }
        scanned = list(scan.get("events") or [])
        if scanned:
            source = "audit-scan" if not turns else "ring"
            seen = {json.dumps(turn, sort_keys=True) for turn in turns}
            for event in scanned:
                key = json.dumps(event, sort_keys=True)
                if key not in seen:
                    turns.append(event)
                    seen.add(key)
            turns = turns[:limit]
            if not _read_voice_turn_ring(path, 1):
                source = "audit-scan"
        scan_complete = bool(scan.get("complete"))
        scan_reason = scan.get("reason")
        if not turns:
            if scan_complete and not error:
                _clear_last_good(path)
                return _voice_turns_payload(
                    turns=[],
                    limit=limit,
                    source="audit-scan",
                    error=None,
                    complete=True,
                    stale=False,
                    audit_path=path,
                )
            diagnostic = error or scan_reason or SCAN_BUDGET_EXHAUSTED
            last_good, gen_mismatch = _load_last_good(path, limit)
            if last_good and not gen_mismatch:
                return _voice_turns_payload(
                    turns=last_good,
                    limit=limit,
                    source="last-good",
                    error=diagnostic,
                    complete=False,
                    stale=True,
                    audit_path=path,
                )
            source = "error" if scan.get("reason") == "scan_error" else SOURCE_INCOMPLETE_AUDIT_SCAN
            return _voice_turns_payload(
                turns=[],
                limit=limit,
                source=source,
                error=diagnostic,
                complete=False,
                stale=False,
                audit_path=path,
            )
        if scanned_history and not scan_complete:
            complete = False
            stale = False
            error = error or scan_reason or SCAN_BUDGET_EXHAUSTED
            if source == "audit-scan":
                source = SOURCE_INCOMPLETE_AUDIT_SCAN
        elif source == "ring":
            complete = True
            stale = False
            error = None
        else:
            complete = scan_complete
            stale = False
            if not scan_complete:
                error = error or scan_reason or SCAN_BUDGET_EXHAUSTED
    scan_token = scan.get("generation") if scanned_history else None
    return _voice_turns_payload(
        turns=turns,
        limit=limit,
        source=source,
        error=error,
        complete=complete,
        stale=stale,
        audit_path=path,
        generation=_scope_generation(path, audit_token=scan_token),
    )
