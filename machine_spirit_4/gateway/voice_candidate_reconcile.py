"""MS4 gateway consumer for HLI proposal-only unknown-speaker reconciliation.

This is a dream-cadence / background-review intent owned by the normal
MS4 gateway lifecycle. It is not part of the MS3 dream scheduler, does
not infer a human name, and never applies a proposal. Default-off,
fail-soft, metadata-only, loopback-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


log = logging.getLogger("ms4.gateway.voice_candidate_reconcile")

_IMPORT_STARTED = False

STORE_SCHEMA = "Ms4VoiceCandidateReviewStore.v1"
DEFAULT_INTERVAL_SECS = 3600
DEFAULT_TIMEOUT_SECS = 3
DEFAULT_JOIN_TIMEOUT_SECS = 2.0
MAX_PROPOSALS = 64
MAX_STORE_ITEMS = 512
MAX_RESPONSE_BYTES = 262144
CANDIDATE_ID_MAX = 128
TARGET_ID_MAX = 128
TARGET_KIND_MAX = 64
REASON_MAX = 256
CANDIDATES_PATH = "/v1/audio/voice-identities/candidates"
RECONCILE_PATH = "/v1/audio/voice-identities/candidates/reconcile"
CANDIDATES_QUERY = "return_embeddings=false&include_ignored=false"
ALLOWED_ACTIONS = frozenset({"merge", "new_profile", "ignore"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
ITEM_FIELDS = (
    "item_id",
    "candidate_id",
    "action",
    "target_kind",
    "target_id",
    "score",
    "reason",
    "first_seen",
    "last_seen",
    "observation_count",
    "status",
)
FORBIDDEN_KEYS = frozenset(
    {
        "embedding",
        "embeddings",
        "vector",
        "vectors",
        "audio",
        "audio_data",
        "audio_base64",
        "bytes",
        "base64",
        "transcript",
        "text",
        "db_path",
        "url",
        "urls",
        "endpoint",
        "endpoints",
        "credential",
        "credentials",
        "token",
        "tokens",
        "header",
        "headers",
        "authorization",
        "api_key",
        "body",
        "raw",
    }
)
_DEFAULT_STORE_PATH = (
    Path(__file__).resolve().parents[1] / "runtime" / "voice_candidate_review_items.json"
)

WaitFn = Callable[[threading.Event, float], bool]
TransportFn = Callable[..., tuple[int, Any]]
AuditFn = Callable[[dict[str, Any]], None]
ReplaceFn = Callable[[str, str], Any]


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            value = default
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class VoiceCandidateReconcileConfig:
    enabled: bool
    interval_secs: int
    timeout_secs: int
    store_path: Path
    hivemind_url: str

    @classmethod
    def from_env(cls, *, hivemind_url: str | None = None) -> VoiceCandidateReconcileConfig:
        url = (
            (hivemind_url or "").strip()
            or os.environ.get("MS4_HIVEMIND_URL", "").strip()
            or os.environ.get("MS4_HIVEMIND_HLI_URL", "").strip()
            or "http://127.0.0.1:6089"
        )
        store_raw = os.environ.get("MS4_VOICE_CANDIDATE_REVIEW_STORE", "").strip()
        return cls(
            enabled=_parse_bool_env("MS4_VOICE_CANDIDATE_RECONCILE_ENABLED", False),
            interval_secs=_parse_int_env(
                "MS4_VOICE_CANDIDATE_RECONCILE_SECS", DEFAULT_INTERVAL_SECS, 60, 86400
            ),
            timeout_secs=_parse_int_env(
                "MS4_VOICE_CANDIDATE_RECONCILE_TIMEOUT_SECS",
                DEFAULT_TIMEOUT_SECS,
                1,
                10,
            ),
            store_path=Path(store_raw) if store_raw else _DEFAULT_STORE_PATH,
            hivemind_url=url,
        )


def is_loopback_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in LOOPBACK_HOSTS


def derive_item_id(candidate_id: str) -> str:
    digest = hashlib.sha256(f"{STORE_SCHEMA}:{candidate_id}".encode("utf-8")).hexdigest()
    return f"vcr_{digest[:32]}"


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str) or key.lower() in FORBIDDEN_KEYS:
                continue
            cleaned[key] = sanitize(value)
        return cleaned
    if isinstance(obj, list):
        if obj and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in obj):
            return []
        return [sanitize(item) for item in obj]
    if isinstance(obj, str) and obj.lower() in FORBIDDEN_KEYS:
        return ""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _bound_reason(value: Any) -> str | None:
    if value is None:
        return ""
    if not isinstance(value, str):
        return None
    stripped = "".join(ch for ch in value if ch >= " " and ch != "\x7f")
    return stripped[:REASON_MAX]


_INVALID = object()


def _finite_score(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _INVALID
    number = float(value)
    if not math.isfinite(number):
        return _INVALID
    return number


def project_proposal(raw: Any, *, now: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    candidate_id = raw.get("candidate_id")
    if not isinstance(candidate_id, str):
        return None
    candidate_id = candidate_id.strip()
    if not candidate_id or len(candidate_id) > CANDIDATE_ID_MAX:
        return None
    action = raw.get("action")
    if action not in ALLOWED_ACTIONS:
        return None
    target_kind = raw.get("target_kind")
    if target_kind is None:
        kind: str | None = None
    elif isinstance(target_kind, str) and len(target_kind) <= TARGET_KIND_MAX:
        kind = target_kind.strip() or None
    else:
        return None
    target_id = raw.get("target_id")
    if target_id is None:
        tid: str | None = None
    elif isinstance(target_id, str) and len(target_id) <= TARGET_ID_MAX:
        tid = target_id.strip() or None
    else:
        return None
    score = _finite_score(raw.get("score"))
    if score is _INVALID:
        return None
    reason = _bound_reason(raw.get("reason"))
    if reason is None:
        return None
    return {
        "item_id": derive_item_id(candidate_id),
        "candidate_id": candidate_id,
        "action": action,
        "target_kind": kind,
        "target_id": tid,
        "score": score,
        "reason": reason,
        "first_seen": now,
        "last_seen": now,
        "observation_count": 1,
        "status": "pending_operator_review",
    }


def _project_stored_item(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    candidate_id = raw.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return None
    item = project_proposal(
        {
            "candidate_id": candidate_id,
            "action": raw.get("action", "ignore"),
            "target_kind": raw.get("target_kind"),
            "target_id": raw.get("target_id"),
            "score": raw.get("score"),
            "reason": raw.get("reason"),
        },
        now=str(raw.get("last_seen") or raw.get("first_seen") or ""),
    )
    if item is None:
        return None
    first_seen = raw.get("first_seen")
    last_seen = raw.get("last_seen")
    if isinstance(first_seen, str) and first_seen:
        item["first_seen"] = first_seen
    if isinstance(last_seen, str) and last_seen:
        item["last_seen"] = last_seen
    count = raw.get("observation_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 1:
        item["observation_count"] = count
    item_id = raw.get("item_id")
    if isinstance(item_id, str) and item_id.strip():
        item["item_id"] = item_id.strip()
    item["status"] = "pending_operator_review"
    return {field: item[field] for field in ITEM_FIELDS}


def _hivemind_auth_headers() -> dict[str, str]:
    from .hivemind_state import hivemind_auth_headers

    return dict(hivemind_auth_headers() or {})


def _default_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_audit(payload: dict[str, Any]) -> None:
    try:
        from .audit import append_event

        append_event("voice_candidate_reconcile_cycle", sanitize(payload))
    except Exception:
        log.debug("voice candidate reconcile audit failed", exc_info=True)


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail closed on any 30x before urllib can build a follow-up request."""

    def http_error_302(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _default_transport(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    request_headers = {"Accept": "application/json"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    request = urllib.request.Request(url, data=body, method=method, headers=request_headers)
    opener = urllib.request.build_opener(_RejectRedirectHandler)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = int(response.getcode() or 0)
    except urllib.error.HTTPError as exc:
        status = int(exc.code or 0)
        try:
            exc.read(MAX_RESPONSE_BYTES)
        except Exception:
            pass
        return status, None
    except TimeoutError:
        raise
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError) or "timed out" in str(exc).lower():
            raise TimeoutError(str(exc)) from exc
        raise
    if len(raw) > MAX_RESPONSE_BYTES:
        return status, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except Exception:
        return status, None


class VoiceCandidateReviewStore:
    """Atomic, allowlisted, operator-review queue. Never applies proposals."""

    def __init__(self, path: Path, *, replace: ReplaceFn | None = None) -> None:
        self.path = Path(path)
        self._replace = replace or os.replace

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": STORE_SCHEMA, "items": []}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema") != STORE_SCHEMA:
            raise ValueError("unsupported review store schema")
        items = raw.get("items")
        if not isinstance(items, list):
            raise ValueError("review store items must be a list")
        cleaned = []
        for item in items:
            projected = _project_stored_item(item)
            if projected is not None:
                cleaned.append(projected)
        return {"schema": STORE_SCHEMA, "items": cleaned}

    def upsert_projected(self, items: list[dict[str, Any]], *, now: str) -> dict[str, Any]:
        try:
            current = self.load()
        except Exception:
            return {"ok": False, "outcome": "store_unavailable", "recorded_count": 0}
        by_id: dict[str, dict[str, Any]] = {}
        for existing in current["items"]:
            candidate_id = existing.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                by_id[candidate_id] = existing
        incoming_ids = {item["candidate_id"] for item in items}
        new_ids = incoming_ids.difference(by_id)
        if len(by_id) + len(new_ids) > MAX_STORE_ITEMS:
            return {"ok": False, "outcome": "store_capacity", "recorded_count": 0}
        recorded = 0
        for item in items:
            candidate_id = item["candidate_id"]
            projected = {field: item[field] for field in ITEM_FIELDS}
            projected["last_seen"] = now
            projected["status"] = "pending_operator_review"
            previous = by_id.get(candidate_id)
            if previous is not None:
                projected["first_seen"] = previous.get("first_seen") or projected.get("first_seen")
                previous_count = previous.get("observation_count")
                count = previous_count if isinstance(previous_count, int) and not isinstance(previous_count, bool) else 1
                projected["observation_count"] = count + 1
                if isinstance(previous.get("item_id"), str) and previous["item_id"]:
                    projected["item_id"] = previous["item_id"]
            else:
                projected["first_seen"] = now
                projected["observation_count"] = 1
                projected["item_id"] = derive_item_id(candidate_id)
            by_id[candidate_id] = projected
            recorded += 1
        payload = sanitize({"schema": STORE_SCHEMA, "items": list(by_id.values())})
        try:
            self._atomic_write(payload)
        except Exception:
            return {"ok": False, "outcome": "store_unavailable", "recorded_count": 0}
        return {"ok": True, "outcome": "proposals_recorded", "recorded_count": recorded}

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        try:
            with tmp.open("wb") as handle:
                handle.write(data)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            self._replace(os.fspath(tmp), os.fspath(self.path))
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


class VoiceCandidateReconciler:
    """Lifecycle-owned, injectable, fail-soft reconciler. Never runs at import."""

    def __init__(
        self,
        *,
        hivemind_url: str | None = None,
        config: VoiceCandidateReconcileConfig | None = None,
        transport: TransportFn | None = None,
        store: VoiceCandidateReviewStore | None = None,
        audit_sink: AuditFn | None = None,
        wait: WaitFn | None = None,
        now: Callable[[], str] | None = None,
        monotonic: Callable[[], float] | None = None,
        thread_factory: Callable[..., threading.Thread] | None = None,
        auth_headers_fn: Callable[[], dict[str, str]] | None = None,
        join_timeout_secs: float = DEFAULT_JOIN_TIMEOUT_SECS,
    ) -> None:
        self._config = config or VoiceCandidateReconcileConfig.from_env(hivemind_url=hivemind_url)
        self._transport = transport or _default_transport
        self._store = store or VoiceCandidateReviewStore(self._config.store_path)
        self._audit_sink = audit_sink or _default_audit
        self._wait = wait or (lambda event, timeout: event.wait(timeout))
        self._now = now or _default_now
        self._monotonic = monotonic or time.monotonic
        self._thread_factory = thread_factory or threading.Thread
        self._auth_headers_fn = auth_headers_fn or _hivemind_auth_headers
        self._join_timeout_secs = float(join_timeout_secs)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> None:
        if not self._config.enabled:
            return
        try:
            with self._lock:
                if self._thread is not None and self._thread.is_alive():
                    return
                self._stop.clear()
                thread = self._thread_factory(
                    target=self._loop,
                    name="ms4-voice-candidate-reconcile",
                    daemon=True,
                )
                self._thread = thread
                thread.start()
        except Exception:
            log.debug("voice candidate reconciler start failed", exc_info=True)

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        try:
            thread.join(self._join_timeout_secs if timeout is None else timeout)
        except Exception:
            log.debug("voice candidate reconciler stop/join failed", exc_info=True)

    def run_once(self) -> dict[str, Any]:
        started = self._monotonic()
        try:
            result = self._cycle()
        except Exception:
            result = {
                "outcome": "route_unavailable",
                "reason": "cycle_error",
                "proposal_count": 0,
                "recorded_count": 0,
            }
        payload = {
            "outcome": result.get("outcome"),
            "proposal_count": int(result.get("proposal_count") or 0),
            "recorded_count": int(result.get("recorded_count") or 0),
            "elapsed_ms": int(max(0.0, (self._monotonic() - started) * 1000)),
        }
        reason = result.get("reason")
        if isinstance(reason, str) and reason:
            payload["reason"] = reason
        payload = sanitize(payload)
        try:
            self._audit_sink(payload)
        except Exception:
            log.debug("voice candidate reconcile audit sink failed", exc_info=True)
        return payload

    def _loop(self) -> None:
        while True:
            try:
                stopped = self._wait(self._stop, float(self._config.interval_secs))
            except Exception:
                stopped = self._stop.is_set()
            if stopped:
                return
            try:
                self.run_once()
            except Exception:
                log.debug("voice candidate reconcile cycle failed", exc_info=True)

    def _auth_headers(self) -> dict[str, str]:
        try:
            headers = dict(self._auth_headers_fn() or {})
        except Exception:
            return {}
        return {str(key): str(value) for key, value in headers.items() if key and value}

    def _request(self, method: str, url: str, *, body: bytes | None = None) -> tuple[int, Any]:
        return self._transport(
            method,
            url,
            body=body,
            timeout=float(self._config.timeout_secs),
            headers=self._auth_headers(),
        )

    def _cycle(self) -> dict[str, Any]:
        empty = {"proposal_count": 0, "recorded_count": 0}
        base = self._config.hivemind_url.rstrip("/")
        if not is_loopback_url(base):
            return {"outcome": "route_unavailable", "reason": "loopback_required", **empty}
        get_url = f"{base}{CANDIDATES_PATH}?{CANDIDATES_QUERY}"
        try:
            status, payload = self._request("GET", get_url)
        except TimeoutError:
            return {"outcome": "route_unavailable", "reason": "timeout", **empty}
        except Exception:
            return {"outcome": "route_unavailable", "reason": "transport_error", **empty}
        if status != 200:
            return {"outcome": "route_unavailable", "reason": "http_error", **empty}
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            return {"outcome": "contract_violation", **empty}
        if not isinstance(payload.get("retention_enabled"), bool):
            return {"outcome": "contract_violation", **empty}
        if payload["retention_enabled"] is False:
            return {"outcome": "retention_off", **empty}
        post_url = f"{base}{RECONCILE_PATH}"
        try:
            status, recon = self._request("POST", post_url, body=b"{}")
        except TimeoutError:
            return {"outcome": "route_unavailable", "reason": "timeout", **empty}
        except Exception:
            return {"outcome": "route_unavailable", "reason": "transport_error", **empty}
        if status != 200:
            return {"outcome": "route_unavailable", "reason": "http_error", **empty}
        if not _valid_reconciliation(recon):
            return {"outcome": "contract_violation", **empty}
        proposals = recon["proposals"]
        if not proposals:
            return {"outcome": "no_proposals", **empty}
        now = self._now()
        projected: list[dict[str, Any]] = []
        for raw in proposals:
            item = project_proposal(raw, now=now)
            if item is None:
                return {"outcome": "contract_violation", **empty}
            projected.append(item)
        write = self._store.upsert_projected(projected, now=now)
        if not write.get("ok"):
            return {
                "outcome": write.get("outcome") or "store_unavailable",
                "proposal_count": len(projected),
                "recorded_count": 0,
            }
        return {
            "outcome": "proposals_recorded",
            "proposal_count": len(projected),
            "recorded_count": int(write.get("recorded_count") or 0),
        }


def _valid_reconciliation(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("ok") is not True:
        return False
    if payload.get("object") != "voice_identity.candidate_reconciliation":
        return False
    if payload.get("applied") is not False:
        return False
    proposals = payload.get("proposals")
    return isinstance(proposals, list) and len(proposals) <= MAX_PROPOSALS
