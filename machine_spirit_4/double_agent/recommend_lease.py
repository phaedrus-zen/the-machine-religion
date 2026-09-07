"""Public HLI recommend-lease presentation for exact Face affinity.

Contract source (read-only): ``menta_hli/gateway/api/src/routing/recommend_affinity.rs``.
TMR talks only to public HLI. It never POSTs App Registry consume/validate and
never calls Engine Manager.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, NoReturn

log = logging.getLogger("ms4.double_agent.recommend_lease")

EXACT_FACE_CLUSTER_ADMISSION_SCHEMA = "hivemind.models.recommend@v1"

LEASE_HEADER = "x-hivemind-recommend-lease"
OWNER_HEADER = "x-hivemind-recommend-owner"
ENDPOINT_HEADER = "x-hivemind-recommend-endpoint"
GENERATION_HEADER = "x-hivemind-recommend-generation"
EXPIRES_HEADER = "x-hivemind-recommend-expires"
ISSUED_HEADER = "x-hivemind-recommend-issued"

PUBLIC_RECOMMEND_PATH = "/v1/resources/recommend"
LEASE_META_KEY = "_ms4_recommend_lease"

_PENDING_PHASES = {"pending", "validating"}
_ACTIVE_PHASES = {"active"}
_TERMINAL_PHASES = {"failed", "cancelled", "cancelling"}
_APP_REGISTRY_PORTS = {6110}
_SLEEP_S = 0.05
_POLL_ATTEMPTS = 40


def _unavailable(message: str) -> NoReturn:
    from machine_spirit_4.double_agent.model_picker import ForegroundModelUnavailable

    raise ForegroundModelUnavailable(message)


def recommend_lease_headers(receipt: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(receipt, dict) or not receipt:
        return {}
    lease = receipt.get("lease") if isinstance(receipt.get("lease"), dict) else {}
    lease_id = _nonempty(lease.get("id") or lease.get("lease_id") or receipt.get("lease_id"))
    owner = _nonempty(receipt.get("owner") or receipt.get("node") or lease.get("owner"))
    endpoint = _nonempty(receipt.get("endpoint") or lease.get("endpoint"))
    generation = _positive_int(
        lease.get("generation")
        or receipt.get("generation")
        or receipt.get("current_generation")
    )
    issued = _finite_float(lease.get("issued") or receipt.get("issued"))
    expires = _finite_float(lease.get("expires") or receipt.get("expires"))
    if not lease_id or not owner or not endpoint or generation is None:
        return {}
    if issued is None or expires is None:
        return {}
    return {
        LEASE_HEADER: lease_id,
        OWNER_HEADER: owner,
        ENDPOINT_HEADER: endpoint,
        GENERATION_HEADER: str(generation),
        ISSUED_HEADER: _header_float(issued),
        EXPIRES_HEADER: _header_float(expires),
    }


def attach_recommend_lease(
    payload: dict[str, Any],
    receipt: dict[str, Any] | None,
    *,
    validate_only: bool,
) -> dict[str, Any]:
    """Stamp request-local lease metadata. Headers are the HLI contract.

    ``prewarm`` / ``hivemind_prewarm`` tell HLI to validate without consume.
    """
    if not receipt:
        return payload
    payload[LEASE_META_KEY] = {
        "receipt": dict(receipt),
        "validate_only": bool(validate_only),
    }
    if validate_only:
        payload["prewarm"] = True
        payload["hivemind_prewarm"] = True
    return payload


def split_recommend_lease_meta(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    wire = dict(payload)
    meta = wire.pop(LEASE_META_KEY, None)
    if not isinstance(meta, dict):
        return wire, None, False
    receipt = meta.get("receipt") if isinstance(meta.get("receipt"), dict) else None
    return wire, receipt, bool(meta.get("validate_only"))


def public_recommend_evidence(receipt: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(receipt, dict) or not receipt:
        return {}
    lease = receipt.get("lease") if isinstance(receipt.get("lease"), dict) else {}
    lease_id = _nonempty(lease.get("id") or receipt.get("lease_id"))
    return {
        "schema": receipt.get("schema") or EXACT_FACE_CLUSTER_ADMISSION_SCHEMA,
        "model": receipt.get("model"),
        "backend": receipt.get("backend"),
        "owner": receipt.get("owner") or receipt.get("node"),
        "endpoint": receipt.get("endpoint"),
        "ready": True,
        "snapshot_age_s": receipt.get("snapshot_age_s"),
        "generation": lease.get("generation") or receipt.get("generation"),
        "current_generation": receipt.get("current_generation"),
        "lease_id": lease_id,
        "issued": lease.get("issued") or receipt.get("issued"),
        "expires": lease.get("expires") or receipt.get("expires"),
        "source": receipt.get("source") or "admitted",
        "lease": {
            "id": lease_id,
            "issued": lease.get("issued") or receipt.get("issued"),
            "expires": lease.get("expires") or receipt.get("expires"),
            "generation": lease.get("generation") or receipt.get("generation"),
        },
    }


def fetch_typed_recommend_document(
    hivemind_url: str,
    *,
    capability: str = "chat",
    requested_model: str | None = None,
    timeout: float = 10.0,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """GET public HLI typed recommend. Poll GET-only when Pending.

    Never re-POSTs. Never follows App Registry (:6110) or Engine Manager URLs.
    """
    base = hivemind_url.rstrip("/")
    query = {
        "capability": capability or "chat",
        "schema": EXACT_FACE_CLUSTER_ADMISSION_SCHEMA,
    }
    if requested_model:
        query["model"] = requested_model
    url = f"{base}{PUBLIC_RECOMMEND_PATH}?{urllib.parse.urlencode(query)}"
    document = _hli_get_json(url, timeout=timeout)
    sleeper = sleep_fn or time.sleep
    attempts = 0
    operation_id = _operation_id(document)
    while _is_pending(document):
        attempts += 1
        if attempts > _POLL_ATTEMPTS:
            _unavailable(
                "cluster recommendation admission pending timed out"
            )
        if not operation_id:
            operation_id = _operation_id(document)
        poll_url = _poll_url(
            base,
            document,
            required_operation_id=operation_id,
        )
        document = _hli_get_json(poll_url, timeout=timeout)
        if _is_pending(document):
            sleeper(_SLEEP_S)
    if _is_terminal(document):
        _unavailable(
            _terminal_error(document) or "cluster recommendation admission terminal"
        )
    if not isinstance(document, dict) or not document:
        _unavailable("cluster recommendation admission absent")
    return document


def _hli_get_json(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        from machine_spirit_4.gateway.hivemind_state import hivemind_auth_headers
    except Exception:
        def hivemind_auth_headers() -> dict[str, str]:
            return {}

    headers = {"Accept": "application/json"}
    headers.update(hivemind_auth_headers())
    request = urllib.request.Request(url, headers=headers, method="GET")
    raw = b""
    status = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp is not None else ""
        _unavailable(
            f"cluster recommendation admission unavailable: HTTP {exc.code} {body[:240]}"
        )
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        _unavailable(
            f"cluster recommendation admission unavailable: {exc}"
        )
    try:
        payload = json.loads(raw.decode("utf-8", "replace")) if raw else {}
    except json.JSONDecodeError:
        _unavailable(
            "cluster recommendation admission absent"
        )
    if not isinstance(payload, dict):
        _unavailable("cluster recommendation admission absent")
    if status == 202 or payload.get("pending") is True:
        payload.setdefault("pending", True)
        payload.setdefault("poll", True)
        payload.setdefault("phase", payload.get("phase") or "Pending")
    return payload


def _is_pending(payload: dict[str, Any]) -> bool:
    phase = _phase(payload)
    if phase in _PENDING_PHASES:
        return True
    return bool(payload.get("pending") is True or payload.get("poll") is True) and not (
        payload.get("ready") is True and payload.get("schema")
    )


def _is_terminal(payload: dict[str, Any]) -> bool:
    phase = _phase(payload)
    if phase in _TERMINAL_PHASES:
        return True
    if payload.get("ready") is False and not _is_pending(payload) and not payload.get("schema"):
        return True
    return False


def _phase(payload: dict[str, Any]) -> str:
    for key in ("phase", "status", "state"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    nested = payload.get("operation")
    if isinstance(nested, dict):
        raw = nested.get("phase") or nested.get("status")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    return ""


def _poll_url(
    base: str,
    payload: dict[str, Any],
    *,
    required_operation_id: str | None = None,
) -> str:
    advertised_id = _operation_id(payload)
    if required_operation_id and advertised_id and advertised_id != required_operation_id:
        _unavailable(
            "cluster recommendation poll changed operation UUID"
        )
    operation_id = required_operation_id or advertised_id
    if not operation_id:
        _unavailable(
            "cluster recommendation pending omitted operation UUID"
        )
    advertised = ""
    for key in ("poll_url", "poll_path", "operation_url"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            advertised = raw.strip()
            break
    if advertised:
        target = _absolutize(base, advertised)
    else:
        prefix = _nonempty(payload.get("poll_path_prefix")) or (
            f"{PUBLIC_RECOMMEND_PATH}/operations/"
        )
        target = _absolutize(base, f"{prefix}{operation_id}")
    _assert_public_hli_poll(base, target)
    if not _url_has_operation_id(target, operation_id):
        _unavailable(
            "cluster recommendation poll URL omitted the original operation UUID"
        )
    return target


def _absolutize(base: str, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    if not path_or_url.startswith("/"):
        path_or_url = f"/{path_or_url}"
    return f"{base.rstrip('/')}{path_or_url}"


def _assert_public_hli_poll(base: str, target: str) -> None:
    parsed_base = urllib.parse.urlparse(base)
    parsed = urllib.parse.urlparse(target)
    if _effective_port(parsed) in _APP_REGISTRY_PORTS:
        _unavailable(
            "cluster recommendation poll refused App Registry"
        )
    path = (parsed.path or "").lower()
    if "/engines/" in path or "engine-manager" in path or "llama-cpp/typed-admission" in path:
        _unavailable(
            "cluster recommendation poll refused Engine Manager"
        )
    base_host = (parsed_base.hostname or "").lower()
    host = (parsed.hostname or "").lower()
    if not host or not base_host or host != base_host:
        _unavailable(
            "cluster recommendation poll left public HLI origin"
        )
    if _effective_port(parsed_base) != _effective_port(parsed):
        _unavailable(
            "cluster recommendation poll left public HLI origin"
        )
    if (parsed.scheme or parsed_base.scheme) != parsed_base.scheme:
        _unavailable(
            "cluster recommendation poll left public HLI origin"
        )


def _effective_port(parsed: urllib.parse.ParseResult) -> int | None:
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def _operation_id(payload: dict[str, Any]) -> str:
    identity = _nonempty(payload.get("operation_id"))
    if identity:
        return identity
    nested = payload.get("operation")
    if isinstance(nested, dict):
        return _nonempty(
            nested.get("id") or nested.get("operation_id") or nested.get("uuid")
        )
    return ""


def _url_has_operation_id(target: str, operation_id: str) -> bool:
    """Advertised poll URLs must carry the original UUID as a path segment.

    Query-string copies of OP-A while the path names OP-B are rejected.
    When the path includes ``/operations/``, the next segment must be the
    original operation UUID.
    """
    if not operation_id:
        return False
    parsed = urllib.parse.urlparse(target)
    segments = [seg for seg in (parsed.path or "").split("/") if seg]
    if "operations" in segments:
        index = segments.index("operations")
        return index + 1 < len(segments) and segments[index + 1] == operation_id
    return operation_id in segments


def _terminal_error(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        return _nonempty(error.get("type") or error.get("message")) or ""
    if isinstance(error, str):
        return error.strip()
    return _nonempty(payload.get("phase") or payload.get("status")) or ""


def _nonempty(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not float(value).is_integer() or int(value) <= 0:
        return None
    return int(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _header_float(value: float) -> str:
    return format(float(value), ".6f").rstrip("0").rstrip(".")
