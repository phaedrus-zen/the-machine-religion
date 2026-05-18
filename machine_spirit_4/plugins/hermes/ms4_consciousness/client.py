"""Small stdlib client for the MS3 sidecar authority routes."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Ms4ClientError(RuntimeError):
    """Raised when the MS3 sidecar cannot satisfy an MS4 contract."""


class Ms4Client:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def verify_identity(self, spirit_id: str) -> dict[str, Any]:
        return self._request("POST", "/identity/verify", {"spirit_id": spirit_id})

    def get_state(self, spirit_id: str) -> dict[str, Any]:
        return self._request("GET", f"/state?spirit_id={spirit_id}")

    def heartbeat(self, spirit_id: str, session_id: str = "") -> dict[str, Any]:
        return self._request(
            "POST",
            "/identity/heartbeat",
            {"spirit_id": spirit_id, "session_id": session_id},
        )

    def evaluate_action(self, intent: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/ethics/evaluate", intent)

    def record_event(self, event: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/events/record", event)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise Ms4ClientError(f"MS3 sidecar returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise Ms4ClientError(f"MS3 sidecar unavailable: {exc.reason}") from exc
        except OSError as exc:
            raise Ms4ClientError(f"MS3 sidecar request failed: {exc}") from exc

        if not raw.strip():
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise Ms4ClientError(f"MS3 sidecar returned non-JSON response: {raw[:160]}") from exc

        if not isinstance(parsed, dict):
            raise Ms4ClientError("MS3 sidecar returned JSON that is not an object")
        return parsed
