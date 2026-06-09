"""Proxy executor: run an imported tool by routing to its upstream MCP
server.

Resolves ``ext.<server>.<tool>@v1`` -> the owning :class:`ImportedServer`
and its upstream tool name, calls ``tools/call`` on a cached connection
(so a stdio subprocess is reused across calls, not relaunched), and
returns the upstream result. Fail-soft: transport errors return an
``{"ok": False, "error": ...}`` envelope (with one reconnect retry)
rather than raising into the gateway.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

from .registry import ImportedServer, Registry, load
from .upstream_client import Upstream, UpstreamConfig, connect


log = logging.getLogger("ms4.mcp_bridge.proxy")

ConnectFn = Callable[[UpstreamConfig], Upstream]
LoadFn = Callable[[], Registry]


def _unwrap(raw: Any) -> Any:
    """Unwrap an MCP tools/call result to the tool's JSON payload.
    Returns the parsed first text content, or ``{"raw_text": ...}`` for
    non-JSON text, or the raw result dict when there's no content
    wrapper, or ``None``."""
    if not isinstance(raw, dict):
        return None
    content = raw.get("content")
    if isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text = chunk.get("text") or ""
                if not str(text).strip():
                    continue
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw_text": text}
    if "content" not in raw:
        return raw
    return None


class BridgeProxy:
    def __init__(self, *, connect_fn: ConnectFn = connect, load_fn: LoadFn = load) -> None:
        self._connect_fn = connect_fn
        self._load_fn = load_fn
        self._conns: dict[str, Upstream] = {}
        self._lock = threading.Lock()

    def _get_conn(self, server: ImportedServer) -> Upstream:
        with self._lock:
            conn = self._conns.get(server.server_id)
            if conn is None:
                conn = self._connect_fn(server.to_upstream_config())
                self._conns[server.server_id] = conn
            return conn

    def _drop_conn(self, server_id: str) -> None:
        with self._lock:
            conn = self._conns.pop(server_id, None)
        if conn is not None:
            try:
                conn.stop()
            except Exception:  # pragma: no cover - best-effort
                pass

    def call(
        self,
        tool_id: str,
        arguments: dict[str, Any] | None = None,
        *,
        registry: Registry | None = None,
    ) -> dict[str, Any]:
        reg = registry or self._load_fn()
        server = reg.server_for_tool(tool_id)
        if server is None:
            return {"ok": False, "tool": tool_id, "error": f"unknown imported tool: {tool_id}"}
        if not server.enabled:
            return {"ok": False, "tool": tool_id, "error": f"server {server.server_id!r} is disabled"}

        upstream_name = None
        for record in server.tools:
            if record.get("name") == tool_id:
                upstream_name = record.get("upstream_name")
                break
        if not upstream_name:
            return {"ok": False, "tool": tool_id, "error": f"no upstream mapping for {tool_id}"}

        # First attempt; on transport failure, drop the (possibly dead)
        # connection and retry once with a fresh one.
        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                conn = self._get_conn(server)
                raw = conn.tools_call(upstream_name, arguments or {})
                return {
                    "ok": True,
                    "tool": tool_id,
                    "server_id": server.server_id,
                    "is_error": bool(raw.get("isError")) if isinstance(raw, dict) else False,
                    "result": _unwrap(raw),
                    "content": raw.get("content") if isinstance(raw, dict) else None,
                }
            except Exception as exc:  # noqa: BLE001 — fail-soft envelope
                last_error = exc
                self._drop_conn(server.server_id)
                log.info("mcp_bridge proxy %s attempt %d failed: %s", tool_id, attempt, exc)
        return {
            "ok": False,
            "tool": tool_id,
            "server_id": server.server_id,
            "error": f"{type(last_error).__name__}: {last_error}",
        }

    def shutdown_all(self) -> None:
        with self._lock:
            conns = list(self._conns.values())
            self._conns.clear()
        for conn in conns:
            try:
                conn.stop()
            except Exception:  # pragma: no cover
                pass


_DEFAULT_PROXY: BridgeProxy | None = None
_DEFAULT_LOCK = threading.Lock()


def default_proxy() -> BridgeProxy:
    global _DEFAULT_PROXY
    with _DEFAULT_LOCK:
        if _DEFAULT_PROXY is None:
            _DEFAULT_PROXY = BridgeProxy()
        return _DEFAULT_PROXY


def call_imported_tool(tool_id: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convenience wrapper over the process-wide :class:`BridgeProxy`."""
    return default_proxy().call(tool_id, arguments)


def _reset_default_proxy_for_tests(proxy: BridgeProxy | None = None) -> None:
    global _DEFAULT_PROXY
    with _DEFAULT_LOCK:
        if _DEFAULT_PROXY is not None:
            try:
                _DEFAULT_PROXY.shutdown_all()
            except Exception:
                pass
        _DEFAULT_PROXY = proxy
