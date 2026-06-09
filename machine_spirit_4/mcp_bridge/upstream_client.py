"""Stdlib MCP client for talking *out* to 3rd-party MCP servers.

MS4 ships with a deliberately tiny dependency footprint (``requests`` +
``jsonschema``, no ``mcp`` SDK) and uses ``urllib`` everywhere, so this
client is built on the stdlib:

* :class:`StdioUpstream` launches a server as a subprocess and speaks
  newline-delimited JSON-RPC 2.0 over stdin/stdout (the MCP stdio
  transport), with the ``initialize`` -> ``notifications/initialized``
  handshake and a background reader thread.
* :class:`HttpUpstream` POSTs JSON-RPC to a Streamable-HTTP endpoint via
  ``urllib``, tolerating both ``application/json`` and ``text/event-stream``
  responses and carrying the ``Mcp-Session-Id`` header.

Both expose the same three operations the importer needs:
``initialize`` / ``tools_list`` / ``tools_call`` (+ ``health`` /
``stop``). Errors raise :class:`UpstreamError`; callers fail-soft.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any


log = logging.getLogger("ms4.mcp_bridge.upstream")


# Match the protocol version MS4's own MCP server advertises
# (``machine_spirit_4/mcp/server.py``) so handshakes are consistent.
MCP_PROTOCOL_VERSION = "2025-11-25"
CLIENT_INFO = {"name": "ms4-mcp-bridge", "version": "0.1.0"}

TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "http"


class UpstreamError(RuntimeError):
    """Any failure talking to an upstream MCP server."""


@dataclass
class UpstreamConfig:
    server_id: str
    transport: str  # "stdio" | "http"
    # --- stdio ---
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    # --- http ---
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # --- common ---
    init_timeout: float = 20.0
    request_timeout: float = 30.0

    def validate(self) -> None:
        if self.transport == TRANSPORT_STDIO:
            if not self.command:
                raise UpstreamError("stdio upstream requires a 'command'")
        elif self.transport == TRANSPORT_HTTP:
            if not self.url:
                raise UpstreamError("http upstream requires a 'url'")
        else:
            raise UpstreamError(f"unknown transport: {self.transport!r}")


class Upstream:
    """Common interface for both transports."""

    config: UpstreamConfig

    def initialize(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def tools_list(self) -> list[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def health(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


class StdioUpstream(Upstream):
    def __init__(self, config: UpstreamConfig) -> None:
        config.validate()
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._write_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._initialized = False

    # -- process lifecycle --

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            merged_env = dict(os.environ)
            merged_env.update(self.config.env or {})
            try:
                self._proc = subprocess.Popen(
                    [self.config.command, *self.config.args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    env=merged_env,
                    cwd=self.config.cwd or None,
                )
            except (OSError, ValueError) as exc:
                raise UpstreamError(
                    f"failed to launch stdio upstream {self.config.server_id!r}: {exc}"
                ) from exc
            self._initialized = False
            threading.Thread(target=self._read_loop, args=(self._proc,), daemon=True).start()
            threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()

    def _read_loop(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("upstream %s emitted non-JSON: %s", self.config.server_id, line[:200])
                    continue
                if not isinstance(msg, dict):
                    continue
                mid = msg.get("id")
                if mid is None:
                    continue  # server-initiated request/notification — ignored
                with self._pending_lock:
                    waiter = self._pending.get(mid)
                if waiter is not None:
                    try:
                        waiter.put_nowait(msg)
                    except queue.Full:  # pragma: no cover - defensive
                        pass
        except Exception as exc:  # pragma: no cover - thread cleanup
            log.debug("upstream %s read loop ended: %s", self.config.server_id, exc)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stderr is not None
            for _line in proc.stderr:
                pass
        except Exception:  # pragma: no cover
            pass

    # -- request/response --

    def _request(self, method: str, params: dict[str, Any] | None, *, timeout: float) -> dict[str, Any]:
        self._ensure_started()
        mid = uuid.uuid4().hex
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[mid] = waiter
        envelope = {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}
        line = json.dumps(envelope) + "\n"
        try:
            with self._write_lock:
                proc = self._proc
                if proc is None or proc.poll() is not None or proc.stdin is None:
                    raise UpstreamError(f"upstream {self.config.server_id!r} is not running")
                proc.stdin.write(line)
                proc.stdin.flush()
            try:
                resp = waiter.get(timeout=timeout)
            except queue.Empty:
                raise UpstreamError(f"timeout after {timeout}s awaiting {method} from {self.config.server_id!r}")
        finally:
            with self._pending_lock:
                self._pending.pop(mid, None)
        if resp.get("error"):
            raise UpstreamError(f"{method} returned error: {resp['error']}")
        result = resp.get("result")
        return result if isinstance(result, dict) else {}

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        envelope = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        line = json.dumps(envelope) + "\n"
        with self._write_lock:
            proc = self._proc
            if proc is not None and proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(line)
                proc.stdin.flush()

    # -- MCP operations --

    def initialize(self) -> dict[str, Any]:
        result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
            timeout=self.config.init_timeout,
        )
        self._notify("notifications/initialized")
        self._initialized = True
        return result

    def tools_list(self) -> list[dict[str, Any]]:
        if not self._initialized:
            self.initialize()
        result = self._request("tools/list", {}, timeout=self.config.request_timeout)
        tools = result.get("tools")
        return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []

    def tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()
        return self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=self.config.request_timeout,
        )

    def health(self) -> dict[str, Any]:
        running = bool(self._proc is not None and self._proc.poll() is None)
        return {
            "transport": TRANSPORT_STDIO,
            "server_id": self.config.server_id,
            "running": running,
            "initialized": self._initialized,
        }

    def stop(self) -> None:
        self._initialized = False
        with self._write_lock:
            proc = self._proc
            if proc is None:
                return
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:  # pragma: no cover - best-effort teardown
                pass


# ---------------------------------------------------------------------------
# Streamable HTTP transport
# ---------------------------------------------------------------------------


class HttpUpstream(Upstream):
    def __init__(self, config: UpstreamConfig) -> None:
        config.validate()
        self.config = config
        self._session_id: str | None = None
        self._initialized = False

    def _post(self, method: str, params: dict[str, Any] | None, *, timeout: float, notification: bool = False) -> dict[str, Any]:
        envelope: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notification:
            envelope["id"] = uuid.uuid4().hex
        body = json.dumps(envelope).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(self.config.headers or {})
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(self.config.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                sid = response.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid
                raw = response.read().decode("utf-8", "replace")
                ctype = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise UpstreamError(f"{method} HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise UpstreamError(f"{method} unreachable: {exc.reason}") from exc
        except OSError as exc:
            raise UpstreamError(f"{method} request failed: {exc}") from exc

        if notification:
            return {}
        parsed = _parse_http_body(raw, ctype)
        if not isinstance(parsed, dict):
            raise UpstreamError(f"{method}: non-JSON response")
        if parsed.get("error"):
            raise UpstreamError(f"{method} returned error: {parsed['error']}")
        result = parsed.get("result")
        return result if isinstance(result, dict) else {}

    def initialize(self) -> dict[str, Any]:
        result = self._post(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
            timeout=self.config.init_timeout,
        )
        try:
            self._post("notifications/initialized", {}, timeout=self.config.init_timeout, notification=True)
        except UpstreamError:
            # Some servers don't accept the notification over HTTP; the
            # session is still usable after initialize.
            pass
        self._initialized = True
        return result

    def tools_list(self) -> list[dict[str, Any]]:
        if not self._initialized:
            self.initialize()
        result = self._post("tools/list", {}, timeout=self.config.request_timeout)
        tools = result.get("tools")
        return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []

    def tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()
        return self._post(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=self.config.request_timeout,
        )

    def health(self) -> dict[str, Any]:
        return {
            "transport": TRANSPORT_HTTP,
            "server_id": self.config.server_id,
            "url": self.config.url,
            "initialized": self._initialized,
            "session_id": self._session_id,
        }

    def stop(self) -> None:
        self._initialized = False
        self._session_id = None


def _parse_http_body(raw: str, content_type: str) -> Any:
    """Parse a Streamable-HTTP response body. Handles plain JSON and
    ``text/event-stream`` (SSE) framing — for SSE we return the JSON
    payload of the last ``data:`` event that parses."""
    raw = raw.strip()
    if not raw:
        return {}
    if "text/event-stream" in content_type.lower() or raw.startswith("event:") or raw.startswith("data:"):
        last: Any = None
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                last = json.loads(data)
            except json.JSONDecodeError:
                continue
        return last
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def connect(config: UpstreamConfig) -> Upstream:
    """Build (but do not yet start/handshake) an upstream client for the
    config's transport. The handshake happens lazily on first
    ``tools_list`` / ``tools_call`` (or an explicit ``initialize``)."""
    config.validate()
    if config.transport == TRANSPORT_STDIO:
        return StdioUpstream(config)
    if config.transport == TRANSPORT_HTTP:
        return HttpUpstream(config)
    raise UpstreamError(f"unknown transport: {config.transport!r}")
