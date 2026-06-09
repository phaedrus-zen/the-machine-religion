"""Registry of imported 3rd-party MCP servers + their converted tools.

Persisted to ``machine_spirit_4/runtime/mcp_imports.json`` (override with
``MS4_MCP_IMPORTS_PATH``). The registry holds *config + converted tool
metadata* only — live connections (stdio subprocesses) live in the
proxy's connection cache, not here.

Secret hygiene: server ``env`` holds only non-secret literals; secrets
are referenced by name via ``env_passthrough`` (a list of env-var names
read from MS4's own process environment at launch time and never
written to disk).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .convert import convert_tools, sanitize_segment
from .upstream_client import Upstream, UpstreamConfig, UpstreamError, connect


log = logging.getLogger("ms4.mcp_bridge.registry")

REGISTRY_SCHEMA = "Ms4McpImports.v1"
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "mcp_imports.json"
_WRITE_LOCK = threading.Lock()


def registry_path() -> Path:
    return Path(os.environ.get("MS4_MCP_IMPORTS_PATH", str(_DEFAULT_PATH)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _audit(event_type: str, data: dict[str, Any]) -> None:
    try:
        from machine_spirit_4.gateway.audit import append_event

        append_event(event_type, data)
    except Exception as exc:  # noqa: BLE001
        log.debug("mcp_bridge audit %s failed: %s", event_type, exc)


@dataclass
class ImportedServer:
    server_id: str
    transport: str  # "stdio" | "http"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    env_passthrough: list[str] = field(default_factory=list)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    allow_inline: bool = False
    enabled: bool = True
    # Populated by refresh():
    tools: list[dict[str, Any]] = field(default_factory=list)
    last_refresh: str | None = None
    last_error: str | None = None

    def to_upstream_config(self) -> UpstreamConfig:
        """Resolve secrets (``env_passthrough`` -> values from MS4's env)
        only at launch time; never persisted."""
        env = dict(self.env or {})
        for name in self.env_passthrough or []:
            value = os.environ.get(name)
            if value is not None:
                env[name] = value
        return UpstreamConfig(
            server_id=self.server_id,
            transport=self.transport,
            command=self.command,
            args=list(self.args or []),
            env=env,
            cwd=self.cwd,
            url=self.url,
            headers=dict(self.headers or {}),
        )

    def to_dict(self, *, include_tools: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "server_id": self.server_id,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args or []),
            "env": dict(self.env or {}),
            "env_passthrough": list(self.env_passthrough or []),
            "cwd": self.cwd,
            "url": self.url,
            "headers": dict(self.headers or {}),
            "allow_inline": bool(self.allow_inline),
            "enabled": bool(self.enabled),
            "last_refresh": self.last_refresh,
            "last_error": self.last_error,
            "tool_count": len(self.tools),
        }
        if include_tools:
            d["tools"] = list(self.tools)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ImportedServer":
        return cls(
            server_id=sanitize_segment(str(raw.get("server_id") or "")),
            transport=str(raw.get("transport") or "stdio"),
            command=(str(raw["command"]) if raw.get("command") else None),
            args=[str(a) for a in (raw.get("args") or []) if isinstance(a, (str, int, float))],
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            env_passthrough=[str(n) for n in (raw.get("env_passthrough") or []) if isinstance(n, str)],
            cwd=(str(raw["cwd"]) if raw.get("cwd") else None),
            url=(str(raw["url"]) if raw.get("url") else None),
            headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()},
            allow_inline=bool(raw.get("allow_inline", False)),
            enabled=bool(raw.get("enabled", True)),
            tools=[t for t in (raw.get("tools") or []) if isinstance(t, dict)],
            last_refresh=(str(raw["last_refresh"]) if raw.get("last_refresh") else None),
            last_error=(str(raw["last_error"]) if raw.get("last_error") else None),
        )


# Injectable so tests can supply a fake upstream without a real server.
ConnectFn = Callable[[UpstreamConfig], Upstream]


class Registry:
    def __init__(self, servers: list[ImportedServer], *, path: Path | None = None) -> None:
        self._servers: dict[str, ImportedServer] = {s.server_id: s for s in servers}
        self._path = path or registry_path()

    # -- queries --

    def list_servers(self) -> list[ImportedServer]:
        return list(self._servers.values())

    def get(self, server_id: str) -> ImportedServer | None:
        return self._servers.get(sanitize_segment(server_id))

    def all_tools(self) -> list[dict[str, Any]]:
        """Flattened converted tool records across all ENABLED servers."""
        out: list[dict[str, Any]] = []
        for server in self._servers.values():
            if server.enabled:
                out.extend(server.tools)
        return out

    def server_for_tool(self, tool_id: str) -> ImportedServer | None:
        for server in self._servers.values():
            if any(t.get("name") == tool_id for t in server.tools):
                return server
        return None

    # -- mutations --

    def register(
        self,
        server: ImportedServer,
        *,
        connect_fn: ConnectFn | None = None,
        refresh: bool = True,
    ) -> ImportedServer:
        server.server_id = sanitize_segment(server.server_id)
        if not server.server_id:
            raise UpstreamError("server_id is required")
        self._servers[server.server_id] = server
        if refresh:
            self.refresh(server.server_id, connect_fn=connect_fn, persist=False)
        self.save()
        _audit("mcp_import_registered", {
            "server_id": server.server_id,
            "transport": server.transport,
            "allow_inline": server.allow_inline,
            "tool_count": len(server.tools),
            "error": server.last_error,
        })
        return server

    def refresh(self, server_id: str, *, connect_fn: ConnectFn | None = None, persist: bool = True) -> ImportedServer:
        server = self.get(server_id)
        if server is None:
            raise UpstreamError(f"unknown server: {server_id!r}")
        # Resolve ``connect`` at call time (not as a default arg) so a
        # monkeypatched ``registry.connect`` takes effect and tests
        # never launch a real subprocess.
        cf = connect_fn if connect_fn is not None else connect
        upstream: Upstream | None = None
        try:
            upstream = cf(server.to_upstream_config())
            raw_tools = upstream.tools_list()
            server.tools = convert_tools(server.server_id, raw_tools, allow_inline=server.allow_inline)
            server.last_refresh = _now()
            server.last_error = None
        except Exception as exc:  # noqa: BLE001 — capture, never raise into the gateway
            server.last_error = f"{type(exc).__name__}: {exc}"
            server.last_refresh = _now()
            log.info("mcp_bridge refresh %s failed: %s", server.server_id, server.last_error)
        finally:
            if upstream is not None:
                try:
                    upstream.stop()
                except Exception:  # pragma: no cover
                    pass
        if persist:
            self.save()
            _audit("mcp_import_refreshed", {
                "server_id": server.server_id,
                "tool_count": len(server.tools),
                "error": server.last_error,
            })
        return server

    def remove(self, server_id: str) -> bool:
        sid = sanitize_segment(server_id)
        existed = self._servers.pop(sid, None) is not None
        if existed:
            self.save()
            _audit("mcp_import_removed", {"server_id": sid})
        return existed

    # -- persistence --

    def save(self) -> None:
        with _WRITE_LOCK:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": REGISTRY_SCHEMA,
                "updated_at": _now(),
                "servers": [s.to_dict(include_tools=True) for s in self._servers.values()],
            }
            import json

            self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REGISTRY_SCHEMA,
            "servers": [s.to_dict(include_tools=False) for s in self._servers.values()],
            "tool_count": len(self.all_tools()),
        }


def load(path: Path | None = None) -> Registry:
    """Load the registry from disk; fail-soft to an empty registry."""
    p = path or registry_path()
    if not p.exists():
        return Registry([], path=p)
    try:
        import json

        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("mcp_bridge registry unreadable at %s: %s", p, exc)
        return Registry([], path=p)
    raw_servers = data.get("servers") if isinstance(data, dict) else None
    servers = [ImportedServer.from_dict(s) for s in raw_servers if isinstance(s, dict)] if isinstance(raw_servers, list) else []
    return Registry(servers, path=p)


def imported_tool_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Convenience for the Quartermaster catalog / MS4 MCP server: the
    flattened converted tool records from the persisted registry."""
    return load(path).all_tools()
