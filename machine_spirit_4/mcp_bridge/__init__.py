"""MS4 MCP importer/bridge.

Ingests 3rd-party MCP servers (stdio or HTTP/SSE), converts their
``tools/list`` into the MS4 / Quartermaster catalog schema, persists a
registry, and proxies ``tools/call`` back to the upstream server.

Imported tools become first-class in MS4's MCP server and the
Quartermaster catalog, but are fail-closed: an imported tool is never
inline-eligible unless the upstream declares ``readOnlyHint`` AND an
operator opts the server in (``allow_inline``). Everything else routes
through the Depth Lobe + MS3 ethics per call.
"""

from __future__ import annotations

from .convert import (
    IMPORTED_TOOL_SCHEMA,
    KIND_READ_ONLY,
    KIND_RUNTIME_ACTION,
    convert_tools,
    imported_tool_id,
    kind_for_tool,
    sanitize_segment,
    upstream_tool_to_record,
)
from .proxy import (
    BridgeProxy,
    call_imported_tool,
    default_proxy,
)
from .registry import (
    ImportedServer,
    Registry,
    imported_tool_records,
    load,
    registry_path,
)
from .upstream_client import (
    CLIENT_INFO,
    MCP_PROTOCOL_VERSION,
    HttpUpstream,
    StdioUpstream,
    Upstream,
    UpstreamConfig,
    UpstreamError,
    connect,
)

__all__ = [
    "CLIENT_INFO",
    "IMPORTED_TOOL_SCHEMA",
    "KIND_READ_ONLY",
    "KIND_RUNTIME_ACTION",
    "MCP_PROTOCOL_VERSION",
    "BridgeProxy",
    "HttpUpstream",
    "ImportedServer",
    "Registry",
    "StdioUpstream",
    "Upstream",
    "UpstreamConfig",
    "UpstreamError",
    "call_imported_tool",
    "connect",
    "convert_tools",
    "default_proxy",
    "imported_tool_id",
    "imported_tool_records",
    "kind_for_tool",
    "load",
    "registry_path",
    "sanitize_segment",
    "upstream_tool_to_record",
]
