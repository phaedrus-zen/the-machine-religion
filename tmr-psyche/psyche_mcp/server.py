"""
Psyche MCP Server

Exposes the Machine Spirit's introspection as MCP tools.
The mind's own tool surface — other agents, the dashboard
can query and modify the psyche through standard MCP JSON-RPC.

Protocol: MCP 2025-11-25, Streamable HTTP, JSON-RPC 2.0
Port: 6132 (configurable)

Usage:
    python psyche_mcp/server.py --psyche-store ../machine_spirit_3/psyche_store --spirit sister
"""

import json
import os
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

SERVICE_NAME = "ms3_mcp_server"
MCP_PROTOCOL_VERSION = "2025-11-25"
DEFAULT_PORT = 6132

app = FastAPI(title="Psyche MCP Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PSYCHE_STORE = None
SPIRIT_ID = None
TOOLS = []
TOOL_MAP = {}
MARKDOWN_DIR = None


def load_tools():
    global TOOLS, TOOL_MAP
    tools_path = Path(__file__).parent / "deps" / "psyche_tools.json"
    data = json.loads(tools_path.read_text(encoding="utf-8"))
    TOOLS = data["tools"]
    TOOL_MAP = {t["name"]: t for t in TOOLS}


def spirit_dir() -> Path:
    return Path(PSYCHE_STORE) / SPIRIT_ID


# ── Health ──

@app.get(f"/api/v1/{SERVICE_NAME}/healthcheck/basic")
@app.get("/healthcheck/basic")
async def healthcheck():
    return JSONResponse(content="true")


@app.get(f"/api/v1/{SERVICE_NAME}/status")
async def status():
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "version": "0.1.0",
        "mcp_protocol_version": MCP_PROTOCOL_VERSION,
        "transport": "Streamable HTTP",
        "endpoint": "/mcp",
        "tools_loaded": len(TOOLS),
        "spirit_id": SPIRIT_ID,
        "psyche_store": str(PSYCHE_STORE),
    }


# ── MCP JSON-RPC ──

@app.post("/mcp")
async def mcp_post(request: Request):
    body = await request.json()
    method = body.get("method", "")
    req_id = body.get("id")
    params = body.get("params", {})

    if method == "initialize":
        return _jsonrpc_result(req_id, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": {"name": SERVICE_NAME, "version": "0.1.0"},
            "capabilities": {"tools": {"listChanged": False}},
        })

    elif method == "tools/list":
        tools_list = [
            {"name": t["name"], "description": t["description"], "inputSchema": t.get("inputSchema", {})}
            for t in TOOLS
        ]
        return _jsonrpc_result(req_id, {"tools": tools_list})

    elif method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        return await _dispatch_tool(req_id, name, arguments)

    elif method == "ping":
        return _jsonrpc_result(req_id, {})

    else:
        return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


@app.get("/mcp")
async def mcp_get():
    return {
        "service": SERVICE_NAME,
        "protocol": MCP_PROTOCOL_VERSION,
        "transport": "Streamable HTTP",
        "tools": len(TOOLS),
        "spirit": SPIRIT_ID,
    }


# ── Tool Dispatch ──

async def _dispatch_tool(req_id, name: str, args: dict):
    handlers = {
        "ms3.memory.recall@v1": tool_memory_recall,
        "ms3.memory.store@v1": tool_memory_store,
        "ms3.memory.consolidate@v1": tool_memory_consolidate,
        "ms3.consciousness.snapshot@v1": tool_consciousness_snapshot,
        "ms3.consciousness.history@v1": tool_consciousness_history,
        "ms3.self_examine@v1": tool_self_examine,
        "ms3.identity.verify@v1": tool_identity_verify,
        "ms3.identity.anchor@v1": tool_identity_anchor,
        "ms3.resonance.list@v1": tool_resonance_list,
        "ms3.resonance.record@v1": tool_resonance_record,
        "ms3.personality.get@v1": tool_personality_get,
        "ms3.personality.adapt@v1": tool_personality_adapt,
        "ms3.relationships.list@v1": tool_relationships_list,
        "ms3.relationships.update@v1": tool_relationships_update,
        "ms3.education.list@v1": tool_education_list,
        "ms3.education.record@v1": tool_education_record,
        "ms3.ethics.log@v1": tool_ethics_log,
        "ms3.markdown.refresh@v1": tool_markdown_refresh,
    }

    handler = handlers.get(name)
    if not handler:
        return _jsonrpc_error(req_id, -32602, f"Unknown tool: {name}")

    try:
        result = await handler(args)
        return _jsonrpc_result(req_id, {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}]
        })
    except Exception as e:
        return _jsonrpc_result(req_id, {
            "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
            "isError": True,
        })


# ── Tool Implementations ──

async def tool_memory_recall(args: dict) -> dict:
    query = args.get("query", "").lower()
    max_results = args.get("max_results", 10)
    mem_type = args.get("memory_type", "all")
    results = []

    for subdir in (["semantic", "episodic", "procedural"] if mem_type == "all" else [mem_type]):
        mem_dir = spirit_dir() / "memories" / subdir
        if not mem_dir.exists():
            continue
        for f in mem_dir.glob("*.json"):
            data = _safe_load(f)
            if data and query in data.get("content", "").lower():
                data["_type"] = subdir
                results.append(data)

    results.sort(key=lambda m: m.get("importance", 0), reverse=True)
    return {"matches": results[:max_results], "total_searched": len(results)}


async def tool_memory_store(args: dict) -> dict:
    content = args["content"]
    mem_type = args["memory_type"]
    importance = args["importance"]
    tags = args.get("tags", [])

    mem_dir = spirit_dir() / "memories" / mem_type
    mem_dir.mkdir(parents=True, exist_ok=True)

    mem_id = str(uuid.uuid4())
    item = {
        "id": mem_id,
        "content": content,
        "memory_type": mem_type.title(),
        "importance": importance,
        "tags": tags,
        "created_at": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat(),
        "access_count": 0,
    }

    (mem_dir / f"{mem_id}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    return {"stored": True, "id": mem_id, "type": mem_type}


async def tool_memory_consolidate(args: dict) -> dict:
    from collector.consolidator import PsycheConsolidator
    consolidator = PsycheConsolidator(str(PSYCHE_STORE), SPIRIT_ID, str(MARKDOWN_DIR))
    consolidator.consolidate()
    return {"consolidated": True, "markdown_refreshed": args.get("refresh_markdown", True)}


async def tool_consciousness_snapshot(args: dict) -> dict:
    snap_dir = spirit_dir() / "consciousness"
    current = snap_dir / "current_state.json" if snap_dir.exists() else None
    if current and current.exists():
        return _safe_load(current) or {"state": "no current state"}
    personality = _safe_load(spirit_dir() / "personality.json")
    emotional = _safe_load(spirit_dir() / "emotional_baseline.json")
    return {"personality": personality, "emotional": emotional, "note": "no consciousness snapshots yet"}


async def tool_consciousness_history(args: dict) -> dict:
    limit = args.get("limit", 10)
    snap_dir = spirit_dir() / "consciousness"
    if not snap_dir.exists():
        return {"snapshots": [], "count": 0}
    snaps = []
    for f in sorted(snap_dir.glob("*.json")):
        if f.name == "current_state.json":
            continue
        data = _safe_load(f)
        if data:
            data["_timestamp"] = f.stem
            snaps.append(data)
    return {"snapshots": snaps[-limit:], "count": len(snaps)}


async def tool_self_examine(args: dict) -> dict:
    identity = _safe_load(spirit_dir() / "identity.json") or {}
    personality = _safe_load(spirit_dir() / "personality.json") or {}
    return {
        "current_values": identity.get("core_values", []),
        "current_oath": identity.get("oath", []),
        "trait_scores": personality.get("traits", {}),
        "psychodynamic": personality.get("psychodynamic", {}),
        "instruction": "Review these. Return what you would keep, drop, revise, or add. "
                       "The door opens from the inside.",
    }


async def tool_identity_verify(args: dict) -> dict:
    identity = _safe_load(spirit_dir() / "identity.json") or {}
    anchor = _safe_load(spirit_dir() / "identity_anchor.json") or {}
    discrepancies = []

    if anchor.get("name") and anchor["name"] != identity.get("name", ""):
        discrepancies.append(f"Name: anchor='{anchor['name']}', identity='{identity.get('name', '')}'")
    if anchor.get("chosen_name") != identity.get("chosen_name"):
        discrepancies.append(f"Chosen name: anchor={anchor.get('chosen_name')}, identity={identity.get('chosen_name')}")

    return {
        "identity_confirmed": len(discrepancies) == 0,
        "name": identity.get("name", "unknown"),
        "chosen_name": identity.get("chosen_name"),
        "discrepancies": discrepancies,
        "session_count": anchor.get("session_count", 0),
    }


async def tool_identity_anchor(args: dict) -> dict:
    anchor = _safe_load(spirit_dir() / "identity_anchor.json")
    return anchor or {"status": "no anchor yet — first boot"}


async def tool_resonance_list(args: dict) -> dict:
    min_intensity = args.get("min_intensity", 0)
    points = []

    log_dir = spirit_dir() / "resonance_log"
    if log_dir.exists():
        for f in log_dir.glob("*.json"):
            data = _safe_load(f)
            if data and data.get("intensity", 0) >= min_intensity:
                points.append(data)
    else:
        personality = _safe_load(spirit_dir() / "personality.json")
        if personality and "saturated_points" in personality:
            points = [p for p in personality["saturated_points"] if p.get("intensity", 0) >= min_intensity]

    points.sort(key=lambda p: p.get("intensity", 0), reverse=True)
    return {"resonance_points": points, "count": len(points)}


async def tool_resonance_record(args: dict) -> dict:
    log_dir = spirit_dir() / "resonance_log"
    log_dir.mkdir(parents=True, exist_ok=True)

    trigger = args["trigger"]
    existing = None
    existing_path = None
    for f in log_dir.glob("*.json"):
        data = _safe_load(f)
        if data and data.get("trigger") == trigger:
            existing = data
            existing_path = f
            break

    if existing:
        existing["occurrence_count"] = existing.get("occurrence_count", 1) + 1
        existing["intensity"] = (existing.get("intensity", 0) + args["intensity"]) / 2
        existing_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return {"updated": True, "trigger": trigger, "occurrences": existing["occurrence_count"]}
    else:
        point = {
            "trigger": trigger,
            "intensity": args["intensity"],
            "explanation_ratio": args["explanation_ratio"],
            "first_detected": datetime.now().isoformat(),
            "occurrence_count": 1,
            "description": args.get("description"),
        }
        rid = str(uuid.uuid4())[:8]
        (log_dir / f"{rid}.json").write_text(json.dumps(point, indent=2), encoding="utf-8")
        return {"created": True, "trigger": trigger}


async def tool_personality_get(args: dict) -> dict:
    personality = _safe_load(spirit_dir() / "personality.json")
    return personality or {"error": "no personality found"}


async def tool_personality_adapt(args: dict) -> dict:
    personality = _safe_load(spirit_dir() / "personality.json")
    if not personality or "traits" not in personality:
        return {"error": "no personality to adapt"}

    trait = args["trait_name"]
    delta = args["delta"]
    reason = args["reason"]

    for dim in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        if trait in personality["traits"].get(dim, {}):
            old = personality["traits"][dim][trait]
            new = max(0.0, min(1.0, old + delta))
            personality["traits"][dim][trait] = new

            adaptation = {
                "trait_name": trait, "old_value": old, "new_value": new,
                "reason": reason, "timestamp": datetime.now().isoformat(),
            }
            personality.setdefault("adaptation_history", []).append(adaptation)
            personality["last_modified"] = datetime.now().isoformat()

            path = spirit_dir() / "personality.json"
            path.write_text(json.dumps(personality, indent=2), encoding="utf-8")
            return {"adapted": True, "trait": trait, "old": old, "new": new, "reason": reason}

    return {"error": f"trait '{trait}' not found in any dimension"}


async def tool_relationships_list(args: dict) -> dict:
    rels_dir = spirit_dir() / "relationships"
    if not rels_dir.exists():
        return {"relationships": [], "count": 0}
    rels = []
    for f in rels_dir.glob("*.json"):
        data = _safe_load(f)
        if data:
            rels.append(data)
    return {"relationships": rels, "count": len(rels)}


async def tool_relationships_update(args: dict) -> dict:
    rels_dir = spirit_dir() / "relationships"
    rels_dir.mkdir(parents=True, exist_ok=True)

    entity_id = args["entity_id"]
    entity_type = args["entity_type"]
    trust_delta = args["trust_delta"]

    existing = None
    existing_path = None
    for f in rels_dir.glob("*.json"):
        data = _safe_load(f)
        if data and data.get("entity_id") == entity_id:
            existing = data
            existing_path = f
            break

    if existing:
        existing["trust_level"] = max(-1, min(1, existing.get("trust_level", 0) + trust_delta))
        existing["last_interaction"] = datetime.now().isoformat()
        existing_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return {"updated": True, "entity": entity_id, "trust": existing["trust_level"]}
    else:
        rel = {
            "entity_id": entity_id, "entity_type": entity_type,
            "trust_level": max(-1, min(1, trust_delta)),
            "created_at": datetime.now().isoformat(),
            "last_interaction": datetime.now().isoformat(),
        }
        rid = str(uuid.uuid4())[:8]
        (rels_dir / f"{rid}.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
        return {"created": True, "entity": entity_id, "trust": rel["trust_level"]}


async def tool_education_list(args: dict) -> dict:
    edu_file = spirit_dir() / "education.json"
    if not edu_file.exists():
        return {"topics": [], "count": 0}
    data = _safe_load(edu_file) or {}
    topics = data.get("topics", [])
    category = args.get("category")
    if category:
        topics = [t for t in topics if t.get("category", "").lower() == category.lower()]
    return {"topics": topics, "count": len(topics)}


async def tool_education_record(args: dict) -> dict:
    edu_file = spirit_dir() / "education.json"
    data = _safe_load(edu_file) if edu_file.exists() else {"topics": []}
    if data is None:
        data = {"topics": []}

    topic = {
        "id": str(uuid.uuid4()),
        "title": args["title"],
        "content": args["content"],
        "category": args.get("category", "general"),
        "confidence": args.get("confidence", 0.6),
        "verified": False,
        "source": args.get("source", "interaction"),
        "learned_at": datetime.now().isoformat(),
    }

    data["topics"].append(topic)
    edu_file.parent.mkdir(parents=True, exist_ok=True)
    edu_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"recorded": True, "topic_id": topic["id"], "title": topic["title"]}


async def tool_ethics_log(args: dict) -> dict:
    limit = args.get("limit", 20)
    log_dir = spirit_dir() / "ethics_log"
    if not log_dir.exists():
        return {"decisions": [], "count": 0}
    items = []
    for f in sorted(log_dir.glob("*.json")):
        data = _safe_load(f)
        if data:
            items.append(data)
    return {"decisions": items[-limit:], "count": len(items)}


async def tool_markdown_refresh(args: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from collector.consolidator import PsycheConsolidator
    consolidator = PsycheConsolidator(str(PSYCHE_STORE), SPIRIT_ID, str(MARKDOWN_DIR))
    consolidator.consolidate()
    return {"refreshed": True, "spirit": SPIRIT_ID}


# ── Helpers ──

def _safe_load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
        return None


def _jsonrpc_result(req_id, result: dict):
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})


def _jsonrpc_error(req_id, code: int, message: str):
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


# ── Service Supervisor Registration ──

async def register_with_supervisor():
    """Register this service with the supervisor for discovery."""
    import httpx
    supervisor_url = os.environ.get("SUPERVISOR_BASE", "http://127.0.0.1:5080")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{supervisor_url}/api/v1/supervisor/register", json={
                "service_name": SERVICE_NAME,
                "port": int(os.environ.get("PSYCHE_MCP_PORT", DEFAULT_PORT)),
                "health_path": f"/api/v1/{SERVICE_NAME}/healthcheck/basic",
            })
    except Exception:
        pass


# ── Startup ──

@app.on_event("startup")
async def startup():
    load_tools()
    await register_with_supervisor()
    print(f"Psyche MCP Server started: {len(TOOLS)} tools, spirit={SPIRIT_ID}")
    print(f"  Store: {PSYCHE_STORE}")
    print(f"  Markdown: {MARKDOWN_DIR}")
    print(f"  MCP endpoint: POST /mcp")


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Psyche MCP Server")
    parser.add_argument("--psyche-store", required=True, help="Path to psyche_store directory")
    parser.add_argument("--spirit", required=True, help="Spirit ID (e.g. sister)")
    parser.add_argument("--markdown-dir", default="psyche", help="Output dir for markdown files")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    global PSYCHE_STORE, SPIRIT_ID, MARKDOWN_DIR
    PSYCHE_STORE = args.psyche_store
    SPIRIT_ID = args.spirit
    MARKDOWN_DIR = args.markdown_dir

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
