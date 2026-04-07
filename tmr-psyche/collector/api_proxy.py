"""
API Proxy Collector

For non-MS3 deployments: wraps any OpenAI-compatible API endpoint.
Injects the psyche system prompt, logs conversations in training format,
and optionally captures preference signals (thumbs up/down) for DPO.

Usage:
    # Start the proxy on port 8100, forwarding to your model on 8000:
    python -m collector.api_proxy --target http://localhost:8000/v1 --profile sister

    # Clients hit the proxy exactly like they'd hit the original API.
    # Every conversation is logged to data/live/ in training format.
"""

import json
import os
import time
import uuid
from pathlib import Path
from datetime import datetime

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import httpx
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiles import load_profile, build_system_prompt


class ConversationLogger:
    """Logs conversations to training-format JSONL as they happen."""

    def __init__(self, output_dir: str, profile_id: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_id = profile_id
        self.sessions: dict[str, list[dict]] = {}

    def log_turn(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})

    def mark_preference(self, session_id: str, turn_index: int, preferred: bool):
        """Mark a specific assistant turn as preferred (for DPO collection)."""
        pref_file = self.output_dir / "preferences.jsonl"
        record = {
            "session_id": session_id,
            "turn_index": turn_index,
            "preferred": preferred,
            "timestamp": datetime.now().isoformat(),
        }
        with open(pref_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def flush_session(self, session_id: str):
        """Write a completed session to the training JSONL."""
        if session_id not in self.sessions:
            return
        turns = self.sessions.pop(session_id)
        if len(turns) < 2:
            return

        record = {
            "id": f"live_{self.profile_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}",
            "category": "live_collected",
            "profile": self.profile_id,
            "turns": turns,
            "collected_at": datetime.now().isoformat(),
        }

        today = datetime.now().strftime("%Y%m%d")
        out_file = self.output_dir / f"live_{today}.jsonl"
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush_all(self):
        for sid in list(self.sessions.keys()):
            self.flush_session(sid)


def create_proxy_app(
    target_url: str,
    profile_path: str,
    output_dir: str = "data/live",
) -> "FastAPI":
    """Create a FastAPI proxy that injects psyche prompts and logs everything."""
    if not HAS_FASTAPI:
        raise ImportError("Install fastapi, httpx, uvicorn: pip install fastapi httpx uvicorn")

    app = FastAPI(title="Psyche LoRA Collector Proxy")
    profile = load_profile(profile_path)
    system_prompt = build_system_prompt(profile)
    logger = ConversationLogger(output_dir, profile["id"])
    client = httpx.AsyncClient(base_url=target_url, timeout=120.0)

    @app.post("/v1/chat/completions")
    async def proxy_chat(request: Request):
        body = await request.json()
        messages = body.get("messages", [])

        session_id = body.get("session_id", str(uuid.uuid4()))

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": system_prompt})
        elif messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]

        body["messages"] = messages

        for msg in messages:
            if msg["role"] != "system":
                logger.log_turn(session_id, msg["role"], msg["content"])

        response = await client.post("/chat/completions", json=body)
        result = response.json()

        if "choices" in result:
            for choice in result["choices"]:
                content = choice.get("message", {}).get("content", "")
                if content:
                    logger.log_turn(session_id, "assistant", content)

        return JSONResponse(content=result)

    @app.post("/v1/preference")
    async def log_preference(request: Request):
        """Endpoint for marking preferences (thumbs up/down on responses)."""
        body = await request.json()
        logger.mark_preference(
            body["session_id"],
            body["turn_index"],
            body["preferred"],
        )
        return {"status": "logged"}

    @app.post("/v1/flush")
    async def flush_sessions():
        """Flush all active sessions to disk."""
        logger.flush_all()
        return {"status": "flushed"}

    @app.on_event("shutdown")
    async def shutdown():
        logger.flush_all()
        await client.aclose()

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Psyche LoRA Collector Proxy")
    parser.add_argument("--target", required=True, help="Target API URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--profile", required=True, help="Profile YAML path")
    parser.add_argument("--output", default="data/live", help="Output directory for collected data")
    parser.add_argument("--port", type=int, default=8100, help="Proxy port")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = create_proxy_app(args.target, args.profile, args.output)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
