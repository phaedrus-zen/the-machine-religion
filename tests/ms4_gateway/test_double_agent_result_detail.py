"""Phase 2 (Double Agent completion loop) — the job-detail endpoint
exposes the full JobResult.text so the UI's completion announcement can
show "Show details"."""

from __future__ import annotations

import io
import json
import types

import machine_spirit_4.gateway.server as srv_module
from machine_spirit_4.double_agent import JobEnvelope, JobResult, default_runner, safety


def _seed_completed_job(*, text: str) -> str:
    runner = default_runner()
    board = runner.blackboard
    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id="conv-detail",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="List the GPUs.",
        internal_goal="List the GPUs.",
    )
    env.validate()
    board.insert_job(env)
    board.update_job_state(env.job_id, state="completed", last_safe_user_status="Found 3 GPUs.")
    result = JobResult(
        job_id=env.job_id,
        status="success",
        summary="Found 3 GPUs.",
        text=text,
        confidence="high",
    )
    board.insert_result(result)
    return env.job_id


def _seed_running_job() -> str:
    runner = default_runner()
    board = runner.blackboard
    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id="conv-running",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="Long task.",
        internal_goal="Long task.",
    )
    env.validate()
    board.insert_job(env)
    board.update_job_state(env.job_id, state="running")
    return env.job_id


def _make_handler(method: str, path: str) -> srv_module.Ms4GatewayHandler:
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.command = method
    handler.path = path
    handler.headers = types.SimpleNamespace(get={"Host": "127.0.0.1:9180"}.get)
    handler.rfile = io.BytesIO(b"")
    handler.wfile = io.BytesIO()
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"
    return handler


def _read_response(handler) -> tuple[int, dict]:
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split(b" ")[1])
    _head, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8") or "{}")


def test_detail_includes_full_result_text():
    job_id = _seed_completed_job(text="Node A: RTX 4090; Node B: A6000; Node C: 3090.")
    h = _make_handler("GET", f"/api/v1/double-agent/jobs/{job_id}")
    h._double_agent_get(job_id)
    status, payload = _read_response(h)
    assert status == 200
    assert payload["state"] == "completed"
    assert payload["result"]["text"] == "Node A: RTX 4090; Node B: A6000; Node C: 3090."
    assert payload["result"]["status"] == "success"


def test_detail_running_job_has_no_result():
    job_id = _seed_running_job()
    h = _make_handler("GET", f"/api/v1/double-agent/jobs/{job_id}")
    h._double_agent_get(job_id)
    status, payload = _read_response(h)
    assert status == 200
    assert payload["state"] == "running"
    assert "result" not in payload  # no JobResult persisted yet


def test_runner_get_result_wrapper():
    job_id = _seed_completed_job(text="hello")
    assert default_runner().get_result(job_id)["text"] == "hello"
    assert default_runner().get_result("da-does-not-exist") is None
