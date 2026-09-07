from __future__ import annotations

from typing import Any

from machine_spirit_4.gateway.face_lobe_chat import (
    FaceLobeChat,
    apply_face_lobe_output_guard,
)


_GROUNDED_JSON = '{"verified":{"nodes":9}}'
_UNGROUNDED_JSON = '{"unverified":{"nodes":999}}'


class _StubFaceLobeChat(FaceLobeChat):
    def __init__(self, reply: str):
        super().__init__(hivemind_url="http://hive:6089", empty_fallback_model="")
        self.reply = reply

    def _post_blocking(self, _payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self.reply, {"http_latency_ms": 1, "bytes_received": len(self.reply)}

    def _post_streaming(self, _payload: dict[str, Any], stream_callback) -> tuple[str, dict[str, Any]]:
        midpoint = max(1, len(self.reply) // 2)
        for fragment in (self.reply[:midpoint], self.reply[midpoint:]):
            stream_callback(fragment)
        return self.reply, {
            "http_latency_ms": 1,
            "bytes_received": len(self.reply),
            "stream_chunks": 2,
            "stream_first_token_ms": 1,
            # This stub represents a successful upstream stream. The product
            # now requires explicit verified completion before buffered JSON
            # or guarded replacement text can be released downstream.
            "stream_completed": True,
            "terminal_source": "finish_reason",
            "finish_reasons": ["stop"],
        }


def test_face_lobe_output_guard_strips_unverified_depth_json():
    text = (
        "The Depth Lobe job da-7bbf1888 already listed the GPUs.\n"
        "For `hivemind.hosts.list@v1`:\n"
        "```json\n[{\"node_id\":\"node-123\",\"gpu_count\":2}]\n```"
    )
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- THIS TURN DID NOT DISPATCH any background work.\n"
        "- terminal Depth Lobe jobs without verified results (MOST RECENT FIRST; do NOT present as successful):\n"
        "    - da-7bbf1888 [completed] goal='run them both'\n"
        "      status: no verified result recorded"
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="Results?",
        extra_system=extra_system,
    )

    assert meta["applied"] is True
    assert meta["reason"] == "unverified_terminal_depth_result"
    assert "node-123" not in guarded
    assert "gpu_count" not in guarded
    assert "verified result payload" in guarded
    assert "will not invent" in guarded


def test_face_lobe_output_guard_allows_verified_depth_result():
    text = "The Depth Lobe found GPU-REAL from hivemind.hosts.list@v1."
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- verified completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-good [completed] goal='list GPUs'\n"
        "      result: GPU-REAL from hivemind.hosts.list@v1"
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="Results?",
        extra_system=extra_system,
    )

    assert meta["applied"] is False
    assert guarded == text


def test_face_lobe_output_guard_allows_architecture_prose_with_result_noun():
    text = (
        "Simultaneously, it sends the detailed query to the Depth lobe for processing. "
        "Once the 35B model completes its analysis, the result is seamlessly integrated "
        "into the conversation flow."
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="Explain the Face and Depth architecture.",
        extra_system=None,
    )

    assert meta["applied"] is False
    assert guarded == text


def test_architecture_prose_stream_and_terminal_text_remain_identical():
    reply = (
        "Simultaneously, it sends the detailed query to the Depth lobe for processing. "
        "Once the 35B model completes its analysis, the result is seamlessly integrated "
        "into the conversation flow."
    )
    chat = _StubFaceLobeChat(reply)
    chunks: list[str] = []

    result = chat.chat(
        "Explain the Face and Depth architecture.",
        session_id="s-architecture-result-noun",
        model="face-model",
        stream_callback=chunks.append,
        extra_system=None,
    )

    assert result["output_guard"]["applied"] is False
    assert result["text"] == reply
    assert "".join(chunks) == result["text"]


def test_face_lobe_output_guard_still_blocks_returned_depth_result_claim():
    text = "The Depth Lobe returned nine active nodes."

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="Results?",
        extra_system=None,
    )

    assert meta["applied"] is True
    assert meta["reason"] == "unsupported_tool_result_claim"
    assert guarded != text


def test_current_dispatch_ack_wins_over_stale_terminal_context():
    current_job_id = "da-11111111-2222-4333-8444-555555555555"
    text = (
        f"The Depth Lobe job {current_job_id} already returned results from "
        "hivemind.cluster.summary@v1."
    )
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        f"- THIS TURN DISPATCHED job {current_job_id} to the Depth Lobe.\n"
        "- active or stale Double Agent jobs:\n"
        f"    - {current_job_id} [queued - deep_chat]: no status yet\n"
        "- terminal Depth Lobe jobs without verified results (MOST RECENT FIRST; do NOT present as successful):\n"
        "    - da-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee [failed] goal='old work'\n"
        "      status: no verified result recorded"
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="/deep run the checks",
        extra_system=extra_system,
    )

    assert meta["applied"] is True
    assert meta["reason"] == "current_turn_depth_dispatch_acknowledgement"
    assert meta["dispatched_job_id"] == current_job_id
    assert current_job_id in guarded
    assert "I dispatched Depth Lobe job" in guarded
    assert "terminal, but not successful" not in guarded
    assert "automatically when it completes" in guarded


def test_face_lobe_output_guard_strips_unverified_weather_json():
    text = (
        "The Depth Lobe job da-62888384 returned this weather result:\n"
        "```json\n"
        "{\"current_weather\":{\"temperature\":\"22C\",\"humidity\":\"60%\"}}\n"
        "```"
    )
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- THIS TURN DID NOT DISPATCH any background work.\n"
        "- terminal Depth Lobe jobs without verified results (MOST RECENT FIRST; do NOT present as successful):\n"
        "    - da-62888384 [completed] goal='verify GPUs'\n"
        "      status: no verified result recorded"
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="weather?",
        extra_system=extra_system,
    )

    assert meta["applied"] is True
    assert meta["tool_json"] is True
    assert "current_weather" not in guarded
    assert "22C" not in guarded
    assert "will not invent" in guarded


def test_face_lobe_output_guard_strips_fabricated_tool_manager_list():
    text = (
        "The available tools include `tool_manager.run_async`, "
        "`ServerSideToolManager`, `browser_snapshot`, and `skills_list`."
    )
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- THIS TURN DID NOT DISPATCH any background work.\n"
        "- no active background jobs"
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="list all tools",
        extra_system=extra_system,
    )

    assert meta["applied"] is True
    assert meta["tool_symbol"] is True
    assert meta["tool_list"] is True
    assert "ServerSideToolManager" not in guarded
    assert "browser_snapshot" not in guarded
    assert "verified tool result" in guarded


def test_face_lobe_output_guard_allows_grounded_tools_list():
    text = "The available tools include MS4 MCP and HiveMind MCP surfaces."
    extra_system = (
        "MS4 capability surface (Quartermaster-curated; answer from this, do not invent tools):\n"
        "- 143 tools across 12 toolboxes (HiveMind 120 + MS4 23).\n"
        "- MS4 MCP server: 23 tools at http://127.0.0.1:9181/mcp\n"
        "- HiveMind MCP server: 120 tools at http://127.0.0.1:6105/mcp"
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="list all tools",
        extra_system=extra_system,
    )

    assert meta["applied"] is False
    assert guarded == text


def test_high_risk_streaming_buffers_until_guarded_text():
    reply = (
        "The Depth Lobe job da-7bbf1888 already listed the GPUs: "
        "[{\"node_id\":\"node-123\",\"gpu_count\":2}]"
    )
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- terminal Depth Lobe jobs without verified results (MOST RECENT FIRST; do NOT present as successful):\n"
        "    - da-7bbf1888 [completed] goal='run them both'\n"
        "      status: no verified result recorded"
    )
    chat = _StubFaceLobeChat(reply)
    chunks: list[str] = []

    result = chat.chat(
        "Results?",
        session_id="s-guard",
        model="face-model",
        stream_callback=chunks.append,
        extra_system=extra_system,
    )

    assert result["output_guard"]["applied"] is True
    assert result["guarded_stream_emitted"] is True
    assert chunks == [result["text"]]
    assert "node-123" not in result["text"]
    assert result["metrics"]["output_guard"]["reason"] == "unverified_terminal_depth_result"


def test_ordinary_streaming_holds_ungrounded_json_until_guarded():
    reply = f"The cluster returned {_UNGROUNDED_JSON}."
    chat = _StubFaceLobeChat(reply)
    chunks: list[str] = []

    result = chat.chat(
        "Tell me about the cluster",
        session_id="s-ordinary-json-guard",
        model="face-model",
        stream_callback=chunks.append,
        extra_system=None,
    )

    emitted = "".join(chunks)
    assert result["output_guard"]["applied"] is True
    assert result["guarded_stream_emitted"] is True
    assert _UNGROUNDED_JSON not in emitted
    assert _UNGROUNDED_JSON not in result["text"]
    assert result["text"] in emitted


def test_ordinary_non_json_prose_keeps_chunked_streaming():
    reply = "The cluster is healthy and ready."
    midpoint = max(1, len(reply) // 2)
    chat = _StubFaceLobeChat(reply)
    chunks: list[str] = []

    result = chat.chat(
        "Tell me about the cluster",
        session_id="s-ordinary-prose-stream",
        model="face-model",
        stream_callback=chunks.append,
        extra_system=None,
    )

    assert result["output_guard"]["applied"] is False
    assert result["guarded_stream_emitted"] is False
    assert result["text"] == reply
    assert chunks == [reply[:midpoint], reply[midpoint:]]


def test_ordinary_streaming_releases_exact_grounded_json_after_guard():
    reply = f"The verified result was {_GROUNDED_JSON}."
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- verified completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-grounded [completed] goal='inspect cluster'\n"
        f"      result: {_GROUNDED_JSON}"
    )
    chat = _StubFaceLobeChat(reply)
    chunks: list[str] = []

    result = chat.chat(
        "Tell me about the cluster",
        session_id="s-ordinary-grounded-json",
        model="face-model",
        stream_callback=chunks.append,
        extra_system=extra_system,
    )

    assert result["output_guard"]["applied"] is False
    assert result["guarded_stream_emitted"] is False
    assert result["text"] == reply
    assert "".join(chunks) == reply


def _assert_ungrounded_json_is_blocked(text: str, claimed_value: str) -> None:
    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="What did the tool return?",
        extra_system=None,
    )

    assert meta["applied"] is True
    assert meta["reason"] == "unsupported_tool_result_claim"
    assert meta["tool_json"] is True
    assert claimed_value not in guarded
    assert "will not invent" in guarded


def _assert_context_result_is_not_grounded(extra_system: str) -> None:
    guarded, meta = apply_face_lobe_output_guard(
        _UNGROUNDED_JSON,
        message="What did the tool return?",
        extra_system=extra_system,
    )

    assert meta["applied"] is True
    assert meta["tool_json"] is True
    assert _UNGROUNDED_JSON not in guarded


def test_verified_depth_section_does_not_authorize_terminal_result_line():
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- verified completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-good [completed] goal='inspect cluster'\n"
        f"      result: {_GROUNDED_JSON}\n"
        "- terminal Depth Lobe jobs without verified results (MOST RECENT FIRST; do NOT present as successful):\n"
        "    - da-bad [failed] goal='failed work'\n"
        f"      result: {_UNGROUNDED_JSON}"
    )

    _assert_context_result_is_not_grounded(extra_system)


def test_verified_depth_section_does_not_authorize_active_result_line():
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- active or stale Double Agent jobs:\n"
        "    - da-active [running - deep_chat]: still working\n"
        f"      result: {_UNGROUNDED_JSON}\n"
        "- verified completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-good [completed] goal='inspect cluster'\n"
        f"      result: {_GROUNDED_JSON}"
    )

    _assert_context_result_is_not_grounded(extra_system)


def test_verified_depth_section_stops_before_unrelated_result_line():
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- verified completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-good [completed] goal='inspect cluster'\n"
        f"      result: {_GROUNDED_JSON}\n\n"
        "Unrelated non-result context:\n"
        f"- result: {_UNGROUNDED_JSON}"
    )

    _assert_context_result_is_not_grounded(extra_system)


def test_quartermaster_block_stops_before_later_grounded_context_result_line():
    extra_system = (
        "MS4 Quartermaster inline tool result (authoritative; quote this faithfully, "
        "do not invent additional tools, data, or fields):\n"
        "- tool: hivemind.cluster.summary@v1  (read-only, executed just now in 8 ms)\n"
        f"- result: {_GROUNDED_JSON}\n"
        "This is the answer to the operator's request. Summarise it in plain language.\n\n"
        "Later grounded context block:\n"
        f"- result: {_UNGROUNDED_JSON}"
    )

    _assert_context_result_is_not_grounded(extra_system)


def test_face_lobe_output_guard_blocks_ungrounded_generic_json_object():
    _assert_ungrounded_json_is_blocked(
        '{"cpu_count":64,"ram_gb":512}',
        "cpu_count",
    )


def test_face_lobe_output_guard_blocks_ungrounded_nested_json_object():
    _assert_ungrounded_json_is_blocked(
        '{"result":{"nodes":9,"healthy":true}}',
        "healthy",
    )


def test_face_lobe_output_guard_blocks_ungrounded_arbitrary_key_json():
    _assert_ungrounded_json_is_blocked(
        '{"zqx_metric_73":{"quux_value":17}}',
        "zqx_metric_73",
    )


def test_face_lobe_output_guard_blocks_ungrounded_prose_plus_json():
    _assert_ungrounded_json_is_blocked(
        'The cluster returned {"workers":12,"ready":11}.',
        "workers",
    )


def test_face_lobe_output_guard_allows_grounded_verified_depth_json_payload():
    text = 'The verified result was {"result":{"nodes":9,"healthy":true}}.'
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- verified completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-grounded [completed] goal='inspect cluster'\n"
        '      result: {"result": {"healthy": true, "nodes": 9}}'
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="Results?",
        extra_system=extra_system,
    )

    assert meta["applied"] is False
    assert guarded == text


def test_face_lobe_output_guard_allows_grounded_inline_tool_json_payload():
    text = 'The tool returned {"cpu_count":64,"ram_gb":512}.'
    extra_system = (
        "MS4 Quartermaster inline tool result (authoritative; quote this faithfully, "
        "do not invent additional tools, data, or fields):\n"
        "- tool: hivemind.cluster.summary@v1  (read-only, executed just now in 8 ms)\n"
        '- result: {"ram_gb": 512, "cpu_count": 64}'
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="Results?",
        extra_system=extra_system,
    )

    assert meta["applied"] is False
    assert guarded == text


def test_face_lobe_output_guard_blocks_json_absent_from_verified_depth_grounding():
    text = '{"result":{"nodes":999,"healthy":true}}'
    extra_system = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- verified completed Depth Lobe jobs (MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-grounded [completed] goal='inspect cluster'\n"
        '      result: {"result":{"nodes":9,"healthy":true}}'
    )

    guarded, meta = apply_face_lobe_output_guard(
        text,
        message="Results?",
        extra_system=extra_system,
    )

    assert meta["applied"] is True
    assert meta["tool_json"] is True
    assert "999" not in guarded


def test_face_lobe_output_guard_leaves_ordinary_non_result_prose_unchanged():
    for text in (
        "I am here and ready to help you think this through.",
        "Use the {name} placeholder when drafting the note.",
        "The choices are [first, second], in that order.",
    ):
        guarded, meta = apply_face_lobe_output_guard(
            text,
            message="Are you there?",
            extra_system=None,
        )

        assert meta["applied"] is False
        assert guarded == text
