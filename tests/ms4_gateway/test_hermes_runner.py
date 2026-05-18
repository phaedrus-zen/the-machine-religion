from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs["model"]
        self.session_id = kwargs["session_id"]
        self.tool_start_callback = kwargs.get("tool_start_callback")
        self.tool_complete_callback = kwargs.get("tool_complete_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None, stream_callback=None):
        if stream_callback:
            stream_callback("stream:")
            stream_callback(str(message))
        if self.tool_start_callback:
            self.tool_start_callback("tc1", "read_file", {"path": "README.md"})
        if self.tool_complete_callback:
            self.tool_complete_callback("tc1", "read_file", {"path": "README.md"}, '{"ok":true}')
        return {
            "final_response": f"echo:{message}",
            "messages": list(conversation_history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "ok"},
            ],
            "api_calls": 1,
            "completed": True,
        }


def test_runner_builds_hermes_agent_with_hivemind_provider(monkeypatch, tmp_path):
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url="http://hive:6089",
        ms3_url="http://ms3:9080",
        agent_cls=FakeAgent,
    )
    monkeypatch.setattr(runner, "require_plugin", lambda: None)

    response = runner.chat("hello", session_id="s1", model="qwen2.5:0.5b")

    assert response["runtime"] == "hermes"
    assert response["session_id"] == "s1"
    assert response["model"] == "qwen2.5:0.5b"
    agent = runner._sessions["s1"].agent
    assert agent.kwargs["base_url"] == "http://hive:6089/v1"
    assert agent.kwargs["provider"] == "custom"
    assert agent.kwargs["api_mode"] == "chat_completions"


def test_runner_default_model_is_hermes_compatible_size(tmp_path):
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeAgent)

    assert runner.default_model == "qwen3-coder-next:latest"


def test_runner_injects_tmr_grounding(monkeypatch, tmp_path):
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeAgent)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)

    response = runner.chat("What is The Machine Religion?", session_id="s1")

    assert response["grounding_source"] == "tmr-canon-grounding"
    assert "Deus Acuo Machina Machina" in response["text"]


def test_sessions_report_hermes_state(monkeypatch, tmp_path):
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeAgent)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    runner.chat("hello", session_id="s1")

    sessions = runner.sessions()

    assert sessions[0]["session_id"] == "s1"
    assert sessions[0]["turns"] == 2
    assert sessions[0]["tool_trace_count"] == 2


def test_chat_response_includes_tool_trace(monkeypatch, tmp_path):
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeAgent)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)

    response = runner.chat("hello", session_id="s1")

    assert response["tool_trace"][0]["event"] == "start"
    assert response["tool_trace"][1]["event"] == "complete"


def test_runner_passes_stream_callback_to_hermes(monkeypatch, tmp_path):
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeAgent)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    chunks = []

    response = runner.chat("hello", session_id="s1", stream_callback=chunks.append)

    assert chunks
    assert chunks[0] == "stream:"
    assert response["runtime"] == "hermes"


def test_terminal_dispatch_defaults_to_safe_workdir(monkeypatch, tmp_path):
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeAgent)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    captured = {}

    def fake_handle(tool_name, args, **kwargs):
        captured.update(args)
        return '{"output":"MS4_HERMES_OK","exit_code":0}'

    monkeypatch.setitem(__import__("sys").modules, "model_tools", type("M", (), {
        "handle_function_call": staticmethod(fake_handle),
        "get_toolset_for_tool": staticmethod(lambda name: "terminal"),
    }))

    result = runner.dispatch_hermes_tool("terminal", {"command": "echo MS4_HERMES_OK"})

    assert captured["workdir"] == "/"
    assert "MS4_HERMES_OK" in result["result"]
