import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from machine_spirit_4.double_agent.blackboard import Blackboard
from machine_spirit_4.double_agent.depth_picker import _pick_from_catalog as pick_depth
from machine_spirit_4.double_agent.model_picker import (
    _is_chat_capable,
    _pick_from_catalog as pick_face,
)
from machine_spirit_4.double_agent.runner import JobRunner
from machine_spirit_4.double_agent.router import route
from machine_spirit_4.double_agent.schemas import JobEnvelope
from machine_spirit_4.gateway.hermes_runner import _canned_chat_failure_text


class ModelPickerChatCapabilityTests(unittest.TestCase):
    def test_face_picker_skips_loaded_qwen3_tts_for_chat_model(self) -> None:
        choice = pick_face([
            {"id": "Qwen3-TTS", "hivemind_status": "running", "capabilities": ["tts"]},
            {"id": "llama3.1:8b", "hivemind_status": "installed"},
        ])

        self.assertIsNotNone(choice)
        self.assertEqual(choice.model_id, "llama3.1:8b")

    def test_depth_picker_skips_loaded_qwen3_tts_for_chat_model(self) -> None:
        choice = pick_depth([
            {"id": "Qwen3-TTS", "hivemind_status": "running", "capabilities": ["tts"]},
            {"id": "qwen3.6:35b", "hivemind_status": "installed"},
        ])

        self.assertIsNotNone(choice)
        self.assertEqual(choice.model_id, "qwen3.6:35b")

    def test_only_qwen3_tts_does_not_get_selected(self) -> None:
        catalog = [{"id": "Qwen3-TTS", "hivemind_status": "running", "capabilities": ["tts"]}]

        self.assertIsNone(pick_face(catalog))
        self.assertIsNone(pick_depth(catalog))

    def test_unknown_chat_model_fails_open(self) -> None:
        self.assertTrue(_is_chat_capable({"id": "mystery-chat:13b"}))


class RouterWordBoundaryTests(unittest.TestCase):
    def test_incidental_tool_words_stay_direct(self) -> None:
        self.assertEqual(route("running back to the store").kind, "direct")
        self.assertEqual(route("the markets are open").kind, "direct")

    def test_real_tool_requests_route_deep(self) -> None:
        self.assertEqual(route("run the cluster summary tool").kind, "deep")
        self.assertEqual(route("open a vm and start it").kind, "deep")
        self.assertEqual(route("what HiveMind MCP tools are available?").kind, "deep")


class CannedFailureTextTests(unittest.TestCase):
    def test_stream_timeout_message_blames_chat_not_tts(self) -> None:
        text = _canned_chat_failure_text(
            RuntimeError("HiveMind /v1/chat/completions (stream) unreachable: timed out")
        )

        self.assertIn("streaming chat endpoint", text)
        self.assertIn("lighter chat model", text)
        self.assertNotIn("TTS", text)
        self.assertNotIn("REST", text)


class DepthWorkerPreflightGateTests(unittest.TestCase):
    def test_failed_depth_preflight_fails_job_without_spawning_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            blackboard = Blackboard(Path(tmpdir) / "double_agent.sqlite3")
            runner = JobRunner(blackboard=blackboard, max_concurrent_jobs=1)
            envelope = JobEnvelope(
                job_id="preflight-fail-job",
                parent_conversation_id="conv-preflight",
                conversation_revision_id=1,
                background_lobe_type="deep_chat",
                user_visible_goal="Check cluster state",
                internal_goal="Check cluster state with depth tools",
            )
            preflight_stdout = json.dumps(
                {
                    "ok": False,
                    "diagnostics": [
                        {
                            "code": "posix_loopback_namespace_unreachable",
                            "severity": "warning",
                        }
                    ],
                    "probes": [
                        {"name": "hivemind_mcp_tools_list", "reachable": False},
                        {"name": "ms3_health", "reachable": False},
                    ],
                }
            )
            completed = SimpleNamespace(returncode=1, stdout=preflight_stdout, stderr="")

            with mock.patch.dict(os.environ, {"MS4_DA_DEPTH_PREFLIGHT": "1"}, clear=False):
                os.environ.pop("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER", None)
                with mock.patch(
                    "machine_spirit_4.double_agent.runner.subprocess.run",
                    return_value=completed,
                ) as run_mock, mock.patch(
                    "machine_spirit_4.double_agent.runner.subprocess.Popen",
                    side_effect=AssertionError("Depth worker should not spawn after failed preflight"),
                ) as popen_mock:
                    snapshot = runner.submit(envelope)

            try:
                self.assertEqual(snapshot["state"], "failed")
                self.assertEqual(snapshot["last_event_type"], "job.failed")
                self.assertIn("worker namespace", snapshot["last_safe_user_status"])
                run_mock.assert_called_once()
                popen_mock.assert_not_called()
                self.assertFalse(runner._procs)

                result = blackboard.get_result(envelope.job_id)
                self.assertIsNotNone(result)
                self.assertEqual(result["status"], "failed")
                self.assertIn("preflight failed", result["summary"])
                self.assertEqual(result["conversation_revision_id"], 1)

                events = blackboard.list_events(envelope.job_id)
                self.assertGreaterEqual(len(events), 2)
                self.assertEqual(events[0]["visibility"], "user_safe")
                self.assertEqual(
                    events[0]["payload"]["diagnostic_codes"],
                    ["posix_loopback_namespace_unreachable"],
                )
                self.assertEqual(
                    events[0]["payload"]["unreachable_probes"],
                    ["hivemind_mcp_tools_list", "ms3_health"],
                )
                self.assertEqual(events[1]["visibility"], "operator_only")
                self.assertIn("preflight", events[1]["payload"])
            finally:
                runner.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
