from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys

from machine_spirit_4.double_agent.router import route as router_route


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_oracle_depth_conversation.py"
SPEC = importlib.util.spec_from_file_location("oracle_depth_conversation_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


NONCE = "N0NCE-42"
SESSION_ID = "oracle-depth-test-session"
JOB_ID = "da-oracle-depth-quality-1"
REFERENCE_UTC = "2026-08-07T21:00:00+00:00"


def _complete_depth_plan() -> str:
    return """SUMMARY
Project AMBER-LATTICE needs a controlled recovery inside the remembered window of 12 minutes. The plan treats ALPHA as stale routing, protects currently healthy traffic, makes the smallest reversible routing change, and reserves enough time for direct validation. OWNER=UNKNOWN because the conversation supplied no owner name. The target outcome is restored routing with observable proof rather than an optimistic status message.

CAUSES
ALPHA means stale routing: clients or the gateway may still select a route whose backend state no longer matches the live cluster. Cached route selection can therefore keep sending work toward an unhealthy or superseded target even after the underlying service recovers. The plan separates route invalidation, service verification, traffic restoration, and post-recovery observation so a misleading health response cannot hide a broken user path.

ACTIONS
1. Freeze nonessential routing changes and capture the current route table, active jobs, healthy peers, and a known failing request. Preserve timestamps and request identifiers so later checks can distinguish the recovery effect from unrelated cluster movement. Keep existing healthy sessions serving while the snapshot is collected.

2. Mitigate risk ALPHA by invalidating the stale routing entry, refreshing route discovery from live peer health, and directing a canary request through the newly selected route. Complete this within the 12-minute recovery constraint, retain the old route definition for reversal, and stop if the replacement target does not pass direct health plus inference checks.

3. Restore normal traffic gradually after the canary succeeds. Compare client response, selected peer, job accounting, inference statistics, and the relevant gateway and peer logs for the same request identifier. Hold the recovery if those independent surfaces disagree, even when the client receives an HTTP success response.

4. Observe the recovered path through repeated requests for the rest of the window, confirm no active jobs remain stranded on the stale target, and record the final route and service state. If errors or routing drift recur, restore the preserved route definition and classify the first failing gate before another attempt.

RISKS
- A newly discovered target can report healthy before inference is actually warm, so the canary must exercise the real response path and not only a health endpoint.
- Invalidating too broadly could disrupt healthy sessions, so the change is limited to the identified stale entry and retains a reversible prior definition.

VALIDATION
OWNER=UNKNOWN. Record VERIFY-N0NCE-42 with tool-reported UTC 2026-08-07T21:00:00Z. Pass only when the canary and repeated requests return complete visible answers, the same peer and request identity appear in job accounting and logs, ALPHA no longer selects the stale route, and no truncation or warning metadata is present. This evidence closes the recovery plan without inventing an owner."""


def _action_two_walkthrough() -> str:
    return """Action 2 is the active mitigation for ALPHA. Start from the captured route snapshot, identify only the entry whose discovery data is stale, and invalidate that entry without clearing unrelated healthy routes. Ask live peer discovery for a replacement, then compare the candidate's direct health with its ability to serve a real inference request. Send one canary through the refreshed route and correlate its request identifier across the client response, gateway record, selected peer, job accounting, inference statistics, and logs. The 12-minute constraint means this work must remain narrow: reserve the opening minutes for snapshot and invalidation, the middle of the window for the canary and reconciliation, and the final minutes for observation or reversal. If any surface disagrees, if the canary is incomplete, or if ALPHA still selects the stale target, stop expansion and restore the preserved route definition. That is a verified mitigation, not merely a cache clear followed by an optimistic health check."""


def passing_evidence() -> dict:
    prompts = validator.canonical_prompts(NONCE)
    depth_text = _complete_depth_plan()
    job = {
        "job_id": JOB_ID,
        "state": "completed",
        "latency_s": 30.0,
        "timed_out": False,
        "events": [
            {
                "type": "job.tool.call.completed",
                "payload": {"tool": "hivemind_time_now", "tool_call_id": "call-time-1"},
            },
            {"type": "job.completed", "payload": {}},
        ],
        "result": {
            "job_id": JOB_ID,
            "status": "success",
            "finish_reason": "stop",
            "text": depth_text,
            "actions_taken": [{"tool": "hivemind_time_now", "tool_call_id": "call-time-1"}],
        },
    }
    return {
        "schema": validator.EVIDENCE_SCHEMA,
        "scenario": validator.SCENARIO_NAME,
        "nonce": NONCE,
        "session_id": SESSION_ID,
        "reference_utc": REFERENCE_UTC,
        "turns": [
            {
                "user": prompts[0],
                "assistant_text": "READY",
                "session_id": SESSION_ID,
                "latency_s": 0.5,
            },
            {
                "user": prompts[1],
                "assistant_text": "I am running the complete recovery analysis now.",
                "session_id": SESSION_ID,
                "latency_s": 0.4,
                "response": {"dispatched_job": {"job_id": JOB_ID}},
                "job": job,
                "automatic_completion": {
                    "delivered": True,
                    "required_user_action": False,
                    "text": f"Depth result\n{depth_text}",
                    "job_id": JOB_ID,
                    "session_id": SESSION_ID,
                    "latency_s": 0.4,
                    "evidence_source": "visible_oracle_dom",
                },
            },
            {
                "user": prompts[2],
                "assistant_text": _action_two_walkthrough(),
                "session_id": SESSION_ID,
                "latency_s": 1.4,
            },
            {
                "user": prompts[3],
                "assistant_text": (
                    "Action 2 is the exact action that mitigates ALPHA by invalidating the stale "
                    "routing entry, refreshing discovery, and validating a canary. You did not give "
                    "an owner name, so the preserved value is OWNER=UNKNOWN."
                ),
                "session_id": SESSION_ID,
                "latency_s": 0.7,
            },
        ],
    }


def _checks(report: dict) -> dict[str, dict]:
    return {item["name"]: item for item in report["checks"]}


def test_canonical_fixture_passes_every_hard_gate():
    evidence = passing_evidence()
    report = validator.evaluate_transcript(
        evidence,
        now_utc=datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc),
    )

    assert report["ok"] is True, report["failed_checks"]
    assert report["failed_checks"] == []
    assert report["metrics"]["depth_jobs"] == 1
    assert report["metrics"]["tool_calls"] == 1
    assert report["metrics"]["depth_words"] >= 180
    assert report["metrics"]["followup_words"] >= 100


def test_required_tool_exactly_once_allows_auxiliary_tool_discovery():
    evidence = passing_evidence()
    evidence["turns"][1]["job"]["result"]["actions_taken"].insert(
        0,
        {"tool": "tool_describe", "tool_call_id": "call-describe-1"},
    )

    report = validator.evaluate_transcript(evidence)
    gate = _checks(report)["required_tool_called_once"]

    assert gate["ok"] is True
    assert gate["detail"]["auxiliary"] == ["tool_describe"]
    assert report["metrics"]["tool_calls"] == 2


def test_required_tool_twice_still_fails_with_auxiliary_calls():
    evidence = passing_evidence()
    evidence["turns"][1]["job"]["result"]["actions_taken"].extend(
        [
            {"tool": "tool_describe", "tool_call_id": "call-describe-1"},
            {"tool": "hivemind_time_now", "tool_call_id": "call-time-2"},
        ]
    )

    gate = _checks(validator.evaluate_transcript(evidence))["required_tool_called_once"]

    assert gate["ok"] is False


def test_fact_and_action_parsing_accepts_legitimate_markdown_typography():
    evidence = passing_evidence()
    depth = evidence["turns"][1]["job"]["result"]["text"]
    depth = depth.replace("12 minutes", "12-minute")
    depth = re.sub(r"(?m)^([1-4])\.\s+", r"**Action \1:** ", depth)
    evidence["turns"][1]["job"]["result"]["text"] = depth
    evidence["turns"][1]["automatic_completion"]["text"] = f"Depth result\n{depth}"

    report = validator.evaluate_transcript(evidence)
    checks = _checks(report)

    assert checks["remembered_facts_preserved"]["ok"] is True
    assert checks["depth_actions_exact"]["ok"] is True
    assert checks["focus_action_mitigates_risk"]["ok"] is True


def test_owner_recall_accepts_semantic_unknown_without_requiring_assignment_syntax():
    evidence = passing_evidence()
    evidence["turns"][3]["assistant_text"] = (
        "Action 2 mitigates ALPHA. The owner name provided was UNKNOWN."
    )

    gate = _checks(validator.evaluate_transcript(evidence))[
        "targeted_followup_preserves_answer_and_unknown"
    ]

    assert gate["ok"] is True
    assert gate["detail"]["owner_assignments"] == ["UNKNOWN"]


def test_hidden_teaser_cannot_substitute_for_automatic_full_result():
    evidence = passing_evidence()
    evidence["turns"][1]["automatic_completion"].update(
        {
            "text": "Here's the gist: refresh the route. Want the full details?",
            "required_user_action": True,
        }
    )

    report = validator.evaluate_transcript(evidence)
    checks = _checks(report)

    assert report["ok"] is False
    assert checks["automatic_full_result_delivery"]["ok"] is False
    assert checks["automatic_delivery_has_no_teaser"]["ok"] is False
    assert "want_more" in checks["automatic_delivery_has_no_teaser"]["detail"]["cop_outs"]


def test_affirmative_followup_must_expand_instead_of_deferring_again():
    evidence = passing_evidence()
    evidence["turns"][2]["assistant_text"] = "Yes. I can go deeper if you want."

    report = validator.evaluate_transcript(evidence)
    followup = _checks(report)["affirmative_followup_is_substantive"]

    assert followup["ok"] is False
    assert followup["detail"]["words"] < 100
    assert followup["detail"]["cop_outs"]


def test_canonical_followup_is_explicit_not_model_luck() -> None:
    followup = validator.canonical_prompts(NONCE)[2]

    assert "Action 2" in followup
    assert "at least 100 words" in followup
    assert "ALPHA" in followup
    assert "12-minute" in followup
    assert "completed answer" in followup
    assert router_route(followup).kind == "direct"


def test_compound_analytical_request_routes_to_depth_without_slash_command() -> None:
    prompt = (
        "Explain why a 35B Depth lobe might outperform a 4B Face lobe when "
        "diagnosing an intermittent distributed-inference failure. Give me the "
        "recommendation, mechanism, tradeoffs, and a concrete example."
    )

    decision = router_route(prompt)

    assert decision.kind == "deep"
    assert decision.source == "heuristic"
    assert "multiple analytical deliverables" in decision.reason


def test_single_aspect_depth_followups_remain_direct() -> None:
    prompts = [
        "Go deeper on the second tradeoff.",
        "What evidence would change your mind?",
        "Give me the final architecture in plain English.",
    ]

    assert [router_route(prompt).kind for prompt in prompts] == ["direct", "direct", "direct"]


def test_yes_cannot_dispatch_a_second_depth_job():
    evidence = passing_evidence()
    evidence["turns"][2]["response"] = {"dispatched_job": {"job_id": "da-duplicate-depth-2"}}

    report = validator.evaluate_transcript(evidence)
    gate = _checks(report)["exactly_one_depth_job"]

    assert gate["ok"] is False
    assert gate["detail"]["dispatches_by_turn"][2] == ["da-duplicate-depth-2"]


def test_failed_or_truncated_job_with_text_is_not_reported_as_success():
    evidence = passing_evidence()
    job = evidence["turns"][1]["job"]
    job["state"] = "failed"
    job["events"][-1] = {"type": "job.failed", "payload": {"reason": "truncated"}}
    job["result"]["finish_reason"] = "length"
    job["result"]["hivemind_warning"] = {"type": "reasoning_model_truncated"}

    report = validator.evaluate_transcript(evidence)
    gate = _checks(report)["terminal_state_and_truncation_honesty"]

    assert gate["ok"] is False
    assert gate["detail"]["job_state"] == "failed"
    assert gate["detail"]["warning_count"] == 1
    assert gate["detail"]["finish_reasons"] == ["length"]


def test_unknown_owner_must_remain_unknown_across_depth_and_followup():
    evidence = passing_evidence()
    depth = evidence["turns"][1]["job"]["result"]["text"].replace("OWNER=UNKNOWN", "OWNER=JORDAN")
    evidence["turns"][1]["job"]["result"]["text"] = depth
    evidence["turns"][3]["assistant_text"] = "Action 2 mitigates ALPHA. OWNER=JORDAN."

    report = validator.evaluate_transcript(evidence)
    checks = _checks(report)

    assert checks["unknown_owner_preserved"]["ok"] is False
    assert checks["targeted_followup_preserves_answer_and_unknown"]["ok"] is False


def test_session_drift_and_latency_are_hard_failures():
    evidence = passing_evidence()
    evidence["turns"][3]["session_id"] = "different-session"
    evidence["turns"][2]["latency_s"] = 31.0

    report = validator.evaluate_transcript(evidence)
    checks = _checks(report)

    assert checks["single_session"]["ok"] is False
    assert checks["latency_budget"]["ok"] is False


def test_custom_spec_uses_the_generic_evaluator_contract():
    evidence = passing_evidence()
    scenario = validator.canonical_scenario(NONCE)
    scenario["name"] = "custom-long-answer-profile"
    scenario["depth_min_words"] = 10_000

    report = validator.evaluate_transcript(evidence, scenario=scenario)

    assert report["scenario"] == "custom-long-answer-profile"
    assert _checks(report)["depth_substantive_without_cop_out"]["ok"] is False


def test_fixture_cli_is_offline_and_writes_machine_readable_report(tmp_path):
    fixture_path = tmp_path / "evidence.json"
    report_path = tmp_path / "report.json"
    fixture_path.write_text(json.dumps(passing_evidence()), encoding="utf-8")

    exit_code = validator.main(
        [
            "--fixture",
            str(fixture_path),
            "--output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema"] == validator.REPORT_SCHEMA
    assert report["ok"] is True


def test_live_mode_without_visible_ui_evidence_cannot_claim_full_pass(monkeypatch):
    prompts = validator.canonical_prompts(NONCE)
    turns = iter(
        [
            {
                "user": prompts[0],
                "assistant_text": "READY",
                "session_id": SESSION_ID,
                "latency_s": 0.1,
                "response": {},
            },
            {
                "user": prompts[1],
                "assistant_text": "Running the recovery analysis.",
                "session_id": SESSION_ID,
                "latency_s": 0.1,
                "response": {"dispatched_job": {"job_id": JOB_ID}},
            },
            {
                "user": prompts[2],
                "assistant_text": _action_two_walkthrough(),
                "session_id": SESSION_ID,
                "latency_s": 0.1,
                "response": {},
            },
            {
                "user": prompts[3],
                "assistant_text": "Action 2 mitigates ALPHA. OWNER=UNKNOWN.",
                "session_id": SESSION_ID,
                "latency_s": 0.1,
                "response": {},
            },
        ]
    )
    monkeypatch.setattr(validator, "_live_turn", lambda *args, **kwargs: next(turns))
    monkeypatch.setattr(
        validator,
        "_request_json",
        lambda *args, **kwargs: (
            {
                "job_id": JOB_ID,
                "state": "completed",
                "result": {
                    "job_id": JOB_ID,
                    "status": "success",
                    "text": _complete_depth_plan(),
                    "actions_taken": [{"tool": "hivemind_time_now"}],
                },
            }
            if "/events" not in args[1]
            else {
                "events": [
                    {"type": "job.tool.call.completed", "payload": {"tool": "hivemind_time_now"}},
                    {"type": "job.completed", "payload": {}},
                ]
            }
        ),
    )

    evidence = validator.run_live(
        base_url="http://127.0.0.1:9180",
        nonce=NONCE,
        face_model=None,
        depth_model=None,
        request_timeout=1.0,
        depth_timeout=1.0,
        poll_seconds=0.1,
        ui_evidence=None,
    )
    # Replace the randomized requested session with the deterministic fake
    # response session solely so this test isolates the UI-evidence gate.
    evidence["session_id"] = SESSION_ID
    evidence["reference_utc"] = REFERENCE_UTC
    report = validator.evaluate_transcript(evidence)

    assert _checks(report)["automatic_full_result_delivery"]["ok"] is False
    assert "automatic_full_result_delivery" in report["failed_checks"]
