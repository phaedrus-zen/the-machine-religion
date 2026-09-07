"""Schema validation tests.

The schema layer is the single chokepoint that enforces the
anti-hallucination event allowlist and the safe-id contract. Tests
here pin both the happy path and the refusal path so a regression in
``safety.EVENT_TYPES`` or the ``is_safe_*`` guards is caught locally
instead of leaking into the blackboard or the gateway.
"""

from __future__ import annotations

import pytest

from machine_spirit_4.double_agent import (
    AuthorityEnvelope,
    JobEnvelope,
    JobEvent,
    JobResult,
    ResourceRequest,
    SchemaError,
    StatusPolicy,
)
from machine_spirit_4.double_agent import safety
from machine_spirit_4.double_agent.schemas import sanitize_prior_context


# ---------------------------------------------------------------------------
# JobEnvelope
# ---------------------------------------------------------------------------


def _good_envelope_kwargs(**overrides):
    base = dict(
        job_id=safety.new_job_id(),
        parent_conversation_id="conv-abc",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="Diagnose why the lights did not turn on.",
        internal_goal="Check bridge, Wi-Fi, device, and command logs.",
    )
    base.update(overrides)
    return base


def test_envelope_happy_path_passes_validate():
    env = JobEnvelope(**_good_envelope_kwargs())
    env.validate()
    dumped = env.to_dict()
    assert dumped["schema"] == "DoubleAgentJobEnvelope.v1"
    assert dumped["status_policy"]["emit_raw_tokens"] is False


@pytest.mark.parametrize("lobe_type", list(safety.BACKGROUND_LOBE_TYPES))
def test_envelope_accepts_each_allowlisted_background_lobe_type(lobe_type):
    env = JobEnvelope(**_good_envelope_kwargs(background_lobe_type=lobe_type))
    env.validate()
    assert env.background_lobe_type == lobe_type


def test_envelope_refuses_unsafe_job_id():
    env = JobEnvelope(**_good_envelope_kwargs(job_id="bad id with spaces;rm -rf"))
    with pytest.raises(SchemaError):
        env.validate()


def test_envelope_refuses_unsafe_conversation_id():
    env = JobEnvelope(**_good_envelope_kwargs(parent_conversation_id="bad id;drop"))
    with pytest.raises(SchemaError):
        env.validate()


def test_envelope_refuses_unknown_background_lobe_type():
    env = JobEnvelope(**_good_envelope_kwargs(background_lobe_type="freeform"))
    with pytest.raises(SchemaError):
        env.validate()


def test_envelope_refuses_can_mutate_world_true_in_phase1():
    env = JobEnvelope(
        **_good_envelope_kwargs(
            authority=AuthorityEnvelope(can_mutate_world=True),
        )
    )
    with pytest.raises(SchemaError):
        env.validate()


def test_envelope_clamps_long_user_visible_goal():
    long_goal = "x" * (safety.USER_VISIBLE_GOAL_MAX_CHARS + 50)
    env = JobEnvelope(**_good_envelope_kwargs(user_visible_goal=long_goal))
    env.validate()
    assert len(env.user_visible_goal) <= safety.USER_VISIBLE_GOAL_MAX_CHARS


def test_envelope_round_trip_json():
    turn_id = "ms4-turn-0123456789abcdef"
    env = JobEnvelope(**_good_envelope_kwargs(turn_id=turn_id))
    env.validate()
    restored = JobEnvelope.from_dict(env.to_dict())
    assert restored.job_id == env.job_id
    assert restored.user_visible_goal == env.user_visible_goal
    assert restored.authority.can_mutate_world is False
    assert restored.turn_id == turn_id


def test_envelope_prior_context_round_trip_filters_and_clamps():
    raw_context = [
        {"role": "system", "content": "hidden grounding must not cross"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "tool", "content": "tool payload"},
        {"role": "user", "content": "x" * 1500},
        {"role": "assistant", "content": "ok\x00done"},
    ]
    env = JobEnvelope(**_good_envelope_kwargs(prior_context=raw_context))
    env.validate()
    assert [m["role"] for m in env.prior_context] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert env.prior_context[0]["content"] == "first question"
    assert len(env.prior_context[2]["content"]) <= 1200
    assert env.prior_context[3]["content"] == "okdone"

    restored = JobEnvelope.from_dict(env.to_dict())
    assert restored.prior_context == env.prior_context


def test_sanitize_prior_context_keeps_recent_tail_under_total_limit():
    raw = [
        {"role": "user", "content": f"old {i} " + ("x" * 700)}
        for i in range(20)
    ]
    sanitized = sanitize_prior_context(raw)
    assert len(sanitized) <= 12
    assert sum(len(m["content"]) for m in sanitized) <= 6000
    assert sanitized[-1]["content"].startswith("old 19")


def test_verified_depth_context_preserves_later_tradeoff_and_conclusion():
    job_id = "da-11111111-1111-4111-8111-111111111111"
    answer = (
        f"[Verified Depth Lobe result; job_id={job_id}]\n\n"
        + ("opening analysis " * 160)
        + "SECOND_TRADEOFF_CRITICAL "
        + ("supporting detail " * 160)
        + "CONCLUSION_TOKEN"
    )

    sanitized = sanitize_prior_context(
        [{"role": "assistant", "content": answer}]
    )

    assert len(sanitized) == 1
    assert len(sanitized[0]["content"]) > 1200
    assert "SECOND_TRADEOFF_CRITICAL" in sanitized[0]["content"]
    assert "CONCLUSION_TOKEN" in sanitized[0]["content"]
    assert len(sanitized[0]["content"]) <= 12000


def test_long_verified_depth_context_does_not_evict_earlier_normal_facts():
    job_id = "da-22222222-2222-4222-8222-222222222222"
    raw = [
        {"role": "user", "content": "Remember ALPHA means stale routing and the window is 12 minutes."},
        {"role": "assistant", "content": "READY"},
        {
            "role": "assistant",
            "content": (
                f"[Verified Depth Lobe result; job_id={job_id}]\n\n"
                + ("complete analysis " * 1200)
                + "FINAL_DEPTH_CONCLUSION"
            ),
        },
    ]

    sanitized = sanitize_prior_context(raw)
    combined = "\n".join(item["content"] for item in sanitized)

    assert "ALPHA means stale routing" in combined
    assert "12 minutes" in combined
    assert "FINAL_DEPTH_CONCLUSION" in combined
    assert sum(len(item["content"]) for item in sanitized) <= 16000


# ---------------------------------------------------------------------------
# JobEvent
# ---------------------------------------------------------------------------


def test_event_refuses_non_allowlisted_type():
    with pytest.raises(SchemaError):
        JobEvent.make(
            job_id="da-1",
            type="raw_reasoning_tokens",
            safe_user_status="I am thinking maybe wifi...",
        )


@pytest.mark.parametrize("event_type", list(safety.EVENT_TYPES))
def test_event_accepts_each_allowlisted_type(event_type):
    event = JobEvent.make(
        job_id="da-1",
        type=event_type,
        safe_user_status="operational fact",
    )
    assert event.type == event_type


def test_event_clamps_long_safe_user_status():
    huge = "x" * (safety.SAFE_USER_STATUS_MAX_CHARS + 100)
    event = JobEvent.make(
        job_id="da-1",
        type="job.checkpoint",
        safe_user_status=huge,
    )
    assert len(event.safe_user_status) <= safety.SAFE_USER_STATUS_MAX_CHARS


def test_event_refuses_unknown_visibility():
    with pytest.raises(SchemaError):
        JobEvent.make(
            job_id="da-1",
            type="job.checkpoint",
            safe_user_status="ok",
            visibility="leak_to_world",
        )


def test_event_refuses_unsafe_job_id():
    with pytest.raises(SchemaError):
        JobEvent.make(
            job_id="bad id;",
            type="job.checkpoint",
            safe_user_status="ok",
        )


def test_event_strips_control_characters_from_status():
    event = JobEvent.make(
        job_id="da-1",
        type="job.checkpoint",
        safe_user_status="a\x00b\x07c\nd",
    )
    assert event.safe_user_status == "abc\nd"


# ---------------------------------------------------------------------------
# JobResult
# ---------------------------------------------------------------------------


def test_result_refuses_non_allowlisted_status():
    with pytest.raises(SchemaError):
        JobResult(
            job_id="da-1",
            status="kind_of_done",
            summary="ok",
        ).validate()


def test_result_refuses_unsafe_job_id():
    with pytest.raises(SchemaError):
        JobResult(
            job_id="bad;rm",
            status="success",
            summary="ok",
        ).validate()


def test_result_round_trip_json():
    turn_id = "ms4-turn-0123456789abcdef"
    res = JobResult(
        job_id="da-1",
        status="success",
        summary="found bridge",
        text="long body of model output",
        confidence="high",
        conversation_revision_id=5,
        turn_id=turn_id,
    )
    res.validate()
    again = JobResult.from_dict(res.to_dict())
    assert again.status == "success"
    assert again.confidence == "high"
    assert again.conversation_revision_id == 5
    assert again.turn_id == turn_id


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


def test_status_policy_pins_emit_raw_tokens_false():
    p = StatusPolicy(emit_raw_tokens=True)
    assert p.to_dict()["emit_raw_tokens"] is False
    restored = StatusPolicy.from_dict({"emit_raw_tokens": True})
    assert restored.emit_raw_tokens is False


def test_resource_request_defaults_disallow_fallback():
    r = ResourceRequest()
    assert r.fallback_allowed is False
    assert r.to_dict()["fallback_allowed"] is False


def test_resource_request_preserves_mcp_hivemind_toolset():
    r = ResourceRequest(enabled_toolsets=["mcp-hivemind"])
    assert r.to_dict()["enabled_toolsets"] == ["mcp-hivemind"]
    restored = ResourceRequest.from_dict(r.to_dict())
    assert restored.enabled_toolsets == ["mcp-hivemind"]


def test_resource_request_preserves_explicit_empty_toolsets():
    request = ResourceRequest(enabled_toolsets=[])
    assert request.to_dict()["enabled_toolsets"] == []
    restored = ResourceRequest.from_dict(request.to_dict())
    assert restored.enabled_toolsets == []
