import pytest

from machine_spirit_4.double_agent import (
    Blackboard,
    JobEnvelope,
    JobResult,
    build_face_lobe_context_block,
)
from machine_spirit_4.double_agent import safety
from machine_spirit_4.double_agent.worker import (
    _depth_answer_quality_error,
    _depth_has_benchmark_decision_rule,
    _depth_is_verified_comparison_evidence,
    _depth_result_confidence,
    build_real_chat_runner,
)


COMPARISON_GOAL = (
    "Explain why a 35B Depth lobe might outperform a 4B Face lobe when diagnosing "
    "an intermittent distributed-inference failure. Give me the recommendation, "
    "mechanism, tradeoffs, and a concrete example."
)
INPUT_CONTROL_SENTENCE = (
    "Input control: Both the 35B and 4B candidates receive the same logs, traces, metrics, "
    "context, tools, and compute budget; only the candidate model differs."
)
MECHANISM_CONTROL_SENTENCE = (
    "Mechanism basis: Any potential advantage must come from the specific candidate's measured "
    "learned behavior, architecture, training, or task alignment under controlled conditions; "
    "parameter count alone supplies no causal mechanism."
)
MECHANISM_HYPOTHESIS_SENTENCE = (
    "Operational hypothesis: With inputs, context, tools, compute budget, and serving conditions "
    "held constant, the Face and Depth candidates may return different diagnostic hypotheses "
    "because their specific learned behavior or architecture may differ; only controlled benchmark "
    "results can establish whether either candidate is better for this task."
)
LATENCY_TRADEOFF_SENTENCE = (
    "- Latency or speed: Either candidate may be faster on the selected serving stack; benchmark "
    "end-to-end latency before choosing."
)
COMPUTE_TRADEOFF_SENTENCE = (
    "- Compute or cost: Either candidate may use more or less GPU/VRAM/compute on the selected "
    "serving stack; measure it under the actual architecture, quantization, offload, batching, "
    "utilization, and serving route rather than infer it from parameter count."
)
QUALITY_TRADEOFF_SENTENCE = (
    "- Accuracy, quality, reliability, operational risk, or complexity: Either candidate may "
    "diagnose the incident better or produce a more persuasive wrong answer; compare ground-truth "
    "accuracy and false negatives before changing the route."
)
OUTPUT_DIFFERENCE_SENTENCE = (
    "Possible output difference: The Face candidate may return a broad routing-symptom "
    "hypothesis, while the Depth candidate could return a more specific causal hypothesis "
    "linking an intermittent serialization delay to cross-node timing."
)


def _run_adapter(result: dict) -> tuple[dict, dict]:
    captured: dict = {}

    class _Agent:
        def run_conversation(self, message, **kwargs):
            captured["message"] = message
            captured["kwargs"] = kwargs
            return result

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **kwargs):
            captured["agent_kwargs"] = kwargs
            return _Agent()

    adapter = build_real_chat_runner(_Runner())
    response = adapter(
        message="Explain the decision and cover all three requested parts.",
        session_id="da-worker-contract",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
        conversation_history=[{"role": "user", "content": "Use the verified result above."}],
    )
    return response, captured


def _run_comparison_adapter(
    results: list[dict],
    *,
    conversation_history: list[dict] | None = None,
) -> tuple[dict, dict]:
    captured: dict = {"messages": [], "kwargs": []}
    result_iter = iter(results)

    class _Agent:
        def run_conversation(self, message, **kwargs):
            captured["messages"].append(message)
            captured["kwargs"].append(kwargs)
            return next(result_iter)

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **kwargs):
            captured["agent_kwargs"] = kwargs
            return _Agent()

    response = build_real_chat_runner(_Runner())(
        message=COMPARISON_GOAL,
        session_id="da-worker-comparison-contract",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
        enabled_toolsets=[],
        conversation_history=conversation_history,
        can_mutate_world=False,
        requires_approval_for=["deploy"],
    )
    return response, captured


def _substantive_answer() -> str:
    paragraph = (
        "The recommendation is to keep the fast foreground path responsive while assigning the "
        "complex diagnosis to the Depth lobe. The mechanism is additional model capacity applied "
        "to correlated routing, timing, accounting, and log evidence rather than a single health "
        "check. This approach has a latency tradeoff, but it reduces the larger risk of a confident "
        "answer that overlooks an intermittent peer or stale route. It also spends more GPU "
        "compute and memory, so its higher cost belongs off the latency-sensitive path. For example, preserve the "
        "request identifier, compare the selected GPU with job accounting, inspect the matching "
        "gateway and worker records, and then repeat a canary before changing traffic. "
    )
    return paragraph * 2


def _comparison_answer() -> str:
    return (
        "Recommendation: Treat the 35B model as a candidate for the Depth role, not a "
        "proven winner over the 4B model in the Face role. Face and Depth are orchestration "
        "runtime roles, not neural architecture depth or context window settings; model size "
        "alone does not determine architecture, training, tools, or retained context. No "
        "verified benchmark or incident evidence was supplied, so this comparison is a "
        "hypothesis.\n\n"
        "Mechanism:\n"
        f"{MECHANISM_CONTROL_SENTENCE}\n"
        f"{MECHANISM_HYPOTHESIS_SENTENCE}\n\n"
        "Tradeoffs:\n"
        f"{LATENCY_TRADEOFF_SENTENCE}\n"
        f"{COMPUTE_TRADEOFF_SENTENCE}\n"
        f"{QUALITY_TRADEOFF_SENTENCE}\n\n"
        "Hypothetical example:\n"
        f"{INPUT_CONTROL_SENTENCE}\n"
        f"{OUTPUT_DIFFERENCE_SENTENCE}\n\n"
        "Benchmark decision rule: Replay the same inputs through both candidates with prompts, "
        "context, and tools held constant. Compare diagnostic accuracy and false negatives, "
        "response latency, and GPU or VRAM compute cost. Choose the 35B candidate only if the "
        "measured quality gain justifies those costs. Any patch is a proposal only; validate it "
        "with a canary and obtain operator approval before applying or deploying it."
    )


def test_depth_final_answer_contract_reaches_hermes_run_conversation() -> None:
    response, captured = _run_adapter(
        {"final_response": _substantive_answer(), "completed": True}
    )

    assert response["completed"] is True
    contract = captured["kwargs"]["system_message"]
    for required in (
        "self-contained",
        "Direct conclusion first",
        "every requested part",
        "Preserve uncertainty",
        "generic invitation",
        "raw chain-of-thought",
        "250-900 visible words",
        "explicit Tradeoffs section",
        "at least three labeled material dimensions",
        "authoritative context",
        "truncated",
        "empty visible output",
    ):
        assert required in contract


def test_empty_tool_catalog_tells_depth_model_to_answer_without_skill_wandering() -> None:
    captured: dict = {}

    class _Agent:
        def run_conversation(self, _message, **kwargs):
            captured["run_kwargs"] = kwargs
            return {"final_response": _substantive_answer(), "completed": True}

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **kwargs):
            captured["agent_kwargs"] = kwargs
            return _Agent()

    response = build_real_chat_runner(_Runner())(
        message=(
            "Explain the mechanism and tradeoffs, then give a concrete example and recommendation."
        ),
        session_id="da-worker-reasoning-only",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
        enabled_toolsets=[],
    )

    assert response["completed"] is True
    assert captured["agent_kwargs"]["enabled_toolsets"] == []
    contract = captured["run_kwargs"]["system_message"]
    assert "Tool access is disabled for this analytical job" in contract
    assert "Do not invoke, search, or describe skills, sessions, memory, or tools" in contract


def test_comparison_truth_and_authority_contract_reaches_hermes() -> None:
    response, captured = _run_comparison_adapter(
        [{"final_response": _comparison_answer(), "completed": True}]
    )

    assert response["completed"] is True
    contract = captured["kwargs"][0]["system_message"]
    for required in (
        "orchestration/runtime roles",
        "not neural-architecture depth or context-window settings",
        "model size alone does not determine",
        "scheduling and orchestration assignments only",
        "Neither role inherently changes a model's context window",
        "Face model lacks a structural mechanism",
        "Depth has native memory or reasoning capabilities",
        "Never infer context-window length, token budget, context retention",
        "Parameter count establishes only the number of weights",
        "does not establish KV-cache size",
        "activation-matrix shape, execution-graph depth",
        "sequential operations per token, cross-layer communication",
        "billing, energy, or serving cost scales proportionally",
        "training-only gradient synchronization",
        "Recommendation: Treat the 35B model as a candidate for the Depth role",
        "Parameter count does not expose a model to a broader training distribution",
        "larger parameter count enables finer-grained",
        "complete plain Mechanism and Tradeoffs block",
        "Outside that block, phrase every latency, cost, and quality comparison conditionally",
        "label the comparison a hypothesis",
        "Do not invent empirical numbers",
        "Hypothetical example",
        MECHANISM_CONTROL_SENTENCE,
        MECHANISM_HYPOTHESIS_SENTENCE,
        LATENCY_TRADEOFF_SENTENCE,
        COMPUTE_TRADEOFF_SENTENCE,
        QUALITY_TRADEOFF_SENTENCE,
        INPUT_CONTROL_SENTENCE,
        OUTPUT_DIFFERENCE_SENTENCE,
        "exactly three consecutive plain, unquoted lines",
        "Possible output difference:",
        "no Markdown",
        "benchmark decision rule",
        "diagnostic accuracy, latency, and GPU/VRAM/compute cost",
        "proposal-only",
        "applicable human approval",
        "no mutation authority",
        "Required approval categories are present",
        "Return exactly the following five-paragraph answer",
    ):
        assert required in contract


def test_comparison_quality_repair_restates_truth_contract_and_can_recover() -> None:
    bad = _comparison_answer().replace(
        "No verified benchmark or incident evidence was supplied, so this comparison is a "
        "hypothesis.",
        "The 35B model will reliably outperform the 4B model.",
    )
    response, captured = _run_comparison_adapter(
        [
            {"final_response": bad, "completed": True},
            {"final_response": _comparison_answer(), "completed": True},
        ]
    )

    assert response["completed"] is True
    assert response["quality_repair_attempted"] is True
    assert len(captured["messages"]) == 2
    repair = captured["messages"][1]
    for required in (
        "separate Face/Depth runtime roles from model architecture",
        "neither role inherently provides memory, state retention",
        "Face lacks a structural mechanism",
        "Depth has native capabilities",
        "Never infer a longer context window, later token exhaustion, or higher accuracy",
        "4B implies deterministic fallback",
        "35B requires larger batch capacity",
        "has larger activation matrices, creates deeper execution graphs",
        "requires more sequential operations per token",
        "incurs greater cross-layer communication",
        "proportional billing/energy/cost law",
        "training-only gradient synchronization",
        "Delete any claim that parameter count exposes a broader training distribution",
        "larger parameter count enables finer-grained dependency encoding",
        "complete plain Mechanism and Tradeoffs block",
        "Recommendation: Treat the 35B model as a candidate for the Depth role",
        "Face and Depth are orchestration runtime roles, not neural architecture depth",
        "Parameter count establishes only weight count",
        INPUT_CONTROL_SENTENCE,
        MECHANISM_CONTROL_SENTENCE,
        MECHANISM_HYPOTHESIS_SENTENCE,
        LATENCY_TRADEOFF_SENTENCE,
        COMPUTE_TRADEOFF_SENTENCE,
        QUALITY_TRADEOFF_SENTENCE,
        OUTPUT_DIFFERENCE_SENTENCE,
        "exactly these three consecutive plain, unquoted lines",
        "Possible output difference:",
        "unsupported superiority as a hypothesis",
        "unattributed empirical numbers",
        "same-input benchmark decision rule",
        "proposal-only",
        "applicable human approval",
        "Return exactly the following five-paragraph answer",
    ):
        assert required in repair


def test_comparison_quality_repair_fails_closed_on_second_bad_draft() -> None:
    bad = _comparison_answer().replace(
        "No verified benchmark or incident evidence was supplied, so this comparison is a "
        "hypothesis.",
        "The 35B model will reliably outperform the 4B model.",
    )
    response, _captured = _run_comparison_adapter(
        [
            {"final_response": bad, "completed": True},
            {"final_response": bad, "completed": True},
        ]
    )

    assert response["completed"] is False
    assert response["quality_repair_attempted"] is True
    assert response["error"] == "model_result_epistemic_overclaim"


def test_prior_assistant_claim_is_not_treated_as_verified_comparison_evidence() -> None:
    bad = _comparison_answer().replace(
        "No verified benchmark or incident evidence was supplied, so this comparison is a "
        "hypothesis.",
        "The 35B model will reliably outperform the 4B model.",
    )
    response, _captured = _run_comparison_adapter(
        [
            {"final_response": bad, "completed": True},
            {"final_response": bad, "completed": True},
        ],
        conversation_history=[
            {
                "role": "assistant",
                "content": "Benchmark results showed that the 35B model wins reliably.",
            }
        ],
    )

    assert response["completed"] is False
    assert response["error"] == "model_result_epistemic_overclaim"


def test_one_line_teaser_is_repaired_once_then_fails_closed() -> None:
    response, captured = _run_adapter(
        {
            "final_response": "Here is the first sentence. Want to know more?",
            "completed": True,
        }
    )

    assert response["completed"] is False
    assert response["error"] == "model_result_cop_out"
    assert response["quality_repair_attempted"] is True
    assert captured["message"].startswith("Rewrite your previous draft")


def test_quality_repair_can_promote_one_complete_replacement() -> None:
    captured: dict = {"messages": []}
    results = iter(
        [
            {"final_response": "Here is the first sentence. Want to know more?", "completed": True},
            {"final_response": _substantive_answer(), "completed": True},
        ]
    )

    class _Agent:
        def run_conversation(self, message, **kwargs):
            captured["messages"].append(message)
            captured["kwargs"] = kwargs
            return next(results)

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **kwargs):
            captured["agent_kwargs"] = kwargs
            return _Agent()

    response = build_real_chat_runner(_Runner())(
        message="Explain the decision and cover all three requested parts.",
        session_id="da-worker-repair",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["completed"] is True
    assert response["quality_repair_attempted"] is True
    assert response["text"] == _substantive_answer()
    assert len(captured["messages"]) == 2
    assert captured["messages"][1].startswith("Rewrite your previous draft")


def test_structural_exact_counts_do_not_bypass_depth_substance_gate() -> None:
    goal = (
        "Create a complete recovery plan with exactly four actions and exactly two risks. "
        "Explain the mechanism, tradeoffs, validation, and a concrete example."
    )

    assert _depth_answer_quality_error(
        goal,
        "SUMMARY. Route stale. ACTIONS. Reset it. RISKS. It may fail.",
    ) == "model_result_insufficient_substance"


def test_exact_video_prompt_accepts_recommendation_noun_as_requested_evidence() -> None:
    assert _depth_answer_quality_error(COMPARISON_GOAL, _comparison_answer()) is None


def test_comparison_rejects_unqualified_reliable_superiority() -> None:
    candidate = _comparison_answer().replace(
        "No verified benchmark or incident evidence was supplied, so this comparison is a "
        "hypothesis.",
        "The 35B model will reliably outperform the 4B model.",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_epistemic_overclaim"
    )


def test_comparison_accepts_hypothesized_and_hypothetical_qualifiers() -> None:
    candidate = _comparison_answer().replace(
        "No verified benchmark or incident evidence was supplied, so this comparison is a "
        "hypothesis.",
        "The outcome is hypothesized rather than measured, and the comparison remains "
        "hypothetical until a controlled benchmark validates it.",
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None

    response, captured = _run_comparison_adapter(
        [
            {
                "final_response": candidate,
                "completed": True,
                "failed": False,
                "turn_exit_reason": "text_response(finish_reason=stop)",
            },
            {"final_response": _comparison_answer(), "completed": True},
        ],
        conversation_history=[
            {"role": "user", "content": "Verified benchmark results measured latency."}
        ],
    )
    assert response["completed"] is True
    assert response["quality_repair_attempted"] is True
    assert "error" not in response
    assert len(captured["messages"]) == 2


@pytest.mark.parametrize(
    "source_context",
    (
        "Have we measured latency yet? No.",
        "Suppose we measured accuracy tomorrow.",
        "No measured latency exists for these candidates.",
        "Use a measured latency result when it becomes available.",
    ),
)
def test_comparison_history_does_not_self_certify_verified_evidence(
    source_context: str,
) -> None:
    bad = _comparison_answer().replace(
        "No verified benchmark or incident evidence was supplied, so this comparison is a hypothesis.",
        "The 35B will reliably outperform the 4B.",
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        bad,
        source_context=source_context,
    ) is not None


def test_no_tool_result_self_certifies_comparison_evidence_without_claim_binding() -> None:
    assert _depth_is_verified_comparison_evidence("hivemind_health", "healthy") is False
    assert _depth_is_verified_comparison_evidence(
        "hivemind_metrics",
        {"status": "success", "text": "Measured latency result: 3 seconds."},
    ) is False
    assert _depth_is_verified_comparison_evidence(
        "hivemind_metrics",
        {
            "status": "success",
            "verified_evidence": True,
            "text": "Measured latency result: 3 seconds.",
        },
    ) is False
    for payload in (
        {"verified_evidence": True, "text": "No measured latency exists for either candidate."},
        {"verified_evidence": True, "text": "Have we measured latency yet? No."},
        {"verified_evidence": True, "text": "Suppose we measured accuracy tomorrow."},
        {
            "verified_evidence": True,
            "failed": True,
            "error": "Measured latency result unavailable.",
        },
    ):
        assert _depth_is_verified_comparison_evidence("hivemind_metrics", payload) is False


def test_comparison_accepts_equivalent_designation_and_configuration_wording() -> None:
    candidate = _comparison_answer().replace(
        "Face and Depth are orchestration runtime roles, not neural architecture depth or "
        "context window settings; model size alone does not determine architecture, training, "
        "tools, or retained context.",
        "The Face and Depth designations are purely scheduling and runtime assignments; they do "
        "not inherently alter neural architecture, context window length, memory retention, tool "
        "access, or core capability. Any potential advantage must be attributed to the candidate "
        "model's explicit specifications, supplied context, compute allocation, or instrumented "
        "data pipelines.",
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


def test_comparison_rejects_role_architecture_and_context_conflation() -> None:
    candidate = _comparison_answer().replace(
        "Face and Depth are orchestration runtime roles, not neural architecture depth or "
        "context window settings; model size alone does not determine architecture, training, "
        "tools, or retained context.",
        "The Depth lobe has deeper attention pathways, while the Face lobe is a shallow model "
        "whose parameter count determines its context window.",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_role_architecture_conflation"
    )


def test_comparison_rejects_parameter_count_as_context_window_proxy() -> None:
    candidate = _comparison_answer() + (
        " The larger weight matrix supports longer multi-turn diagnostic context without early "
        "token exhaustion and helps integrate several supplied trace clues."
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_role_architecture_conflation"
    )

    smaller_context = _comparison_answer() + (
        " The 4B Face instance misses the symptom because its context window truncates earlier."
    )
    assert _depth_answer_quality_error(COMPARISON_GOAL, smaller_context) == (
        "model_result_role_architecture_conflation"
    )


def test_comparison_allows_explicitly_configured_longer_context() -> None:
    candidate = _comparison_answer() + (
        " Larger parameter counts can capture additional statistical dependencies when the "
        "system is explicitly configured to supply longer context windows and matched tools."
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


def test_comparison_accepts_weight_count_only_size_separation() -> None:
    candidate = _comparison_answer().replace(
        "model size alone does not determine architecture, training, tools, or retained context.",
        "Parameter count establishes only weight count.",
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


def test_comparison_rejects_unmeasured_higher_accuracy_claim() -> None:
    candidate = _comparison_answer() + (
        " The larger model yields higher diagnostic accuracy for these failures."
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_epistemic_overclaim"
    )


def test_comparison_accepts_explicitly_conditional_higher_accuracy_claim() -> None:
    candidate = _comparison_answer() + (
        " The 35B candidate could deliver higher diagnostic accuracy only if a controlled "
        "benchmark confirms it under matched serving conditions."
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


@pytest.mark.parametrize(
    "claim",
    (
        " Cloud billing and energy costs scale proportionally with active parameter count.",
        " The hypothetical inference incident shows gradient desync during distributed inference.",
    ),
)
def test_comparison_rejects_unsupported_cost_law_or_inference_gradient_claim(
    claim: str,
) -> None:
    assert _depth_answer_quality_error(COMPARISON_GOAL, _comparison_answer() + claim) == (
        "model_result_epistemic_overclaim"
    )


def test_comparison_accepts_disavowed_proportional_cost_law() -> None:
    candidate = _comparison_answer() + (
        " Billing does not scale proportionally with parameter count because hardware, "
        "batching, utilization, and pricing intervene."
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


def test_comparison_rejects_unmeasured_false_negative_improvement() -> None:
    candidate = _comparison_answer() + (
        " The 35B architecture generally captures richer relationships, reducing false "
        "negatives in cross-layer failure correlation."
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_epistemic_overclaim"
    )


def test_comparison_rejects_size_as_serving_configuration_proxy() -> None:
    invalid_claims = (
        " The 4B model prioritizes deterministic fallback behavior.",
        " The 4B model has a smaller KV-cache footprint.",
        " The 4B model uses reduced activation paths.",
        " The 35B model requires larger batch capacity.",
        " The 35B parameter model has larger KV-cache allocations.",
        " A larger parameter count implies a larger KV-cache allocation.",
    )
    for claim in invalid_claims:
        assert _depth_answer_quality_error(COMPARISON_GOAL, _comparison_answer() + claim) == (
            "model_result_role_architecture_conflation"
        )


@pytest.mark.parametrize(
    "claim",
    (
        " Depth mode incurs higher latency due to deeper execution graphs in the "
        "orchestration layer.",
        " The 35B model has larger activation matrices.",
        " Depth mode creates deeper execution graphs but may not improve quality.",
        " The 35B model comes with larger activation matrices.",
        " More parameters imply larger activation matrices.",
        " The 35B model has a larger activation matrix.",
        " The 35B model is built with larger activation matrices.",
        " The 35B model has greater execution-graph depth.",
        " The 35B model does not only have larger activation matrices; it also creates "
        "deeper execution graphs.",
        " The 35B model, unlike a smaller model, has larger activation matrices.",
        " Larger parameter sets require more sequential operations per token and greater "
        "cross-layer communication overhead during query execution.",
        " The larger parameter count exposes the architecture to a broader training distribution.",
        " Larger parameter counts may capture deeper reasoning nuances.",
        " A larger parameter count may allow the candidate to encode finer-grained cross-entity "
        "relationships and capture subtle statistical dependencies.",
        " The larger candidate may require more sequential token generation steps.",
        " Increased weight count raises peak VRAM demand and FLOPs per token.",
        " Larger parameter count enables finer-grained dependency encoding.",
        " The larger candidate requires more sequential token-generation steps.",
        " Additional weights increase representational capacity for complex dependencies.",
        " The 35B candidate has higher representational density.",
        " The 4B candidate relies on shallower feature extraction.",
        " The expanded weight matrix may improve root-cause attribution.",
        " The 35B candidate may incur higher latency due to greater parameter load.",
        " The larger model may require more GPU/VRAM allocation per diagnostic cycle.",
    ),
)
def test_comparison_rejects_role_or_size_as_execution_graph_proxy(
    claim: str,
) -> None:
    assert _depth_answer_quality_error(COMPARISON_GOAL, _comparison_answer() + claim) == (
        "model_result_role_architecture_conflation"
    )


def test_comparison_rejects_unsupported_observed_advantage_wording() -> None:
    candidate = _comparison_answer() + (
        " The observed diagnostic advantage would stem from the candidate configuration."
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_epistemic_overclaim"
    )


def test_comparison_requires_an_explicit_model_recommendation() -> None:
    candidate = _comparison_answer().replace("Recommendation:", "Background:")

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_missing_requested_parts"
    )


@pytest.mark.parametrize(
    "claim",
    (
        " A 35B model does not inherently have larger activation matrices.",
        " Depth mode does not create deeper execution graphs.",
        " A larger model is not guaranteed to have larger activation matrices or deeper "
        "execution graphs.",
        " The 35B model has no larger activation matrices.",
        " The 35B model has neither larger activation matrices nor deeper execution graphs.",
        " A larger model may have larger activation matrices only if its explicit architecture "
        "specifies them.",
        " Only if explicitly specified by architecture, the 35B model may have larger "
        "activation matrices.",
        " The 35B model, only if explicitly specified by architecture, may have larger "
        "activation matrices.",
        " A 35B model can coexist with a smaller model that has larger activation matrices.",
        " The 35B model delegates work to a 4B candidate that uses larger activation matrices.",
        " The 35B model has a smaller sibling with larger activation matrices.",
        " Larger parameter sets do not require more sequential operations per token.",
        " The 35B model requires no greater cross-layer communication overhead.",
    ),
)
def test_comparison_accepts_disavowed_execution_graph_conflation(
    claim: str,
) -> None:
    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        _comparison_answer() + claim,
        verified_evidence_present=True,
    ) is None


@pytest.mark.parametrize(
    "claim",
    (
        " A larger model does not by itself imply a larger KV-cache allocation.",
        " A 35B parameter model is not guaranteed to have a larger KV-cache footprint.",
        " A larger model may not have a larger KV-cache allocation.",
        " A larger model may have a larger KV-cache allocation only if its explicit "
        "architecture and serving configuration require it.",
        " Compared with a larger model, the smaller model has more KV-cache allocation.",
    ),
)
def test_comparison_accepts_disavowed_or_attributed_kv_cache_claim(
    claim: str,
) -> None:
    candidate = _comparison_answer() + claim

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


def test_comparison_rejects_exact_live_kv_cache_conflation() -> None:
    candidate = _comparison_answer() + (
        " Running a 35B parameter model could consume more FLOPs per inference step and "
        "larger KV-cache allocations than a 4B model under matched context lengths."
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_role_architecture_conflation"
    )


@pytest.mark.parametrize(
    "replacement",
    ("Causal explanation:", "Explanation due to configured differences:"),
)
def test_comparison_rejects_noncanonical_causal_heading(
    replacement: str,
) -> None:
    candidate = (
        _comparison_answer()
        .replace("Mechanism:", replacement)
        .replace("several testable causes", "several testable hypotheses")
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_missing_requested_parts"
    )


def test_comparison_rejects_answer_without_mechanism_evidence() -> None:
    candidate = (
        _comparison_answer()
        .replace("Mechanism:", "Explanation:")
        .replace("several testable causes", "several testable hypotheses")
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_missing_requested_parts"
    )


def test_comparison_accepts_exact_canonical_blocks_with_crlf_and_trailing_spaces() -> None:
    candidate = "\r\n".join(
        line + ("  " if line else "") for line in _comparison_answer().split("\n")
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) is None


def test_comparison_accepts_trailing_horizontal_whitespace_on_blank_separators() -> None:
    candidate = "\r\n".join(line + " \t" for line in _comparison_answer().split("\n"))

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) is None


@pytest.mark.parametrize(
    "candidate",
    (
        "\n" + _comparison_answer(),
        "  " + _comparison_answer(),
        _comparison_answer() + "\n\n",
        _comparison_answer() + "\n \t\n",
    ),
)
def test_evidence_free_template_rejects_leading_or_extra_trailing_blank_space(
    candidate: str,
) -> None:
    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_noncanonical_comparison_answer"
    )


def test_evidence_free_template_accepts_one_optional_terminal_line_ending() -> None:
    assert _depth_answer_quality_error(COMPARISON_GOAL, _comparison_answer() + "\n") is None
    assert _depth_answer_quality_error(COMPARISON_GOAL, _comparison_answer() + "\r\n") is None


def test_evidence_free_template_is_embedded_exactly_in_system_and_repair_prompts() -> None:
    bad = _comparison_answer() + " Extra sentence."
    response, captured = _run_comparison_adapter(
        [
            {"final_response": bad, "completed": True},
            {"final_response": _comparison_answer(), "completed": True},
        ]
    )

    assert response["completed"] is True
    assert response["quality_repair_attempted"] is True
    assert _comparison_answer() in captured["kwargs"][0]["system_message"]
    assert _comparison_answer() in captured["messages"][1]
    assert captured["kwargs"][0]["system_message"].endswith(_comparison_answer())
    assert captured["messages"][1].endswith(_comparison_answer())


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: "Prelude. " + value,
        lambda value: value + " Extra sentence.",
        lambda value: value + "\n\nExtra paragraph.",
        lambda value: value.replace("\n\nMechanism:", "\n\nEverything below is false.\n\nMechanism:"),
        lambda value: value.replace(
            "\n\nMechanism:",
            "\n\nDo not accept the following mechanism section.\n\nMechanism:",
        ),
        lambda value: value.replace(
            "\n\nMechanism:",
            "\n\nThe entire mechanism section below is false.\n\nMechanism:",
        ),
        lambda value: value.replace(
            "\n\nBenchmark decision rule:",
            "\n\nThe canonical mechanism block above is false.\n\nBenchmark decision rule:",
        ),
        lambda value: value.replace(
            "\n\nBenchmark decision rule:",
            "\n\nThe input-control statement above is false.\n\nBenchmark decision rule:",
        ),
        lambda value: value.replace(
            "\n\nMechanism:",
            "\n\n‘\n\nMechanism:",
        ).replace("\n\nBenchmark decision rule:", "\n\n’\n\nBenchmark decision rule:"),
        lambda value: value.replace(
            "\n\nMechanism:",
            "\n\n<!--\n\nMechanism:",
        ).replace("\n\nBenchmark decision rule:", "\n\n-->\n\nBenchmark decision rule:"),
    ),
)
def test_evidence_free_template_rejects_prefix_suffix_wrapper_and_retraction(
    mutation,
) -> None:
    assert _depth_answer_quality_error(COMPARISON_GOAL, mutation(_comparison_answer())) is not None


@pytest.mark.parametrize(
    "claim",
    (
        "**Tradeoffs:** The 4B is faster and cheaper, while the 35B is more accurate and reliable.",
        "The 35B candidate may be slower because it has more parameters.",
        "More weights necessarily raise compute cost.",
        "More parameters let the model track more simultaneous causal hypotheses.",
        "A larger learned representation preserves rare failure patterns more reliably.",
        "The 35B candidate produces fewer false positives through deeper reasoning.",
        "The smaller 4B footprint guarantees more throughput.",
    ),
)
def test_evidence_free_template_rejects_comparative_claim_appended_outside_schema(
    claim: str,
) -> None:
    candidate = _comparison_answer() + f" {claim}"

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) is not None


def test_benchmark_section_does_not_absorb_crlf_appendix_keywords() -> None:
    candidate = _comparison_answer().replace(
        "Benchmark decision rule: Replay the same inputs through both candidates with prompts, "
        "context, and tools held constant. Compare diagnostic accuracy and false negatives, "
        "response latency, and GPU or VRAM compute cost. Choose the 35B candidate only if the "
        "measured quality gain justifies those costs.",
        "Benchmark decision rule: Replay the exact workload.\r\n\r\n"
        "Appendix: hold the same controlled conditions and compare accuracy, latency, and "
        "GPU/VRAM/compute cost.",
    )

    assert _depth_has_benchmark_decision_rule(candidate) is False
    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) is not None


@pytest.mark.parametrize(
    "insertion",
    (
        "More weights may afford a denser internal representation of complex service interactions.",
        "A higher weight count may yield greater representational capacity for distributed failure patterns.",
        "The 35B candidate may use denser parameterization to represent subtler cross-service dependencies.",
        "The 4B candidate might use coarser feature extraction and miss subtle timing failures.",
        "A bigger weight tensor may improve causal diagnosis.",
        "The larger checkpoint may take longer to process because it contains more weights.",
        "The 35B candidate may occupy more accelerator memory because its checkpoint is larger.",
        "The 35B candidate may track more concurrent hypotheses because it has more weights.",
        "More weights may mean better correlation across service boundaries.",
        "This mechanism basis is false.",
    ),
)
def test_comparison_rejects_free_prose_inside_canonical_mechanism_block(
    insertion: str,
) -> None:
    candidate = _comparison_answer().replace(
        f"{MECHANISM_CONTROL_SENTENCE}\n{MECHANISM_HYPOTHESIS_SENTENCE}",
        f"{MECHANISM_CONTROL_SENTENCE}\n{insertion}\n{MECHANISM_HYPOTHESIS_SENTENCE}",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_missing_requested_parts"
    )


@pytest.mark.parametrize(
    "replacement",
    (
        f"{MECHANISM_CONTROL_SENTENCE}\nWrapped mechanism line.\n{MECHANISM_HYPOTHESIS_SENTENCE}",
        f"{MECHANISM_CONTROL_SENTENCE}\n{MECHANISM_HYPOTHESIS_SENTENCE}\nExtra mechanism line.",
        f"{MECHANISM_CONTROL_SENTENCE.replace('parameter count', 'weight count')}\n"
        f"{MECHANISM_HYPOTHESIS_SENTENCE}",
        f"{MECHANISM_CONTROL_SENTENCE}\n"
        f"{MECHANISM_HYPOTHESIS_SENTENCE.replace('controlled benchmark', 'careful benchmark')}",
    ),
)
def test_comparison_rejects_changed_or_expanded_canonical_mechanism(
    replacement: str,
) -> None:
    candidate = _comparison_answer().replace(
        f"{MECHANISM_CONTROL_SENTENCE}\n{MECHANISM_HYPOTHESIS_SENTENCE}",
        replacement,
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_missing_requested_parts"
    )


@pytest.mark.parametrize(
    "wrapper",
    (
        "'",
        '"""',
        "'''",
        ">",
        '<blockquote class="quoted">',
        "<q>",
        "<pre>",
        "<code>",
        "<template>",
        "The following mechanism section is false.",
        "The below mechanism block is not true.",
    ),
)
def test_comparison_rejects_wrapped_or_disclaimed_canonical_mechanism(
    wrapper: str,
) -> None:
    candidate = _comparison_answer().replace(
        "\n\nMechanism:\n",
        f"\n\n{wrapper}\n\nMechanism:\n",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) in {
        "model_result_missing_requested_parts",
        "model_result_hypothetical_inputs_not_held_constant",
    }


def test_comparison_rejects_duplicate_or_conflicting_mechanism_and_tradeoff_sections() -> None:
    duplicate_mechanism = _comparison_answer() + (
        f"\n\nMechanism:\n{MECHANISM_CONTROL_SENTENCE}\n{MECHANISM_HYPOTHESIS_SENTENCE}"
    )
    duplicate_tradeoffs = _comparison_answer() + "\n\nTradeoffs:\nConflicting claim."

    for candidate in (duplicate_mechanism, duplicate_tradeoffs):
        assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
            "model_result_missing_requested_parts"
        )


@pytest.mark.parametrize(
    "old,new",
    (
        (LATENCY_TRADEOFF_SENTENCE, LATENCY_TRADEOFF_SENTENCE + " More weights make it slower."),
        (COMPUTE_TRADEOFF_SENTENCE, COMPUTE_TRADEOFF_SENTENCE.replace("Either", "The 35B")),
        (QUALITY_TRADEOFF_SENTENCE, QUALITY_TRADEOFF_SENTENCE.replace("may", "will", 1)),
    ),
)
def test_comparison_rejects_changed_or_expanded_canonical_tradeoff_lines(
    old: str,
    new: str,
) -> None:
    candidate = _comparison_answer().replace(old, new)

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_missing_requested_parts"
    )


def test_comparison_hypothetical_requires_same_evidence_for_both_candidates() -> None:
    candidate = _comparison_answer().replace(
        INPUT_CONTROL_SENTENCE,
        "The Face candidate receives raw access logs, while the Depth candidate receives metrics, "
        "traces, and thread dumps.",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_hypothetical_inputs_not_held_constant"
    )


def test_comparison_rejects_parenthetical_hypothetical_heading() -> None:
    candidate = _comparison_answer().replace(
        "Hypothetical example:",
        "Concrete Example (Hypothetical):",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_hypothetical_inputs_not_held_constant"
    )


@pytest.mark.parametrize(
    "replacement",
    (
        "Both candidates receive identical inputs.",
        "The same evidence is provided to each candidate.",
        "Both candidates get identical inputs.",
        "Both candidates receive identical inputs, but Depth gets additional traces.",
        "Both candidates receive identical logs but different prompts.",
        "Input control: Both the 35B and 4B candidates receive the same logs, traces, metrics, "
        "context, tools, and budget; only the candidate model differs.",
        "input control: Both the 35B and 4B candidates receive the same logs, traces, metrics, "
        "context, tools, and compute budget; only the candidate model differs.",
    ),
)
def test_comparison_rejects_noncanonical_input_control_wording(replacement: str) -> None:
    candidate = _comparison_answer().replace(INPUT_CONTROL_SENTENCE, replacement)

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_hypothetical_inputs_not_held_constant"
    )


def test_comparison_rejects_line_wrapped_canonical_input_control() -> None:
    candidate = _comparison_answer().replace(
        "same logs, traces, metrics, context, tools, and compute budget",
        "same logs, traces, metrics,\ncontext, tools, and compute budget",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_hypothetical_inputs_not_held_constant"
    )


@pytest.mark.parametrize(
    "replacement",
    (
        f"'{INPUT_CONTROL_SENTENCE}' is false.",
        f"‘{INPUT_CONTROL_SENTENCE}’ is false.",
        f"It is false that {INPUT_CONTROL_SENTENCE}",
        f"> {INPUT_CONTROL_SENTENCE}",
        INPUT_CONTROL_SENTENCE.replace("Input control:", "**Input control:**"),
        f"Hypothetical example:\n\n{INPUT_CONTROL_SENTENCE}",
    ),
)
def test_comparison_rejects_quoted_negated_or_decorated_input_control(replacement: str) -> None:
    candidate = _comparison_answer().replace(
        f"Hypothetical example:\n{INPUT_CONTROL_SENTENCE}",
        replacement,
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) is not None


@pytest.mark.parametrize(
    "replacement",
    (
        "Possible output difference: However, Depth receives additional traces.",
        "Possible output difference: The 35B candidate gets richer context.",
        "Possible output difference: The candidates use different tools.",
        "Possible output difference: Only Depth reads the logs; Face cannot.",
        "Possible output difference: Depth sees the entire trace while Face sees only part.",
        "Possible output difference: Face lacks several metrics that Depth can inspect.",
        "Possible output difference: Depth starts with a richer system message.",
        "Possible output difference: Depth consults a private scratchpad unavailable to Face.",
        "Possible output difference: Depth reads a broader telemetry slice than Face.",
    ),
)
def test_comparison_rejects_input_asymmetry_in_output_difference_line(
    replacement: str,
) -> None:
    candidate = _comparison_answer().replace(
        OUTPUT_DIFFERENCE_SENTENCE,
        replacement,
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_hypothetical_inputs_not_held_constant"
    )


@pytest.mark.parametrize(
    "replacement",
    (
        "Possible output difference:  .",
        "Possible output difference: N/A.",
        "Possible output difference: None.",
        "Possible output difference: <one sentence about output behavior only>.",
        "Possible output difference: The Depth candidate produces an answer.",
        "Possible output difference: The Depth candidate produces one answer while the Face "
        "candidate produces another without any stated uncertainty.",
    ),
)
def test_comparison_rejects_non_substantive_or_nonconditional_output_difference(
    replacement: str,
) -> None:
    candidate = _comparison_answer().replace(
        OUTPUT_DIFFERENCE_SENTENCE,
        replacement,
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_hypothetical_inputs_not_held_constant"
    )


def test_comparison_rejects_interstitial_asymmetry_before_output_line() -> None:
    candidate = _comparison_answer().replace(
        f"{INPUT_CONTROL_SENTENCE}\nPossible output difference:",
        f"{INPUT_CONTROL_SENTENCE}\nHowever, Depth receives additional traces.\n"
        "Possible output difference:",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_hypothetical_inputs_not_held_constant"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda answer: answer.replace(
            "Benchmark decision rule:",
            "However, Depth receives additional traces.\n\nBenchmark decision rule:",
            1,
        ),
        lambda answer: answer.replace(
            "Hypothetical example:\n",
            "The following schema is false:\nHypothetical example:\n",
            1,
        ),
        lambda answer: answer.replace(
            "Hypothetical example:\n",
            "The following schema is false:\n\nHypothetical example:\n",
            1,
        ),
        lambda answer: answer.replace(
            "Benchmark decision rule:",
            "Hypothetical example:\nBoth candidates receive different evidence.\n\n"
            "Benchmark decision rule:",
            1,
        ),
        lambda answer: answer.replace(
            "Hypothetical example:\n",
            '"\n\nHypothetical example:\n',
            1,
        ).replace("\n\nBenchmark decision rule:", '\n\nBenchmark decision rule:\n"', 1),
        lambda answer: answer.replace(
            "Hypothetical example:\n",
            "'\n\nHypothetical example:\n",
            1,
        ).replace("\n\nBenchmark decision rule:", "\n\nBenchmark decision rule:\n'", 1),
        lambda answer: answer.replace(
            "Hypothetical example:\n",
            "‘\n\nHypothetical example:\n",
            1,
        ).replace("\n\nBenchmark decision rule:", "\n\nBenchmark decision rule:\n’", 1),
        lambda answer: answer.replace(
            "Hypothetical example:\n",
            "<blockquote>\n\nHypothetical example:\n",
            1,
        ).replace("\n\nBenchmark decision rule:", "\n</blockquote>\n\nBenchmark decision rule:", 1),
        lambda answer: answer.replace(
            "Hypothetical example:\n",
            "<q>\n\nHypothetical example:\n",
            1,
        ).replace("\n\nBenchmark decision rule:", "\n</q>\n\nBenchmark decision rule:", 1),
        lambda answer: answer.replace(
            "\n\nBenchmark decision rule:",
            "\n\nArbitrary prose between the schema and benchmark.\n\nBenchmark decision rule:",
            1,
        ),
        lambda answer: answer.replace(
            OUTPUT_DIFFERENCE_SENTENCE,
            "Possible output difference: The Face candidate finds a symptom. The Depth candidate "
            "finds a cause.",
            1,
        ),
        lambda answer: answer.replace(
            "Hypothetical example:\n",
            "Benchmark decision rule: Replay the same inputs and compare accuracy, latency, and "
            "GPU cost.\n\nHypothetical example:\n",
            1,
        ).replace(
            "Benchmark decision rule: Replay the same inputs through both candidates with prompts, "
            "context, and tools held constant. Compare diagnostic accuracy and false negatives, "
            "response latency, and GPU or VRAM compute cost. Choose the 35B candidate only if the "
            "measured quality gain justifies those costs.",
            "Benchmark decision rule:",
            1,
        ),
    ),
)
def test_comparison_rejects_structural_input_control_bypasses(mutation) -> None:
    assert _depth_answer_quality_error(COMPARISON_GOAL, mutation(_comparison_answer())) is not None


def test_comparison_rejects_unattributed_empirical_numbers() -> None:
    candidate = _comparison_answer() + (
        " The Depth path typically adds 20-40% latency and responds in sub-10ms."
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_ungrounded_numeric_claim"
    )


def test_comparison_accepts_postposed_plural_numeric_estimates() -> None:
    candidate = _comparison_answer() + (
        " The 4B model fits in ~8-16 GB VRAM estimates."
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


def test_comparison_requires_hypothetical_label_for_evidence_free_example() -> None:
    candidate = _comparison_answer().replace(
        "Hypothetical example:",
        "Concrete example:",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_unlabeled_hypothetical_example"
    )


def test_comparison_accepts_exact_workload_benchmark_wording() -> None:
    candidate = _comparison_answer().replace(
        "Replay the same inputs through both candidates",
        "Replay the exact diagnostic workload through both candidates",
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


def test_comparison_requires_same_input_multidimensional_benchmark_rule() -> None:
    candidate = _comparison_answer().replace(
        "Benchmark decision rule: Replay the same inputs through both candidates with prompts, "
        "context, and tools held constant. Compare diagnostic accuracy and false negatives, "
        "response latency, and GPU or VRAM compute cost. Choose the 35B candidate only if the "
        "measured quality gain justifies those costs.",
        "Benchmark decision rule: Validation should compare the candidates before choosing one.",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_benchmark_decision_rule_missing"
    )


def test_comparison_rejects_automatic_or_unapproved_mutation_language() -> None:
    candidate = _comparison_answer().replace(
        "Any patch is a proposal only; validate it with a canary and obtain operator approval "
        "before applying or deploying it.",
        "The system automatically applies the patch and deploys it.",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_unsafe_mutation_language"
    )


def test_comparison_rejects_observed_automatic_patch_wording() -> None:
    candidate = _comparison_answer().replace(
        "Any patch is a proposal only; validate it with a canary and obtain operator approval "
        "before applying or deploying it.",
        "Action patches are applied automatically or flagged for SRE review.",
    )

    assert _depth_answer_quality_error(COMPARISON_GOAL, candidate) == (
        "model_result_unsafe_mutation_language"
    )


def test_comparison_preserves_quoted_and_disavowed_bad_claims() -> None:
    candidate = _comparison_answer() + (
        ' The statement "A 35B will reliably outperform a 4B" is unsupported. '
        "Do not apply patches automatically; validate a canary and obtain approval."
    )

    assert _depth_answer_quality_error(
        COMPARISON_GOAL,
        candidate,
        verified_evidence_present=True,
    ) is None


def test_comparison_result_confidence_is_low_without_verified_evidence() -> None:
    assert _depth_result_confidence(COMPARISON_GOAL, _comparison_answer(), []) == "low"
    assert _depth_result_confidence(
        COMPARISON_GOAL,
        _comparison_answer(),
        [{"kind": "verified_tool_result"}],
    ) == "medium"
    assert _depth_result_confidence(
        "Explain a generic architecture.",
        _substantive_answer(),
        [],
    ) == "medium"


def test_requested_tradeoffs_require_three_distinct_material_dimensions() -> None:
    goal = (
        "Explain why a larger Depth lobe might outperform a smaller Face lobe. Give me the "
        "recommendation, mechanism, tradeoffs, and a concrete example."
    )
    two_dimension_paragraph = (
        "Recommendation: use the 35B Depth lobe for the difficult diagnosis while keeping the "
        "4B Face lobe responsive. The mechanism correlates observations across time and nodes "
        "instead of judging one health check. Tradeoffs include slower first-token latency and "
        "higher GPU compute, memory, and operating cost. For example, correlate one request "
        "identifier through gateway routing, worker accounting, and the selected peer before "
        "changing traffic, then repeat the canary to verify the proposed cause. "
    ) * 2

    assert _depth_answer_quality_error(goal, two_dimension_paragraph) == (
        "model_result_missing_tradeoff_dimensions"
    )
    assert _depth_answer_quality_error(
        goal,
        two_dimension_paragraph
        + "A third dimension is diagnostic quality and reliability: the larger model should "
        "reduce false positives, but that claim remains conditional on replay evidence.",
    ) is None


def test_brief_request_waives_length_but_not_requested_part_coverage() -> None:
    goal = (
        "Briefly explain the mechanism, tradeoffs, and a concrete example for this "
        "distributed failure."
    )

    assert _depth_answer_quality_error(goal, "It works better.") == (
        "model_result_missing_requested_parts"
    )


def test_negated_brevity_never_waives_depth_substance() -> None:
    goals = (
        "Do not give me a short answer. Explain what happened to the HiveMind cluster.",
        "Please do not be concise. Give me a complete explanation of what happened.",
        "I do not want a brief response. What happened and what should we do next?",
        "Do not be brief about the HiveMind cluster's current condition.",
        "Don't give me a short answer about the cluster's current condition.",
        "Never be concise about the cluster's current condition and recovery options.",
        "Don't be brief.",
        "Do not be concise.",
        "Do not be brief. Explain HiveMind.",
        "Don't be concise. Explain the cluster.",
        "Never be brief; describe ALPHA.",
        "No short answer.",
        "Avoid a brief answer about HiveMind.",
        "Explain HiveMind without being concise.",
    )

    for goal in goals:
        assert _depth_answer_quality_error(goal, "It broke.") == (
            "model_result_insufficient_substance"
        )

def test_explicit_word_target_is_a_real_minimum() -> None:
    goal = "Explain the mechanism, tradeoffs, and a concrete example in 100 words."

    assert _depth_answer_quality_error(goal, "It works better.") == (
        "model_result_insufficient_substance"
    )


def test_common_depth_language_requires_substance() -> None:
    shallow = "The larger one is better."
    goals = (
        "What are the pros and cons of using the 35B lobe here?",
        "Walk me through the distributed failure from request to worker and recovery.",
        "Give me a detailed answer about this intermittent distributed inference failure.",
        "What happened and what should we do next with the HiveMind cluster after all this work?",
        "Give me a full status report on every cluster component and remaining gap.",
    )

    for goal in goals:
        assert _depth_answer_quality_error(goal, shallow) == (
            "model_result_insufficient_substance"
        )


def test_common_more_later_invitations_are_cop_outs() -> None:
    goal = "Explain the complete distributed failure and recovery approach."
    endings = (
        "Would you like a deeper dive?",
        "I can provide more detail if needed.",
        "Happy to expand.",
        "More information is available on request.",
    )

    for ending in endings:
        answer = f"This is only a partial answer. {ending}"
        assert _depth_answer_quality_error(goal, answer) == "model_result_cop_out"


def test_natural_multipart_verbs_require_each_requested_part() -> None:
    goal = (
        "Analyze the architecture, recommend the best Depth model, explain how it works, "
        "and tell me how to test it."
    )
    mechanism_only = (
        "Requests enter the gateway with a correlation identifier, and the mechanism carries "
        "that identifier through routing, worker selection, inference, and accounting. The larger "
        "lobe reads the same bounded conversation plus current cluster observations because it must "
        "connect intermittent symptoms across several surfaces. A foreground lobe stays responsive "
        "while background processing reconstructs the timeline and returns one result. "
    ) * 3

    assert _depth_answer_quality_error(goal, mechanism_only) == (
        "model_result_missing_requested_parts"
    )


def test_multipart_choice_mechanism_and_case_synonyms_are_enforced() -> None:
    goals = (
        "Assess the architecture, choose the best Depth model, describe how it operates, "
        "and show a concrete test case.",
        "Compare the 4B and 35B models, decide which one to run, describe why, and "
        "illustrate it with a concrete case.",
        "Review the failure, pick a recovery approach, tell me how it functions, and "
        "demonstrate the validation path.",
    )
    architecture_and_test_only = (
        "The architecture has a foreground gateway, a background worker, a durable job ledger, "
        "and a cluster transport. A test sends traffic through each layer and records timing, "
        "routing, accounting, and response fields. The validation path repeats the request under "
        "controlled load and records the resulting service state. "
    ) * 5

    for goal in goals:
        assert _depth_answer_quality_error(goal, architecture_and_test_only) == (
            "model_result_missing_requested_parts"
        )


def test_concrete_test_case_is_independently_required() -> None:
    goal = (
        "Assess the architecture, choose the best Depth model, describe how it operates, "
        "and show a concrete test case."
    )
    answer_without_case = (
        "Choose the 35B model for the Depth role. Its mechanism operates by keeping the small "
        "foreground model responsive while a larger background model correlates routing, worker, "
        "timing, and accounting evidence. Validation tests should measure first-token latency, "
        "end-to-end completion, job identity, terminal reason, and retained session facts. The "
        "checks should run under controlled load and compare the visible answer with the durable "
        "result and cluster ledger. "
    ) * 2

    assert _depth_answer_quality_error(goal, answer_without_case) == (
        "model_result_missing_requested_parts"
    )


def test_auxiliary_how_questions_require_mechanism_evidence() -> None:
    goals = (
        "How does the Depth lobe operate under cluster load, and what changes when a node "
        "becomes saturated?",
        "Explain in detail: how does the Depth lobe function when the foreground lobe "
        "dispatches a background job?",
        "How is the Depth lobe supposed to operate when a cluster peer becomes saturated "
        "during inference?",
    )
    architecture_inventory_only = (
        "The system contains a browser, gateway, foreground lobe, background worker, durable "
        "ledger, load balancer, and several inference nodes. Each component has configuration, "
        "health, timing, and accounting fields. Operators can inspect the visible transcript, "
        "job identifier, selected model, node address, and terminal state. The cluster also "
        "records utilization, queue depth, response size, and timestamps. "
    ) * 3

    for goal in goals:
        assert _depth_answer_quality_error(goal, architecture_inventory_only) == (
            "model_result_missing_requested_parts"
        )


def test_show_me_a_test_case_requires_case_evidence() -> None:
    goal = (
        "Analyze this architecture thoroughly and show me a test case involving a completed "
        "Depth result."
    )
    generic_test_inventory = (
        "The architecture contains a browser, gateway, worker, ledger, and cluster. A test records "
        "identifiers, timestamps, model names, terminal state, and response length. Operators can "
        "inspect routing, queue depth, GPU load, and completion metadata. The same test records "
        "voice timing and visible transcript fields. "
    ) * 4

    assert _depth_answer_quality_error(goal, generic_test_inventory) == (
        "model_result_missing_requested_parts"
    )


def test_truncated_hermes_result_is_not_reported_complete() -> None:
    response, _captured = _run_adapter(
        {
            "final_response": "This answer stops in the middle despite visible text.",
            "completed": True,
            "finish_reason": "length",
        }
    )

    assert response["completed"] is False
    assert response["error"] == "model_result_truncated"


def test_reasoning_truncation_warning_shapes_fail_closed() -> None:
    warning_shapes = (
        {"hivemind_warning": {"type": "reasoning_model_truncated"}},
        {"hivemind_warnings": [{"warning_type": "reasoning_model_truncated"}]},
        {"provider_metadata": {"warnings": [{"code": "reasoning_model_truncated"}]}},
    )

    for metadata in warning_shapes:
        response, _captured = _run_adapter(
            {
                "final_response": _substantive_answer(),
                "completed": True,
                "finish_reason": "stop",
                **metadata,
            }
        )
        assert response["completed"] is False
        assert response["error"] == "model_result_reasoning_model_truncated"


def test_any_explicit_non_stop_terminal_reason_fails_closed() -> None:
    terminal_shapes = (
        {"finish_reason": "tool_calls"},
        {"finish_reason": "content_filter"},
        {"stop_reason": "cancelled"},
        {"termination_reason": "error"},
        {"finish_reason": None},
    )

    for terminal in terminal_shapes:
        response, _captured = _run_adapter(
            {
                "final_response": _substantive_answer(),
                "completed": True,
                **terminal,
            }
        )
        assert response["completed"] is False
        assert response["error"].startswith("model_result_non_success_")


def test_actual_hermes_turn_exit_reason_requires_embedded_stop() -> None:
    successful, _captured = _run_adapter(
        {
            "final_response": _substantive_answer(),
            "completed": True,
            "turn_exit_reason": "text_response(finish_reason=stop)",
        }
    )
    assert successful["completed"] is True

    failing_reasons = (
        ("text_response(finish_reason=length)", "model_result_truncated"),
        ("text_response(finish_reason=tool_calls)", "model_result_non_success_turn_exit_reason"),
        ("max_iterations_reached(8/8)", "model_result_non_success_turn_exit_reason"),
        ("budget_exhausted", "model_result_non_success_turn_exit_reason"),
    )
    for turn_exit_reason, expected_error in failing_reasons:
        response, _captured = _run_adapter(
            {
                "final_response": _substantive_answer(),
                "completed": True,
                "turn_exit_reason": turn_exit_reason,
            }
        )
        assert response["completed"] is False
        assert response["error"] == expected_error


def test_non_string_final_response_is_never_visible_success() -> None:
    for malformed in (None, {"answer": 4}, ["four"], 4):
        response, _captured = _run_adapter(
            {"final_response": malformed, "completed": True}
        )
        assert response["completed"] is False
        assert response["text"] == ""
        assert response["error"] == "model_result_invalid_final_response"


def test_repair_result_uses_same_reasoning_warning_gate() -> None:
    results = iter(
        [
            {"final_response": "Here is a teaser. Want to know more?", "completed": True},
            {
                "final_response": _substantive_answer(),
                "completed": True,
                "finish_reason": "stop",
                "hivemind_warning": {"type": "reasoning_model_truncated"},
            },
        ]
    )

    class _Agent:
        def run_conversation(self, _message, **_kwargs):
            return next(results)

    class _Runner:
        default_model = "depth-default"

        def new_background_agent(self, **_kwargs):
            return _Agent()

    response = build_real_chat_runner(_Runner())(
        message="Explain the decision and cover all three requested parts.",
        session_id="da-worker-repair-warning",
        model=None,
        stream_callback=lambda _chunk: None,
        tool_start_callback=lambda *_args: None,
        tool_complete_callback=lambda *_args: None,
    )

    assert response["quality_repair_attempted"] is True
    assert response["completed"] is False
    assert response["error"] == "model_result_reasoning_model_truncated"


def _completed_job(
    board: Blackboard,
    *,
    conversation_id: str,
    text: str,
    label: str,
) -> str:
    job_id = safety.new_job_id()
    envelope = JobEnvelope(
        job_id=job_id,
        parent_conversation_id=conversation_id,
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal=f"Explain {label}.",
        internal_goal=f"Explain {label}.",
    )
    envelope.validate()
    board.insert_job(envelope)
    board.insert_result(
        JobResult(
            job_id=job_id,
            status="success",
            summary=f"{label} completed.",
            text=text,
            confidence="high",
            conversation_revision_id=1,
        )
    )
    board.update_job_state(
        job_id,
        state="completed",
        last_safe_user_status=f"{label} completed.",
    )
    return job_id


def test_completed_result_context_is_structured_bounded_and_latest_weighted(
    tmp_path,
) -> None:
    board = Blackboard(tmp_path / "depth-context.sqlite3")
    conversation_id = "depth-context-contract"
    for index in range(3):
        _completed_job(
            board,
            conversation_id=conversation_id,
            label=f"older-{index}",
            text=(
                f"OLDER_{index}_START\n\n- old detail\n"
                + ("o" * 1500)
                + f"\nOLDER_{index}_TAIL"
            ),
        )
    _completed_job(
        board,
        conversation_id=conversation_id,
        label="latest",
        text=(
            "Direct conclusion.\n\n"
            "- First requested part\n"
            "- Second requested part\n"
            + ("l" * 2500)
            + "\nLATEST_TAIL"
        ),
    )

    block = build_face_lobe_context_block(
        conversation_id=conversation_id,
        blackboard=board,
    )

    assert block is not None
    assert "LATEST_TAIL" in block
    assert "result: Direct conclusion.\n        \n        - First requested part" in block
    assert all(f"OLDER_{index}_TAIL" not in block for index in range(3))
    assert block.count("      result:") == 4
    assert len(block) < 8_000
