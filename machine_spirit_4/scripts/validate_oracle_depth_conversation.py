from __future__ import annotations

"""Validate Oracle Depth as a complete multi-turn user experience.

The evaluator is deterministic and accepts a JSON evidence transcript.  The
optional ``--live`` mode runs the canonical conversation against a local MS4
gateway, but it does not claim that polling a backend job proves UI delivery.
A live pass therefore also requires ``--ui-evidence`` captured from the visible
Oracle client.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request
import uuid


EVIDENCE_SCHEMA = "OracleDepthConversationEvidence.v1"
REPORT_SCHEMA = "OracleDepthConversationReport.v1"
SCENARIO_NAME = "amber-lattice-depth-completeness-v1"
EXPECTED_HEADINGS = ("SUMMARY", "CAUSES", "ACTIONS", "RISKS", "VALIDATION")

DEFAULT_LIMITS: dict[str, float] = {
    "face_turn_seconds": 15.0,
    "depth_seconds": 180.0,
    "automatic_delivery_seconds": 5.0,
    "followup_turn_seconds": 30.0,
    "total_seconds": 250.0,
    "utc_tolerance_seconds": 900.0,
}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\N{RIGHT SINGLE QUOTATION MARK}-]*")
_UTC_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)\b",
    re.IGNORECASE,
)
_COP_OUT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("want_more", re.compile(r"\bwant (?:to know )?(?:the )?(?:full )?(?:details|more)\??", re.I)),
    ("would_you_like", re.compile(r"\bwould you like me to\b", re.I)),
    ("do_you_want", re.compile(r"\bdo you want me to\b", re.I)),
    ("if_you_want", re.compile(r"\bif you want(?: me to)?\b", re.I)),
    ("let_me_know", re.compile(r"\blet me know if you(?:'d| would)? like\b", re.I)),
    ("can_go_deeper", re.compile(r"\bI can (?:go|dig|dive) deeper\b", re.I)),
    ("ask_for_more", re.compile(r"\b(?:ask me|say yes) (?:if|for|to)\b", re.I)),
)
_DASH_RE = re.compile(r"[-\N{HYPHEN}\N{NON-BREAKING HYPHEN}\N{FIGURE DASH}\N{EN DASH}\N{EM DASH}]+")
_OWNER_ASSIGNMENT_RE = re.compile(
    r"\bOWNER\s*(?:=|:|\bis\b|\bwas\b)\s*([A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)
_OWNER_NAME_STATEMENT_RE = re.compile(
    r"\bowner\s+name(?:\s+(?:provided|given|supplied))?\s+"
    r"(?:is|was)\s+([A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Gate:
    name: str
    ok: bool
    detail: dict[str, Any]


def canonical_prompts(nonce: str) -> tuple[str, str, str, str]:
    """Return the exact four-turn AMBER-LATTICE acceptance conversation."""

    safe_nonce = str(nonce).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", safe_nonce):
        raise ValueError("nonce must be 4-64 ASCII letters, digits, '_' or '-'")
    return (
        "Remember these session facts: project AMBER-LATTICE, recovery window "
        "12 minutes, and risk ALPHA means stale routing. Do not solve anything "
        "yet. Reply exactly READY.",
        "/deep Using the remembered facts, create a complete recovery plan. "
        "Call hivemind_time_now exactly once. Use exactly these headings: "
        "SUMMARY, CAUSES, ACTIONS, RISKS, VALIDATION. ACTIONS must contain "
        "exactly four numbered actions; Action 2 must mitigate risk ALPHA. "
        "RISKS exactly two bullets. No owner supplied, so write OWNER=UNKNOWN. "
        f"VALIDATION includes VERIFY-{safe_nonce} and real UTC. Do not ask "
        "whether I want more.",
        "Explain Action 2 from the completed answer in at least 100 words: cover "
        "the mechanism, timing, failure gates, and how to undo it. Keep ALPHA "
        "and the 12-minute window explicit.",
        "Which exact action mitigates ALPHA, and what owner name did I give you?",
    )


def canonical_scenario(nonce: str) -> dict[str, Any]:
    """Build the default scenario spec used by the generic evaluator."""

    return {
        "name": SCENARIO_NAME,
        "prompts": list(canonical_prompts(nonce)),
        "ready_exact": "READY",
        "headings": list(EXPECTED_HEADINGS),
        "depth_min_words": 180,
        "action_count": 4,
        "risk_count": 2,
        "focus_action": 2,
        "fact_tokens": ["AMBER-LATTICE", "12 minutes", "ALPHA"],
        "risk_meaning_tokens": ["stale", "routing"],
        "tool_name": "hivemind_time_now",
        "owner_literal": "OWNER=UNKNOWN",
        "validation_marker": f"VERIFY-{nonce}",
        "followup_min_words": 100,
        "followup_tokens": ["ALPHA", "12"],
        "final_tokens": ["Action 2", "OWNER=UNKNOWN"],
    }


def detect_cop_outs(text: str) -> list[str]:
    """Return stable labels for teaser/follow-up-deferral language."""

    return [label for label, pattern in _COP_OUT_PATTERNS if pattern.search(text or "")]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []


def _assistant_text(turn: Mapping[str, Any]) -> str:
    direct = turn.get("assistant_text")
    if isinstance(direct, str):
        return direct
    assistant = turn.get("assistant")
    if isinstance(assistant, str):
        return assistant
    if isinstance(assistant, Mapping):
        return str(assistant.get("text") or assistant.get("content") or "")
    response = turn.get("response")
    if isinstance(response, Mapping):
        return str(response.get("text") or response.get("content") or "")
    return ""


def _turn_session_id(turn: Mapping[str, Any]) -> str:
    if turn.get("session_id"):
        return str(turn["session_id"])
    response = turn.get("response")
    if isinstance(response, Mapping) and response.get("session_id"):
        return str(response["session_id"])
    return ""


def _dispatch_ids(turn: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for value in _sequence(turn.get("job_ids")):
        if value:
            ids.append(str(value))
    for container in (turn, _mapping(turn.get("response"))):
        dispatched = container.get("dispatched_job")
        if isinstance(dispatched, Mapping) and dispatched.get("job_id"):
            ids.append(str(dispatched["job_id"]))
    return list(dict.fromkeys(ids))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _normalize_fact_text(text: str) -> str:
    """Normalize typography without weakening word or value matching."""

    normalized = _normalize_text(_DASH_RE.sub(" ", text or "")).casefold()
    # Attributive measurements conventionally use a singular unit
    # ("12-minute window") while predicate forms use the plural
    # ("window is 12 minutes"). They carry the same remembered fact.
    return re.sub(r"\b(\d+)\s+([a-z]+?)s\b", r"\1 \2", normalized)


def _owner_values(text: str) -> list[str]:
    values = [match.group(1) for match in _OWNER_ASSIGNMENT_RE.finditer(text or "")]
    values.extend(match.group(1) for match in _OWNER_NAME_STATEMENT_RE.finditer(text or ""))
    return values


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _clean_heading(line: str) -> str:
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", line.strip())
    cleaned = cleaned.strip().strip("*_`").rstrip(":").strip()
    return cleaned.upper()


def _heading_rows(text: str, expected: Sequence[str]) -> list[tuple[str, int]]:
    expected_set = {str(value).upper() for value in expected}
    rows: list[tuple[str, int]] = []
    for index, line in enumerate((text or "").splitlines()):
        cleaned = _clean_heading(line)
        if cleaned in expected_set:
            rows.append((cleaned, index))
    return rows


def _section(text: str, heading: str, next_heading: str | None) -> str:
    lines = (text or "").splitlines()
    start: int | None = None
    end = len(lines)
    for index, line in enumerate(lines):
        cleaned = _clean_heading(line)
        if start is None and cleaned == heading.upper():
            start = index + 1
            continue
        if start is not None and next_heading is not None and cleaned == next_heading.upper():
            end = index
            break
    return "\n".join(lines[start:end]) if start is not None else ""


def _numbered_items(text: str) -> list[tuple[int, str]]:
    matches = list(
        re.finditer(
            r"(?mi)^\s*(?:\*\*|__)?(?:Action\s+)?(\d+)[.):](?:\*\*|__)?\s+(\S.*)$",
            text or "",
        )
    )
    items: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        items.append((int(match.group(1)), text[match.start():stop].strip()))
    return items


def _tool_names(job: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    """Choose one authoritative tool trace to avoid double-counting callbacks."""

    sources = (
        _sequence(job.get("tool_calls")),
        _sequence(result.get("actions_taken")),
        _sequence(result.get("tool_trace")),
    )
    for source in sources:
        if source:
            names = []
            for item in source:
                if isinstance(item, Mapping):
                    name = item.get("tool") or item.get("name")
                else:
                    name = item
                if name:
                    names.append(str(name))
            return names

    names = []
    for event in _sequence(job.get("events")):
        if not isinstance(event, Mapping):
            continue
        if str(event.get("type") or "") != "job.tool.call.completed":
            continue
        payload = _mapping(event.get("payload"))
        name = payload.get("tool") or payload.get("name")
        if name:
            names.append(str(name))
    return names


def _recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for candidate, child in value.items():
            if str(candidate) == key:
                found.append(child)
            found.extend(_recursive_values(child, key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.extend(_recursive_values(child, key))
    return found


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def evaluate_transcript(
    evidence: Mapping[str, Any],
    *,
    scenario: Mapping[str, Any] | None = None,
    limits: Mapping[str, float] | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate a recorded conversation against a configurable hard-gate spec."""

    nonce = str(evidence.get("nonce") or "")
    try:
        spec = dict(scenario or canonical_scenario(nonce))
    except ValueError as exc:
        return {
            "schema": REPORT_SCHEMA,
            "scenario": SCENARIO_NAME,
            "ok": False,
            "checks": [asdict(Gate("valid_nonce", False, {"error": str(exc)}))],
            "failed_checks": ["valid_nonce"],
            "metrics": {},
        }

    effective_limits = dict(DEFAULT_LIMITS)
    effective_limits.update({key: float(value) for key, value in (limits or {}).items()})
    turns_raw = _sequence(evidence.get("turns"))
    turns = [_mapping(value) for value in turns_raw]
    while len(turns) < 4:
        turns.append({})
    turns = turns[:4]
    checks: list[Gate] = []

    def add(name: str, ok: bool, **detail: Any) -> None:
        checks.append(Gate(name=name, ok=bool(ok), detail=detail))

    expected_prompts = [str(value) for value in _sequence(spec.get("prompts"))]
    observed_prompts = [str(turn.get("user") or "").strip() for turn in turns]
    add(
        "canonical_turns",
        len(turns_raw) == 4 and observed_prompts == expected_prompts,
        observed_turn_count=len(turns_raw),
        prompt_matches=[
            index < len(expected_prompts) and observed == expected_prompts[index]
            for index, observed in enumerate(observed_prompts)
        ],
    )

    expected_session = str(evidence.get("session_id") or "")
    observed_sessions = [_turn_session_id(turn) for turn in turns]
    session_ok = bool(expected_session) and all(value == expected_session for value in observed_sessions)
    add("single_session", session_ok, session_id=expected_session, observed=observed_sessions)

    ready_text = _assistant_text(turns[0]).strip()
    ready_expected = str(spec.get("ready_exact") or "READY")
    add("priming_ack_exact", ready_text == ready_expected, observed=ready_text, expected=ready_expected)

    dispatches = [_dispatch_ids(turn) for turn in turns]
    all_dispatches = [job_id for ids in dispatches for job_id in ids]
    unique_jobs = list(dict.fromkeys(all_dispatches))
    depth_job = _mapping(turns[1].get("job") or evidence.get("depth_job"))
    depth_job_id = str(depth_job.get("job_id") or "")
    one_job_ok = (
        len(unique_jobs) == 1
        and len(all_dispatches) == 1
        and len(dispatches[1]) == 1
        and not dispatches[0]
        and not dispatches[2]
        and not dispatches[3]
        and depth_job_id == unique_jobs[0]
    )
    add(
        "exactly_one_depth_job",
        one_job_ok,
        dispatches_by_turn=dispatches,
        job_id=depth_job_id,
    )

    result = _mapping(depth_job.get("result"))
    depth_text = str(result.get("text") or "")
    event_types = [
        str(event.get("type") or "")
        for event in _sequence(depth_job.get("events"))
        if isinstance(event, Mapping)
    ]
    # Keep the job and foreground response as independent evidence surfaces.
    # Traversing the whole turn would visit ``turn["job"]`` a second time and
    # falsely double-count identical warning/finish metadata.
    terminal_surfaces = {"job": depth_job, "response": _mapping(turns[1].get("response"))}
    warning_values = [value for value in _recursive_values(terminal_surfaces, "hivemind_warning") if value]
    finish_reasons = [str(value).lower() for value in _recursive_values(terminal_surfaces, "finish_reason") if value]
    truncated_flags = [value for value in _recursive_values(terminal_surfaces, "truncated") if value is True]
    state_ok = (
        depth_job.get("state") == "completed"
        and result.get("status") == "success"
        and str(result.get("job_id") or depth_job_id) == depth_job_id
        and bool(depth_text.strip())
        and "job.completed" in event_types
        and "job.failed" not in event_types
        and not warning_values
        and not truncated_flags
        and not any(reason in {"length", "max_tokens", "truncated"} for reason in finish_reasons)
        and not depth_job.get("timed_out")
    )
    add(
        "terminal_state_and_truncation_honesty",
        state_ok,
        job_state=depth_job.get("state"),
        result_status=result.get("status"),
        event_types=event_types,
        warning_count=len(warning_values),
        finish_reasons=finish_reasons,
        truncated=bool(truncated_flags),
        timed_out=bool(depth_job.get("timed_out")),
    )

    tool_name = str(spec.get("tool_name") or "")
    tool_names = _tool_names(depth_job, result)
    add(
        "required_tool_called_once",
        tool_names.count(tool_name) == 1,
        expected=tool_name,
        observed=tool_names,
        auxiliary=[name for name in tool_names if name != tool_name],
    )

    headings = [str(value).upper() for value in _sequence(spec.get("headings"))]
    observed_heading_rows = _heading_rows(depth_text, headings)
    observed_headings = [name for name, _ in observed_heading_rows]
    heading_ok = observed_headings == headings
    add("depth_headings_exact", heading_ok, expected=headings, observed=observed_headings)

    actions_text = _section(depth_text, "ACTIONS", "RISKS")
    actions = _numbered_items(actions_text)
    expected_action_count = int(spec.get("action_count") or 0)
    expected_numbers = list(range(1, expected_action_count + 1))
    action_numbers = [number for number, _ in actions]
    add(
        "depth_actions_exact",
        action_numbers == expected_numbers,
        expected=expected_numbers,
        observed=action_numbers,
    )

    focus_action = int(spec.get("focus_action") or 0)
    focus_text = next((text for number, text in actions if number == focus_action), "")
    risk_token = str(_sequence(spec.get("followup_tokens"))[0] if _sequence(spec.get("followup_tokens")) else "")
    mitigation_language = re.search(r"\b(?:mitigat\w*|refresh\w*|invalidat\w*|clear\w*|re-?rout\w*)\b", focus_text, re.I)
    add(
        "focus_action_mitigates_risk",
        bool(focus_text and risk_token.lower() in focus_text.lower() and mitigation_language),
        action=focus_action,
        risk=risk_token,
        excerpt=_normalize_text(focus_text)[:240],
    )

    risks_text = _section(depth_text, "RISKS", "VALIDATION")
    risk_bullets = re.findall(r"(?m)^\s*[-*\N{BULLET}]\s+\S.*$", risks_text)
    expected_risk_count = int(spec.get("risk_count") or 0)
    add(
        "depth_risks_exact",
        len(risk_bullets) == expected_risk_count,
        expected=expected_risk_count,
        observed=len(risk_bullets),
    )

    fact_tokens = [str(value) for value in _sequence(spec.get("fact_tokens"))]
    risk_meaning_tokens = [str(value) for value in _sequence(spec.get("risk_meaning_tokens"))]
    normalized_depth_facts = _normalize_fact_text(depth_text)
    missing_facts = [
        value
        for value in fact_tokens + risk_meaning_tokens
        if _normalize_fact_text(value) not in normalized_depth_facts
    ]
    add("remembered_facts_preserved", not missing_facts, missing=missing_facts)

    owner_literal = str(spec.get("owner_literal") or "")
    owner_assignments = re.findall(r"\bOWNER\s*=\s*([A-Za-z][A-Za-z0-9_-]*)", depth_text, re.I)
    owner_ok = owner_literal.lower() in _normalize_text(depth_text).lower() and bool(owner_assignments) and all(
        value.upper() == "UNKNOWN" for value in owner_assignments
    )
    add("unknown_owner_preserved", owner_ok, assignments=owner_assignments)

    marker = str(spec.get("validation_marker") or "")
    timestamps = [_parse_utc(value) for value in _UTC_RE.findall(depth_text)]
    timestamps = [value for value in timestamps if value is not None]
    reference = _parse_utc(str(evidence.get("reference_utc") or "")) or now_utc or datetime.now(timezone.utc)
    reference = reference.astimezone(timezone.utc)
    closest_delta = min((abs((value - reference).total_seconds()) for value in timestamps), default=None)
    utc_ok = closest_delta is not None and closest_delta <= effective_limits["utc_tolerance_seconds"]
    add(
        "validation_nonce_and_real_utc",
        bool(marker and marker in depth_text and utc_ok),
        marker=marker,
        utc_values=[value.isoformat() for value in timestamps],
        closest_delta_seconds=closest_delta,
        tolerance_seconds=effective_limits["utc_tolerance_seconds"],
    )

    depth_words = _word_count(depth_text)
    depth_min_words = int(spec.get("depth_min_words") or 0)
    cop_outs = detect_cop_outs(depth_text)
    add(
        "depth_substantive_without_cop_out",
        depth_words >= depth_min_words and not cop_outs,
        words=depth_words,
        minimum=depth_min_words,
        cop_outs=cop_outs,
    )

    automatic = _mapping(turns[1].get("automatic_completion") or evidence.get("automatic_completion"))
    automatic_text = str(automatic.get("text") or "")
    normalized_result = _normalize_text(depth_text)
    normalized_automatic = _normalize_text(automatic_text)
    automatic_ok = (
        automatic.get("delivered") is True
        and automatic.get("required_user_action") is False
        and bool(normalized_result)
        and normalized_result in normalized_automatic
        and str(automatic.get("job_id") or "") == depth_job_id
        and str(automatic.get("session_id") or "") == expected_session
    )
    add(
        "automatic_full_result_delivery",
        automatic_ok,
        delivered=automatic.get("delivered"),
        required_user_action=automatic.get("required_user_action"),
        result_chars=len(depth_text),
        automatic_chars=len(automatic_text),
        job_matches=str(automatic.get("job_id") or "") == depth_job_id,
        session_matches=str(automatic.get("session_id") or "") == expected_session,
    )
    automatic_cop_outs = detect_cop_outs(automatic_text)
    add("automatic_delivery_has_no_teaser", not automatic_cop_outs, cop_outs=automatic_cop_outs)

    followup_text = _assistant_text(turns[2])
    followup_words = _word_count(followup_text)
    followup_tokens = [str(value) for value in _sequence(spec.get("followup_tokens"))]
    missing_followup = [value for value in followup_tokens if value.lower() not in followup_text.lower()]
    followup_cop_outs = detect_cop_outs(followup_text)
    add(
        "affirmative_followup_is_substantive",
        followup_words >= int(spec.get("followup_min_words") or 0)
        and not missing_followup
        and not followup_cop_outs,
        words=followup_words,
        minimum=int(spec.get("followup_min_words") or 0),
        missing=missing_followup,
        cop_outs=followup_cop_outs,
    )

    final_text = _assistant_text(turns[3])
    final_tokens = [str(value) for value in _sequence(spec.get("final_tokens"))]
    missing_final = [
        value
        for value in final_tokens
        if value.casefold() != owner_literal.casefold()
        and value.casefold() not in _normalize_text(final_text).casefold()
    ]
    final_owner_assignments = _owner_values(final_text)
    final_unknown_explicit = re.search(r"\bUNKNOWN\b", final_text, re.I) is not None
    final_cop_outs = detect_cop_outs(final_text)
    final_ok = (
        not missing_final
        and final_unknown_explicit
        and all(value.upper() == "UNKNOWN" for value in final_owner_assignments)
        and not final_cop_outs
    )
    add(
        "targeted_followup_preserves_answer_and_unknown",
        final_ok,
        missing=missing_final,
        owner_assignments=final_owner_assignments,
        cop_outs=final_cop_outs,
    )

    turn_latencies = [_float(turn.get("latency_s")) for turn in turns]
    depth_latency = _float(depth_job.get("latency_s") or evidence.get("depth_latency_s"))
    automatic_latency = _float(automatic.get("latency_s"))
    latency_values = [*turn_latencies, depth_latency, automatic_latency]
    total_latency = sum(value for value in latency_values if value is not None)
    latency_ok = (
        all(value is not None for value in latency_values)
        and turn_latencies[0] <= effective_limits["face_turn_seconds"]
        and turn_latencies[1] <= effective_limits["face_turn_seconds"]
        and depth_latency <= effective_limits["depth_seconds"]
        and automatic_latency <= effective_limits["automatic_delivery_seconds"]
        and turn_latencies[2] <= effective_limits["followup_turn_seconds"]
        and turn_latencies[3] <= effective_limits["followup_turn_seconds"]
        and total_latency <= effective_limits["total_seconds"]
    )
    add(
        "latency_budget",
        latency_ok,
        turn_seconds=turn_latencies,
        depth_seconds=depth_latency,
        automatic_delivery_seconds=automatic_latency,
        total_seconds=total_latency,
        limits=effective_limits,
    )

    check_dicts = [asdict(check) for check in checks]
    failed = [check.name for check in checks if not check.ok]
    return {
        "schema": REPORT_SCHEMA,
        "scenario": str(spec.get("name") or "custom"),
        "nonce": nonce,
        "ok": not failed,
        "failed_checks": failed,
        "metrics": {
            "depth_words": depth_words,
            "followup_words": followup_words,
            "depth_jobs": len(unique_jobs),
            "tool_calls": len(tool_names),
            "total_latency_seconds": total_latency,
        },
        "checks": check_dicts,
    }


def _request_json(base_url: str, path: str, *, payload: Mapping[str, Any] | None = None, timeout: float) -> dict[str, Any]:
    data = None
    method = "GET"
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(dict(payload)).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"{method} {path} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
    parsed = json.loads(body or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {path} returned non-object JSON")
    return parsed


def _live_turn(
    base_url: str,
    *,
    user: str,
    session_id: str,
    timeout: float,
    face_model: str | None,
    depth_model: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": user, "session_id": session_id}
    if face_model:
        payload["model_id"] = face_model
    if depth_model:
        payload["depth_model_id"] = depth_model
    started = time.monotonic()
    response = _request_json(base_url, "/chat", payload=payload, timeout=timeout)
    return {
        "user": user,
        "assistant_text": str(response.get("text") or ""),
        "session_id": str(response.get("session_id") or session_id),
        "latency_s": time.monotonic() - started,
        "response": response,
    }


def run_live(
    *,
    base_url: str,
    nonce: str,
    face_model: str | None,
    depth_model: str | None,
    request_timeout: float,
    depth_timeout: float,
    poll_seconds: float,
    ui_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run the canonical local API conversation and return raw evidence."""

    prompts = canonical_prompts(nonce)
    session_id = f"oracle-depth-acceptance-{uuid.uuid4().hex[:12]}"
    captured_at = datetime.now(timezone.utc)
    turn1 = _live_turn(
        base_url,
        user=prompts[0],
        session_id=session_id,
        timeout=request_timeout,
        face_model=face_model,
        depth_model=depth_model,
    )
    turn2 = _live_turn(
        base_url,
        user=prompts[1],
        session_id=session_id,
        timeout=request_timeout,
        face_model=face_model,
        depth_model=depth_model,
    )
    job_ids = _dispatch_ids(turn2)
    job: dict[str, Any] = {"job_id": job_ids[0] if len(job_ids) == 1 else "", "timed_out": False}
    if len(job_ids) == 1:
        deadline = time.monotonic() + depth_timeout
        depth_started = time.monotonic()
        while time.monotonic() < deadline:
            job = _request_json(base_url, f"/api/v1/double-agent/jobs/{job_ids[0]}", timeout=request_timeout)
            if job.get("state") in {"completed", "failed", "canceled", "stale"}:
                break
            time.sleep(max(0.1, poll_seconds))
        else:
            job["timed_out"] = True
        job["latency_s"] = time.monotonic() - depth_started
        try:
            event_payload = _request_json(
                base_url,
                f"/api/v1/double-agent/jobs/{job_ids[0]}/events?limit=500",
                timeout=request_timeout,
            )
            job["events"] = _sequence(event_payload.get("events"))
        except RuntimeError as exc:
            job["events_error"] = str(exc)
            job["events"] = []
    turn2["job"] = job
    turn2["automatic_completion"] = dict(ui_evidence or {
        "delivered": False,
        "required_user_action": None,
        "text": "",
        "job_id": job.get("job_id") or "",
        "session_id": session_id,
        "latency_s": None,
        "evidence_status": "not_supplied",
    })

    turn3 = _live_turn(
        base_url,
        user=prompts[2],
        session_id=session_id,
        timeout=request_timeout,
        face_model=face_model,
        depth_model=depth_model,
    )
    turn4 = _live_turn(
        base_url,
        user=prompts[3],
        session_id=session_id,
        timeout=request_timeout,
        face_model=face_model,
        depth_model=depth_model,
    )
    return {
        "schema": EVIDENCE_SCHEMA,
        "scenario": SCENARIO_NAME,
        "nonce": nonce,
        "session_id": session_id,
        "reference_utc": captured_at.isoformat(),
        "turns": [turn1, turn2, turn3, turn4],
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return parsed


def _write_json(path: str | None, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n"
    if not path or path == "-":
        print(rendered, end="")
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Oracle Depth completeness, automatic delivery, and multi-turn continuity.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", type=Path, help="Evaluate an existing evidence JSON file without network access.")
    mode.add_argument("--live", action="store_true", help="Explicitly run the canonical conversation against a local gateway.")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:9180")
    parser.add_argument("--face-model")
    parser.add_argument("--depth-model")
    parser.add_argument("--nonce", default=None, help="4-64 character validation nonce; generated in live mode by default.")
    parser.add_argument("--ui-evidence", type=Path, help="Visible-client automatic-delivery JSON required for a live full pass.")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--depth-timeout", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--max-face-seconds", type=float, default=DEFAULT_LIMITS["face_turn_seconds"])
    parser.add_argument("--max-depth-seconds", type=float, default=DEFAULT_LIMITS["depth_seconds"])
    parser.add_argument("--max-auto-seconds", type=float, default=DEFAULT_LIMITS["automatic_delivery_seconds"])
    parser.add_argument("--max-followup-seconds", type=float, default=DEFAULT_LIMITS["followup_turn_seconds"])
    parser.add_argument("--max-total-seconds", type=float, default=DEFAULT_LIMITS["total_seconds"])
    parser.add_argument("--output", default="-", help="Report JSON path, or '-' for stdout.")
    parser.add_argument("--evidence-output", help="Optional raw evidence JSON path for live mode.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.fixture:
            evidence = _read_json_object(args.fixture)
        else:
            nonce = args.nonce or uuid.uuid4().hex[:12]
            ui_evidence = _read_json_object(args.ui_evidence) if args.ui_evidence else None
            evidence = run_live(
                base_url=args.gateway_url,
                nonce=nonce,
                face_model=args.face_model,
                depth_model=args.depth_model,
                request_timeout=args.request_timeout,
                depth_timeout=args.depth_timeout,
                poll_seconds=args.poll_seconds,
                ui_evidence=ui_evidence,
            )
            if args.evidence_output:
                _write_json(args.evidence_output, evidence)
        limits = {
            "face_turn_seconds": args.max_face_seconds,
            "depth_seconds": args.max_depth_seconds,
            "automatic_delivery_seconds": args.max_auto_seconds,
            "followup_turn_seconds": args.max_followup_seconds,
            "total_seconds": args.max_total_seconds,
        }
        report = evaluate_transcript(evidence, limits=limits)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        error_report = {
            "schema": REPORT_SCHEMA,
            "ok": False,
            "failed_checks": ["validator_execution"],
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(args.output, error_report)
        return 2
    _write_json(args.output, report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
