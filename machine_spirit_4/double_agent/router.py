"""Auto-router: decides whether to dispatch a Depth Lobe alongside the
Face Lobe reply.

The router never replaces the foreground turn. It only decides whether
to also spawn a background Double Agent job for the same user
message. The foreground always replies; the background, when fired,
runs in parallel and surfaces via the Double Agent panel + the Face
Lobe context block on subsequent turns.

Decision layers (highest priority wins):

1. **Slash commands**: ``/deep <msg>`` forces dispatch, ``/direct <msg>``
   forces no dispatch. Strips the prefix from the cleaned message.
2. **Heuristic**: deterministic rules over the user message
   (length, code blocks, action verbs, multi-step asks).
3. **LLM classifier**: only consulted when the heuristic confidence
   is below threshold AND a classifier callable is supplied. Avoids
   adding latency to obvious cases. Disabled by default; the gateway
   wires a small-model classifier when ``MS4_ROUTER_LLM_CLASSIFY=1``.

Routing decisions are surfaced on the chat response under
``router.{kind,confidence,source,reason,goal,override}`` so the
operator can see *why* a turn dispatched (or didn't).

Mirrors artifact §15 router decision policy, but stays MS4-specific:
no fanout in phase 2 (one Depth Lobe per turn), no blocking-mode
override (foreground always replies first), no approval gate (those
still go through MS3 ethics inside the worker).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable


# Public knob: turn the LLM classifier layer on without redeploying.
ENV_LLM_CLASSIFY = "MS4_ROUTER_LLM_CLASSIFY"

# Heuristic score at or above this threshold confidently routes to deep.
HEURISTIC_DEEP_THRESHOLD = 0.5

# Heuristic confidence at or above this threshold (on the "direct" side)
# skips the LLM classifier. Anything else inside the heuristic-uncertain
# band consults the LLM when one is supplied AND
# ``MS4_ROUTER_LLM_CLASSIFY`` is on. Greetings / explicit "direct hints"
# are routed direct deterministically and never reach the LLM.
HEURISTIC_CONFIDENT_DIRECT_THRESHOLD = 0.9


@dataclass
class RouteDecision:
    kind: str  # "direct" | "deep"
    confidence: float  # 0.0..1.0
    source: str  # "slash" | "heuristic" | "llm" | "default"
    reason: str
    goal: str  # short user_visible_goal for the Depth Lobe job
    cleaned_message: str  # message stripped of slash command, if any
    override: bool = False
    raw_score: float = 0.0  # heuristic-internal: probability of "deep"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "Ms4RouteDecision.v1",
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "reason": self.reason,
            "goal": self.goal,
            "override": self.override,
        }


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

_SLASH_DEEP_RE = re.compile(r"^\s*/deep\b\s*", re.IGNORECASE)
_SLASH_DIRECT_RE = re.compile(r"^\s*/direct\b\s*", re.IGNORECASE)


def _parse_slash(message: str) -> RouteDecision | None:
    if not message:
        return None
    deep_match = _SLASH_DEEP_RE.match(message)
    if deep_match:
        cleaned = message[deep_match.end():].strip() or message.strip()
        return RouteDecision(
            kind="deep",
            confidence=1.0,
            source="slash",
            reason="/deep prefix",
            goal=_summary_goal(cleaned),
            cleaned_message=cleaned,
            override=True,
        )
    direct_match = _SLASH_DIRECT_RE.match(message)
    if direct_match:
        cleaned = message[direct_match.end():].strip() or message.strip()
        return RouteDecision(
            kind="direct",
            confidence=1.0,
            source="slash",
            reason="/direct prefix",
            goal="",
            cleaned_message=cleaned,
            override=True,
        )
    return None


# ---------------------------------------------------------------------------
# Heuristic layer
# ---------------------------------------------------------------------------

_ACTION_VERBS = (
    "implement",
    "audit",
    "research",
    "investigate",
    "debug",
    "diagnose",
    "refactor",
    "review",
    "design",
    "plan",
    "analyze",
    "compare",
    "benchmark",
    "profile",
    "trace",
    "find the",
    "find why",
    "figure out",
    "explain why",
    "walk through",
    "build",
    "scaffold",
    "migrate",
    "port",
    "rewrite",
    "summarize the",
    "summarise the",
)
_DEEP_NOUNS = (
    "module",
    "codebase",
    "architecture",
    "pipeline",
    "schema",
    "deadlock",
    "performance",
    "regression",
    "stack trace",
    "logs",
    "repository",
    "data model",
    "design doc",
    "specification",
    # Operational targets (vm, container, service, process). Hits here
    # combined with an action verb push score above the deep threshold;
    # alone they're only a weak signal.
    "vm",
    "virtual machine",
    "container",
    "docker",
    "kubernetes",
    "k8s",
    "service",
    "daemon",
    "process",
    "database",
    "package",
    "dependency",
)
# Short questions about runtime state / tools / hardware should ALSO route
# deep because they almost always need tool access (terminal, MCP calls,
# file I/O) that the Face Lobe alone shouldn't do. Without this axis the
# heuristic missed "do you see any GPUs?" and "what tools do you have?"
# in live testing and the model hallucinated tool names.
_TOOL_INTENT_VERBS = (
    # Process / execution verbs.
    "run", "execute", "open", "launch", "spawn", "kill", "restart",
    "deploy", "rollback", "browse", "navigate", "click", "type", "scroll",
    # Operational verbs (start a vm, stop a service, etc.). Missing these
    # was the root cause of "I want you to start a vm" routing direct.
    "start ", "stop ", "boot ", "shut down", "shutdown",
    "create ", "make a ", "make the ", "build me ",
    "install ", "uninstall ", "remove ", "delete ", "wipe ",
    "edit ", "modify ", "change the ", "update the ",
    "configure ", "provision ", "scale ", "tune ",
    "terminate ", "halt ", "pause ", "resume ",
    "mount ", "unmount ", "format ", "partition ",
    "connect ", "disconnect ", "ssh ",
    "i want you to ",  # explicit imperative marker
    "please ",  # polite imperative; combined with the above ones below
    # File / system / capture verbs.
    "read file", "write file", "list files", "cat ", "ls ",
    "show me", "look up", "look at", "fetch", "download ", "upload ",
    "take a screenshot", "screenshot", "capture the",
    # Capabilities / introspection variants.
    "what tools", "list tools", "any tools", "available tools",
    "tools do you", "what mcp", "mcp servers", "your tools",
    "actual tools", "the tools", "full list of tools",
    "give me the tools", "give me the actual tools", "give me a list",
    "give me the full list", "the full list",
    "what gpus", "any gpus", "list gpus",
    "what nodes", "list nodes",
    "what models", "list models",
    "do you see", "do you have", "can you call", "can you run",
    "what can you do", "capabilities",
)
_DIRECT_HINTS = (
    "what time is",
    "what is the date",
    "hello",
    "hi there",
    "thanks",
    "thank you",
    "yes",
    "no",
    "ok",
    "good morning",
    "good night",
)
# Matched against the lowercased message with word boundaries so that
# "look" doesn't trigger on "ok" and "idiomatic" doesn't trigger on "no".
_DIRECT_HINT_RES = tuple(re.compile(r"\b" + re.escape(h) + r"\b") for h in _DIRECT_HINTS)


def _has_code_block(message: str) -> bool:
    if "```" in message:
        return True
    # heuristic: 3+ consecutive indented lines look code-ish
    lines = message.splitlines()
    indented = sum(1 for line in lines if line.startswith(("    ", "\t")))
    return indented >= 3


def _looks_multistep(message: str) -> bool:
    if " and also " in message.lower():
        return True
    bullets = sum(1 for line in message.splitlines() if line.lstrip().startswith(("-", "*", "1.", "2.")))
    return bullets >= 2


def _heuristic_route(message: str) -> RouteDecision:
    lower = message.lower().strip()
    length = len(message.strip())

    if length == 0:
        return RouteDecision(
            kind="direct",
            confidence=1.0,
            source="heuristic",
            reason="empty message",
            goal="",
            cleaned_message=message,
            raw_score=-1.0,  # sentinel: "definitely direct", skip LLM
        )

    if any(hint_re.search(lower) for hint_re in _DIRECT_HINT_RES) and length < 80:
        return RouteDecision(
            kind="direct",
            confidence=0.95,
            source="heuristic",
            reason="conversational greeting/short ack",
            goal="",
            cleaned_message=message,
            raw_score=-1.0,  # sentinel: "definitely direct", skip LLM
        )

    score = 0.0
    reasons: list[str] = []

    if length >= 280:
        score += 0.35
        reasons.append(f"long message ({length} chars)")
    elif length >= 140:
        score += 0.20
        reasons.append(f"medium-length message ({length} chars)")

    if _has_code_block(message):
        # Code blocks are a strong signal: the user wants the model to read
        # or write code, which is exactly what the Depth Lobe is for.
        score += 0.50
        reasons.append("contains a code block")

    action_hits = [v for v in _ACTION_VERBS if v in lower]
    if action_hits:
        score += min(0.30, 0.10 * len(action_hits))
        reasons.append(f"action verbs: {', '.join(action_hits[:3])}")

    noun_hits = [n for n in _DEEP_NOUNS if n in lower]
    if noun_hits:
        score += min(0.20, 0.07 * len(noun_hits))
        reasons.append(f"deep nouns: {', '.join(noun_hits[:3])}")

    if _looks_multistep(message):
        score += 0.20
        reasons.append("multi-step request")

    if lower.count("?") >= 3:
        score += 0.15
        reasons.append("multiple questions")

    tool_hits = [verb for verb in _TOOL_INTENT_VERBS if verb in lower]
    if tool_hits:
        # Tool-requiring intents are the most common failure mode of a
        # Face-Lobe-only reply: the model invents tool names. Any hit
        # in this allowlist short-circuits to deep so even short
        # questions like "do you see any GPUs?" or "what tools do you
        # have?" go to a Depth Lobe that actually has tool access.
        return RouteDecision(
            kind="deep",
            confidence=min(1.0, 0.7 + 0.1 * len(tool_hits)),
            source="heuristic",
            reason=f"tool-requiring intent: {', '.join(tool_hits[:3])}",
            goal=_summary_goal(message),
            cleaned_message=message,
            raw_score=0.9,
        )

    score = min(1.0, score)
    kind = "deep" if score >= 0.5 else "direct"
    return RouteDecision(
        kind=kind,
        confidence=score if kind == "deep" else max(0.0, 1.0 - score),
        source="heuristic",
        reason="; ".join(reasons) if reasons else "no deep signals",
        goal=_summary_goal(message) if kind == "deep" else "",
        cleaned_message=message,
        raw_score=score,
    )


def _summary_goal(message: str) -> str:
    """Distill a user_visible_goal string from the user message.

    Used as the Depth Lobe job's user-facing goal label. Stays
    template-driven (no model call) per the anti-hallucination contract."""
    clean = " ".join(message.split())
    if not clean:
        return "Background work"
    if len(clean) <= 180:
        return clean
    return clean[:177].rstrip() + "..."


# ---------------------------------------------------------------------------
# LLM classifier (optional layer)
# ---------------------------------------------------------------------------

LlmClassifier = Callable[[str], dict[str, Any]]


def _llm_classify(
    message: str,
    classifier: LlmClassifier | None,
    heuristic: RouteDecision,
) -> RouteDecision | None:
    if classifier is None:
        return None
    try:
        raw = classifier(message) or {}
    except Exception as exc:
        return RouteDecision(
            kind=heuristic.kind,
            confidence=heuristic.confidence,
            source="heuristic",
            reason=f"{heuristic.reason} (llm classifier raised {type(exc).__name__})",
            goal=heuristic.goal,
            cleaned_message=heuristic.cleaned_message,
        )
    kind = str(raw.get("kind") or "").lower()
    if kind not in {"direct", "deep"}:
        return None
    confidence = float(raw.get("confidence") or 0.0)
    reason = str(raw.get("reason") or "model classified the message")
    goal = str(raw.get("goal") or _summary_goal(message))
    return RouteDecision(
        kind=kind,
        confidence=max(0.0, min(1.0, confidence)),
        source="llm",
        reason=reason[:240],
        goal=goal,
        cleaned_message=heuristic.cleaned_message,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def route(
    message: str,
    *,
    classifier: LlmClassifier | None = None,
) -> RouteDecision:
    """Decide ``direct`` vs ``deep`` for a Face Lobe user turn.

    Slash command always wins. Otherwise the heuristic runs; if its
    confidence is below ``HEURISTIC_DEFER_THRESHOLD`` AND a classifier
    is supplied AND the ``MS4_ROUTER_LLM_CLASSIFY`` env var is truthy,
    the classifier is consulted. The classifier's verdict overrides
    the heuristic when present and well-formed.
    """
    slash = _parse_slash(message or "")
    if slash is not None:
        return slash
    heuristic = _heuristic_route(message or "")
    # `raw_score == -1.0` is the "definitely direct" sentinel set by
    # _heuristic_route when the message is empty or matches a DIRECT_HINT.
    if heuristic.raw_score < 0.0:
        return heuristic
    # Confident-deep heuristic short-circuits the LLM.
    if heuristic.raw_score >= HEURISTIC_DEEP_THRESHOLD:
        return heuristic
    if os.environ.get(ENV_LLM_CLASSIFY, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return heuristic
    llm = _llm_classify(message or "", classifier, heuristic)
    return llm or heuristic
