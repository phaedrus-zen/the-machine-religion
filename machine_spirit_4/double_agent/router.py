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
    "cluster",
    "tool",
    "tools",
    "toolbox",
    "toolboxes",
    "mcp",
    "gpu",
    "gpus",
    "node",
    "nodes",
    "model",
    "models",
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
# A user can request several distinct analytical deliverables without using a
# long prompt, code block, or operational tool verb.  Those compound asks are
# exactly where a fast Face-only answer tends to become shallow.  Count the
# requested outputs independently so a recommendation + mechanism + tradeoffs
# + example crosses the Depth threshold, while a short single-aspect follow-up
# remains direct and can use the completed Depth answer already in history.
_ANALYTICAL_DELIVERABLES = (
    "recommendation",
    "mechanism",
    "tradeoff",
    "concrete example",
    "failure mode",
    "evidence",
    "constraint",
    "architecture",
)

# Depth is a model-capacity decision, not automatically a tool-permission
# decision.  These independent categories let the gateway/worker recognize a
# compound explanatory deliverable that the model can answer from the prompt
# and already-supplied conversation context.  Keep this deliberately narrower
# than the deep-routing vocabulary: a bare "diagnose the issue" remains
# tool-capable because it does not prove that the requested work is abstract.
_REASONING_ONLY_ANALYTICAL_CATEGORIES = (
    re.compile(r"\b(?:recommend(?:ation|ed|ing)?|should (?:i|we|you)|choose|pick|select|decide)\b", re.I),
    re.compile(r"\b(?:mechanism|explain (?:how|why)|describe (?:how|why)|how (?:does|do|would|will|can)|why (?:does|do|would|will|can))\b", re.I),
    re.compile(r"\b(?:trade-?offs?|pros? and cons?|advantages? (?:and|/) disadvantages?|downsides?)\b", re.I),
    re.compile(r"\b(?:concrete )?(?:example|scenario|case study|test case|worked case|use case)s?\b", re.I),
    re.compile(r"\b(?:architecture|design|strategy|approach)\b", re.I),
    re.compile(r"\b(?:compare|contrast|evaluate|assess)\b", re.I),
)

# Positive evidence that answering requires observations outside the prompt or
# an effect in the world.  Ambiguous jobs keep their tools; the deny-all path is
# entered only when a compound analytical request has none of these signals.
_DEPTH_EXTERNAL_OR_EFFECT_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:current(?:ly)?|right now|today|latest|live|real[- ]time|up[- ]to[- ]date|"
    r"online|internet|web|external sources?|citations?|cite|browse|search|look up|"
    r"fetch|retrieve)\b|"
    r"\b(?:inspect|check|verify|validate|query|read|tail|grep|scan|list)\b|"
    r"\b(?:run|execute|launch|restart|deploy|rollback|install|uninstall|edit|modify|"
    r"update|write|delete|remove|create|start|stop|open|connect|disconnect|ssh|"
    r"download|upload|capture|screenshot)\b|"
    r"\b(?:tools?|skills?|sessions?|files?|folders?|logs?|metrics?|process(?:es)?|"
    r"services?|endpoints?|databases?|repositories|repo|codebase|workspace|"
    r"filesystem|hosts?|machines?|nodes?|gpus?|cluster status|runtime status)\b|"
    r"https?://|(?:[a-zA-Z]:[\\/])|(?:^|[\s`'\"])(?:\.{0,2}/)[^\s`'\"]+"
    r")",
    re.IGNORECASE,
)

# These verbs can describe either an abstract scenario ("when diagnosing") or
# an instruction to gather evidence ("Diagnose why ...").  Matching them
# globally would break the analytical demo prompt, so recognize only a
# request/imperative at the start of the message or a new sentence/clause.
_DEPTH_REQUEST_FORM_OPERATIONAL_RE = re.compile(
    r"(?:\A|[.!?;]\s+|,?\s+(?:and\s+)?then\s+)"
    r"(?:(?:please|now|first|next|then)\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"(?:i|we)\s+(?:want|need|would like)(?:\s+you)?\s+to\s+)?"
    r"(?:implement(?:ing)?|audit(?:ing)?|research(?:ing)?|investigat(?:e|ing)|"
    r"debug(?:ging)?|diagnos(?:e|ing)|refactor(?:ing)?|benchmark(?:ing)?|"
    r"profil(?:e|ing)|trac(?:e|ing)|build(?:ing)?|scaffold(?:ing)?|"
    r"migrat(?:e|ing)|port(?:ing)?|rewrit(?:e|ing)|find\s+(?:the|why)|figure\s+out)\b",
    re.IGNORECASE,
)

# Negated catalog words are constraints, not positive tool intent.  Remove only
# these bounded phrases before scanning for external/effect signals; any
# affirmative signal elsewhere in the request remains visible and therefore
# fails open to tool access.
_DEPTH_NEGATED_TOOL_INTENT_RES = (
    re.compile(
        r"\b(?:do not|don't|dont|never)\s+"
        r"(?:use|call|invoke|search|browse)(?:\s+(?:for|through))?\s+"
        r"(?:(?:any|the)\s+)?"
        r"(?:tools?|skills?|sessions?|memory|external sources?|web(?:\s+browsing)?|internet)"
        r"(?:\s+(?:or|and)\s+(?:"
        r"(?:use|call|invoke|search|browse)(?:\s+(?:for|through))?\s+"
        r"(?:(?:any|the)\s+)?"
        r"(?:tools?|skills?|sessions?|memory|external sources?|web(?:\s+browsing)?|internet)"
        r"|(?:(?:any|the)\s+)?"
        r"(?:tools?|skills?|sessions?|memory|external sources?|web(?:\s+browsing)?|internet)"
        r"))*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwithout\s+(?:using\s+)?(?:(?:any|the)\s+)?"
        r"(?:tools?|browsing|searching|skills?|sessions?|memory|external sources?|"
        r"web(?:\s+browsing)?|internet)"
        r"(?:\s+(?:or|and)\s+(?:(?:any|the)\s+)?"
        r"(?:tools?|browsing|searching|skills?|sessions?|memory|external sources?|"
        r"web(?:\s+browsing)?|internet))*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:(?:any|the)\s+)?"
        r"(?:tools?|skills?|sessions?|memory|external sources?|web(?:\s+browsing)?|internet)"
        r"(?:\s+(?:or|and)\s+(?:(?:any|the)\s+)?"
        r"(?:tools?|skills?|sessions?|memory|external sources?|web(?:\s+browsing)?|internet))*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\banswer\s+only\s+from\s+(?:the\s+)?supplied\s+(?:context|information|text)\b",
        re.IGNORECASE,
    ),
)


def _without_negated_tool_intent(message: str) -> str:
    text = message
    for pattern in _DEPTH_NEGATED_TOOL_INTENT_RES:
        text = pattern.sub(" ", text)
    return text
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
    "hivemind tools", "hivemind mcp", "mcp tools",
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
# Bounded positive imperative/request for an explicitly named tool. Anchoring
# the command shape rejects negated or quoted mentions that merely contain
# "use ... tool"; the helper also rejects word/phrase meta-language before
# requiring at least two identity/descriptor words. Identity words remain
# generic, so canonical and ASR-alias spellings match by shape without
# duplicating voice normalization in this text router.
_EXPLICIT_NAMED_TOOL_REQUEST_RE = re.compile(
    r"^\s*(?:(?:now|then|hey)\s*,?\s+)?"
    r"(?:oracle\s*,\s*)?"
    r"(?:(?:please)\s+|(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"i\s+want\s+you\s+to\s+)?(?:use|invoke|call|execute)\s+"
    r"(?P<identity>[a-z0-9][a-z0-9_.@-]*(?:\s+[a-z0-9][a-z0-9_.@-]*){0,8})"
    r"\s+tool(?!\w)",
    re.IGNORECASE,
)
_EXPLICIT_TOOL_META_NOUNS = frozenset({
    "word", "phrase", "term", "text",
    "sentence", "example", "quote", "quotation",
})
_EXPLICIT_TOOL_META_DEMONSTRATIVES = frozenset({"this", "that", "following"})


def _has_explicit_named_tool_request(message: str) -> bool:
    match = _EXPLICIT_NAMED_TOOL_REQUEST_RE.match(message)
    if match is None:
        return False
    words = match.group("identity").lower().split()
    if words and words[0] in {"the", "a", "an"}:
        words = words[1:]
    if not words or words[0] in _EXPLICIT_TOOL_META_NOUNS:
        return False
    if (
        len(words) >= 2
        and words[0] in _EXPLICIT_TOOL_META_DEMONSTRATIVES
        and words[1] in _EXPLICIT_TOOL_META_NOUNS
    ):
        return False
    return len(words) >= 2


def is_reasoning_only_analytical(message: str) -> bool:
    """Return whether a Depth request must run with an empty tool catalog.

    A request qualifies only when it asks for at least two independent
    analytical deliverables and contains no positive signal for current or
    external evidence, named tools, system inspection, or real-world effects.
    This intentionally fails open to tool access for ambiguous diagnostics;
    it exists to stop abstract explanatory jobs from wandering through skills
    and old sessions merely because they were routed to the larger model.
    """
    text = str(message or "").strip()
    if not text or _has_explicit_named_tool_request(text):
        return False
    positive_intent_text = _without_negated_tool_intent(text)
    if (
        _DEPTH_EXTERNAL_OR_EFFECT_INTENT_RE.search(positive_intent_text)
        or _DEPTH_REQUEST_FORM_OPERATIONAL_RE.search(positive_intent_text)
    ):
        return False
    category_count = sum(
        1 for pattern in _REASONING_ONLY_ANALYTICAL_CATEGORIES if pattern.search(text)
    )
    return category_count >= 2


def _phrase_re(phrase: str) -> re.Pattern[str]:
    cleaned = " ".join(phrase.strip().split())
    return re.compile(r"(?<!\w)" + re.escape(cleaned) + r"(?!\w)")


_TOOL_INTENT_RES = tuple(
    (phrase.strip(), _phrase_re(phrase))
    for phrase in _TOOL_INTENT_VERBS
    if phrase.strip()
)

_OPERATIONAL_TOOL_INTENTS = {
    "run", "execute", "launch", "spawn", "kill", "restart", "deploy",
    "rollback", "click", "type", "scroll", "start", "stop", "boot",
    "shut down", "shutdown", "create", "install", "uninstall", "remove",
    "delete", "wipe", "edit", "modify", "update the", "configure",
    "provision", "scale", "tune", "terminate", "halt", "pause", "resume",
    "mount", "unmount", "format", "partition", "connect", "disconnect",
    "ssh", "read file", "write file", "list files", "cat", "ls",
    "download", "upload", "take a screenshot", "screenshot", "capture the",
}


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

    if _has_explicit_named_tool_request(message):
        return RouteDecision(
            kind="deep",
            confidence=0.95,
            source="heuristic",
            reason="explicit named-tool request",
            goal=_summary_goal(message),
            cleaned_message=message,
            raw_score=0.95,
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

    analytical_hits = [item for item in _ANALYTICAL_DELIVERABLES if item in lower]
    if analytical_hits:
        score += min(0.40, 0.10 * len(analytical_hits))
        if len(analytical_hits) >= 3:
            reasons.append(
                "multiple analytical deliverables: " + ", ".join(analytical_hits[:4])
            )

    if _looks_multistep(message):
        score += 0.20
        reasons.append("multi-step request")

    if lower.count("?") >= 3:
        score += 0.15
        reasons.append("multiple questions")

    tool_hits = [phrase for phrase, pattern in _TOOL_INTENT_RES if pattern.search(lower)]
    if tool_hits:
        # Tool words are strong only when paired with an object or a
        # multi-word intent. A lone verb such as "open" in "markets are
        # open" must not dispatch a depth job.
        score += min(0.30, 0.12 * len(tool_hits))
        strong_tool_hits = [hit for hit in tool_hits if " " in hit]
        if strong_tool_hits:
            score += 0.28
        if any(hit in _OPERATIONAL_TOOL_INTENTS for hit in tool_hits):
            score += 0.40
        if noun_hits:
            score += 0.28
        reasons.append(f"tool-requiring intent: {', '.join(tool_hits[:3])}")

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
