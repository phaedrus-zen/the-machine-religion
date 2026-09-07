"""Continuation detection for Double Agent revision bumps.

Problem
-------

Every Face Lobe turn bumps the conversation revision, and by default any
in-flight Depth Lobe job from an older revision is marked ``stale`` (the
user changed direction, so the old background work is probably moot).

But that is exactly wrong when the user's new message is a *continuation*
— "ok, do that", "yes please", "any update?". There the user is WAITING
for the background work, and staling it mid-stream is the worst possible
behavior (observed live).

Two-layer detection
--------------------

1. :func:`phrase_is_continuation` — a zero-latency, zero-dependency
   phrase fast-path. Catches the obvious short affirmations/prompts.
   This is the only layer that runs in tests and when no classifier is
   wired, so behavior stays deterministic and hermetic.

2. :func:`make_llm_continuation_classifier` — an *optional*, injectable
   LLM classifier for the ambiguous middle ground (paraphrases the
   phrase list can't enumerate, other languages, etc.). It is:

   * **Bounded** — the runner only consults it when there are
     staleable jobs AND the phrase path missed AND the message is short
     enough to plausibly be a continuation (``CLASSIFIER_CONSULT_MAX_LEN``).
   * **Fail-safe** — any error / timeout / unparseable answer returns
     ``None`` ("unsure"), and the runner treats unsure exactly like the
     pre-classifier behavior (stale). A down cluster therefore degrades
     to the old regex-only semantics, never to a hang.
   * **Cheap** — strict one-word output, tiny ``max_tokens``, short
     timeout, ideally pointed at a small model via
     ``MS4_DA_CONTINUATION_MODEL``.

The classifier returns ``True`` (continuation — keep work), ``False``
(new direction — stale), or ``None`` (unsure — caller decides; the
runner stales, matching legacy behavior).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional


log = logging.getLogger("ms4.double_agent.continuation")


# A classifier takes the user message excerpt and returns:
#   True  -> continuation (do NOT stale in-flight work)
#   False -> new direction (stale as usual)
#   None  -> unsure / unavailable (caller falls back to legacy behavior)
ContinuationClassifier = Callable[[str], Optional[bool]]


# ---------------------------------------------------------------------------
# Layer 1: phrase fast-path (moved verbatim from runner._is_short_continuation)
# ---------------------------------------------------------------------------

# Short-continuation phrases that should NOT stale an in-flight Depth Lobe
# job. Matching contract:
#  * lowercased, stripped of trailing punctuation
#  * exact match against the whole message (so "do it" doesn't also fire
#    on "do it differently") OR the message starts with the phrase
#    followed by an allowed tail connector (please / now / etc).
#  * length-capped so a long sentence containing "yes" as a word doesn't
#    accidentally count.
_CONTINUATION_PHRASES: tuple[str, ...] = (
    "do that",
    "yes",
    "yeah",
    "yep",
    "yup",
    "ok",
    "okay",
    "sure",
    "go ahead",
    "go on",
    "continue",
    "keep going",
    "please do",
    "do it",
    "please continue",
    "and?",
    "and then?",
    "what next",
    "what's next",
    "and after",
    "any update",
    "any updates",
    "do you have an update",
    "any progress",
    "what's the status",
    "status update",
)
_CONTINUATION_MAX_LEN = 50
# Words allowed AFTER a continuation phrase without flipping it into a
# new-direction message. E.g. "yes please", "do it now", "ok thanks".
_CONTINUATION_TAIL_TOKENS: tuple[str, ...] = (
    "please", "now", "thanks", "thank you", "sir", "ma'am",
    "if you can", "if you could", "for me", "go", "do",
)

# The runner only spends an LLM classification when the message is at
# most this long. Anything longer is almost certainly a new direction
# (and staling is correct), so we don't pay for a classification.
CLASSIFIER_CONSULT_MAX_LEN = 160


def phrase_is_continuation(message: str) -> bool:
    """Return True when the message is an obvious short continuation.

    Pure, deterministic, no I/O. This is the fast-path used everywhere;
    the LLM classifier only runs when this returns False.
    """
    if not message:
        return False
    text = message.strip().lower().rstrip(".!?,;: \t\n")
    if not text or len(text) > _CONTINUATION_MAX_LEN:
        return False
    for phrase in _CONTINUATION_PHRASES:
        if text == phrase:
            return True
        if text.startswith(phrase + " "):
            tail = text[len(phrase) + 1:].strip()
            if tail in _CONTINUATION_TAIL_TOKENS:
                return True
            # Also accept a tail that's itself another continuation
            # phrase ("yes please continue") — recurse once.
            if any(tail == p or tail.startswith(p + " ") for p in _CONTINUATION_PHRASES):
                return True
    return False


# ---------------------------------------------------------------------------
# Layer 2: optional LLM classifier
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You classify a user's reply in an ongoing conversation where some "
    "background work is currently running.\n"
    "Decide if the reply means CONTINUE (let the running background work "
    "keep going / the user is waiting for it / acknowledging it) or NEW "
    "(the user is changing direction, so the running work is no longer "
    "wanted).\n"
    "Answer with EXACTLY one word: CONTINUE or NEW. No punctuation, no "
    "explanation."
)


def _parse_verdict(text: str) -> Optional[bool]:
    """Map a model reply to True/False/None. Strict + fail-safe."""
    if not text:
        return None
    t = text.strip().lower()
    # Look at the first ~12 chars so trailing noise can't flip it.
    head = t[:12]
    if "continue" in head:
        return True
    if "new" in head:
        return False
    return None


def make_llm_continuation_classifier(
    hivemind_url: str,
    *,
    model: str | None = None,
    timeout: float | None = None,
) -> ContinuationClassifier:
    """Build a fail-safe LLM continuation classifier.

    ``model`` defaults to ``MS4_DA_CONTINUATION_MODEL`` (operators should
    point this at a small/fast model). ``timeout`` defaults to
    ``MS4_DA_CONTINUATION_TIMEOUT`` seconds (2.5s). Any failure path
    returns ``None`` so the runner falls back to legacy (stale) behavior
    — the classifier can only ever *rescue* a continuation, never hang a
    turn or invent a new failure mode.
    """
    resolved_model = model or os.environ.get("MS4_DA_CONTINUATION_MODEL") or ""
    if timeout is None:
        try:
            timeout = float(os.environ.get("MS4_DA_CONTINUATION_TIMEOUT", "2.5"))
        except (TypeError, ValueError):
            timeout = 2.5

    def _classify(message: str) -> Optional[bool]:
        if not message or not message.strip():
            return None
        # Import lazily so the double_agent package doesn't hard-depend on
        # the gateway import chain (keeps worker subprocess + tests light).
        try:
            from machine_spirit_4.gateway import hivemind_tools
        except Exception as exc:  # pragma: no cover - import guard
            log.debug("continuation classifier import failed: %s", exc)
            return None

        chat_model = resolved_model
        if not chat_model:
            # No explicit model: ask HiveMind to recommend a small chat
            # model once. Best-effort; if it fails we return unsure.
            try:
                rec = hivemind_tools.models_recommend(
                    hivemind_url, capability="chat"
                )
                if isinstance(rec, dict):
                    from machine_spirit_4.double_agent.model_picker import (
                        _is_chat_capable,
                    )

                    required_values = (
                        rec.get("capability"),
                        rec.get("recommended_model"),
                        rec.get("backend"),
                    )
                    if all(
                        isinstance(value, str) and value.strip()
                        for value in required_values
                    ) and _is_chat_capable(rec):
                        chat_model = str(rec.get("recommended_model") or "").strip()
            except Exception as exc:
                log.debug("continuation classifier model recommend failed: %s", exc)
                return None
        if not chat_model:
            return None

        try:
            resp = hivemind_tools.inference_chat(
                hivemind_url,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": message.strip()[:CLASSIFIER_CONSULT_MAX_LEN]},
                ],
                model=chat_model,
                max_tokens=4,
                temperature=0.0,
                timeout=timeout,
            )
        except Exception as exc:
            log.info("continuation classifier call failed (unsure): %s", exc)
            return None

        text = _extract_text(resp)
        verdict = _parse_verdict(text)
        log.debug("continuation classifier: %r -> %r (%r)", message[:40], verdict, text[:20])
        return verdict

    return _classify


def _extract_text(resp: Any) -> str:
    """Pull the assistant text out of an OpenAI-shaped chat response."""
    if isinstance(resp, str):
        return resp
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    for key in ("text", "content", "output"):
        if isinstance(resp.get(key), str):
            return resp[key]
    return ""
