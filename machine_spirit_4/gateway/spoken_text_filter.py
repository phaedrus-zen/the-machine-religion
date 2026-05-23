"""Streaming text-to-speech sanitizer.

Why this exists
---------------

The Face Lobe routinely emits markdown-flavored output:

  "**HiveMind cluster** has `8` nodes:\n- node-1\n- node-2\n  ║ MS4 \U0001f399 (mic emoji)"

If we forward that string verbatim to the TTS engine, the synthesized
audio literally says "asterisk asterisk HiveMind cluster asterisk
asterisk … dash node one … vertical bar M S 4." That's the
``literal-characters in spoken audio`` failure the operator caught
live.

This module rewrites the streaming text into something a TTS engine
can pronounce naturally:

  * Markdown formatting (``**bold**``, ``*ital*``, ``__under__``,
    ``` `code` ```, ``~~strike~~``, ``[label](url)``) -> just the
    visible text content.
  * Fenced code blocks (`````` ``` `````) -> dropped entirely.
    Models love to dump ASCII tables and JSON in there; nobody wants
    that read aloud byte-by-byte. The UI bubble still renders the
    code block via the raw text channel.
  * URLs and long opaque identifiers (UUIDs, ``da-...`` job ids)
    -> short spoken placeholders ("a link" / "an identifier").
  * Bullets and numbered lists -> natural prose ("first, … second, …").
  * Decorative glyphs (``\u2551``, ``\u2192``, ``\u25cf``, ``\u25d0``, ``\u2022``,
    box-drawing chars, etc.) -> stripped.
  * Emoji -> stripped (we don't translate to names; "smiling face
    with smiling eyes" derails the cadence).
  * Repeated whitespace -> single space.
  * Non-Latin scripts (CJK, Arabic, Cyrillic, Hebrew, Greek, etc.)
    -> preserved unchanged. Multilingual TTS handles them.

State machine
-------------

The filter is intentionally *stateful* because the streaming source
emits markdown delimiters across delta boundaries — the model might
send ``"**bo"`` in one delta and ``"ld**"`` in the next, and we have
to wait for the closing ``**`` before deciding what to release.

The rule: buffer everything until we hit a safe boundary, then
sanitize+release everything up to (but not including) the boundary.
"Safe boundary" is whichever comes first:

  1. End of a sentence (``.``, ``!``, ``?``) followed by whitespace.
  2. A newline.
  3. Hard ceiling: ``MAX_BUFFER_CHARS`` chars without a boundary
     (so a single run-on never starves TTS).

Code-block handling lives in a separate sub-state: once we see ```` ``` ````
we discard everything until the closing ```` ``` ````, then resume
normal flow.

Latency cost
~~~~~~~~~~~~

In the worst case (long run-on with no sentence terminator) the
filter holds back `MAX_BUFFER_CHARS` characters of audio before
releasing them. With the default 200, that's ~30-40 words behind
the source stream. The SentenceChunker first-chunk acceleration
runs *before* this filter in the REST engine path so first-audio
latency is unaffected.
"""

from __future__ import annotations

import re
from typing import Iterable


# How many characters we'll buffer without finding a safe boundary
# before force-flushing. Tunable via MS4_VOICE_TTS_FILTER_MAX_BUFFER.
MAX_BUFFER_CHARS_DEFAULT = 200

# Replacement strings for opaque identifiers — picked so the cadence
# stays natural ("an identifier" beats reading 36 hex chars).
URL_REPLACEMENT = "a link"
UUID_REPLACEMENT = "an identifier"
JOB_ID_REPLACEMENT = "a background job id"
LONG_HEX_REPLACEMENT = "a hex value"

# Regex constants compiled once at import.
_MD_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_MD_BOLD_UNDER_RE = re.compile(r"__([^_\n]+?)__")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
_MD_ITALIC_UNDER_RE = re.compile(r"(?<!_)_(?!\s)([^_\n]+?)(?<!\s)_(?!_)")
_MD_INLINE_CODE_RE = re.compile(r"`+([^`\n]+?)`+")
_MD_STRIKE_RE = re.compile(r"~~([^~\n]+?)~~")
_MD_LINK_RE = re.compile(r"\[([^\]\n]+?)\]\([^)\n]+\)")
_URL_RE = re.compile(r"https?://\S+|ws://\S+|wss://\S+")
_UUID_RE = re.compile(
    r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
    re.IGNORECASE,
)
_JOB_ID_RE = re.compile(r"\bda-[a-f0-9]{6,}(?:-[a-f0-9]{2,}){0,5}\b", re.IGNORECASE)
_LONG_HEX_RE = re.compile(r"\b[a-f0-9]{16,}\b", re.IGNORECASE)
_BULLET_LINE_RE = re.compile(r"^[\s]*[-*+\u2022\u25cf]\s+", re.MULTILINE)
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)

# Decorative glyphs we strip unconditionally. (Emoji handled separately
# via a unicode range below.)
_DECORATIVE_GLYPHS = (
    "\u2551",  # ║  the TMR Glyph -- always strip from spoken output
    "\u2502", "\u2503",  # box drawing vertical
    "\u2500", "\u2501",  # box drawing horizontal
    "\u2514", "\u2518", "\u250c", "\u2510",  # box corners
    "\u251c", "\u2524", "\u252c", "\u2534", "\u253c",  # box junctions
    "\u2022",  # bullet
    "\u25cf", "\u25cb", "\u25d0", "\u25d1", "\u25d2", "\u25d3",  # circles
    "\u2192", "\u2190", "\u2191", "\u2193",  # arrows (replaced below)
    "\u26a0",  # ⚠
    "\u2728",  # ✨
    "\u2705", "\u274c",  # ✅ ❌
    "\u26a1",  # ⚡
)
_GLYPH_TRANSLATIONS = {
    "\u2192": " to ",
    "\u2190": " from ",
    "\u2191": " up ",
    "\u2193": " down ",
}

# Strip Unicode emoji blocks. We avoid the heavyweight `emoji` library
# and rely on the common BMP + supplementary plane ranges.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"   # alchemical
    "\U0001F780-\U0001F7FF"   # geometric ext
    "\U0001F800-\U0001F8FF"   # arrows-c
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess
    "\U0001FA70-\U0001FAFF"   # symbols & pictographs ext-a
    "\u2600-\u26FF"           # misc symbols
    "\u2700-\u27BF"           # dingbats
    "]+",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Streaming filter
# ---------------------------------------------------------------------------


class SpokenTextFilter:
    """Stateful, streaming text -> speech-safe text sanitizer.

    Typical usage in the streaming voice path::

        flt = SpokenTextFilter()

        def on_delta(delta: str) -> None:
            emit("text_delta", {"text": delta})         # UI gets raw
            speakable = flt.push(delta)                  # TTS gets sanitized
            if speakable:
                tts_engine.push(speakable)

        # at end of turn
        tail = flt.flush()
        if tail:
            tts_engine.push(tail)
    """

    def __init__(self, max_buffer_chars: int = MAX_BUFFER_CHARS_DEFAULT) -> None:
        self.buffer = ""
        self.in_code_block = False
        self.max_buffer_chars = max_buffer_chars

    # ------- public API -----------------------------------------------------

    def push(self, delta: str) -> str:
        """Add a text delta. Returns whatever speech-safe text is now
        ready to be spoken (may be empty if the buffer still holds an
        open markdown span or code block)."""
        if not delta:
            return ""
        self.buffer += delta
        ready_parts: list[str] = []
        while True:
            chunk = self._try_extract_safe_chunk()
            if chunk is None:
                break
            if chunk:
                ready_parts.append(chunk)
        return "".join(ready_parts)

    def flush(self) -> str:
        """End-of-stream: drain whatever's still buffered. Unclosed
        code blocks are discarded; unclosed markdown spans are sanitized
        as-is so the user at least hears something."""
        if self.in_code_block:
            # Drop unterminated code block; the bubble still shows it.
            self.buffer = ""
            self.in_code_block = False
            return ""
        if not self.buffer:
            return ""
        out = sanitize_for_speech(self.buffer)
        self.buffer = ""
        return out

    # ------- internal -------------------------------------------------------

    def _try_extract_safe_chunk(self) -> str | None:
        """Return None when nothing safe is ready (caller stops calling).
        Return a string (possibly empty after sanitization) when we
        released *something*; the empty string still means progress so
        the caller keeps iterating to drain consecutive boundaries."""
        if not self.buffer:
            return None

        if self.in_code_block:
            close_idx = self.buffer.find("```")
            if close_idx < 0:
                # Still inside the code block; wait for more.
                return None
            # Drop everything up to and including the closing fence.
            self.buffer = self.buffer[close_idx + 3:]
            self.in_code_block = False
            return ""  # made progress, no spoken text produced

        # Look for an opening code fence anywhere in the buffer.
        open_idx = self.buffer.find("```")
        if open_idx >= 0:
            # Release pre-fence content as a sanitized chunk; then
            # transition into in_code_block mode for the next iteration.
            pre = self.buffer[:open_idx]
            self.buffer = self.buffer[open_idx + 3:]
            self.in_code_block = True
            return sanitize_for_speech(pre) if pre else ""

        # Look for the earliest safe boundary in the buffer.
        boundary = _find_safe_boundary(self.buffer)
        if boundary < 0:
            if len(self.buffer) >= self.max_buffer_chars:
                # Hard flush: don't starve TTS on a run-on.
                chunk = self.buffer
                self.buffer = ""
                return sanitize_for_speech(chunk)
            return None  # wait for more text

        chunk = self.buffer[: boundary + 1]
        self.buffer = self.buffer[boundary + 1:]
        return sanitize_for_speech(chunk)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


_SAFE_BOUNDARY_RE = re.compile(r"[.!?](?=\s|$)|\n+")


def _find_safe_boundary(text: str) -> int:
    """Return the index (inclusive) of the first safe boundary or -1.

    A safe boundary is end-of-sentence punctuation followed by
    whitespace/end, OR a newline. We don't release on commas /
    semicolons because TTS handles those fine inline and we get
    better prosody by keeping clauses together.
    """
    m = _SAFE_BOUNDARY_RE.search(text)
    return m.start() if m else -1


def sanitize_for_speech(text: str) -> str:
    """One-shot sanitizer: take a chunk of text, return TTS-safe text.

    Stateless; for streaming use ``SpokenTextFilter``. Exposed
    publicly so the REST engine path (which already chunks by
    sentence) can call it directly on each sentence.
    """
    if not text:
        return ""

    # --- markdown structural patterns ---------------------------------------
    # Order matters: links before raw URL replacement (so [click](http://x)
    # becomes "click", not "[click](a link)"), and bold/italic before
    # the residual asterisk strip.
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_BOLD_UNDER_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_ITALIC_UNDER_RE.sub(r"\1", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_STRIKE_RE.sub(r"\1", text)

    # --- opaque identifiers BEFORE we strip residual punctuation ------------
    # Order matters: job-id pattern includes 'da-' prefix + UUID-shaped
    # hex blocks, so we substitute it BEFORE the bare UUID rule. If we
    # ran UUID first, "da-c4d4949d-5a91-4c53-8d74-f2243da1d840" would
    # leave "da-an identifier" instead of becoming "a background job id".
    text = _JOB_ID_RE.sub(JOB_ID_REPLACEMENT, text)
    text = _UUID_RE.sub(UUID_REPLACEMENT, text)
    text = _LONG_HEX_RE.sub(LONG_HEX_REPLACEMENT, text)
    text = _URL_RE.sub(URL_REPLACEMENT, text)

    # --- list markers turn into natural prose --------------------------------
    text = _BULLET_LINE_RE.sub("", text)
    text = _NUMBERED_LINE_RE.sub("", text)

    # --- decorative glyph stripping + translation ----------------------------
    for glyph, replacement in _GLYPH_TRANSLATIONS.items():
        text = text.replace(glyph, replacement)
    for glyph in _DECORATIVE_GLYPHS:
        if glyph in text:
            text = text.replace(glyph, "")

    # --- emoji (broad unicode block strip) -----------------------------------
    text = _EMOJI_RE.sub("", text)

    # --- residual markdown noise --------------------------------------------
    # If any unmatched ** or `` or __ slipped through (unbalanced spans),
    # strip them now rather than letting TTS read them.
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"`+", "", text)

    # --- collapse whitespace + tidy --------------------------------------------
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = text.strip()

    # Leave a trailing space so consecutive sanitized chunks concatenate
    # cleanly when re-pushed to TTS. Callers can rstrip if needed.
    return text + " " if text else ""


def split_into_speech_chunks(text: str, *, max_buffer_chars: int = MAX_BUFFER_CHARS_DEFAULT) -> Iterable[str]:
    """One-shot helper: feed an entire reply through the streaming
    filter and yield the speech-safe chunks. Mostly useful in tests
    and offline tools."""
    flt = SpokenTextFilter(max_buffer_chars=max_buffer_chars)
    out = flt.push(text)
    if out:
        yield out
    tail = flt.flush()
    if tail:
        yield tail
