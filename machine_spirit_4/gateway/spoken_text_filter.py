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
  * ATX headings and Markdown tables -> natural cell prose; heading markers,
    pipes, and alignment/separator runs never reach TTS.
  * HTML/XML-like tags and comments -> dropped from speech while their visible
    text children remain; state survives arbitrary model-delta boundaries.
  * Fenced code blocks (`````` ``` `````) -> dropped entirely.
    Models love to dump ASCII tables and JSON in there; nobody wants
    that read aloud byte-by-byte. The UI bubble still renders the
    code block via the raw text channel.
  * URLs and long opaque identifiers (UUIDs, ``da-...`` job ids)
    -> short spoken placeholders ("a link" / "an identifier").
  * Technical names and initialisms (``HiveMind``, ``GPU``, ``API``, etc.)
    -> backend-neutral plain-text aliases ("Hive Mind", "G P U").
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
     (so a single run-on never starves TTS). Trailing alias and protected-
     span prefixes are retained until the next character disambiguates them.

Code-block handling lives in a separate sub-state: once we see ```` ``` ````
we discard everything until the closing ```` ``` ````, retaining only a
possible split closing fence between deltas, then resume normal flow.
URL recognition has lexical priority over sentence punctuation: once a
supported scheme starts, one placeholder is emitted and all non-whitespace
continuation is discarded across deltas.

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
_MD_ATX_HEADING_RE = re.compile(r"(?<!\S)#{1,6}(?=\s)")
_MD_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")
_URL_RE = re.compile(r"https?://\S+|ws://\S+|wss://\S+")
_URL_AT_BUFFER_END_RE = re.compile(r"(?:https?|wss?)://\S+$")
_PROTECTED_STREAM_MARKERS = ("http://", "https://", "ws://", "wss://", "```")
_PROTECTED_STREAM_PREFIXES = frozenset(
    marker[:length]
    for marker in _PROTECTED_STREAM_MARKERS
    for length in range(1, len(marker) + 1)
)
_UUID_RE = re.compile(
    r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
    re.IGNORECASE,
)
_JOB_ID_RE = re.compile(r"\bda-[a-f0-9]{6,}(?:-[a-f0-9]{2,}){0,5}\b", re.IGNORECASE)
_LONG_HEX_RE = re.compile(r"\b[a-f0-9]{16,}\b", re.IGNORECASE)
_SPOKEN_ALIASES = {
    # Longest forms are selected by the compiled expression below.  Keep the
    # audible plural/possessive form stable for ASCII and typographic marks.
    "GPUs'": "G P U's",
    "GPUs\u2019": "G P U's",
    "GPU's": "G P U's",
    "GPU\u2019s": "G P U's",
    # Approved model names are exact aliases; decorated/suffixed identifiers
    # are deliberately rejected by the replacement callback.
    "Whisper large-v3-turbo": "Whisper large version three turbo",
    "Llama 3.1 8B": "Llama three point one eight B",
    "gpt-oss:20b": "G P T O S S twenty B",
    "Qwen3-TTS": "Qwen three T T S",
    "XTTS-v2": "X T T S version two",
    "HiveMind": "Hive Mind",
    "HTTPS": "H T T P S",
    "HTTP": "H T T P",
    "CUDA": "coo duh",
    "ROCm": "rock em",
    "JSON": "jay son",
    "SSH": "S S H",
    "VRAM": "V ram",
    "RAM": "ram",
    "PCIe": "P C I E",
    "FP16": "F P sixteen",
    "BF16": "B F sixteen",
    "C++": "C plus plus",
    "C#": "C sharp",
    "GPUs": "G P U's",
    "CPUs": "C P U's",
    "NPUs": "N P U's",
    "APIs": "A P I's",
    "LLMs": "L L M's",
    "VMs": "V M's",
    "UIs": "U I's",
    "GPU": "G P U",
    "CPU": "C P U",
    "NPU": "N P U",
    "API": "A P I",
    "LLM": "L L M",
    "VM": "V M",
    "UI": "U I",
    "TTS": "T T S",
    "ASR": "A S R",
    "VAD": "V A D",
    "MCP": "M C P",
}
_MODEL_ALIAS_KEYS = frozenset({
    "Whisper large-v3-turbo",
    "Llama 3.1 8B",
    "gpt-oss:20b",
    "Qwen3-TTS",
    "XTTS-v2",
})
_SPOKEN_ALIAS_PATTERN = "|".join(
    re.escape(alias)
    for alias in sorted(_SPOKEN_ALIASES, key=lambda alias: (-len(alias), alias))
)
_SPOKEN_ALIAS_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{_SPOKEN_ALIAS_PATTERN})(?![A-Za-z0-9_])"
)
_SEMVER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])v((?:[0-9]{1,3}\.){1,2}[0-9]{1,3})"
    r"(?![A-Za-z0-9_+-]|\.[0-9])"
)
_TEMPERATURE_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([0-9]{1,4}(?:\.[0-9]{1,2})?)\u00b0([CF])(?![A-Za-z0-9_])"
)
_PERCENT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([0-9]{1,6}(?:\.[0-9]{1,3})?)%(?![A-Za-z0-9_])"
)
_NUMERIC_EQUALS_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([0-9]{1,6}(?:\.[0-9]{1,3})?)\s*=\s*"
    r"([0-9]{1,6}(?:\.[0-9]{1,3})?)(?![A-Za-z0-9_.])"
)
_NUMERIC_TIMES_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([0-9]{1,6}(?:\.[0-9]{1,3})?)\s*\u00d7\s*"
    r"([0-9]{1,6}(?:\.[0-9]{1,3})?)(?![A-Za-z0-9_.])"
)
_NUMERIC_COMPARISON_RE = re.compile(
    r"(?<![A-Za-z0-9_.<>=!])(?P<operator><=|>=|\u2264|\u2265|<|>)\s*"
    r"(?P<value>[+-]?[0-9]{1,6}(?:\.[0-9]{1,3})?[A-Za-z]{0,8})"
    r"(?![A-Za-z0-9_>]|\.[0-9])"
)
_SPOKEN_NUMERIC_COMPARISONS = {
    "<": "less than",
    ">": "greater than",
    "<=": "less than or equal to",
    ">=": "greater than or equal to",
    "\u2264": "less than or equal to",
    "\u2265": "greater than or equal to",
}
_SENSITIVE_LABEL_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|token|secret|password|passwd|bearer)[=:]",
    re.IGNORECASE,
)
_STREAM_DYNAMIC_PREFIX_RE = re.compile(
    r"(?:"
    r"v[0-9]{0,3}(?:\.[0-9]{0,3}){0,2}"
    r"|[0-9]{1,6}(?:\.[0-9]{0,3})?"
    r"(?:[%=\u00d7][0-9]{0,6}(?:\.[0-9]{0,3})?|\u00b0[CF]?)?"
    r")\Z"
)
_STREAM_NUMERIC_COMPARISON_PREFIX_RE = re.compile(
    r"(?:<=?|>=?|\u2264|\u2265)\s*[+-]?[0-9]{0,6}"
    r"(?:\.[0-9]{0,3})?[A-Za-z]{0,8}\Z"
)
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


_MAX_PENDING_MARKUP_CHARS = 4096
_HTML_LIKE_OPEN_TAG_RE = re.compile(
    r"<[A-Za-z][A-Za-z0-9:_.-]*"
    r"(?:\s+[A-Za-z_:][A-Za-z0-9:_.-]*"
    r"(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+))?)*"
    r"\s*/?>",
    flags=re.DOTALL,
)
_HTML_BOOLEAN_ATTRIBUTES = frozenset(
    {
        "allowfullscreen",
        "async",
        "autofocus",
        "autoplay",
        "checked",
        "controls",
        "default",
        "defer",
        "disabled",
        "formnovalidate",
        "hidden",
        "inert",
        "ismap",
        "itemscope",
        "loop",
        "multiple",
        "muted",
        "nomodule",
        "novalidate",
        "open",
        "playsinline",
        "readonly",
        "required",
        "reversed",
        "selected",
    }
)
_HTML_TAG_NAMES = frozenset(
    {
        "a", "abbr", "address", "area", "article", "aside", "audio",
        "b", "base", "bdi", "bdo", "blockquote", "body", "br", "button",
        "canvas", "caption", "cite", "code", "col", "colgroup",
        "data", "datalist", "dd", "del", "details", "dfn", "dialog", "div",
        "dl", "dt", "em", "embed", "fieldset", "figcaption", "figure",
        "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "head",
        "header", "hgroup", "hr", "html", "i", "iframe", "img", "input",
        "ins", "kbd", "label", "legend", "li", "link", "main", "map",
        "mark", "menu", "meta", "meter", "nav", "noscript", "object", "ol",
        "optgroup", "option", "output", "p", "picture", "pre", "progress",
        "q", "rp", "rt", "ruby", "s", "samp", "script", "search",
        "section", "select", "slot", "small", "source", "span", "strong",
        "style", "sub", "summary", "sup", "table", "tbody", "td",
        "template", "textarea", "tfoot", "th", "thead", "time", "title",
        "tr", "track", "u", "ul", "var", "video", "wbr",
    }
)


def _is_markup_name_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in ":_.-")


def _is_markup_attribute_start(char: str) -> bool:
    return char.isascii() and (char.isalpha() or char in "_:")


class _StreamingMarkupFilter:
    """Drop HTML-like tags/comments with bounded state across text deltas.

    The visible assistant transcript keeps the raw model text.  Speech should
    neither read markup nor reinterpret a tag delimiter such as ``>42`` as a
    numeric comparison.  Opening tags are confirmed only after a syntactically
    complete, quote-aware ``>`` delimiter, which preserves compact prose such
    as ``x<y`` while still recognizing ``text<p ...>`` without whitespace.
    The pending candidate is capped, and comments retain only a three-character
    tail, so an unclosed or adversarial tag cannot grow memory without bound.
    """

    def __init__(self) -> None:
        self.in_tag = False
        self.in_comment = False
        self.quote: str | None = None
        self.comment_tail = ""
        self.pending = ""
        self.pending_boundary = True
        self.last_visible = ""
        self.pending_tag_state: str | None = None
        self.pending_tag_name = ""
        self.pending_attr_name = ""
        self.pending_has_assignment = False
        self.confirmed_custom_tags: list[str] = []
        self.deferred_custom_tag_name = ""
        self.in_tag_last_nonspace = ""

    def _emit(self, out: list[str], text: str) -> None:
        if not text:
            return
        out.append(text)
        self.last_visible = text[-1]

    def _clear_pending_tag_state(self, *, preserve_quote: bool = False) -> None:
        self.pending_tag_state = None
        self.pending_tag_name = ""
        self.pending_attr_name = ""
        self.pending_has_assignment = False
        if not preserve_quote:
            self.quote = None

    def _advance_pending_open_tag(self, char: str) -> str:
        """Advance a conservative opening-tag grammar.

        Returns ``pending``, ``complete``, or ``invalid``. Boolean attributes
        are accepted only from the HTML allowlist; that lets ``<input
        disabled>`` stream correctly without mistaking prose such as ``x<y and
        z >5`` for markup.
        """
        state = self.pending_tag_state
        if state == "name":
            if _is_markup_name_char(char):
                self.pending_tag_name += char
                return "pending"
            if char.isspace():
                self.pending_tag_state = "before_attr"
                return "pending"
            if char == ">":
                return "complete"
            if char == "/":
                self.pending_tag_state = "self_closing"
                return "pending"
            return "invalid"

        if state == "before_attr":
            if char.isspace():
                return "pending"
            if char == ">":
                return "complete"
            if char == "/":
                self.pending_tag_state = "self_closing"
                return "pending"
            if _is_markup_attribute_start(char):
                self.pending_tag_state = "attr_name"
                self.pending_attr_name = char
                return "pending"
            return "invalid"

        if state == "attr_name":
            if _is_markup_name_char(char):
                self.pending_attr_name += char
                return "pending"
            if char == "=":
                self.pending_has_assignment = True
                self.pending_tag_state = "before_value"
                return "pending"
            if char.isspace():
                self.pending_tag_state = "after_attr_name"
                return "pending"
            if char in {">", "/"} and (
                self.pending_attr_name.lower() in _HTML_BOOLEAN_ATTRIBUTES
            ):
                if char == ">":
                    return "complete"
                self.pending_tag_state = "self_closing"
                return "pending"
            return "invalid"

        if state == "after_attr_name":
            if char.isspace():
                return "pending"
            if char == "=":
                self.pending_has_assignment = True
                self.pending_tag_state = "before_value"
                return "pending"
            is_boolean = self.pending_attr_name.lower() in _HTML_BOOLEAN_ATTRIBUTES
            if char == ">" and is_boolean:
                return "complete"
            if char == "/" and is_boolean:
                self.pending_tag_state = "self_closing"
                return "pending"
            if _is_markup_attribute_start(char) and is_boolean:
                self.pending_tag_state = "attr_name"
                self.pending_attr_name = char
                return "pending"
            return "invalid"

        if state == "before_value":
            if char.isspace():
                return "pending"
            if char in {"'", '"'}:
                self.quote = char
                self.pending_tag_state = "quoted_value"
                return "pending"
            if char not in "\"'=<>`" and not char.isspace():
                self.pending_tag_state = "unquoted_value"
                return "pending"
            return "invalid"

        if state == "quoted_value":
            if char == self.quote:
                self.quote = None
                self.pending_tag_state = "after_value"
            return "pending"

        if state == "unquoted_value":
            if char == ">":
                return "complete"
            if char.isspace():
                self.pending_tag_state = "before_attr"
                return "pending"
            if char in "\"'=<>`":
                return "invalid"
            return "pending"

        if state == "after_value":
            if char.isspace():
                self.pending_tag_state = "before_attr"
                return "pending"
            if char == ">":
                return "complete"
            if char == "/":
                self.pending_tag_state = "self_closing"
                return "pending"
            return "invalid"

        if state == "self_closing":
            if char.isspace():
                return "pending"
            return "complete" if char == ">" else "invalid"

        return "invalid"

    def _advance_pending_closing_tag(self, char: str) -> str:
        state = self.pending_tag_state
        if state == "closing_start":
            if char.isascii() and char.isalpha():
                self.pending_tag_state = "closing_name"
                self.pending_tag_name = char
                return "pending"
            return "invalid"
        if state == "closing_name":
            if _is_markup_name_char(char):
                self.pending_tag_name += char
                return "pending"
            if char.isspace():
                self.pending_tag_state = "closing_after_name"
                return "pending"
            return "complete" if char == ">" else "invalid"
        if state == "closing_after_name":
            if char.isspace():
                return "pending"
            return "complete" if char == ">" else "invalid"
        return "invalid"

    def _confirmed_open_tag(self) -> bool:
        return (
            self.pending_tag_name.lower() in _HTML_TAG_NAMES
            or self.pending_has_assignment
        )

    def _confirmed_closing_tag(self) -> bool:
        name = self.pending_tag_name.lower()
        return name in _HTML_TAG_NAMES or name in self.confirmed_custom_tags

    def _remember_custom_open_tag(self, candidate: str) -> None:
        name = self.pending_tag_name.lower()
        if (
            name
            and name not in _HTML_TAG_NAMES
            and self.pending_has_assignment
            and not candidate.rstrip().endswith("/>")
        ):
            self._remember_custom_tag_name(name)

    def _remember_custom_tag_name(self, name: str) -> None:
        self.confirmed_custom_tags.append(name)
        del self.confirmed_custom_tags[:-32]

    def _consume_custom_closing_tag(self, name: str) -> None:
        if name in _HTML_TAG_NAMES:
            return
        for index in range(len(self.confirmed_custom_tags) - 1, -1, -1):
            if self.confirmed_custom_tags[index] == name:
                del self.confirmed_custom_tags[index]
                return

    def push(self, text: str) -> str:
        out: list[str] = []
        for char in text:
            if self.in_comment:
                self.comment_tail = (self.comment_tail + char)[-3:]
                if self.comment_tail == "-->":
                    self.in_comment = False
                    self.comment_tail = ""
                continue

            if self.in_tag:
                if self.quote is not None:
                    if char == self.quote:
                        self.quote = None
                elif char in {"'", '"'}:
                    self.quote = char
                elif char == ">":
                    self.in_tag = False
                    if (
                        self.deferred_custom_tag_name
                        and self.in_tag_last_nonspace != "/"
                    ):
                        self._remember_custom_tag_name(
                            self.deferred_custom_tag_name
                        )
                    self.deferred_custom_tag_name = ""
                    self.in_tag_last_nonspace = ""
                elif not char.isspace():
                    self.in_tag_last_nonspace = char
                continue

            if self.pending:
                self.pending += char
                if "<!--".startswith(self.pending):
                    if self.pending == "<!--":
                        self.pending = ""
                        self.in_comment = True
                        self.comment_tail = ""
                    continue

                opener = self.pending[1:2]
                if opener in "?!":
                    self.pending = ""
                    self._clear_pending_tag_state()
                    self.in_tag = True
                    self.deferred_custom_tag_name = ""
                    self.in_tag_last_nonspace = ""
                    continue

                if opener == "/":
                    if len(self.pending) == 2:
                        self.pending_tag_state = "closing_start"
                        continue
                    verdict = self._advance_pending_closing_tag(char)
                    if verdict == "complete":
                        candidate = self.pending
                        closing_name = self.pending_tag_name.lower()
                        confirmed = self._confirmed_closing_tag()
                        self.pending = ""
                        self._clear_pending_tag_state()
                        if confirmed:
                            self._consume_custom_closing_tag(closing_name)
                        else:
                            self._emit(out, candidate)
                        continue
                    if verdict == "invalid":
                        candidate = self.pending
                        self.pending = ""
                        self._clear_pending_tag_state()
                        self._emit(out, candidate)
                        continue
                    if len(self.pending) > _MAX_PENDING_MARKUP_CHARS:
                        candidate = self.pending
                        self.pending = ""
                        self._clear_pending_tag_state()
                        self._emit(out, candidate)
                    continue

                if opener.isalpha():
                    if len(self.pending) == 2:
                        self.pending_tag_state = "name"
                        self.pending_tag_name = opener
                        continue
                    verdict = self._advance_pending_open_tag(char)
                    if verdict == "complete":
                        candidate = self.pending
                        confirmed = self._confirmed_open_tag()
                        is_markup = (
                            confirmed
                            and _HTML_LIKE_OPEN_TAG_RE.fullmatch(candidate) is not None
                        )
                        if is_markup:
                            self._remember_custom_open_tag(candidate)
                        self.pending = ""
                        self._clear_pending_tag_state()
                        if not is_markup:
                            self._emit(out, candidate)
                        continue
                    if verdict == "invalid":
                        candidate = self.pending
                        self.pending = ""
                        self._clear_pending_tag_state()
                        self._emit(out, candidate)
                        continue
                    if len(self.pending) > _MAX_PENDING_MARKUP_CHARS:
                        # A multi-kilobyte alpha-led candidate is safer to drop
                        # from speech than to release as raw markup. Continue in
                        # bounded tag state until its closing delimiter, while
                        # preserving the quote phase already reached by the
                        # threshold-crossing character.
                        if self._confirmed_open_tag():
                            name = self.pending_tag_name.lower()
                            is_custom = name not in _HTML_TAG_NAMES
                            was_self_closing = (
                                self.pending_tag_state == "self_closing"
                            )
                            self.pending = ""
                            self.in_tag = True
                            self._clear_pending_tag_state(preserve_quote=True)
                            self.deferred_custom_tag_name = (
                                name if is_custom else ""
                            )
                            self.in_tag_last_nonspace = (
                                "/" if was_self_closing else ""
                            )
                        else:
                            candidate = self.pending
                            self.pending = ""
                            self._clear_pending_tag_state()
                            self._emit(out, candidate)
                        continue
                    continue

                pending = self.pending
                self.pending = ""
                self._clear_pending_tag_state()
                self._emit(out, pending)
                continue

            if char == "<":
                self.pending = "<"
                self.pending_boundary = not (
                    self.last_visible
                    and (
                        self.last_visible.isalnum()
                        or self.last_visible in "_."
                    )
                )
                continue
            self._emit(out, char)
        return "".join(out)

    def flush(self) -> str:
        out = ""
        if self.pending and not (
            self.pending_boundary and self.pending in {"<!", "<!-"}
        ):
            out = self.pending
            self.last_visible = out[-1]
        self.pending = ""
        self._clear_pending_tag_state()
        self.in_tag = False
        self.in_comment = False
        self.quote = None
        self.comment_tail = ""
        self.deferred_custom_tag_name = ""
        self.in_tag_last_nonspace = ""
        return out


def _strip_markup_for_speech(text: str) -> str:
    markup = _StreamingMarkupFilter()
    return markup.push(text) + markup.flush()


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
        self.in_url = False
        self.max_buffer_chars = max_buffer_chars
        self._markup_filter = _StreamingMarkupFilter()

    def _sanitize_stream_chunk(self, text: str) -> str:
        return sanitize_for_speech(text) if text else ""

    # ------- public API -----------------------------------------------------

    def push(self, delta: str) -> str:
        """Add a text delta. Returns whatever speech-safe text is now
        ready to be spoken (may be empty if the buffer still holds an
        open markdown span or code block)."""
        if not delta:
            return ""
        # Remove hidden markup before looking for sentence and hard-flush
        # boundaries.  Attribute values and comments may contain periods or be
        # much longer than the speech buffer; letting those raw bytes reach the
        # chunker can split the markup state and even split the first visible
        # word after a long tag.
        outside_markup = self._markup_filter.push(delta)
        if not outside_markup:
            return ""
        self.buffer += outside_markup
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
        markup_tail = self._markup_filter.flush()
        if markup_tail:
            self.buffer += markup_tail
        if self.in_code_block or self.in_url:
            # Drop unterminated protected content; the bubble still shows it.
            self.buffer = ""
            self.in_code_block = False
            self.in_url = False
            return ""
        if not self.buffer:
            return ""
        out = self._sanitize_stream_chunk(self.buffer)
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

        if self.in_url:
            terminator = re.search(r"\s", self.buffer)
            if terminator is None:
                # The URL replacement was already emitted. Discard its
                # continuation immediately so URL length cannot grow memory.
                self.buffer = ""
                return None
            self.buffer = self.buffer[terminator.start():]
            self.in_url = False
            return ""

        if self.in_code_block:
            close_idx = self.buffer.find("```")
            if close_idx < 0:
                # Discard code immediately, retaining only a possible closing
                # fence prefix so content length cannot grow memory.
                if self.buffer.endswith("``"):
                    self.buffer = "``"
                elif self.buffer.endswith("`"):
                    self.buffer = "`"
                else:
                    self.buffer = ""
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
            return self._sanitize_stream_chunk(pre) if pre else ""

        # Look for the earliest safe boundary in the buffer. A URL that runs
        # through the end of the current delta takes lexical precedence over
        # punctuation inside that URL. Ordinary sentence boundaries before
        # the URL are still released first.
        boundary = _find_safe_boundary(self.buffer)
        url_match = _URL_AT_BUFFER_END_RE.search(self.buffer)
        if url_match is not None and (boundary < 0 or boundary >= url_match.start()):
            chunk = self.buffer
            self.buffer = ""
            self.in_url = True
            return self._sanitize_stream_chunk(chunk)

        pronunciation_prefix_start = _find_trailing_pronunciation_prefix_start(
            self.buffer
        )
        if (
            boundary >= 0
            and pronunciation_prefix_start is not None
            and boundary >= pronunciation_prefix_start
            and not _ends_with_complete_pronunciation_token(
                self.buffer[:boundary]
            )
        ):
            # A period can be part of a model/version token (``Llama 3.1`` or
            # ``v1.2``). Wait for one more character instead of releasing a
            # permanently unnormalizable prefix.
            boundary = -1

        if boundary < 0:
            if len(self.buffer) >= self.max_buffer_chars:
                # Hard flush without splitting a possible spoken alias,
                # URL scheme, or fenced-code delimiter.
                cutoff = _find_hard_flush_cutoff(self.buffer)
                if cutoff == 0:
                    return None
                chunk = self.buffer[:cutoff]
                self.buffer = self.buffer[cutoff:]
                return self._sanitize_stream_chunk(chunk)
            return None  # wait for more text

        chunk = self.buffer[: boundary + 1]
        self.buffer = self.buffer[boundary + 1:]
        return self._sanitize_stream_chunk(chunk)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


_SAFE_BOUNDARY_RE = re.compile(r"[.!?](?=\s|$)|\n+")


def _ends_with_complete_pronunciation_token(text: str) -> bool:
    """Return true when sentence punctuation follows a complete known token."""
    for alias in _SPOKEN_ALIASES:
        if not text.endswith(alias):
            continue
        start = len(text) - len(alias)
        if start == 0 or not (text[start - 1].isalnum() or text[start - 1] == "_"):
            return True
    semver_match = _SEMVER_RE.search(text)
    if (
        semver_match is not None
        and semver_match.end() == len(text)
        and semver_match.group(1).count(".") == 2
    ):
        # Three components are a complete semantic version. A two-component
        # form remains held because it can still become ``v1.2.3`` on the next
        # stream delta.
        return True
    return False


def _find_trailing_pronunciation_prefix_start(text: str) -> int | None:
    """Return the start of a trailing approved-token prefix, if present."""
    max_candidate_chars = max(max(map(len, _SPOKEN_ALIASES)), 32)
    for suffix_chars in range(min(len(text), max_candidate_chars), 0, -1):
        suffix_start = len(text) - suffix_chars
        suffix = text[suffix_start:]
        if not (
            any(alias.startswith(suffix) for alias in _SPOKEN_ALIASES)
            or _STREAM_DYNAMIC_PREFIX_RE.fullmatch(suffix) is not None
            or _STREAM_NUMERIC_COMPARISON_PREFIX_RE.fullmatch(suffix) is not None
        ):
            continue
        if suffix_start == 0:
            return 0
        previous = text[suffix_start - 1]
        if not (previous.isalnum() or previous == "_"):
            return suffix_start
    return None


def _find_hard_flush_cutoff(text: str) -> int:
    """Return a cutoff retaining a trailing alias or protected-span prefix.

    A complete alias is retained too: without the next character, ``GPU``
    could still become plural ``GPUs`` or ordinary embedded text ``GPU2``.
    """
    max_candidate_chars = max(
        max(len(alias) for alias in _SPOKEN_ALIASES),
        max(len(prefix) for prefix in _PROTECTED_STREAM_PREFIXES),
        32,
    )
    for suffix_chars in range(min(len(text), max_candidate_chars), 0, -1):
        suffix_start = len(text) - suffix_chars
        suffix = text[suffix_start:]
        if suffix in _PROTECTED_STREAM_PREFIXES:
            return suffix_start
        if (
            any(alias.startswith(suffix) for alias in _SPOKEN_ALIASES)
            or _STREAM_DYNAMIC_PREFIX_RE.fullmatch(suffix) is not None
            or _STREAM_NUMERIC_COMPARISON_PREFIX_RE.fullmatch(suffix) is not None
        ):
            if suffix_start == 0:
                return suffix_start
            previous = text[suffix_start - 1]
            if not (previous.isalnum() or previous == "_"):
                return suffix_start
    return len(text)


def _find_safe_boundary(text: str) -> int:
    """Return the index (inclusive) of the first safe boundary or -1.

    A safe boundary is end-of-sentence punctuation followed by
    whitespace/end, OR a newline. We don't release on commas /
    semicolons because TTS handles those fine inline and we get
    better prosody by keeping clauses together.
    """
    m = _SAFE_BOUNDARY_RE.search(text)
    return m.start() if m else -1


def _whitespace_span(text: str, start: int, end: int) -> str:
    """Return the enclosing whitespace-delimited token for a match."""
    left = start
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    right = end
    while right < len(text) and not text[right].isspace():
        right += 1
    return text[left:right]


def _alias_is_in_sensitive_context(
    text: str,
    start: int,
    end: int,
    alias: str,
) -> bool:
    """Reject path, credential, secret, and decorated-model contexts.

    Spoken aliases are narration helpers, not identifier rewriters.  Looking at
    the small enclosing token keeps paths and secret-shaped values byte-stable
    without adding a general parser or an external dependency.
    """
    token = _whitespace_span(text, start, end)
    token_lower = token.lower()
    if any(delimiter in token for delimiter in ("\\", "/", "@")):
        return True
    if _SENSITIVE_LABEL_RE.search(token) is not None:
        return True
    if token_lower.startswith(("sk-", "pk-", "ghp_", "github_pat_")):
        return True
    if alias in _MODEL_ALIAS_KEYS:
        if start > 0 and text[start - 1] in "-_:":
            return True
        if end < len(text) and text[end] in "-_:":
            return True
    return False


def _replace_spoken_alias(match: re.Match[str]) -> str:
    alias = match.group(0)
    if _alias_is_in_sensitive_context(
        match.string,
        match.start(),
        match.end(),
        alias,
    ):
        return alias
    return _SPOKEN_ALIASES[alias]


def _replace_numeric_comparison(match: re.Match[str]) -> str:
    """Verbalize a numeric comparator after markup has been removed."""
    return (
        f"{_SPOKEN_NUMERIC_COMPARISONS[match.group('operator')]} "
        f"{match.group('value')}"
    )


def _normalize_pronunciation_tokens(text: str) -> str:
    """Apply only compiled, deterministic, context-bounded speech aliases."""
    text = _SPOKEN_ALIAS_RE.sub(_replace_spoken_alias, text)
    text = _SEMVER_RE.sub(
        lambda match: "version " + " point ".join(match.group(1).split(".")),
        text,
    )
    text = _TEMPERATURE_RE.sub(
        lambda match: (
            f"{match.group(1)} degrees "
            f"{'Celsius' if match.group(2) == 'C' else 'Fahrenheit'}"
        ),
        text,
    )
    text = _PERCENT_RE.sub(r"\1 percent", text)
    text = _NUMERIC_EQUALS_RE.sub(r"\1 equals \2", text)
    text = _NUMERIC_TIMES_RE.sub(r"\1 times \2", text)
    # Angle-bracket comparisons are unsafe as literal TTS input: at least one
    # live backend treated ``<4 seconds`` like markup, stopped speaking before
    # the constraint, and returned a long silent tail.  Verbalize only bounded
    # numeric comparisons so the synthesizer and duration gate receive the same
    # canonical, speakable text.  Markup tags and comparison-like identifiers
    # remain untouched because the right-hand side must start with a number.
    text = _NUMERIC_COMPARISON_RE.sub(_replace_numeric_comparison, text)
    return text


def _normalize_markdown_layout(text: str) -> str:
    """Turn headings and pipe-table fragments into speech-safe prose.

    The sentence chunker can split one Markdown separator row across adjacent
    TTS requests (for example, one chunk ending in ``|-----`` and the next
    starting ``-|-----|``).  Treat each pipe-delimited fragment independently:
    discard empty/alignment-only cells and keep semantic cells as short spoken
    sentences.  This also removes ATX markers after streaming has collapsed the
    original newline, while leaving identifiers such as ``C#`` and ``#tag``
    untouched.
    """
    text = _MD_ATX_HEADING_RE.sub("", text)
    if "|" not in text:
        return text

    spoken_cells: list[str] = []
    raw_cells = text.split("|")
    for index, raw_cell in enumerate(raw_cells):
        cell = raw_cell.strip()
        if not cell or _MD_TABLE_SEPARATOR_CELL_RE.fullmatch(cell):
            continue
        # A following pipe proves the table cell is complete.  The final
        # fragment may instead be a sentence-chunker split in the middle of a
        # cell (``4B Face`` / ``Lobe |``); do not inject a false full stop there.
        cell_is_closed = index < len(raw_cells) - 1
        if not cell_is_closed or cell.endswith((".", "!", "?", ":", ";")):
            spoken_cells.append(cell)
        else:
            spoken_cells.append(cell + ".")
    return " ".join(spoken_cells)


def sanitize_for_speech(text: str) -> str:
    """One-shot sanitizer: take a chunk of text, return TTS-safe text.

    Stateless; for streaming use ``SpokenTextFilter``. Exposed
    publicly so the REST engine path (which already chunks by
    sentence) can call it directly on each sentence.
    """
    if not text:
        return ""

    # Raw markup remains visible in the text transcript but is not useful or
    # safe speech.  Strip tags/comments before numeric-operator verbalization;
    # the stateful streaming path applies the same scanner across deltas.
    text = _strip_markup_for_speech(text)

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

    # --- backend-neutral spoken aliases --------------------------------------
    # Run after markdown/link and opaque-value cleanup so aliases inside URL
    # targets stay protected. The case-sensitive token boundaries leave
    # lowercase, mixed-case, and embedded ordinary text untouched.
    text = _normalize_pronunciation_tokens(text)

    # --- headings and pipe tables --------------------------------------------
    # Run after protected-value and pronunciation normalization so code names,
    # URLs, and C# retain their established spoken forms.  The returned text is
    # the exact text sent both to TTS and to the duration gate, keeping the gate's
    # unchanged word-based bound aligned with what the synthesizer is asked to say.
    text = _normalize_markdown_layout(text)

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
