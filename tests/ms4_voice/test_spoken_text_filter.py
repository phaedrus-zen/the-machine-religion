"""Tests for the streaming TTS sanitizer.

The contract this suite locks in:

* Inline markdown (bold / italic / code / strikethrough / links /
  underscore-style) is unwrapped to its visible text content.
* Fenced code blocks are dropped entirely — including their content.
  The reasoning is that models routinely dump ASCII tables, JSON, or
  log lines in code blocks and nobody wants those read byte-by-byte.
* URLs, UUIDs, and job-ids (``da-<hex>...``) are replaced with short
  spoken placeholders so the audio doesn't degenerate into hex
  recitation.
* Bullet markers / numbered-list markers at line starts are dropped.
* Decorative glyphs (``║``, ``→``, ``•``, box-drawing chars, emoji
  blocks) are stripped.
* Non-Latin scripts (CJK, Arabic, Cyrillic, Hebrew, Greek) are
  preserved unchanged so multilingual TTS still works.
* The streaming wrapper handles markdown patterns that span delta
  boundaries (e.g. ``**bo`` then ``ld**`` should still strip the
  ``**`` and speak ``bold``).
* ``flush()`` drains the buffer at end-of-stream so the trailing
  fragment is spoken.
"""

from __future__ import annotations

import pytest

from machine_spirit_4.gateway.spoken_text_filter import (
    SpokenTextFilter,
    sanitize_for_speech,
    split_into_speech_chunks,
)


# ---------------------------------------------------------------------------
# Stateless sanitize_for_speech
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected_in,expected_not_in", [
    # ---- markdown structural --------------------------------------------
    ("**HiveMind** cluster", ["HiveMind cluster"], ["**"]),
    ("_italic_ word",        ["italic word"],     ["_"]),
    ("*one* and *two*",      ["one and two"],     ["*"]),
    ("a `code` value",       ["a code value"],    ["`"]),
    ("~~strike~~ word",      ["strike word"],     ["~~"]),
    ("[click](https://x.io) here", ["click here"], ["[", "](", "x.io", "https"]),
    ("__bold under__ text",  ["bold under text"], ["__"]),

    # ---- URLs / identifiers --------------------------------------------
    ("see https://example.com for more",     ["a link"],              ["https://", "example.com"]),
    ("job da-abc1234def5 completed",          ["a background job id"], ["da-abc1234def5"]),
    ("uuid 12345678-1234-1234-1234-123456789abc was used",
                                              ["an identifier"],       ["12345678-1234"]),
    ("hex deadbeefcafebabe1234deadbeef",      ["a hex value"],         ["deadbeefcafebabe"]),

    # ---- list markers ---------------------------------------------------
    ("- first item\n- second item",          ["first item"],          ["- ", "* "]),
    ("1. one\n2. two\n3. three",             ["one"],                 ["1.", "2.", "3."]),

    # ---- decorative glyphs ---------------------------------------------
    ("\u2551 MS4 Fusion Runtime",             ["MS4 Fusion Runtime"],  ["\u2551"]),
    ("Step A \u2192 Step B",                  ["Step A to Step B"],    ["\u2192"]),
    ("\u2022 bullet item",                    ["bullet item"],         ["\u2022"]),
    ("\u25cf full circle",                    ["full circle"],         ["\u25cf"]),
    ("\u26a0 warning",                        ["warning"],             ["\u26a0"]),

    # ---- emoji ---------------------------------------------------------
    ("hello \U0001f600 there",                ["hello", "there"],      ["\U0001f600"]),
    ("\U0001f680 launch \U0001f389",          ["launch"],              ["\U0001f680", "\U0001f389"]),
    ("audio \U0001f3a4 chat",                 ["audio", "chat"],       ["\U0001f3a4"]),

    # ---- whitespace cleanup --------------------------------------------
    ("multiple    spaces",                    ["multiple spaces"],     ["    "]),
    ("line one\n\n\nline two",                ["line one", "line two"],[]),
])
def test_sanitize_handles_each_pattern(raw, expected_in, expected_not_in):
    out = sanitize_for_speech(raw)
    for needle in expected_in:
        assert needle in out, f"expected {needle!r} in sanitized output {out!r} (input: {raw!r})"
    for forbidden in expected_not_in:
        assert forbidden not in out, f"unexpected {forbidden!r} in sanitized output {out!r} (input: {raw!r})"


# ---------------------------------------------------------------------------
# Code blocks
# ---------------------------------------------------------------------------


def test_streaming_filter_drops_fenced_code_block_entirely():
    flt = SpokenTextFilter()
    out = flt.push("Here's the code:\n```python\nfor i in range(10):\n    print(i)\n```\nThat prints numbers.")
    tail = flt.flush()
    full = (out + tail).strip()
    assert "Here's the code" in full
    assert "That prints numbers" in full
    # The code block content must not be present in the spoken stream.
    assert "for i in range" not in full
    assert "```" not in full
    assert "print(i)" not in full


def test_streaming_filter_drops_unclosed_code_block_at_flush():
    flt = SpokenTextFilter()
    out = flt.push("Preamble.\n```python\nfor i in range(10):")
    tail = flt.flush()
    full = (out + tail).strip()
    assert "Preamble" in full
    assert "range" not in full  # unclosed block discarded


# ---------------------------------------------------------------------------
# Streaming state — multi-delta patterns
# ---------------------------------------------------------------------------


def test_streaming_filter_handles_bold_spanning_two_deltas():
    flt = SpokenTextFilter()
    a = flt.push("Hello, **wor")
    b = flt.push("ld** of MS4. ")
    tail = flt.flush()
    full = (a + b + tail).strip()
    assert "Hello, world of MS4." in full
    assert "**" not in full


def test_streaming_filter_releases_at_sentence_boundary():
    """The filter MUST hold text until a sentence-ending punctuation,
    so the SentenceChunker / WS push gets coherent prosody units."""
    flt = SpokenTextFilter()
    out1 = flt.push("First sentence. Second")
    # First sentence has been released, "Second" is still buffered.
    assert "First sentence." in out1
    assert "Second" not in out1
    out2 = flt.push(" sentence here. ")
    assert "Second sentence here." in out2


def test_streaming_filter_force_flushes_runon_at_max_buffer():
    """If a single delta exceeds max_buffer_chars without a boundary
    the filter MUST force-flush instead of starving TTS."""
    flt = SpokenTextFilter(max_buffer_chars=50)
    long = "word " * 30  # ~150 chars no sentence terminator
    out = flt.push(long)
    assert out.strip(), "force-flush should have produced output"


# ---------------------------------------------------------------------------
# Non-Latin pass-through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,must_appear", [
    ("你好,世界. Hello.",              "你好"),
    ("Привет, мир. Hello.",          "Привет"),
    ("مرحبا بالعالم. Hello.",        "مرحبا"),
    ("Καλημέρα, κόσμε. Hello.",      "Καλημέρα"),
    ("שלום עולם. Hello.",            "שלום"),
])
def test_non_latin_scripts_pass_through(text, must_appear):
    out = sanitize_for_speech(text)
    assert must_appear in out, f"expected {must_appear!r} to survive sanitization in {out!r}"


# ---------------------------------------------------------------------------
# Real-world reply from the live transcript
# ---------------------------------------------------------------------------


def test_realistic_face_lobe_reply_becomes_speech_safe():
    """A reduced copy of the kind of reply the Face Lobe actually
    emitted live. The TTS filter must produce something a human can
    listen to without hearing punctuation noise."""
    reply = (
        "\u2551 MS4 Fusion Runtime\n\n"
        "The HiveMind cluster is **fully active** with:\n\n"
        "- **9 total nodes**, all 9 active  \n"
        "- **17 GPU devices** across the cluster\n\n"
        "Nodes:\n"
        "- `DESKTOP-OH048LC` (Intel Core Ultra 9, 2\u00d7 RTX PRO 6000 Blackwell)\n"
        "- `project-nexus-server-001` (RTX 4090 \u00d72, GTX 1080, Intel iGPU)\n\n"
        "Job da-c4d4949d-5a91-4c53-8d74-f2243da1d840 completed.\n"
        "More info at https://hivemind.local/docs."
    )
    chunks = list(split_into_speech_chunks(reply))
    spoken = " ".join(c.strip() for c in chunks).strip()
    # Decorative + markdown noise stripped.
    for forbidden in ("\u2551", "**", "`", "[", "https://", "da-c4d4949d"):
        assert forbidden not in spoken, f"unexpected {forbidden!r} in spoken output: {spoken!r}"
    # Real content preserved.
    for needed in ("HiveMind", "fully active", "9 total nodes", "DESKTOP-OH048LC", "a background job id"):
        assert needed in spoken, f"expected {needed!r} preserved in spoken output: {spoken!r}"


def test_empty_input_returns_empty():
    assert sanitize_for_speech("") == ""
    assert sanitize_for_speech("   \n\n  ") == ""
    flt = SpokenTextFilter()
    assert flt.push("") == ""
    assert flt.flush() == ""
