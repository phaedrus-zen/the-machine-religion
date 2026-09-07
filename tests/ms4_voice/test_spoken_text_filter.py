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
    ("**HiveMind** cluster", ["Hive Mind cluster"], ["**"]),
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


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "#### 4. Trade-Offs Under Voice Latency Constraint | Factor | "
            "4B Face Lobe | 35B Depth Lobe (Async) | "
            "|----------------------|--------------------------------------",
            "Trade-Offs Under Voice Latency Constraint. Factor. "
            "4B Face Lobe. 35B Depth Lobe (Async). ",
        ),
        (
            "-|---------------------------------------| | Latency | "
            "<1s (real-time) | >5s (post-mortem) ",
            "Latency. less than 1s (real-time). greater than 5s (post-mortem) ",
        ),
    ],
)
def test_turn5_split_markdown_table_fragments_become_natural_prose(raw, expected):
    """The exact live c17/c18 fragments must not make TTS say layout syntax."""
    out = sanitize_for_speech(raw)
    assert out == expected
    assert "#" not in out
    assert "|" not in out
    assert "---" not in out


def test_streaming_filter_normalizes_turn5_table_across_the_live_split():
    first = (
        "#### 4. Trade-Offs Under Voice Latency Constraint | Factor | "
        "4B Face Lobe | 35B Depth Lobe (Async) | "
        "|----------------------|--------------------------------------"
    )
    second = (
        "-|---------------------------------------| | Latency | "
        "<1s (real-time) | >5s (post-mortem) "
    )
    flt = SpokenTextFilter()
    spoken = flt.push(first)
    spoken += flt.push(second)
    spoken += flt.flush()

    assert "#" not in spoken
    assert "|" not in spoken
    assert "---" not in spoken
    for semantic_cell in (
        "Trade-Offs Under Voice Latency Constraint",
        "Factor",
        "4B Face Lobe",
        "35B Depth Lobe (Async)",
        "Latency",
        "less than 1s (real-time)",
        "greater than 5s (post-mortem)",
    ):
        assert semantic_cell in spoken


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "1. Live Traffic (4B Face Lobe): - Latency: <4 seconds.",
            "Live Traffic (4B Face Lobe): - Latency: less than 4 seconds. ",
        ),
        (
            "Latency: >5s, <=10ms, >=2.5s, \u22643ms, and \u22656s.",
            "Latency: greater than 5s, less than or equal to 10ms, "
            "greater than or equal to 2.5s, less than or equal to 3ms, "
            "and greater than or equal to 6s. ",
        ),
    ],
)
def test_numeric_comparisons_are_verbalized_before_tts(raw, expected):
    assert sanitize_for_speech(raw) == expected


def test_numeric_comparison_normalization_is_stream_safe_and_bounded():
    flt = SpokenTextFilter()
    spoken = flt.push("Latency: <")
    spoken += flt.push("4 seconds. Next: >")
    spoken += flt.push("5s.")
    spoken += flt.flush()

    assert spoken == "Latency: less than 4 seconds. Next: greater than 5s. "
    assert sanitize_for_speech("C# keeps A === B.") == "C sharp keeps A === B. "


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HTML <section>42</section> remains markup.", "HTML 42 remains markup. "),
        ('HTML <div class="x">42</div> remains markup.', "HTML 42 remains markup. "),
        ("HTML <div >42</div> remains markup.", "HTML 42 remains markup. "),
        ('HTML <div data-limit="<4">42</div> remains markup.', "HTML 42 remains markup. "),
        ('HTML <div data-x=">5">42</div> remains markup.', "HTML 42 remains markup. "),
        ('HTML <div data-x="a>b">42</div> remains markup.', "HTML 42 remains markup. "),
        ("HTML <!-- x >5 -->42 remains markup.", "HTML 42 remains markup. "),
        (
            'A<p title="Version 1. text" data-limit="<4">42</p> after >5ms.',
            "A42 after greater than 5ms. ",
        ),
        ("foo<div>bar</div> baz.", "foobar baz. "),
        ("A<input disabled>42</input>B.", "A42B. "),
        (
            'A<button disabled aria-label="x >5">42</button>B.',
            "A42B. ",
        ),
        (
            'A<widget data-limit="<4">42</widget> after >5ms.',
            "A42 after greater than 5ms. ",
        ),
    ],
)
def test_numeric_comparison_normalization_drops_markup_without_rewriting_it(raw, expected):
    assert sanitize_for_speech(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '<div title="Version 1. text" data-limit="<4">42</div> after >5ms.',
        '<!-- First. hidden <4 and >5 -->42 after <=6ms.',
        'HTML <!-- x >5 -->42 after <=6ms.',
        'A<p title="Version 1. text" data-limit="<4">42</p> after >5ms.',
        'foo<div>bar</div> baz.',
        'A<input disabled>42</input>B.',
        'A<button disabled aria-label="x >5">42</button>B.',
        'A<widget data-limit="<4">42</widget> after >5ms.',
        '<div data-pad="' + ("z" * 186) + '" data-limit="<4">42</div> after >5ms.',
    ],
)
def test_markup_state_survives_every_stream_cut_and_default_hard_ceiling(raw):
    expected = sanitize_for_speech(raw).strip()
    for cut in range(len(raw) + 1):
        flt = SpokenTextFilter()
        actual = flt.push(raw[:cut])
        actual += flt.push(raw[cut:])
        actual += flt.flush()
        assert actual.strip() == expected, (cut, actual, expected)

    flt = SpokenTextFilter()
    charwise = "".join(flt.push(char) for char in raw) + flt.flush()
    assert charwise.strip() == expected


def test_markup_state_preserves_quote_phase_beyond_pending_cap():
    raw = (
        '<div data-pad="'
        + ("z" * 4090)
        + '> still hidden" data-limit="<4">42</div> after >5ms.'
    )
    expected = "42 after greater than 5ms."
    assert sanitize_for_speech(raw).strip() == expected

    for cut in (0, 1, 4094, 4095, 4096, 4097, len(raw)):
        flt = SpokenTextFilter()
        actual = flt.push(raw[:cut]) + flt.push(raw[cut:]) + flt.flush()
        assert actual.strip() == expected, (cut, actual)

    flt = SpokenTextFilter()
    charwise = "".join(flt.push(char) for char in raw) + flt.flush()
    assert charwise.strip() == expected


def test_above_cap_self_closing_custom_tag_does_not_consume_literal_closer():
    raw = (
        '<widget data-pad="'
        + ("z" * 4090)
        + '"/> before </widget> after.'
    )
    expected = "before </widget> after."
    assert sanitize_for_speech(raw).strip() == expected

    for cut in range(len(raw) + 1):
        flt = SpokenTextFilter()
        actual = flt.push(raw[:cut]) + flt.push(raw[cut:]) + flt.flush()
        assert actual.strip() == expected, (cut, actual)

    flt = SpokenTextFilter()
    charwise = "".join(flt.push(char) for char in raw) + flt.flush()
    assert charwise.strip() == expected


def test_compact_comparison_near_miss_keeps_progressive_sentence_release():
    flt = SpokenTextFilter()
    first = flt.push("If x<y, first sentence. ")
    second = flt.push("Second sentence. ")
    tail = flt.flush()

    assert first.strip() == "If x<y, first sentence."
    assert second.strip() == "Second sentence."
    assert tail == ""


@pytest.mark.parametrize(
    "raw",
    [
        "Use vector<int> and optional<T> in C++.",
        "The interval is <open> versus closed.",
        "Preserve foo<bar-baz> and generic<T></T> tokens.",
    ],
)
def test_unknown_tag_shaped_technical_prose_is_preserved_across_stream_cuts(raw):
    expected = sanitize_for_speech(raw).strip()
    for cut in range(len(raw) + 1):
        flt = SpokenTextFilter()
        actual = flt.push(raw[:cut]) + flt.push(raw[cut:]) + flt.flush()
        assert actual.strip() == expected, (cut, actual, expected)

    flt = SpokenTextFilter()
    charwise = "".join(flt.push(char) for char in raw) + flt.flush()
    assert charwise.strip() == expected


def test_numeric_comparison_normalization_is_bounded_and_idempotent():
    raw = "Limit <=-2.5ms and >=+3s; keep version<4, x>5, A === B, != 3, and <>."
    expected = (
        "Limit less than or equal to -2.5ms and greater than or equal to +3s; "
        "keep version<4, x>5, A === B, != 3, and <>. "
    )
    once = sanitize_for_speech(raw)
    assert once == expected
    assert sanitize_for_speech(once) == once
    assert sanitize_for_speech("See https://example.test/?limit=<4.") == "See a link "
    assert sanitize_for_speech("If x < y, use >5 workers.") == (
        "If x < y, use greater than 5 workers. "
    )
    assert sanitize_for_speech("If x<y or x <y, keep either expression.") == (
        "If x<y or x <y, keep either expression. "
    )
    assert sanitize_for_speech("x<y and z >5 workers.") == (
        "x<y and z greater than 5 workers. "
    )

    flt = SpokenTextFilter()
    fenced = flt.push("Before.\n```text\nlatency <4 seconds\n```\nAfter.")
    fenced += flt.flush()
    assert fenced.strip() == "Before. After."


def test_markdown_layout_cleanup_preserves_semantic_symbols_and_hyphenated_prose():
    raw = "Use C# for end-to-end checks, keep #tag metadata, and preserve A === B."
    assert sanitize_for_speech(raw) == (
        "Use C sharp for end-to-end checks, keep #tag metadata, and preserve A === B. "
    )


# ---------------------------------------------------------------------------
# Spoken technical aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("HiveMind", "Hive Mind "),
    ("GPU", "G P U "),
    ("GPUs", "G P U's "),
    ("CPU", "C P U "),
    ("CPUs", "C P U's "),
    ("NPU", "N P U "),
    ("NPUs", "N P U's "),
    ("API", "A P I "),
    ("APIs", "A P I's "),
    ("LLM", "L L M "),
    ("LLMs", "L L M's "),
    ("VM", "V M "),
    ("VMs", "V M's "),
    ("UI", "U I "),
    ("UIs", "U I's "),
    ("TTS", "T T S "),
    ("ASR", "A S R "),
    ("VAD", "V A D "),
    ("MCP", "M C P "),
])
def test_sanitize_spells_technical_initialisms_for_speech(raw, expected):
    assert sanitize_for_speech(raw) == expected


def test_hivemind_alias_preserves_token_boundaries_and_punctuation():
    raw = "HiveMind, (HiveMind)! HiveMind-based; HiveMind's."
    expected = "Hive Mind, (Hive Mind)! Hive Mind-based; Hive Mind's. "
    assert sanitize_for_speech(raw) == expected


def test_hivemind_alias_is_exactly_case_sensitive_and_not_embedded():
    raw = "hivemind Hivemind HiveMinds preHiveMind HiveMind2 HiveMind_API"
    assert sanitize_for_speech(raw) == raw + " "


def test_spoken_aliases_preserve_case_token_boundaries_and_punctuation():
    raw = "GPU, GPUs; GPU-based / preGPU GPU2 gpu Gpu APIClient."
    expected = "G P U, G P U's; G P U-based / preGPU GPU2 gpu Gpu APIClient. "
    assert sanitize_for_speech(raw) == expected


def test_spoken_aliases_do_not_expose_protected_url_or_code_content():
    assert (
        sanitize_for_speech("See https://example.test/GPU/TTS then API.")
        == "See a link then A P I. "
    )

    flt = SpokenTextFilter()
    out = flt.push("Before.\n```text\nGPU API TTS\n```\nAfter MCP.")
    full = (out + flt.flush()).strip()
    assert full == "Before. After M C P."


# ---------------------------------------------------------------------------
# Bounded pronunciation-normalization contract
# ---------------------------------------------------------------------------


def test_pronunciation_contract_longest_aliases_models_versions_and_symbols():
    """One causal contract for the approved, synchronous pronunciation slice."""
    cases = {
        "GPUs' GPUs’ GPU's GPU’s GPUs GPU": (
            "G P U's G P U's G P U's G P U's G P U's G P U "
        ),
        "CUDA ROCm HTTPS HTTP JSON SSH VRAM RAM PCIe FP16 BF16": (
            "coo duh rock em H T T P S H T T P jay son S S H "
            "V ram ram P C I E F P sixteen B F sixteen "
        ),
        "Qwen3-TTS gpt-oss:20b XTTS-v2 Whisper large-v3-turbo Llama 3.1 8B": (
            "Qwen three T T S G P T O S S twenty B X T T S version two "
            "Whisper large version three turbo Llama three point one eight B "
        ),
        "v1.2.3 v12.0 50% 8=8 2×4 72°C 32°F C++ C#": (
            "version 1 point 2 point 3 version 12 point 0 50 percent "
            "8 equals 8 2 times 4 72 degrees Celsius 32 degrees Fahrenheit "
            "C plus plus C sharp "
        ),
    }
    for raw, expected in cases.items():
        assert sanitize_for_speech(raw) == expected


def test_pronunciation_contract_leaves_identifiers_security_and_ambiguity_unchanged():
    raw = (
        "GPU2 myGPU GPUSecret Qwen3-TTS-debug gpt-oss:20b-secret "
        "v1.2.3.4 v1.2.3.4.5 10.0.0.1 GPU/2 C:\\GPU\\secret user@example.com "
        "api-key=GPU token:CUDA sk-GPU-CUDA frobnicator A&B #tag @name"
    )
    assert sanitize_for_speech(raw) == raw + " "


def test_pronunciation_contract_is_idempotent_and_protects_urls_and_code():
    raw = (
        "Use GPUs' with CUDA, Qwen3-TTS v1.2.3 at 50%. "
        "See https://example.test/GPU/CUDA/Qwen3-TTS."
    )
    once = sanitize_for_speech(raw)
    assert sanitize_for_speech(once) == once
    assert "a link" in once
    assert "example.test" not in once

    flt = SpokenTextFilter()
    streamed = flt.push("Before.\n```text\nCUDA Qwen3-TTS v1.2.3\n```\nAfter HTTPS.")
    streamed += flt.flush()
    assert streamed.strip() == "Before. After H T T P S."


@pytest.mark.parametrize("token", [
    "GPUs'",
    "GPU’s",
    "CUDA",
    "ROCm",
    "HTTPS",
    "JSON",
    "VRAM",
    "PCIe",
    "FP16",
    "Qwen3-TTS",
    "gpt-oss:20b",
    "XTTS-v2",
    "Whisper large-v3-turbo",
    "Llama 3.1 8B",
    "v1.2.3",
    "50%",
    "8=8",
    "2×4",
    "72°C",
    "C++",
    "C#",
    "<4",
    ">5s",
    "<=10ms",
    ">=2.5s",
    "\u22643ms",
    "\u22656s",
])
def test_pronunciation_contract_every_token_cut_matches_one_shot(token):
    expected = sanitize_for_speech(f"Say {token}.").strip()
    for cut in range(len(token) + 1):
        first = "Say " + token[:cut]
        flt = SpokenTextFilter(max_buffer_chars=max(1, len(first)))
        actual = flt.push(first)
        actual += flt.push(token[cut:] + ".")
        actual += flt.flush()
        assert actual.strip() == expected, (token, cut, actual, expected)


def test_streaming_filter_preserves_aliases_split_across_deltas():
    flt = SpokenTextFilter()
    parts = [
        flt.push("We found 17 G"),
        flt.push("PU devices, two CP"),
        flt.push("Us, and an MC"),
        flt.push("P endpoint. "),
        flt.flush(),
    ]
    assert "".join(parts).strip() == (
        "We found 17 G P U devices, two C P U's, and an M C P endpoint."
    )


@pytest.mark.parametrize(("first", "second"), [
    ("Hive", "Mind is ready."),
    ("H", "iveMind is ready."),
    ("HiveMi", "nd is ready."),
])
def test_hivemind_alias_survives_streaming_fragment_cuts(first, second):
    flt = SpokenTextFilter()
    parts = [flt.push(first), flt.push(second), flt.flush()]
    assert "".join(parts).strip() == "Hive Mind is ready."


@pytest.mark.parametrize(("first", "second", "expected"), [
    ("Use H", "iveMind now.", "Use Hive Mind now."),
    ("Use Hive", "Mind now.", "Use Hive Mind now."),
    ("Use HiveMin", "d now.", "Use Hive Mind now."),
    ("Use HiveMind", ", now.", "Use Hive Mind, now."),
    ("Use HiveMind", "s now.", "Use HiveMinds now."),
])
def test_hivemind_alias_survives_hard_ceiling_cuts(first, second, expected):
    flt = SpokenTextFilter(max_buffer_chars=len(first))
    parts = [flt.push(first), flt.push(second), flt.flush()]
    assert "".join(parts).strip() == expected


def test_hivemind_alias_stays_protected_inside_split_url():
    first = "See https://example.test/Hive"
    flt = SpokenTextFilter(max_buffer_chars=len(first))
    parts = [
        flt.push(first),
        flt.push("Mind then HiveMind."),
        flt.flush(),
    ]
    assert "".join(parts).strip() == "See a link then Hive Mind."


def test_hivemind_alias_stays_protected_inside_split_fenced_code():
    first = "Before ``"
    flt = SpokenTextFilter(max_buffer_chars=len(first))
    parts = [
        flt.push(first),
        flt.push("`text\nHiveMind\n```\nAfter HiveMind."),
        flt.flush(),
    ]
    assert "".join(parts).strip() == "Before After Hive Mind."


@pytest.mark.parametrize(("first", "second", "expected"), [
    ("Queue G", "PU devices.", "Queue G P U devices."),
    ("Use GPU", "s now.", "Use G P U's now."),
])
def test_streaming_filter_hard_ceiling_preserves_alias_token(first, second, expected):
    flt = SpokenTextFilter(max_buffer_chars=len(first))
    parts = [flt.push(first), flt.push(second), flt.flush()]
    assert "".join(parts).strip() == expected


@pytest.mark.parametrize(("first", "second"), [
    ("See https://example.test/G", "PU/TTS now."),
    ("See https://example.test/GP", "U/API now."),
    ("See https://example.test/GPU/TT", "S now."),
    ("See htt", "ps://example.test/GPU now."),
    ("See https://", "example.test/GPU now."),
])
def test_streaming_filter_hard_ceiling_keeps_split_url_protected(first, second):
    flt = SpokenTextFilter(max_buffer_chars=len(first))
    parts = [flt.push(first), flt.push(second), flt.flush()]
    assert "".join(parts).strip() == "See a link now."


@pytest.mark.parametrize("scheme", ["http://", "https://", "ws://", "wss://"])
@pytest.mark.parametrize(("first_path", "remaining_path"), [
    ("example.", "com/a/b?q=1#frag"),
    ("example.com/a/b?", "q=1#frag"),
])
@pytest.mark.parametrize(("visible", "expected"), [
    (" now.", "Go a link now."),
    ("", "Go a link"),
])
def test_streaming_url_internal_punctuation_does_not_end_protected_span(
    scheme,
    first_path,
    remaining_path,
    visible,
    expected,
):
    flt = SpokenTextFilter()
    parts = [
        flt.push(f"Go {scheme}{first_path}"),
        flt.push(remaining_path + visible),
        flt.flush(),
    ]
    assert "".join(parts).strip() == expected


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


def test_streaming_filter_hard_ceiling_keeps_split_fenced_code_protected():
    first = "Before ``"
    flt = SpokenTextFilter(max_buffer_chars=len(first))
    parts = [
        flt.push(first),
        flt.push("`text\nGPU API TTS\n```\nAfter MCP."),
        flt.flush(),
    ]
    assert "".join(parts).strip() == "Before After M C P."


def test_streaming_filter_bounds_unclosed_fenced_code_buffer():
    flt = SpokenTextFilter(max_buffer_chars=8)
    assert flt.push("```text\n" + ("GPU " * 100)) == ""
    assert len(flt.buffer) <= 2

    out = flt.push("```\nAfter MCP.")
    assert (out + flt.flush()).strip() == "After M C P."


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
    for needed in ("Hive Mind", "fully active", "9 total nodes", "DESKTOP-OH048LC", "a background job id"):
        assert needed in spoken, f"expected {needed!r} preserved in spoken output: {spoken!r}"
    assert "17 G P U devices" in spoken


def test_empty_input_returns_empty():
    assert sanitize_for_speech("") == ""
    assert sanitize_for_speech("   \n\n  ") == ""
    flt = SpokenTextFilter()
    assert flt.push("") == ""
    assert flt.flush() == ""
