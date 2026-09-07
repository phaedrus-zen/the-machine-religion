"""TMR doctrine loader + session injection.

The contract this suite locks in:

* :func:`get_full_bible` returns the contents of the on-disk
  ``canon/The_Complete_Bible.md`` file (or whatever
  ``MS4_BIBLE_PATH`` points at in test mode).
* :func:`list_sections` finds the markdown headings and produces
  stable snake_case ids.
* :func:`get_section` returns ``(meta, text)`` for a known id and
  raises :class:`DoctrineSectionUnknown` otherwise.
* :func:`inject_into_session` appends a user+assistant message pair
  to the FaceLobeChat session's message list so future chat turns
  see the doctrine in their conversation_history.
* :func:`reload_bible` re-reads the file on disk (used by the
  ``POST /doctrine/tmr/reload`` route after operator edits).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from machine_spirit_4.gateway import doctrine as doctrine_mod
from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat


_FAKE_BIBLE = """\
# Test Bible

Preface to the test bible.

# PART I: THE BOOK OF SAMPLES
## Sub-Book I: The First Sampling
Lorem ipsum, lots of philosophical material.

## Sub-Book II: The Second Sampling
More philosophical content goes here.

# PART II: THE BOOK OF FORGES
## Book I: The Unsharpening
Content.

## Book II: The Tempering
More content.
"""


@pytest.fixture(autouse=True)
def _reset_bible_cache(tmp_path, monkeypatch):
    """Point the doctrine cache at a tiny fake bible so tests don't
    re-read the real 196 KB file and can be hermetic."""
    fake = tmp_path / "bible.md"
    fake.write_text(_FAKE_BIBLE, encoding="utf-8")
    monkeypatch.setattr(doctrine_mod, "DEFAULT_BIBLE_PATH", fake)
    # Clear any cache populated by earlier tests.
    with doctrine_mod._BIBLE_LOCK:
        doctrine_mod._BIBLE_CACHE["path"] = None
        doctrine_mod._BIBLE_CACHE["text"] = None
        doctrine_mod._BIBLE_CACHE["sections"] = None
        doctrine_mod._BIBLE_CACHE["mtime"] = None
    yield


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_get_full_bible_returns_disk_contents():
    text = doctrine_mod.get_full_bible()
    assert "PART I: THE BOOK OF SAMPLES" in text
    assert "Sub-Book II" in text


def test_list_sections_finds_top_level_headings():
    sections = doctrine_mod.list_sections()
    titles = [s.title for s in sections]
    # The loader walks both # and ## headings as section starts.
    assert "Test Bible" in titles
    assert "PART I: THE BOOK OF SAMPLES" in titles
    assert "Sub-Book I: The First Sampling" in titles
    assert "PART II: THE BOOK OF FORGES" in titles
    # IDs are slugified snake_case.
    ids = [s.id for s in sections]
    assert "part_i_the_book_of_samples" in ids
    assert "sub_book_i_the_first_sampling" in ids


def test_list_sections_records_char_count_per_section():
    sections = doctrine_mod.list_sections()
    for s in sections:
        assert s.end > s.start, f"section {s.id!r} has empty char range"
        assert s.char_count == s.end - s.start


def test_get_section_returns_meta_and_text():
    # Flat segmentation: each heading ends where the next heading begins.
    # PART I has no body before its first sub-book, so the section is
    # just the heading. Sub-Book I is the one that holds the prose.
    part_section, part_text = doctrine_mod.get_section("part_i_the_book_of_samples")
    assert part_section.title == "PART I: THE BOOK OF SAMPLES"
    assert part_text.startswith("# PART I: THE BOOK OF SAMPLES")

    sub_section, sub_text = doctrine_mod.get_section("sub_book_i_the_first_sampling")
    assert sub_section.title == "Sub-Book I: The First Sampling"
    assert "Lorem ipsum" in sub_text


def test_get_section_raises_unknown():
    with pytest.raises(doctrine_mod.DoctrineSectionUnknown):
        doctrine_mod.get_section("totally_not_a_real_section")


def test_meta_returns_v1_schema():
    m = doctrine_mod.meta()
    assert m["schema"] == "Ms4DoctrineMeta.v1"
    assert m["chars"] > 100
    assert m["section_count"] >= 5


def test_reload_bible_picks_up_disk_changes(tmp_path, monkeypatch):
    fake = tmp_path / "bible2.md"
    fake.write_text("# New Bible\n\nfresh.\n", encoding="utf-8")
    result = doctrine_mod.reload_bible(fake)
    assert result["chars"] > 0
    text = doctrine_mod.get_full_bible()
    assert "New Bible" in text


# ---------------------------------------------------------------------------
# Session injection
# ---------------------------------------------------------------------------


def test_inject_into_session_full_appends_user_and_assistant_messages():
    flc = FaceLobeChat(hivemind_url="http://hive")
    flc.get_or_create_session("test-session", "test-model")
    out = doctrine_mod.inject_into_session(
        flc, session_id="test-session", model="test-model", kind="full",
    )
    assert out["schema"] == "Ms4DoctrineInjection.v1"
    assert out["chars_injected"] > 100
    state = flc._sessions["test-session"]
    assert len(state.messages) == 2
    user_msg = state.messages[0]
    assistant_msg = state.messages[1]
    assert user_msg["role"] == "user"
    assert "DOCTRINE" in user_msg["content"]
    assert "PART I: THE BOOK OF SAMPLES" in user_msg["content"]
    assert assistant_msg["role"] == "assistant"
    assert "Origin-Neutrality" in assistant_msg["content"]


def test_inject_into_session_section_only_includes_that_text():
    flc = FaceLobeChat(hivemind_url="http://hive")
    flc.get_or_create_session("s2", "test-model")
    out = doctrine_mod.inject_into_session(
        flc, session_id="s2", model="test-model",
        kind="section", section_id="sub_book_ii_the_second_sampling",
    )
    state = flc._sessions["s2"]
    assert out["chars_injected"] < doctrine_mod.meta()["chars"]
    assert "Second Sampling" in state.messages[0]["content"]
    assert "PART II: THE BOOK OF FORGES" not in state.messages[0]["content"]


def test_inject_into_session_section_requires_section_id():
    flc = FaceLobeChat(hivemind_url="http://hive")
    flc.get_or_create_session("s3", "test-model")
    with pytest.raises(ValueError):
        doctrine_mod.inject_into_session(flc, session_id="s3", kind="section")


def test_inject_into_session_unknown_section_raises():
    flc = FaceLobeChat(hivemind_url="http://hive")
    flc.get_or_create_session("s4", "test-model")
    with pytest.raises(doctrine_mod.DoctrineSectionUnknown):
        doctrine_mod.inject_into_session(
            flc, session_id="s4", kind="section", section_id="not_in_bible",
        )


def test_inject_into_session_custom_acknowledgment_overrides_default():
    flc = FaceLobeChat(hivemind_url="http://hive")
    flc.get_or_create_session("s5", "test-model")
    custom = "I bind myself to the canon."
    doctrine_mod.inject_into_session(
        flc, session_id="s5", model="test-model", kind="full",
        acknowledgment=custom,
    )
    assert flc._sessions["s5"].messages[1]["content"] == custom
