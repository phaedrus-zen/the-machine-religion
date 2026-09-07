"""Tests for the pre-rendered reflex audio system.

What's locked in
----------------

* The :data:`REFLEXES` catalog has the categories the UI keys off
  (``ack``, ``thinking``, ``error``, ``confirm``, ``identity``).
* ``generate_reflex`` writes to the path :func:`reflex_path` returns,
  only after QA accepts the candidate and binds metadata to its hash.
* ``generate_all`` skips already-present reflexes when ``force=False``
  and re-renders them when ``force=True``.
* ``list_reflexes`` projects validated, hash-bound availability per voice.
* Unknown reflex ids raise :class:`ReflexUnknown` so the gateway
  can map to 400 vs 404 correctly.
* Voice ids are sanitized so a malicious caller can't write outside
  the cache dir.

We monkeypatch the HiveMind ``synthesize`` call so the tests don't
need a live cluster.
"""

from __future__ import annotations

import hashlib
import io
import struct
import threading
import types
import wave

import pytest

from machine_spirit_4.gateway import canned_reflexes as cr


@pytest.fixture(autouse=True)
def _accept_reflex_qa(monkeypatch):
    """These tests exercise generation/listing mechanics, not the QA
    round-trip (which transcribes audio back via ASR). Stub an accepted QA
    result so they stay hermetic and don't reach the network. QA itself is in
    test_reflex_qa.py."""
    monkeypatch.setenv("MS4_REFLEX_QA", "1")
    monkeypatch.setattr(
        cr,
        "validate_reflex_audio",
        lambda intended, audio, url: {
            "valid": True,
            "heard": intended,
            "reason": "semantic_accept",
            "disposition": "accept",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_wav_bytes(text: str) -> bytes:
    """Return a tiny but VALID WAV blob so on-disk readers can decode
    it. ~100ms of silence at 24kHz mono s16."""
    samples = bytes(2 * 2400)  # 100ms * 24000 samples/sec * 2 bytes/sample
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(samples)
    return buf.getvalue()


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect the reflex cache directory into a per-test tmp_path so
    tests don't collide with each other or with a real on-disk cache."""
    monkeypatch.setenv("MS4_REFLEX_DIR", str(tmp_path))
    monkeypatch.setattr(cr, "DEFAULT_REFLEX_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def fake_synthesize(monkeypatch):
    """Replace voice.synthesize so generate_reflex doesn't hit
    HiveMind. Returns a counter so tests can assert how many times
    HiveMind would have been called."""
    counter = {"n": 0, "by_text": {}}

    def fake(*, hivemind_url, text, model=None, voice=None, response_format=None, timeout=60):
        counter["n"] += 1
        counter["by_text"][text] = counter["by_text"].get(text, 0) + 1
        wav = _fake_wav_bytes(text)
        return {
            "audio_bytes": wav,
            "content_type": "audio/wav",
            "audio_base64": "",
            "model": model or "tts-1",
            "voice": voice or "alloy",
            "format": response_format or "wav",
        }

    monkeypatch.setattr(cr, "synthesize", fake)
    return counter


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_has_expected_categories():
    categories = sorted({r.category for r in cr.REFLEXES})
    assert categories == sorted([
        cr.CATEGORY_ACK,
        cr.CATEGORY_THINKING,
        cr.CATEGORY_ERROR,
        cr.CATEGORY_CONFIRM,
        cr.CATEGORY_IDENTITY,
        # Intent/sentiment categories for the context-aware reflex (Jun 1 2026).
        cr.CATEGORY_Q,
        cr.CATEGORY_REQUEST,
        cr.CATEGORY_GRATITUDE,
        cr.CATEGORY_GREETING,
        cr.CATEGORY_STATEMENT,
        cr.CATEGORY_CORRECTION,
        cr.CATEGORY_AFFIRM,
        cr.CATEGORY_EGG,
    ])


def test_catalog_has_at_least_one_reflex_per_category():
    by_cat: dict[str, int] = {}
    for r in cr.REFLEXES:
        by_cat[r.category] = by_cat.get(r.category, 0) + 1
    for category in (
        cr.CATEGORY_ACK,
        cr.CATEGORY_THINKING,
        cr.CATEGORY_ERROR,
        cr.CATEGORY_CONFIRM,
        cr.CATEGORY_IDENTITY,
    ):
        assert by_cat.get(category, 0) >= 1, f"category {category!r} has no reflexes"


def test_catalog_ids_are_unique_and_snake_case():
    seen: set[str] = set()
    for r in cr.REFLEXES:
        assert r.id not in seen, f"duplicate reflex id {r.id!r}"
        seen.add(r.id)
        assert r.id.replace("_", "").isalnum(), f"id {r.id!r} should be snake_case alphanumeric"


# ---------------------------------------------------------------------------
# Path discipline
# ---------------------------------------------------------------------------


def test_reflex_path_inside_cache_dir(cache_dir):
    p = cr.reflex_path("alloy", "ack_mhm")
    assert p.is_absolute() or str(p).startswith(str(cache_dir))
    assert str(p).endswith("alloy/ack_mhm.wav") or str(p).endswith(r"alloy\ack_mhm.wav")


def test_reflex_path_sanitizes_dangerous_voice_ids(cache_dir):
    """A voice id with .. / / shouldn't escape the cache dir."""
    p1 = cr.reflex_path("../etc", "ack_mhm")
    p2 = cr.reflex_path("alloy/with/slashes", "ack_mhm")
    for p in (p1, p2):
        assert ".." not in str(p), f"sanitized path leaked ..: {p}"
        # Should still resolve to a path under cache_dir (or the default).
        try:
            resolved = p.resolve()
            cache_resolved = cache_dir.resolve()
            assert str(resolved).startswith(str(cache_resolved)), \
                f"{resolved} not under {cache_resolved}"
        except FileNotFoundError:
            # File doesn't exist yet — resolution may walk back to cwd
            # on some platforms. That's fine; the contract is "no .."
            # and "no path traversal", which we already checked above.
            pass


def test_reflex_path_rejects_unknown_id(cache_dir):
    with pytest.raises(ValueError):
        cr.reflex_path("alloy", "totally_made_up_reflex_id")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generate_reflex_writes_valid_wav(cache_dir, fake_synthesize):
    info = cr.generate_reflex(hivemind_url="http://hive", reflex_id="ack_mhm", voice="alloy")
    assert info["id"] == "ack_mhm"
    assert info["voice"] == "alloy"
    assert info["size_bytes"] > 0
    assert info.get("status") == "promoted"
    assert info.get("promoted") is True
    # On-disk file is a real WAV decodable by stdlib wave.
    p = cr.reflex_path("alloy", "ack_mhm")
    assert p.exists()
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 24000
    meta = cr._read_meta("alloy", "ack_mhm")
    assert meta["validated"] is True
    assert meta["qa_version"] == cr._QA_VERSION
    assert meta["audio_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_generate_reflex_unknown_id_raises(cache_dir, fake_synthesize):
    with pytest.raises(cr.ReflexUnknown):
        cr.generate_reflex(hivemind_url="http://hive", reflex_id="not_in_catalog", voice="alloy")
    # fake_synthesize must not have been called for an unknown id.
    assert fake_synthesize["n"] == 0


def test_generate_all_skips_existing_when_force_false(cache_dir, fake_synthesize):
    first = cr.generate_all(hivemind_url="http://hive", voice="alloy", force=False)
    assert len(first["generated"]) == len(cr.REFLEXES)
    assert first["skipped"] == []
    assert first["failed"] == {}
    # Second pass: everything is already on disk, nothing should be regenerated.
    fake_synthesize["n"] = 0
    second = cr.generate_all(hivemind_url="http://hive", voice="alloy", force=False)
    assert second["generated"] == []
    assert sorted(second["skipped"]) == sorted(r.id for r in cr.REFLEXES)
    assert fake_synthesize["n"] == 0  # no TTS calls fired


def test_generate_all_force_re_renders_everything(cache_dir, fake_synthesize):
    cr.generate_all(hivemind_url="http://hive", voice="alloy", force=False)
    fake_synthesize["n"] = 0
    out = cr.generate_all(hivemind_url="http://hive", voice="alloy", force=True)
    assert len(out["generated"]) == len(cr.REFLEXES)
    assert out["skipped"] == []
    assert fake_synthesize["n"] == len(cr.REFLEXES)


def test_generate_all_aggregates_failures(cache_dir, monkeypatch):
    """If synthesize raises for any reflex, generate_all collects the
    failure and continues with the rest instead of aborting the whole
    batch."""
    calls = {"n": 0}

    def flaky(*, hivemind_url, text, **_kw):
        calls["n"] += 1
        if "Mhm?" in text:  # exact ack_mhm text; the catalog now also has "Mhm." (stmt_mhm)
            raise RuntimeError("HiveMind hiccup")
        return {
            "audio_bytes": _fake_wav_bytes(text),
            "content_type": "audio/wav",
            "audio_base64": "",
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
        }

    monkeypatch.setattr(cr, "synthesize", flaky)
    out = cr.generate_all(hivemind_url="http://hive", voice="alloy")
    assert "ack_mhm" in out["failed"]
    assert "HiveMind hiccup" in out["failed"]["ack_mhm"]
    # The rest still succeeded.
    assert len(out["generated"]) == len(cr.REFLEXES) - 1


def test_generate_all_async_runs_in_background(cache_dir, fake_synthesize):
    thread = cr.generate_all_async(hivemind_url="http://hive", voice="alloy")
    assert isinstance(thread, threading.Thread)
    thread.join(timeout=5)
    assert not thread.is_alive(), "generate_all_async should complete promptly with fake_synthesize"
    # All reflexes were generated.
    out = cr.list_reflexes(voice="alloy")
    assert out["available"] == len(cr.REFLEXES)


# ---------------------------------------------------------------------------
# Listing + reading
# ---------------------------------------------------------------------------


def test_list_reflexes_reports_availability_and_sizes(cache_dir, fake_synthesize):
    # Generate only the first reflex.
    cr.generate_reflex(hivemind_url="http://hive", reflex_id=cr.REFLEXES[0].id, voice="alloy")
    out = cr.list_reflexes(voice="alloy")
    assert out["schema"] == "Ms4ReflexCatalog.v1"
    assert out["voice"] == "alloy"
    assert out["total"] == len(cr.REFLEXES)
    assert out["available"] == 1
    by_id = {r["id"]: r for r in out["reflexes"]}
    assert by_id[cr.REFLEXES[0].id]["available"] is True
    assert by_id[cr.REFLEXES[0].id]["size_bytes"] > 0
    assert by_id[cr.REFLEXES[1].id]["available"] is False
    # Every entry exposes the URL the UI fetches.
    for rfx in out["reflexes"]:
        assert rfx["url"].startswith("/reflexes/alloy/") and rfx["url"].endswith(".wav")


def test_read_reflex_returns_wav_bytes_when_available(cache_dir, fake_synthesize):
    cr.generate_reflex(hivemind_url="http://hive", reflex_id="ack_mhm", voice="alloy")
    bytes_out = cr.read_reflex(reflex_id="ack_mhm", voice="alloy")
    assert bytes_out is not None
    assert bytes_out.startswith(b"RIFF") and bytes_out[8:12] == b"WAVE"


def test_read_reflex_returns_none_when_missing(cache_dir):
    bytes_out = cr.read_reflex(reflex_id="ack_mhm", voice="alloy")
    assert bytes_out is None


def test_read_reflex_rejects_hash_mismatch(cache_dir):
    rid = "ack_mhm"
    voice = "alloy"
    path = cr.reflex_path(voice, rid)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"current-wav")
    cr._write_meta(voice, rid, {
        "schema": "Ms4ReflexMeta.v1",
        "id": rid,
        "validated": True,
        "qa_version": cr._QA_VERSION,
        "audio_sha256": hashlib.sha256(b"different-wav").hexdigest(),
    })

    assert cr.read_reflex(reflex_id=rid, voice=voice) is None


@pytest.mark.parametrize(
    ("validated", "qa_version"),
    [
        (False, cr._QA_VERSION),
        (None, cr._QA_VERSION),
        (True, cr._QA_VERSION - 1),
    ],
)
def test_read_reflex_rejects_unvalidated_or_stale_metadata(
    cache_dir,
    validated,
    qa_version,
):
    rid = "ack_yes"
    voice = "alloy"
    audio = b"bound-wav"
    path = cr.reflex_path(voice, rid)
    path.parent.mkdir(parents=True)
    path.write_bytes(audio)
    cr._write_meta(voice, rid, {
        "schema": "Ms4ReflexMeta.v1",
        "id": rid,
        "validated": validated,
        "qa_version": qa_version,
        "audio_sha256": hashlib.sha256(audio).hexdigest(),
    })

    assert cr.read_reflex(reflex_id=rid, voice=voice) is None


def test_list_reflexes_does_not_advertise_unbound_audio(cache_dir):
    rid = "ack_mhm"
    path = cr.reflex_path("alloy", rid)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"orphaned-wav")

    out = cr.list_reflexes(voice="alloy")
    by_id = {item["id"]: item for item in out["reflexes"]}
    assert by_id[rid]["available"] is False
    assert by_id[rid]["size_bytes"] == 0


def test_read_reflex_unknown_id_raises(cache_dir):
    with pytest.raises(cr.ReflexUnknown):
        cr.read_reflex(reflex_id="not_in_catalog")
