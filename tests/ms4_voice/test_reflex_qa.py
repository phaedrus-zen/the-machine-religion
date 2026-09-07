"""Round-trip QA for generated canned speech.

Render to an isolated candidate -> transcribe back with real ASR evidence ->
require semantic acceptance -> atomically promote. Missing or ambiguous
evidence holds the candidate and preserves the last-known-good render.
"""

from __future__ import annotations

import hashlib
import io
import wave

import pytest

from machine_spirit_4.gateway import canned_reflexes as cr


def _wav_of_secs(secs: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * secs))
    return buf.getvalue()


@pytest.fixture
def tmp_reflex_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_REFLEX_DIR", str(tmp_path))
    return tmp_path


def _raise(*a, **k):
    raise RuntimeError("unavailable")


def _asr(text: str, **raw_extra):
    """Production-shaped ASR evidence: outer text plus the raw service body."""
    return {
        "text": text,
        "model": "whisper-1",
        "raw": {"text": text, **raw_extra},
    }


def _judge(monkeypatch, answer: str = "yes"):
    import machine_spirit_4.gateway.hivemind_tools as ht

    monkeypatch.setattr(
        ht,
        "inference_chat",
        lambda *a, **k: {"choices": [{"message": {"content": answer}}]},
    )


# ---------------------------------------------------------------------------
# Fuzzy verdict
# ---------------------------------------------------------------------------


def test_qa_fuzzy_exact_match():
    assert cr._qa_fuzzy("On it.", "on it") is True


def test_qa_fuzzy_length_blowup_is_bad():
    assert cr._qa_fuzzy("On it.", "on it and then we head to the store for apples today") is False


def test_qa_fuzzy_empty_is_bad():
    assert cr._qa_fuzzy("On it.", "") is False


def test_qa_fuzzy_ambiguous_defers_to_llm():
    assert cr._qa_fuzzy("On it.", "honor") is None


def test_qa_fuzzy_repetition_is_bad():
    # "On it. On it." (doubled) must NOT count as a clean match.
    assert cr._qa_fuzzy("On it.", "on it on it") is False


# ---------------------------------------------------------------------------
# Duration gate (the strongest, model-free garble signal)
# ---------------------------------------------------------------------------


def test_validate_rejects_too_long(monkeypatch):
    # A 5s render for a 2-word phrase is a babble tail -> rejected outright,
    # WITHOUT even calling ASR.
    monkeypatch.setattr(cr, "transcribe", lambda **k: pytest.fail("ASR must not run for an over-long clip"))
    v = cr.validate_reflex_audio("On it.", _wav_of_secs(5.0), "http://hive")
    assert v["valid"] is False and v["reason"].startswith("too_long")


def test_validate_ok_duration_passes_to_asr(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: _asr("On it"))
    _judge(monkeypatch)
    v = cr.validate_reflex_audio("On it.", _wav_of_secs(1.0), "http://hive")
    assert v["valid"] is True and v["reason"] == "semantic_accept"


# ---------------------------------------------------------------------------
# validate_reflex_audio
# ---------------------------------------------------------------------------


def test_validate_clean_match(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: _asr("On it"))
    _judge(monkeypatch)
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is True and v["reason"] == "semantic_accept"


def test_validate_garble_rejected(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: _asr("on it " * 10))
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat", _raise)  # no LLM -> trust the fuzzy bad verdict
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is False


def test_validate_asr_unavailable_holds(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", _raise)
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is False
    assert v["reason"] == "asr_unavailable"
    assert v.get("disposition") == "hold"


def test_validate_rejects_text_without_real_asr_evidence(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: {"text": "On it"})
    _judge(monkeypatch)
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is False
    assert v["reason"] == "asr_evidence_unavailable"


def test_validate_rejects_transcript_hint_as_fabricated_evidence(monkeypatch):
    monkeypatch.setattr(
        cr,
        "transcribe",
        lambda **k: _asr("On it", transcript_hint="On it", source="request_hint"),
    )
    _judge(monkeypatch)
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is False
    assert v["reason"] == "asr_evidence_ambiguous"


def test_validate_ambiguous_transcript_holds_even_if_judge_would_accept(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: _asr("honor"))
    _judge(monkeypatch)
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is False
    assert v["reason"] == "transcript_ambiguous"


def test_validate_absent_semantic_judge_holds(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: _asr("On it"))
    import machine_spirit_4.gateway.hivemind_tools as ht

    monkeypatch.setattr(ht, "inference_chat", _raise)
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is False
    assert v["reason"] == "semantic_judge_unavailable"


def test_validate_semantic_rejection_holds(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: _asr("On it"))
    _judge(monkeypatch, "no")
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is False
    assert v["reason"] == "semantic_rejected"


# ---------------------------------------------------------------------------
# rephrase_reflex_text
# ---------------------------------------------------------------------------


def test_rephrase_uses_llm(monkeypatch):
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat", lambda *a, **k: {"choices": [{"message": {"content": "I'm on it."}}]})
    assert cr.rephrase_reflex_text("On it.", "http://hive") == "I'm on it."


def test_rephrase_falls_back_to_manual(monkeypatch):
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat", _raise)
    assert cr.rephrase_reflex_text("On it.", "http://hive") == cr._MANUAL_REPHRASE["On it."]


# ---------------------------------------------------------------------------
# generate_reflex with QA retry/rephrase
# ---------------------------------------------------------------------------


def test_generate_reflex_rephrases_a_garbled_phrase(tmp_reflex_dir, monkeypatch):
    rid = "req_on_it"
    original = cr.get_reflex(rid).text  # "On it."
    rephrased = "I'm on it."

    # synth encodes the text into the bytes; transcribe decodes it back, but
    # the ORIGINAL text "garbles" (length blow-up) while the rephrase is clean.
    monkeypatch.setattr(cr, "synthesize", lambda **k: {"audio_bytes": k["text"].encode("utf-8")})

    def fake_transcribe(**k):
        said = k["audio"].decode("utf-8")
        return _asr("on it " * 10) if said == original else _asr(said)

    monkeypatch.setattr(cr, "transcribe", fake_transcribe)
    _judge(monkeypatch)
    monkeypatch.setattr(cr, "rephrase_reflex_text", lambda original_, url, avoid=(): rephrased)

    info = cr.generate_reflex(hivemind_url="http://hive", reflex_id=rid)
    assert info.get("status") == "promoted"
    assert info.get("promoted") is True
    assert info["validated"] is True
    assert info["effective_text"] == rephrased
    # sidecar written + audio written
    meta = cr._read_meta(cr.DEFAULT_REFLEX_VOICE, rid)
    audio = cr.reflex_path(cr.DEFAULT_REFLEX_VOICE, rid).read_bytes()
    assert meta["effective_text"] == rephrased
    assert meta["audio_sha256"] == hashlib.sha256(audio).hexdigest()
    assert cr.read_reflex(reflex_id=rid) == audio


def test_generate_reflex_retry_exhaustion_holds_without_promotion(tmp_reflex_dir, monkeypatch):
    rid = "stmt_mhm"
    monkeypatch.setattr(cr, "synthesize", lambda **k: {"audio_bytes": k["text"].encode("utf-8")})
    monkeypatch.setattr(cr, "transcribe", lambda **k: _asr("blah " * 12))  # always garble
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat", _raise)
    monkeypatch.setattr(cr, "rephrase_reflex_text", lambda o, u, avoid=(): f"variant {len(list(avoid))}")
    monkeypatch.setenv("MS4_REFLEX_QA_RETRIES", "2")

    info = cr.generate_reflex(hivemind_url="http://hive", reflex_id=rid)
    assert info.get("status") == "hold"
    assert info.get("promoted") is False
    assert info.get("hold_reason") == "retry_exhausted"
    assert info["validated"] is False
    assert not cr.reflex_path(cr.DEFAULT_REFLEX_VOICE, rid).exists()
    assert cr._read_meta(cr.DEFAULT_REFLEX_VOICE, rid) is None


def test_generate_reflex_qa_off_holds_without_promotion(tmp_reflex_dir, monkeypatch):
    monkeypatch.setenv("MS4_REFLEX_QA", "0")
    calls = {"asr": 0}

    def counting_transcribe(**k):
        calls["asr"] += 1
        return {"text": "whatever"}

    monkeypatch.setattr(cr, "synthesize", lambda **k: {"audio_bytes": b"AUDIO"})
    monkeypatch.setattr(cr, "transcribe", counting_transcribe)
    info = cr.generate_reflex(hivemind_url="http://hive", reflex_id="conf_okay")
    assert info.get("status") == "hold"
    assert info.get("hold_reason") == "qa_disabled"
    assert info.get("promoted") is False
    assert calls["asr"] == 0
    assert not cr.reflex_path(cr.DEFAULT_REFLEX_VOICE, "conf_okay").exists()


def test_successful_generation_validates_candidate_before_promotion(tmp_reflex_dir, monkeypatch):
    rid = "conf_okay"
    audio = b"candidate-audio"
    final_path = cr.reflex_path(cr.DEFAULT_REFLEX_VOICE, rid)
    monkeypatch.setattr(cr, "synthesize", lambda **k: {"audio_bytes": audio})
    monkeypatch.setattr(cr, "transcribe", lambda **k: _asr("Okay"))
    _judge(monkeypatch)
    real_validate = cr.validate_reflex_audio

    def validate_while_candidate_isolated(intended, candidate_audio, hivemind_url):
        candidates = [
            path for path in final_path.parent.glob("*.wav")
            if "candidate" in path.name
        ]
        assert len(candidates) == 1
        assert candidates[0].read_bytes() == audio
        assert not final_path.exists()
        return real_validate(intended, candidate_audio, hivemind_url)

    monkeypatch.setattr(cr, "validate_reflex_audio", validate_while_candidate_isolated)
    info = cr.generate_reflex(
        hivemind_url="http://hive",
        reflex_id=rid,
        retries=0,
    )

    assert info.get("status") == "promoted"
    assert final_path.read_bytes() == audio
    assert not list(final_path.parent.glob("*candidate*"))


def test_failed_regeneration_preserves_last_known_good_pair(tmp_reflex_dir, monkeypatch):
    rid = "conf_done"
    voice = cr.DEFAULT_REFLEX_VOICE
    path = cr.reflex_path(voice, rid)
    path.parent.mkdir(parents=True)
    good_audio = b"last-known-good"
    path.write_bytes(good_audio)
    good_meta = {
        "schema": "Ms4ReflexMeta.v1",
        "id": rid,
        "validated": True,
        "qa_version": cr._QA_VERSION,
        "audio_sha256": hashlib.sha256(good_audio).hexdigest(),
    }
    cr._write_meta(voice, rid, good_meta)
    before_meta = cr._meta_path(voice, rid).read_bytes()
    monkeypatch.setattr(cr, "synthesize", lambda **k: {"audio_bytes": b"bad-candidate"})

    def unavailable_while_good_pair_stays_live(**kwargs):
        assert path.read_bytes() == good_audio
        assert cr._meta_path(voice, rid).read_bytes() == before_meta
        raise RuntimeError("unavailable")

    monkeypatch.setattr(cr, "transcribe", unavailable_while_good_pair_stays_live)

    info = cr.generate_reflex(
        hivemind_url="http://hive",
        reflex_id=rid,
        retries=0,
    )

    assert info.get("status") == "hold"
    assert info.get("hold_reason") == "asr_unavailable"
    assert path.read_bytes() == good_audio
    assert cr._meta_path(voice, rid).read_bytes() == before_meta
    assert cr.read_reflex(reflex_id=rid, voice=voice) == good_audio
    assert not list(path.parent.glob("*candidate*"))


def test_stale_revalidation_cannot_clobber_concurrent_promotion(tmp_reflex_dir, monkeypatch):
    rid = "conf_okay"
    voice = cr.DEFAULT_REFLEX_VOICE
    reflex = cr.get_reflex(rid)
    path = cr.reflex_path(voice, rid)
    path.parent.mkdir(parents=True)
    old_audio = b"unbound-old-audio"
    new_audio = b"concurrently-promoted-audio"
    path.write_bytes(old_audio)
    monkeypatch.setattr(cr, "REFLEXES", (reflex,))

    def validate_then_promote(intended, audio, hivemind_url):
        assert audio == old_audio
        candidate = cr._write_candidate_audio(path, new_audio)
        cr._promote_candidate(
            candidate_path=candidate,
            voice=voice,
            reflex_id=rid,
            meta={
                "schema": "Ms4ReflexMeta.v1",
                "id": rid,
                "validated": True,
                "qa_version": cr._QA_VERSION,
                "audio_sha256": hashlib.sha256(new_audio).hexdigest(),
            },
        )
        return {
            "valid": True,
            "heard": intended,
            "reason": "semantic_accept",
            "disposition": "accept",
        }

    monkeypatch.setattr(cr, "validate_reflex_audio", validate_then_promote)
    out = cr.generate_all(
        hivemind_url="http://hive",
        voice=voice,
        force=False,
        qa=True,
    )

    assert cr.read_reflex(reflex_id=rid, voice=voice) == new_audio
    assert out["revalidated"] == []
    assert out["skipped"] == [rid]
