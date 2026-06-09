"""Round-trip QA for generated canned speech.

Render -> transcribe back -> fuzzy + tiny-LLM judge -> if bad, re-render
with an LLM-rephrased equivalent. Everything is fail-OPEN: if ASR or the
LLM are unavailable, QA accepts the render rather than rejecting it.
"""

from __future__ import annotations

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
    monkeypatch.setattr(cr, "transcribe", lambda **k: {"text": "On it"})
    v = cr.validate_reflex_audio("On it.", _wav_of_secs(1.0), "http://hive")
    assert v["valid"] is True and v["reason"] == "match"


# ---------------------------------------------------------------------------
# validate_reflex_audio
# ---------------------------------------------------------------------------


def test_validate_clean_match(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: {"text": "On it"})
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is True and v["reason"] == "match"


def test_validate_garble_rejected(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: {"text": "on it " * 10})
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat", _raise)  # no LLM -> trust the fuzzy bad verdict
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is False


def test_validate_asr_unavailable_fails_open(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", _raise)
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is True and v["reason"] == "asr_unavailable"


def test_validate_llm_judge_yes(monkeypatch):
    monkeypatch.setattr(cr, "transcribe", lambda **k: {"text": "honor"})  # ambiguous fuzzy
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat", lambda *a, **k: {"choices": [{"message": {"content": "yes"}}]})
    v = cr.validate_reflex_audio("On it.", b"WAV", "http://hive")
    assert v["valid"] is True and v["reason"] == "llm"


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
        return {"text": "on it " * 10} if said == original else {"text": said}

    monkeypatch.setattr(cr, "transcribe", fake_transcribe)
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat", _raise)  # judge unavailable -> trust fuzzy
    monkeypatch.setattr(cr, "rephrase_reflex_text", lambda original_, url, avoid=(): rephrased)

    info = cr.generate_reflex(hivemind_url="http://hive", reflex_id=rid)
    assert info["validated"] is True
    assert info["effective_text"] == rephrased
    # sidecar written + audio written
    assert cr._read_meta(cr.DEFAULT_REFLEX_VOICE, rid)["effective_text"] == rephrased
    assert cr.reflex_path(cr.DEFAULT_REFLEX_VOICE, rid).exists()


def test_generate_reflex_flags_when_all_attempts_fail(tmp_reflex_dir, monkeypatch):
    rid = "stmt_mhm"
    original = cr.get_reflex(rid).text
    monkeypatch.setattr(cr, "synthesize", lambda **k: {"audio_bytes": k["text"].encode("utf-8")})
    monkeypatch.setattr(cr, "transcribe", lambda **k: {"text": "blah " * 12})  # always garble
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat", _raise)
    monkeypatch.setattr(cr, "rephrase_reflex_text", lambda o, u, avoid=(): f"variant {len(list(avoid))}")
    monkeypatch.setenv("MS4_REFLEX_QA_RETRIES", "2")

    info = cr.generate_reflex(hivemind_url="http://hive", reflex_id=rid)
    assert info["validated"] is False
    # On total failure we fall back to the FIRST (original-text) render.
    assert info["effective_text"] == original


def test_generate_reflex_qa_off_skips_validation(tmp_reflex_dir, monkeypatch):
    monkeypatch.setenv("MS4_REFLEX_QA", "0")
    calls = {"asr": 0}

    def counting_transcribe(**k):
        calls["asr"] += 1
        return {"text": "whatever"}

    monkeypatch.setattr(cr, "synthesize", lambda **k: {"audio_bytes": b"AUDIO"})
    monkeypatch.setattr(cr, "transcribe", counting_transcribe)
    info = cr.generate_reflex(hivemind_url="http://hive", reflex_id="conf_okay")
    assert info["validated"] is None      # QA off -> unknown
    assert calls["asr"] == 0              # never transcribed
