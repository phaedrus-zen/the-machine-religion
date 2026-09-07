"""Context-aware "buying time" reflex.

Covers the tiny-model sentiment/intent classifier (with its keyword
fail-closed fallback), the varied reflex picker (no-repeat), and the
`reflex` SSE event emission. The classifier must NEVER block or crash a
turn, so the model path is best-effort and falls back to the heuristic.
"""

from __future__ import annotations

import pytest

from machine_spirit_4.gateway import voice
from machine_spirit_4.gateway import canned_reflexes as cr


# ---------------------------------------------------------------------------
# Heuristic intent (zero-latency fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("what time is it?", "question"),
    ("how does this work", "question"),
    ("can you open the browser", "request"),   # "can you" -> a request to DO
    ("run the benchmark", "request"),
    ("open gamestop.com", "request"),
    ("thanks so much", "gratitude"),
    ("thank you", "gratitude"),
    ("hey there", "greeting"),
    ("hello", "greeting"),
    ("no that's wrong", "correction"),
    ("actually not that", "correction"),
    ("yes please", "affirmation"),
    ("okay sounds good", "affirmation"),
    ("the weather is nice today", "statement"),
    ("", "statement"),
])
def test_heuristic_intent(text, expected):
    assert voice._heuristic_intent(text) == expected


# ---------------------------------------------------------------------------
# classify_voice_intent
# ---------------------------------------------------------------------------


def test_classify_uses_heuristic_when_model_disabled():
    # timeout<=0 -> skip the model entirely, return the heuristic.
    intent, source = voice.classify_voice_intent("http://hive:6089", "what is the date?", timeout=0)
    assert intent == "question"
    assert source == "heuristic"


def test_classify_model_path(monkeypatch):
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat",
                        lambda *a, **k: {"choices": [{"message": {"content": "request"}}]})
    intent, source = voice.classify_voice_intent("http://hive:6089", "do the thing", timeout=2.0)
    assert intent == "request"
    assert source == "model"


def test_classify_fails_closed_to_heuristic(monkeypatch):
    import machine_spirit_4.gateway.hivemind_tools as ht

    def boom(*a, **k):
        raise RuntimeError("cluster down")

    monkeypatch.setattr(ht, "inference_chat", boom)
    intent, source = voice.classify_voice_intent("http://hive:6089", "thanks!", timeout=2.0)
    assert intent == "gratitude"
    assert source == "heuristic"


def test_classify_garbage_model_output_falls_back(monkeypatch):
    import machine_spirit_4.gateway.hivemind_tools as ht
    monkeypatch.setattr(ht, "inference_chat",
                        lambda *a, **k: {"choices": [{"message": {"content": "purple monkey"}}]})
    intent, source = voice.classify_voice_intent("http://hive:6089", "hey there", timeout=2.0)
    # Unrecognized label -> heuristic wins (greeting), never crashes.
    assert intent == "greeting"
    assert source == "heuristic"


# ---------------------------------------------------------------------------
# Reflex picker + variety
# ---------------------------------------------------------------------------


def test_every_intent_maps_to_a_category_with_reflexes():
    for intent in cr.VALID_INTENTS:
        cat = cr.INTENT_TO_CATEGORY[intent]
        assert cr.reflex_ids_for_category(cat), f"no reflexes for intent {intent!r} -> {cat!r}"


def test_pick_reflex_for_intent_in_category():
    rid = cr.pick_reflex_for_intent("question")
    assert rid in cr.reflex_ids_for_category(cr.CATEGORY_Q)


def test_pick_reflex_variety_excludes_recent():
    ids = cr.reflex_ids_for_category(cr.CATEGORY_GREETING)
    assert len(ids) >= 2
    # Exclude all but the last -> the picker must return the remaining one.
    rid = cr.pick_reflex_for_intent("greeting", exclude=ids[:-1])
    assert rid == ids[-1]


def test_pick_reflex_unknown_intent_falls_back_to_thinking():
    rid = cr.pick_reflex_for_intent("nonsense-intent")
    assert rid in cr.reflex_ids_for_category(cr.CATEGORY_THINKING)


# ---------------------------------------------------------------------------
# emit_smart_reflex -> `reflex` SSE event
# ---------------------------------------------------------------------------


def test_emit_smart_reflex_emits_reflex_event(monkeypatch):
    events: list[tuple[str, dict]] = []
    seen: dict[str, float] = {}

    def classify(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return "question", "heuristic"

    monkeypatch.delenv("MS4_VOICE_REFLEX_CLASSIFIER_BUDGET_S", raising=False)
    monkeypatch.setattr(voice, "classify_voice_intent", classify)
    turn_started_at = voice.time.monotonic()
    voice.emit_smart_reflex(
        emit=lambda ev, payload: (events.append((ev, payload)), True)[1],
        hivemind_url="http://hive:6089", transcript="what is it?", session_id="s1",
        turn_started_at=turn_started_at,
    )
    assert seen["timeout"] == 0.0
    assert events
    ev, payload = events[0]
    assert ev == "reflex"
    assert payload["id"] in cr.reflex_ids_for_category(cr.CATEGORY_Q)
    assert payload["intent"] == "question"
    assert payload["category"] == cr.CATEGORY_Q
    assert payload["classification_ms"] >= 0
    assert payload["turn_ms"] >= 0


def test_emit_smart_reflex_model_refinement_is_explicit_opt_in(monkeypatch):
    seen: dict[str, float] = {}

    def classify(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return "statement", "model"

    monkeypatch.setenv("MS4_VOICE_REFLEX_CLASSIFIER_BUDGET_S", "0.25")
    monkeypatch.setattr(voice, "classify_voice_intent", classify)
    voice.emit_smart_reflex(
        emit=lambda _ev, _payload: True,
        hivemind_url="http://hive:6089",
        transcript="the circuit is ready",
        session_id="s-model-opt-in",
    )
    assert seen["timeout"] == 0.25


def test_emit_smart_reflex_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("classify exploded")

    monkeypatch.setattr(voice, "classify_voice_intent", boom)
    # Must swallow everything — a reflex must never break a turn.
    voice.emit_smart_reflex(
        emit=lambda ev, payload: True,
        hivemind_url="http://hive:6089", transcript="hello", session_id="s1",
    )


# ---------------------------------------------------------------------------
# "Honorable" easter egg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "hey agent, continue the phrase",
    "I want you to finish the phrase",
    "complete the sentence for me",
    "honorable mode",
])
def test_egg_arming_detected(text):
    assert voice._is_egg_arming(text) is True


@pytest.mark.parametrize("text", [
    "what's the weather",
    "continue running that job",     # 'continue' but not the phrase
    "tell me a sentence about cats",
])
def test_egg_arming_not_detected(text):
    assert voice._is_egg_arming(text) is False


def test_word_is_honorable():
    assert voice._word_is_honorable("honorable")
    assert voice._word_is_honorable("Honorable.")
    assert voice._word_is_honorable("honourable")  # UK spelling
    assert not voice._word_is_honorable("honor")
    assert not voice._word_is_honorable("noble")


def test_egg_arming_turn_arms_and_acks(monkeypatch):
    events = []
    sess = "egg-arm-1"
    voice._disarm_egg(sess)
    res = voice.handle_honorable_egg(
        emit=lambda ev, p: (events.append((ev, p)), True)[1],
        hivemind_url="http://hive:6089", transcript="hey agent, continue the phrase",
        session_id=sess, asr_ms=120,
    )
    assert res is not None  # arming handled -> normal reply skipped
    assert voice._egg_is_armed(sess) is True
    assert any(ev == "reflex" for ev, _ in events)  # "go ahead" ack
    voice._disarm_egg(sess)


def test_egg_fires_on_honorable_prediction(monkeypatch):
    events = []
    sess = "egg-fire-1"
    monkeypatch.setattr(voice, "_egg_should_fire", lambda *a, **k: (True, "predicted"))
    voice._arm_egg(sess)
    res = voice.handle_honorable_egg(
        emit=lambda ev, p: (events.append((ev, p)), True)[1],
        hivemind_url="http://hive:6089", transcript="a knight of unwavering virtue is truly",
        session_id=sess, asr_ms=200,
    )
    assert res is not None  # egg fired -> reply replaced
    egg_events = [p for ev, p in events if ev == "egg"]
    assert egg_events and egg_events[0]["clip"] == "/easter/honorable"
    assert voice._egg_is_armed(sess) is False  # disarmed ONLY on a hit


def test_egg_fires_when_user_says_the_word():
    # The surest trigger: while armed, the user literally says "honorable".
    # No model needed (string match), so no monkeypatch.
    events = []
    sess = "egg-saidit-1"
    voice._arm_egg(sess)
    res = voice.handle_honorable_egg(
        emit=lambda ev, p: (events.append((ev, p)), True)[1],
        hivemind_url="http://hive:6089", transcript="can you say honorable please",
        session_id=sess, asr_ms=200,
    )
    assert res is not None
    assert any(ev == "egg" for ev, _ in events)
    assert voice._egg_is_armed(sess) is False


def test_egg_miss_STAYS_armed(monkeypatch):
    # A miss must NOT disarm — you can keep trying without re-arming.
    sess = "egg-miss-1"
    monkeypatch.setattr(voice, "_egg_should_fire", lambda *a, **k: (False, ""))
    voice._arm_egg(sess)
    res = voice.handle_honorable_egg(
        emit=lambda ev, p: True,
        hivemind_url="http://hive:6089", transcript="a knight is",
        session_id=sess, asr_ms=200,
    )
    assert res is None  # miss -> normal reply proceeds
    assert voice._egg_is_armed(sess) is True  # STILL armed (key fix)
    voice._disarm_egg(sess)


def test_egg_should_fire_string_match():
    fired, reason = voice._egg_should_fire("http://hive:6089", "the word is honorable")
    assert fired and reason == "said_it"


def test_egg_yes_no_fails_closed(monkeypatch):
    import machine_spirit_4.gateway.hivemind_tools as ht

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(ht, "inference_chat", boom)
    # No literal "honorable" -> falls to the probe -> probe fails -> no fire.
    fired, _ = voice._egg_should_fire("http://hive:6089", "a person of great virtue is", )
    assert fired is False


def test_egg_not_armed_is_noop():
    sess = "egg-noop-1"
    voice._disarm_egg(sess)
    res = voice.handle_honorable_egg(
        emit=lambda ev, p: True,
        hivemind_url="http://hive:6089", transcript="just a normal sentence",
        session_id=sess, asr_ms=100,
    )
    assert res is None  # not armed, not arming -> normal turn


def test_predict_next_word_fails_closed(monkeypatch):
    import machine_spirit_4.gateway.hivemind_tools as ht

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(ht, "inference_chat", boom)
    assert voice.predict_next_word("http://hive:6089", "the knight is", timeout=2.0) == ""
