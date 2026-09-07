"""Isolate voice tests from shell-level env vars.

The streaming voice engine selector (MS4_VOICE_TTS_ENGINE) governs which TTS
path a turn takes. The PRODUCTION default is now ``rest`` (2026-07-06
audio-stutter repair: ws_super produced audio slower than real time and
stuttered; see voice._default_tts_engine and DEFAULT_ENGINE). If an operator
overrides the env for their own gateway, that value is inherited by every
subsequent pytest run from the same shell — and tests would silently route
through the wrong engine, then fail with confusing errors about missing
parallelism / chunker behavior.

Strip the relevant env vars at the start of every voice test so the tests
always reflect the code's defaults, not the shell's state.
"""

from __future__ import annotations

import pytest


_ENV_VARS_TO_ISOLATE = (
    "MS4_VOICE_TTS_ENGINE",
    "MS4_VOICE_FIRST_CHUNK_MIN_WORDS",
    "MS4_VOICE_MAX_CHUNK_WORDS",
    "MS4_VOICE_MIN_CHUNK_WORDS",
    "MS4_VOICE_TTS_POOL_SIZE",
    "MS4_VOICE_TTS_CAPACITY",
    "MS4_VOICE_TTS_CAPACITY_MIN_SPEEDUP",
    "MS4_VOICE_TTS_CAPACITY_PROBE_TIMEOUT_S",
    "MS4_VOICE_TTS_CAPACITY_PROOF_TTL_S",
    "MS4_TTS_LOCATION_POLICY",
    "MS4_VOICE_TTS_MODEL",
    "MS4_VOICE_TTS_VOICE",
    "MS4_VOICE_TTS_FORMAT",
    "MS4_VOICE_ASR_MODEL",
    "MS4_VOICE_SPEAKER_ID_ASYNC",
    "MS4_VOICE_SPEAKER_ID_TERMINAL_BUDGET_S",
    "MS4_TTS_WS_SEND_TIMEOUT_S",
    "MS4_TTS_WS_SENDER_CLEANUP_TIMEOUT_S",
    "MS4_VOICE_SMART_REFLEX",
    "MS4_VOICE_EGG_HONORABLE",
)


@pytest.fixture(autouse=True)
def _isolate_voice_env(monkeypatch):
    for name in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(name, raising=False)
    import machine_spirit_4.gateway.voice as voice_module

    # Replica availability is process-local production state populated by the
    # scale endpoint. Tests must not inherit an observation made by an earlier
    # test in the same pytest worker.
    monkeypatch.setattr(
        voice_module,
        "_VOICE_REST_CAPACITY_OBSERVATION",
        (1, "fail_closed_default", None),
    )
    yield


@pytest.fixture
def pin_ws_super_engine(monkeypatch):
    """Deterministically route the WS-engine path for tests that drive a
    ws_engine_factory (or exercise WS setup/fallback) WITHOUT an explicit
    engine=. The PRODUCTION default is REST (voice._default_tts_engine), so WS
    pinning is EXPLICIT and per-test rather than an autouse global force. That
    keeps the DEFAULT_ENGINE constant unmasked for the REST-default test while
    preserving deterministic WS-path tests. REST-path tests pass engine="rest"
    explicitly and do not request this fixture."""
    import machine_spirit_4.gateway.voice as voice_module
    monkeypatch.setattr(voice_module, "DEFAULT_ENGINE", "ws_super")
    yield
