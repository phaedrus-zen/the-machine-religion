"""Isolate voice tests from shell-level env vars.

The streaming voice engine selector (MS4_VOICE_TTS_ENGINE) defaults to
the REST path for backward compatibility. If an operator sets the env
to ws_super for their own gateway, that var is inherited by every
subsequent pytest run from the same shell — and the REST-engine tests
in test_voice_stream.py will silently route through the WS engine
factory instead of their fake REST one, then fail with confusing
errors about missing parallelism / chunker behavior.

Strip the relevant env vars at the start of every voice test so the
tests always reflect the code's defaults, not the shell's state.
"""

from __future__ import annotations

import os

import pytest


_ENV_VARS_TO_ISOLATE = (
    "MS4_VOICE_TTS_ENGINE",
    "MS4_VOICE_FIRST_CHUNK_MIN_WORDS",
    "MS4_VOICE_MAX_CHUNK_WORDS",
    "MS4_VOICE_TTS_POOL_SIZE",
    "MS4_VOICE_TTS_MODEL",
    "MS4_VOICE_TTS_VOICE",
    "MS4_VOICE_TTS_FORMAT",
    "MS4_VOICE_ASR_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_voice_env(monkeypatch):
    for name in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(name, raising=False)
    # Force the in-module DEFAULT_ENGINE constant to the canonical default
    # in case the module was imported BEFORE we cleared the env. The voice
    # module reads MS4_VOICE_TTS_ENGINE once at import; if a stale value
    # was captured we'd still need to override it for the duration of the
    # test session.
    import machine_spirit_4.gateway.voice as voice_module
    monkeypatch.setattr(voice_module, "DEFAULT_ENGINE", "rest")
    yield
