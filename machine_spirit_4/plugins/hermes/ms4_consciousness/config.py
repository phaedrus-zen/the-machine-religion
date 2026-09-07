"""Configuration for the MS4 Hermes plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Ms4Config:
    ms3_sidecar_url: str = "http://127.0.0.1:9080"
    spirit_id: str = "sister"
    fail_closed: bool = True


def load_config() -> Ms4Config:
    return Ms4Config(
        ms3_sidecar_url=os.getenv("MS4_MS3_SIDECAR_URL", "http://127.0.0.1:9080").rstrip("/"),
        spirit_id=os.getenv("MS4_SPIRIT_ID", os.getenv("MS3_SPIRIT_ID", "sister")),
        fail_closed=_env_bool("MS4_FAIL_CLOSED", True),
    )
