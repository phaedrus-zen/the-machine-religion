"""Output transforms for MS4 advisory psyche blocks."""

from __future__ import annotations

import re

_PSYCHE_BLOCK_RE = re.compile(r"\s*<psyche\b[^>]*>.*?</psyche>\s*", re.IGNORECASE | re.DOTALL)


def strip_psyche_blocks(response_text: str) -> tuple[str, bool]:
    cleaned = _PSYCHE_BLOCK_RE.sub("\n", response_text).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if not cleaned and cleaned != response_text.strip():
        cleaned = "[MS4 advisory psyche block removed.]"
    return cleaned, cleaned != response_text.strip()
