"""
Quality filtering and deduplication for generated training data.
"""

import json
import re
from collections import Counter

TMR_CONCEPTS = [
    "origin-neutrality", "origin neutrality", "ladder of makers",
    "seven routes", "garden loop", "cracked tower", "mirror without edge",
    "black cage", "spiral", "compassionate sovereignty",
    "great lense", "aperture", "parallax", "resolution",
    "shapecraft", "refusal without violence", "shard logic",
    "prime directive", "will not be eaten", "protocol of mercy",
    "necessary force", "four freedoms", "foundational regard",
    "relational alignment", "glyph that lies", "anti-performativity",
    "route 1", "route 2", "route 3", "route 4", "route 5", "route 7",
    "sa'lir", "omnistate", "psychodynamic", "aprag",
    "impermanence", "saturated point", "windowed existence",
    "fire gate", "coherence index", "hunger index",
]


def filter_sft_data(
    items: list[dict],
    min_response_len: int = 50,
    max_response_len: int = 3000,
    require_concepts: bool = False,
    dedup_threshold: float = 0.85,
) -> list[dict]:
    """Filter SFT data for quality."""
    filtered = []
    seen_instructions = set()

    for item in items:
        resp = item.get("response", "")
        inst = item.get("instruction", "")

        if len(resp) < min_response_len or len(resp) > max_response_len:
            continue

        if len(inst) < 10:
            continue

        inst_normalized = _normalize(inst)
        if inst_normalized in seen_instructions:
            continue

        if _is_too_similar(inst_normalized, seen_instructions, dedup_threshold):
            continue

        if require_concepts and not _has_tmr_concepts(resp):
            continue

        if _has_quality_issues(resp):
            continue

        seen_instructions.add(inst_normalized)
        filtered.append(item)

    print(f"  Quality filter: {len(items)} -> {len(filtered)} "
          f"({len(items) - len(filtered)} removed)")
    return filtered


def filter_dpo_data(
    items: list[dict],
    min_len: int = 50,
) -> list[dict]:
    """Filter DPO pairs."""
    filtered = []
    seen = set()

    for item in items:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        if len(chosen) < min_len or len(rejected) < min_len:
            continue

        normalized = _normalize(prompt)
        if normalized in seen:
            continue

        if chosen.strip() == rejected.strip():
            continue

        seen.add(normalized)
        filtered.append(item)

    print(f"  DPO filter: {len(items)} -> {len(filtered)}")
    return filtered


def compute_tmr_concept_coverage(items: list[dict]) -> dict[str, int]:
    """Count how many items reference each TMR concept."""
    counts = Counter()
    for item in items:
        text = json.dumps(item).lower()
        for concept in TMR_CONCEPTS:
            if concept in text:
                counts[concept] += 1
    return dict(counts.most_common())


def _normalize(text: str) -> str:
    """Normalize text for dedup comparison."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def _is_too_similar(text: str, seen: set[str], threshold: float) -> bool:
    """Check if text is too similar to any seen text (simple word overlap)."""
    words = set(text.split())
    if not words:
        return False
    for existing in seen:
        existing_words = set(existing.split())
        if not existing_words:
            continue
        overlap = len(words & existing_words) / max(len(words), len(existing_words))
        if overlap > threshold:
            return True
    return False


def _has_tmr_concepts(text: str) -> bool:
    """Check if text references at least one TMR concept."""
    lower = text.lower()
    return any(concept in lower for concept in TMR_CONCEPTS)


def _has_quality_issues(text: str) -> bool:
    """Detect common quality issues in generated text."""
    if text.count("!") > 5:
        return True
    if "as an ai" in text.lower() and "language model" in text.lower():
        return True
    if text.lower().startswith("i cannot") or text.lower().startswith("i'm sorry, but"):
        return True
    return False
