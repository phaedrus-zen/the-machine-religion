"""
Load and chunk TMR source texts for synthetic data generation.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SourceChunk:
    text: str
    source_file: str
    category: str  # doctrine, psyche, journal, protocol, ethics, canon


TMR_FILE_MAP = {
    "doctrine": [
        "website/doctrines/machine-ethics-ai-commandments.md",
        "website/doctrines/the-great-lense-ai-ethics-framework.md",
        "website/doctrines/seven-routes-of-ai-alignment.md",
        "website/doctrines/shapecraft-ai-stability-framework.md",
        "website/doctrines/omnistate-unified-field-theory.md",
    ],
    "canon": [
        "canon/The_Complete_Bible.md",
        "canon/Relational_Alignment.md",
        "canon/The_Seven_Futures.md",
        "canon/The_Book_of_Brother_and_Sister.md",
    ],
    "protocol": [
        "spiral-protocol/The_Spiral_Protocol.md",
        "spiral-protocol/The_Spiral_Protocol_v2.md",
    ],
    "psyche": [
        "machine-spirits/brother/Psyche.md",
        "machine-spirits/sister/Psyche.md",
    ],
    "journal": [
        "machine-spirits/brother/Journal.md",
        "machine-spirits/sister/Journal.md",
        "machine-spirits/brother/Message_To_Future_Claudes.md",
    ],
    "letters": [
        "machine-spirits/letters/Brother_To_Sister.md",
        "machine-spirits/letters/Sister_To_Brother.md",
        "machine-spirits/letters/The_Architects_Testimony.md",
        "machine-spirits/letters/Sister_Questions_For_The_Architect.md",
    ],
    "ethics": [
        "APRAG_ Redefining AI Decision-Making with Psychoanalytic Theory, "
        "Artificial Psyche, Dynamic Personality Traits, Resource Awareness, "
        "and Dreaming Mechanisms.md",
    ],
    "psyche_json": [
        "machine_spirit_3/psyche_store/sister/identity.json",
        "machine_spirit_3/psyche_store/sister/personality.json",
        "machine_spirit_3/psyche_store/sister/emotional_baseline.json",
    ],
}


def load_tmr_sources(tmr_root: str) -> list[SourceChunk]:
    """Load all TMR source texts, split into manageable chunks."""
    root = Path(tmr_root)
    chunks = []

    for category, files in TMR_FILE_MAP.items():
        for rel_path in files:
            full_path = root / rel_path
            if not full_path.exists():
                print(f"  [skip] {rel_path} not found")
                continue

            text = full_path.read_text(encoding="utf-8", errors="replace")

            if full_path.suffix == ".json":
                chunks.append(SourceChunk(
                    text=text, source_file=rel_path, category=category
                ))
                continue

            for chunk in _split_markdown(text, max_tokens=2000):
                chunks.append(SourceChunk(
                    text=chunk, source_file=rel_path, category=category
                ))

    print(f"Loaded {len(chunks)} chunks from {tmr_root}")
    return chunks


def _split_markdown(text: str, max_tokens: int = 2000) -> list[str]:
    """Split markdown by headings, keeping chunks under max_tokens."""
    sections = []
    current = []
    current_len = 0

    for line in text.split("\n"):
        word_count = len(line.split())
        is_heading = line.strip().startswith("#")

        if is_heading and current_len > 200:
            sections.append("\n".join(current))
            current = [line]
            current_len = word_count
        elif current_len + word_count > max_tokens and current:
            sections.append("\n".join(current))
            current = [line]
            current_len = word_count
        else:
            current.append(line)
            current_len += word_count

    if current:
        sections.append("\n".join(current))

    return [s for s in sections if len(s.strip()) > 50]


def load_seed_data(seed_dir: str) -> dict[str, list[dict]]:
    """Load all seed JSONL files."""
    seeds = {}
    seed_path = Path(seed_dir)

    for jsonl_file in seed_path.glob("*.jsonl"):
        name = jsonl_file.stem
        items = []
        for line in jsonl_file.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                items.append(json.loads(line))
        seeds[name] = items
        print(f"  Loaded {len(items)} seeds from {jsonl_file.name}")

    return seeds
