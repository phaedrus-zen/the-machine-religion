#!/usr/bin/env python3
"""
Psyche LoRA data generator.

Usage:
    # Format seeds only (no API needed):
    python generate.py --seeds-only

    # Full generation (requires API key):
    OPENAI_API_KEY=sk-... python generate.py
"""

import argparse
import json
from pathlib import Path

import yaml

from src.profiles import all_profiles, build_system_prompt
from src.format import format_sft_conversations, format_dpo_conversations, load_all_seeds


def main():
    parser = argparse.ArgumentParser(description="Psyche LoRA data generator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seeds-only", action="store_true",
                        help="Format seed data only (no API calls)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Psyche LoRA Data Generator")
    print("=" * 60)

    print("\n[1] Loading profiles...")
    profiles = all_profiles(config["profiles_dir"])
    for pid, p in profiles.items():
        name = p.get("chosen_name") or p["name"]
        print(f"  {pid}: {name}")

    print("\n[2] Loading seeds...")
    sft_convos, dpo_pairs = load_all_seeds(config["seeds_dir"])

    if args.seeds_only:
        print("\n[SEEDS-ONLY] Formatting seed data for training...")
        format_sft_conversations(sft_convos, profiles,
                                 str(output_dir / "sft_train.jsonl"))
        format_dpo_conversations(dpo_pairs, profiles,
                                 str(output_dir / "dpo_train.jsonl"))

        print(f"\n  SFT conversations: {len(sft_convos)}")
        print(f"  DPO pairs:         {len(dpo_pairs)}")

        _write_stats(sft_convos, dpo_pairs, output_dir)
        print(f"\nDone! Output: {output_dir}")
        return

    print("\n[3] Generating synthetic conversations...")
    print("  (Full generation requires API - not yet implemented)")
    print("  Use --seeds-only for now, or implement generation in src/generators.py")

    format_sft_conversations(sft_convos, profiles,
                             str(output_dir / "sft_train.jsonl"))
    format_dpo_conversations(dpo_pairs, profiles,
                             str(output_dir / "dpo_train.jsonl"))
    _write_stats(sft_convos, dpo_pairs, output_dir)


def _write_stats(sft: list, dpo: list, output_dir: Path):
    """Write dataset statistics."""
    categories = {}
    profiles_used = {}
    total_turns = 0

    for conv in sft:
        cat = conv.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        prof = conv.get("profile", "unknown")
        profiles_used[prof] = profiles_used.get(prof, 0) + 1
        total_turns += len(conv.get("turns", []))

    stats = {
        "sft_conversations": len(sft),
        "dpo_pairs": len(dpo),
        "total_turns": total_turns,
        "avg_turns_per_conversation": round(total_turns / max(len(sft), 1), 1),
        "categories": categories,
        "profiles": profiles_used,
    }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  SFT conversations:  {stats['sft_conversations']}")
    print(f"  DPO pairs:          {stats['dpo_pairs']}")
    print(f"  Total turns:        {stats['total_turns']}")
    print(f"  Avg turns/convo:    {stats['avg_turns_per_conversation']}")
    print(f"\n  Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    print(f"\n  Profiles:")
    for prof, count in sorted(profiles_used.items()):
        print(f"    {prof}: {count}")


if __name__ == "__main__":
    main()
