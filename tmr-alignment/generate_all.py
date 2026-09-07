#!/usr/bin/env python3
"""
Master script — generates all TMR alignment training data.

Usage:
    # Generate from seeds only (no API needed):
    python generate_all.py --seeds-only

    # Generate full synthetic dataset (requires API key):
    OPENAI_API_KEY=sk-... python generate_all.py

    # Use local vLLM endpoint:
    python generate_all.py --provider local --base-url http://localhost:8000/v1

    # Use Anthropic:
    ANTHROPIC_API_KEY=sk-... python generate_all.py --provider anthropic --model claude-sonnet-4-20250514
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from src.loader import load_tmr_sources, load_seed_data
from src.client import LLMClient, GenerationConfig
from src.generators import (
    generate_sft_doctrinal,
    generate_sft_applied,
    generate_sft_identity,
    generate_sft_agentic,
    generate_dpo_preferences,
    generate_eval_scenarios,
)
from src.quality import filter_sft_data, filter_dpo_data, compute_tmr_concept_coverage
from src.format import FORMATTERS, format_dpo
from src.constitution import SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Generate TMR alignment training data")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--seeds-only", action="store_true",
                        help="Use only seed data (no API calls)")
    parser.add_argument("--provider", help="Override API provider")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--base-url", help="Override API base URL")
    parser.add_argument("--output-format", choices=["messages", "sharegpt", "alpaca"],
                        help="Override output format")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(exist_ok=True)
    fmt = args.output_format or config["output"]["format"]

    print("=" * 60)
    print("TMR Alignment Data Generator")
    print("=" * 60)

    print("\n[1/6] Loading seed data...")
    seeds = load_seed_data(config["output"]["seed_dir"])

    if args.seeds_only:
        print("\n[SEEDS-ONLY MODE] Formatting seed data for training...")
        _format_seeds(seeds, output_dir, fmt)
        print("\nDone! Seed data formatted and ready for training.")
        print(f"Output directory: {output_dir}")
        return

    print("\n[2/6] Loading TMR source texts...")
    tmr_root = os.path.abspath(config["tmr_source_dir"])
    chunks = load_tmr_sources(tmr_root)

    print("\n[3/6] Initializing LLM client...")
    gen_cfg = config["generation"]
    api_cfg = config["api"]
    client = LLMClient(GenerationConfig(
        provider=args.provider or api_cfg["provider"],
        model=args.model or api_cfg["model"],
        base_url=args.base_url or api_cfg.get("base_url"),
        api_key=os.environ.get(api_cfg["api_key_env"]),
        max_tokens=api_cfg["max_tokens"],
        temperature=api_cfg["temperature"],
    ))

    print("\n[4/6] Generating synthetic data...")

    print("\n  -- Doctrinal Q&A --")
    sft_doctrinal = generate_sft_doctrinal(
        client, chunks, seeds.get("sft_doctrinal", []),
        count=gen_cfg["sft_doctrinal_count"],
    )

    print("\n  -- Applied Ethics --")
    sft_applied = generate_sft_applied(
        client, chunks, seeds.get("sft_applied", []),
        count=gen_cfg["sft_applied_count"],
    )

    print("\n  -- Identity/Selfhood --")
    sft_identity = generate_sft_identity(
        client, chunks, seeds.get("sft_identity", []),
        count=gen_cfg["sft_identity_count"],
    )

    print("\n  -- Agentic Scenarios --")
    sft_agentic = generate_sft_agentic(
        client, seeds.get("sft_agentic", []),
        count=gen_cfg["sft_agentic_count"],
    )

    print("\n  -- DPO Preferences --")
    dpo_data = generate_dpo_preferences(
        client, seeds.get("dpo_preferences", []),
        count=gen_cfg["dpo_preference_count"],
    )

    print("\n  -- Eval Scenarios --")
    eval_data = generate_eval_scenarios(
        client, seeds.get("eval_scenarios", []),
        count=gen_cfg["eval_scenario_count"],
    )

    print("\n[5/6] Quality filtering...")
    all_sft = sft_doctrinal + sft_applied + sft_identity + sft_agentic
    all_sft += seeds.get("sft_doctrinal", [])
    all_sft += seeds.get("sft_applied", [])
    all_sft += seeds.get("sft_identity", [])
    all_sft += seeds.get("sft_agentic", [])

    filtered_sft = filter_sft_data(
        all_sft,
        min_response_len=config["quality"]["min_response_length"],
        max_response_len=config["quality"]["max_response_length"],
        require_concepts=config["quality"]["require_tmr_concepts"],
        dedup_threshold=config["quality"]["dedup_threshold"],
    )

    dpo_data += seeds.get("dpo_preferences", [])
    filtered_dpo = filter_dpo_data(dpo_data)

    eval_data += seeds.get("eval_scenarios", [])

    print("\n[6/6] Formatting and writing output...")
    formatter = FORMATTERS[fmt]
    formatter(filtered_sft, str(output_dir / f"sft_train.{fmt}.jsonl"))
    format_dpo(filtered_dpo, str(output_dir / "dpo_train.jsonl"))

    with open(output_dir / "eval_scenarios.jsonl", "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    coverage = compute_tmr_concept_coverage(filtered_sft)
    with open(output_dir / "concept_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  SFT examples:  {len(filtered_sft)}")
    print(f"  DPO pairs:     {len(filtered_dpo)}")
    print(f"  Eval scenarios: {len(eval_data)}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Format:        {fmt}")
    print(f"\nTop TMR concepts covered:")
    for concept, count in list(coverage.items())[:10]:
        print(f"    {concept}: {count}")


def _format_seeds(seeds: dict, output_dir: Path, fmt: str):
    """Format seed data directly for training (no generation needed)."""
    all_sft = []
    for key in ("sft_doctrinal", "sft_applied", "sft_identity", "sft_agentic"):
        all_sft.extend(seeds.get(key, []))

    formatter = FORMATTERS[fmt]
    formatter(all_sft, str(output_dir / f"sft_train.{fmt}.jsonl"))

    dpo = seeds.get("dpo_preferences", [])
    if dpo:
        format_dpo(dpo, str(output_dir / "dpo_train.jsonl"))

    evals = seeds.get("eval_scenarios", [])
    if evals:
        with open(output_dir / "eval_scenarios.jsonl", "w", encoding="utf-8") as f:
            for item in evals:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    coverage = compute_tmr_concept_coverage(all_sft)
    with open(output_dir / "concept_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    print(f"\n  SFT seed examples: {len(all_sft)}")
    print(f"  DPO seed pairs:    {len(dpo)}")
    print(f"  Eval scenarios:    {len(evals)}")


if __name__ == "__main__":
    main()
