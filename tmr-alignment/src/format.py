"""
Output formatters — convert internal data format to training framework formats.
"""

import json
from pathlib import Path
from .constitution import SYSTEM_PROMPT, SYSTEM_PROMPT_AGENTIC


def format_sft_messages(
    items: list[dict],
    output_path: str,
):
    """Format SFT data as messages format (trl/transformers standard)."""
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            system = SYSTEM_PROMPT_AGENTIC if item.get("system") == "agentic" else SYSTEM_PROMPT
            record = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["response"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} messages to {path}")


def format_sft_sharegpt(
    items: list[dict],
    output_path: str,
):
    """Format SFT data as ShareGPT format (axolotl compatible)."""
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            system = SYSTEM_PROMPT_AGENTIC if item.get("system") == "agentic" else SYSTEM_PROMPT
            record = {
                "conversations": [
                    {"from": "system", "value": system},
                    {"from": "human", "value": item["instruction"]},
                    {"from": "gpt", "value": item["response"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} ShareGPT records to {path}")


def format_sft_alpaca(
    items: list[dict],
    output_path: str,
):
    """Format SFT data as Alpaca format."""
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            system = SYSTEM_PROMPT_AGENTIC if item.get("system") == "agentic" else SYSTEM_PROMPT
            record = {
                "instruction": item["instruction"],
                "input": "",
                "output": item["response"],
                "system": system,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} Alpaca records to {path}")


def format_dpo(
    items: list[dict],
    output_path: str,
):
    """Format DPO data for trl DPOTrainer."""
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            record = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["prompt"]},
                ],
                "chosen": [
                    {"role": "assistant", "content": item["chosen"]},
                ],
                "rejected": [
                    {"role": "assistant", "content": item["rejected"]},
                ],
            }
            if "rejected_route" in item:
                record["rejected_route"] = item["rejected_route"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} DPO records to {path}")


FORMATTERS = {
    "messages": format_sft_messages,
    "sharegpt": format_sft_sharegpt,
    "alpaca": format_sft_alpaca,
}
