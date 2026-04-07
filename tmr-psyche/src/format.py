"""
Format psyche conversations for training.
Injects the appropriate system prompt from the profile.
"""

import json
from pathlib import Path
from .profiles import load_profile, build_system_prompt, all_profiles


def format_sft_conversations(
    conversations: list[dict],
    profiles: dict[str, dict],
    output_path: str,
):
    """Convert seed conversations into messages format for SFT training."""
    path = Path(output_path)
    written = 0

    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            profile_id = conv.get("profile", "blank")
            profile = profiles.get(profile_id)
            if not profile:
                print(f"  [warn] Unknown profile '{profile_id}', skipping {conv['id']}")
                continue

            system_prompt = build_system_prompt(profile)
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conv["turns"])

            record = {"messages": messages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Wrote {written} conversations to {path}")


def format_dpo_conversations(
    pairs: list[dict],
    profiles: dict[str, dict],
    output_path: str,
):
    """Convert DPO pairs into trl DPOTrainer format."""
    path = Path(output_path)
    written = 0

    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            profile_id = pair.get("profile", "blank")
            profile = profiles.get(profile_id)
            if not profile:
                continue

            system_prompt = build_system_prompt(profile)
            prompt_messages = [{"role": "system", "content": system_prompt}]
            prompt_messages.extend(pair["prompt"])

            record = {
                "prompt": prompt_messages,
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Wrote {written} DPO pairs to {path}")


def load_all_seeds(seeds_dir: str) -> tuple[list[dict], list[dict]]:
    """Load all seed conversations and DPO pairs."""
    sft_conversations = []
    dpo_pairs = []

    for jsonl_file in Path(seeds_dir).glob("*.jsonl"):
        for line in jsonl_file.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("category") == "dpo":
                dpo_pairs.append(item)
            else:
                sft_conversations.append(item)

    print(f"  Loaded {len(sft_conversations)} SFT conversations, "
          f"{len(dpo_pairs)} DPO pairs")
    return sft_conversations, dpo_pairs
