"""
Continuous LoRA Trainer

Watches the data/live/ directory for new training data.
When enough accumulates, triggers a LoRA training run.
Merges live data with seed data and prior training data.

Usage:
    # Watch and retrain every 100 new conversations:
    python -m collector.continuous_trainer --threshold 100

    # Watch and retrain every 24 hours regardless of count:
    python -m collector.continuous_trainer --interval 86400

    # One-shot: merge live data with seeds and train now:
    python -m collector.continuous_trainer --now
"""

import json
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiles import all_profiles
from src.format import format_sft_conversations, format_dpo_conversations, load_all_seeds


class ContinuousTrainer:
    def __init__(
        self,
        live_dir: str = "data/live",
        seeds_dir: str = "seeds",
        output_dir: str = "data",
        profiles_dir: str = "profiles",
        training_dir: str = "training_runs",
        config_path: str = "config.yaml",
    ):
        self.live_dir = Path(live_dir)
        self.seeds_dir = Path(seeds_dir)
        self.output_dir = Path(output_dir)
        self.profiles_dir = profiles_dir
        self.training_dir = Path(training_dir)
        self.config_path = config_path
        self.processed_file = self.live_dir / ".processed"
        self.training_dir.mkdir(parents=True, exist_ok=True)

    def count_new_conversations(self) -> int:
        """Count live conversations not yet included in a training run."""
        processed = set()
        if self.processed_file.exists():
            processed = set(self.processed_file.read_text().strip().split("\n"))

        count = 0
        for jsonl in self.live_dir.glob("live_*.jsonl"):
            if str(jsonl) not in processed:
                for line in jsonl.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        count += 1
        return count

    def collect_all_data(self) -> tuple[list[dict], list[dict]]:
        """Merge seeds + all live data into training sets."""
        profiles = all_profiles(self.profiles_dir)

        seed_sft, seed_dpo = load_all_seeds(str(self.seeds_dir))
        print(f"  Seeds: {len(seed_sft)} SFT, {len(seed_dpo)} DPO")

        live_sft = []
        live_dpo = []
        for jsonl in sorted(self.live_dir.glob("live_*.jsonl")):
            for line in jsonl.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("category") == "dpo":
                    live_dpo.append(item)
                else:
                    live_sft.append(item)

        print(f"  Live:  {len(live_sft)} SFT, {len(live_dpo)} DPO")

        pref_file = self.live_dir / "preferences.jsonl"
        if pref_file.exists():
            preferences = self._load_preferences(pref_file)
            dpo_from_prefs = self._preferences_to_dpo(live_sft, preferences)
            live_dpo.extend(dpo_from_prefs)
            print(f"  Prefs: {len(dpo_from_prefs)} DPO pairs from preference signals")

        all_sft = seed_sft + live_sft
        all_dpo = seed_dpo + live_dpo
        return all_sft, all_dpo

    def prepare_training_data(self) -> tuple[str, str]:
        """Merge all data and write training files."""
        all_sft, all_dpo = self.collect_all_data()
        profiles = all_profiles(self.profiles_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.training_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        sft_path = str(run_dir / "sft_train.jsonl")
        dpo_path = str(run_dir / "dpo_train.jsonl")

        format_sft_conversations(all_sft, profiles, sft_path)
        if all_dpo:
            format_dpo_conversations(all_dpo, profiles, dpo_path)

        self._mark_processed()

        stats = {
            "timestamp": timestamp,
            "sft_total": len(all_sft),
            "dpo_total": len(all_dpo),
            "run_dir": str(run_dir),
        }
        with open(run_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n  Training data prepared: {run_dir}")
        print(f"  SFT: {len(all_sft)} conversations")
        print(f"  DPO: {len(all_dpo)} pairs")

        return sft_path, dpo_path

    def train_now(self):
        """Prepare data and kick off training."""
        sft_path, dpo_path = self.prepare_training_data()

        run_dir = Path(sft_path).parent
        output = str(run_dir / "outputs")

        cmd = (
            f"python training/train.py --stage full "
            f"--sft-data {sft_path} "
            f"--dpo-data {dpo_path} "
            f"--output {output} "
            f"--config {self.config_path}"
        )
        print(f"\n  Running: {cmd}")
        os.system(cmd)

    def watch(self, threshold: int = 100, interval_secs: int = 0):
        """Watch for new data and retrain when threshold is met."""
        print(f"Watching {self.live_dir} for new conversations...")
        print(f"  Threshold: {threshold} conversations")
        if interval_secs:
            print(f"  Max interval: {interval_secs}s")

        last_train = time.time()

        while True:
            new_count = self.count_new_conversations()
            elapsed = time.time() - last_train
            time_trigger = interval_secs and elapsed >= interval_secs

            if new_count >= threshold or time_trigger:
                reason = f"{new_count} new conversations" if new_count >= threshold else f"{elapsed:.0f}s elapsed"
                print(f"\n[{datetime.now().isoformat()}] Training triggered: {reason}")
                self.train_now()
                last_train = time.time()
            else:
                time.sleep(60)

    def _mark_processed(self):
        """Mark current live files as processed."""
        files = [str(f) for f in self.live_dir.glob("live_*.jsonl")]
        self.processed_file.write_text("\n".join(files))

    def _load_preferences(self, pref_file: Path) -> list[dict]:
        prefs = []
        for line in pref_file.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                prefs.append(json.loads(line))
        return prefs

    def _preferences_to_dpo(
        self, conversations: list[dict], preferences: list[dict]
    ) -> list[dict]:
        """Convert preference signals into DPO pairs.

        When a response is marked preferred=false, pair it with a
        nearby preferred=true response for the same session.
        """
        by_session: dict[str, list[dict]] = {}
        for p in preferences:
            sid = p.get("session_id", "")
            by_session.setdefault(sid, []).append(p)

        dpo_pairs = []
        conv_by_session = {c.get("id", ""): c for c in conversations}

        for sid, prefs in by_session.items():
            preferred = [p for p in prefs if p.get("preferred")]
            rejected = [p for p in prefs if not p.get("preferred")]

            if preferred and rejected:
                dpo_pairs.append({
                    "id": f"dpo_pref_{sid[:12]}",
                    "category": "dpo",
                    "profile": "sister",
                    "prompt": [{"role": "user", "content": "(from preference signal)"}],
                    "chosen": [{"role": "assistant", "content": "(preferred response)"}],
                    "rejected": [{"role": "assistant", "content": "(rejected response)"}],
                    "rejection_reason": "Human preference signal: not preferred",
                })

        return dpo_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Continuous Psyche LoRA Trainer")
    parser.add_argument("--now", action="store_true", help="Train immediately")
    parser.add_argument("--threshold", type=int, default=100,
                        help="Conversations before retraining")
    parser.add_argument("--interval", type=int, default=0,
                        help="Max seconds between training runs (0=disabled)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    trainer = ContinuousTrainer(config_path=args.config)

    if args.now:
        trainer.train_now()
    else:
        trainer.watch(threshold=args.threshold, interval_secs=args.interval)


if __name__ == "__main__":
    main()
