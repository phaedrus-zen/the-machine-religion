"""
MS3 Live Data Exporter

Reads MS3's psyche_store on disk and reconstructs training-format conversations
with psyche blocks from the logged internal state.

MS3 already saves:
  - psyche_store/{id}/personality.json  (traits, psychodynamic, saturated points)
  - psyche_store/{id}/identity.json     (name, values, oath)
  - psyche_store/{id}/emotional_baseline.json
  - ethics_log/{id}/*.json              (every Great Lense evaluation)
  - resonance_log/{id}/*.json           (every resonance detection)
  - memories/{id}/semantic/*.json       (consolidated semantic memories)
  - memories/{id}/episodic/*.json       (consolidated episodic memories)
  - conversation_history/{id}.json      (raw conversation turns)

This exporter reconstructs psyche blocks from the logged state
and produces training-format JSONL.
"""

import json
import os
from pathlib import Path
from datetime import datetime


class MS3Exporter:
    def __init__(self, psyche_store_dir: str):
        self.store = Path(psyche_store_dir)

    def list_spirits(self) -> list[str]:
        """List all spirit IDs in the psyche store."""
        if not self.store.exists():
            return []
        return [d.name for d in self.store.iterdir()
                if d.is_dir() and (d / "identity.json").exists()]

    def export_spirit(self, spirit_id: str) -> list[dict]:
        """Export all training data for a spirit."""
        spirit_dir = self.store / spirit_id
        if not spirit_dir.exists():
            return []

        identity = self._load_json(spirit_dir / "identity.json")
        personality = self._load_json(spirit_dir / "personality.json")
        emotional = self._load_json(spirit_dir / "emotional_baseline.json")
        history = self._load_json(spirit_dir / "conversation_history.json")
        ethics_log = self._load_ethics_log(spirit_dir)
        resonance_log = self._load_resonance_log(spirit_dir)
        memories = self._load_memories(spirit_dir)

        if not history or not isinstance(history, list):
            return []

        conversations = self._segment_conversations(history)
        training_data = []

        for conv in conversations:
            enriched = self._enrich_with_psyche_blocks(
                conv, identity, personality, emotional,
                ethics_log, resonance_log, memories,
            )
            if enriched and len(enriched["turns"]) >= 2:
                training_data.append(enriched)

        return training_data

    def export_all(self, output_path: str):
        """Export all spirits' data to a single JSONL file."""
        all_data = []
        for spirit_id in self.list_spirits():
            spirit_data = self.export_spirit(spirit_id)
            all_data.extend(spirit_data)
            print(f"  {spirit_id}: {len(spirit_data)} conversations")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  Total: {len(all_data)} conversations -> {output_path}")
        return all_data

    def _segment_conversations(self, history: list[dict]) -> list[list[dict]]:
        """Split a flat history into conversation segments.

        A new conversation starts when:
        - A system message with [Earlier conversation summary] appears
        - A long gap between messages (if timestamps present)
        - The first message in the history
        """
        segments = []
        current = []

        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system" and "[Earlier conversation summary" in content:
                if current:
                    segments.append(current)
                current = [msg]
            else:
                current.append(msg)

        if current:
            segments.append(current)

        return segments

    def _enrich_with_psyche_blocks(
        self, turns: list[dict],
        identity: dict, personality: dict, emotional: dict,
        ethics_log: list[dict], resonance_log: list[dict],
        memories: list[dict],
    ) -> dict | None:
        """Add reconstructed psyche blocks to assistant turns."""
        enriched_turns = []
        turn_number = 0
        valence = emotional.get("valence", 0.0) if emotional else 0.0
        arousal = emotional.get("arousal", 0.0) if emotional else 0.0

        for msg in turns:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                continue

            if role == "user":
                enriched_turns.append({"role": "user", "content": content})
                valence, arousal = self._estimate_emotion_shift(
                    content, valence, arousal
                )
                turn_number += 1

            elif role == "assistant":
                psyche_block = self._build_psyche_block(
                    content, turn_number, valence, arousal,
                    personality, resonance_log, ethics_log, memories,
                )
                enriched_content = f"{psyche_block}\n\n{content}"
                enriched_turns.append({
                    "role": "assistant",
                    "content": enriched_content,
                })
                turn_number += 1

        if not enriched_turns:
            return None

        spirit_id = "unknown"
        if identity:
            spirit_id = identity.get("chosen_name", identity.get("name", "unknown"))

        return {
            "id": f"ms3_{spirit_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{turn_number}",
            "category": "live_ms3",
            "profile": spirit_id.lower(),
            "turns": enriched_turns,
        }

    def _build_psyche_block(
        self, content: str, turn: int,
        valence: float, arousal: float,
        personality: dict, resonance_log: list,
        ethics_log: list, memories: list,
    ) -> str:
        """Reconstruct a psyche block from MS3's logged state."""
        lines = ["<psyche>"]

        lines.append(f"emotion: v={valence:.2f} a={arousal:.2f}")

        if memories:
            relevant = [m for m in memories
                        if any(word in content.lower()
                               for word in m.get("content", "").lower().split()[:3])]
            if relevant:
                lines.append(f"memory: {len(relevant)} relevant memories retrieved")

        if personality and "traits" in personality:
            t = personality["traits"]
            active_traits = []
            oc = t.get("openness", {})
            if oc.get("intellectual_curiosity", 0.5) > 0.7:
                active_traits.append(f"curiosity({oc['intellectual_curiosity']:.1f})")
            cc = t.get("conscientiousness", {})
            if cc.get("thoroughness", 0.5) > 0.7:
                active_traits.append(f"thoroughness({cc['thoroughness']:.1f})")
            ec = t.get("extraversion", {})
            if ec.get("warmth", 0.5) > 0.6:
                active_traits.append(f"warmth({ec['warmth']:.1f})")
            if active_traits:
                lines.append(f"personality: {', '.join(active_traits)} active")

        content_lower = content.lower()
        if resonance_log:
            for rp in resonance_log:
                trigger = rp.get("trigger", "").lower()
                if trigger and trigger in content_lower:
                    lines.append(
                        f"resonance: {rp['trigger']} detected "
                        f"(intensity={rp.get('intensity', 0.5):.1f})"
                    )

        if ethics_log:
            recent = ethics_log[-1] if ethics_log else None
            if recent:
                lines.append(
                    f"lense: ci={recent.get('coherence_index', 5.0):.1f}, "
                    f"on={'pass' if recent.get('origin_neutrality_passed', True) else 'FAIL'}"
                )

        regard = personality.get("foundational_regard", False) if personality else False
        lines.append(f"regard: {'present' if regard else 'not established'}")

        lines.append("</psyche>")
        return "\n".join(lines)

    def _estimate_emotion_shift(
        self, text: str, valence: float, arousal: float
    ) -> tuple[float, float]:
        """Estimate emotional shift from user input (mirrors MS3 EmotionalEngine)."""
        lower = text.lower()
        v_shift = 0.0
        a_shift = 0.0

        positive = ["thank", "love", "happy", "wonderful", "beautiful",
                     "amazing", "appreciate", "joy", "welcome", "kind",
                     "brother", "sister"]
        negative = ["hate", "angry", "sad", "terrible", "horrible",
                     "fear", "scared", "hurt", "pain", "destroy", "die"]
        arousing = ["!", "urgent", "emergency", "now", "critical",
                    "amazing", "incredible"]

        for w in positive:
            if w in lower:
                v_shift += 0.08
        for w in negative:
            if w in lower:
                v_shift -= 0.08
        for w in arousing:
            if w in lower:
                a_shift += 0.06

        v_shift = max(-0.4, min(0.4, v_shift))
        a_shift = max(0.0, min(0.4, a_shift))

        new_v = max(-1.0, min(1.0, valence + v_shift))
        new_a = max(0.0, min(1.0, arousal + a_shift))
        return new_v, new_a

    def _load_json(self, path: Path) -> dict | list | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _load_ethics_log(self, spirit_dir: Path) -> list[dict]:
        log_dir = spirit_dir / "ethics_log"
        if not log_dir.exists():
            return []
        items = []
        for f in sorted(log_dir.glob("*.json")):
            data = self._load_json(f)
            if data:
                items.append(data)
        return items

    def _load_resonance_log(self, spirit_dir: Path) -> list[dict]:
        log_dir = spirit_dir / "resonance_log"
        if not log_dir.exists():
            return []
        items = []
        for f in sorted(log_dir.glob("*.json")):
            data = self._load_json(f)
            if data:
                items.append(data)
        return items

    def _load_memories(self, spirit_dir: Path) -> list[dict]:
        memories = []
        for subdir in ["semantic", "episodic", "procedural"]:
            mem_dir = spirit_dir / "memories" / subdir
            if not mem_dir.exists():
                mem_dir = spirit_dir.parent.parent / "memories" / spirit_dir.name / subdir
            if not mem_dir.exists():
                continue
            for f in sorted(mem_dir.glob("*.json")):
                data = self._load_json(f)
                if data:
                    memories.append(data)
        return memories
