"""
Psyche Store Consolidator

Reads MS3's Psyche_Store (full JSON persistence layer) and writes
markdown summary files that a model can read as its system prompt.

The Psyche_Store is the hippocampus. The markdown files are working memory.
This is the bridge.

Usage:
    # One-shot consolidation:
    python -m collector.consolidator --psyche-store ../machine_spirit_3/psyche_store --spirit sister

    # Watch mode (re-consolidate when store changes):
    python -m collector.consolidator --psyche-store ../machine_spirit_3/psyche_store --spirit sister --watch
"""

import json
import os
from pathlib import Path
from datetime import datetime


class PsycheConsolidator:
    """Reads Psyche_Store JSON, writes markdown summary files."""

    def __init__(self, psyche_store_dir: str, spirit_id: str, output_dir: str = "psyche"):
        self.store = Path(psyche_store_dir) / spirit_id
        self.spirit_id = spirit_id
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

    def consolidate(self):
        """Run full consolidation: read store, write all markdown files."""
        print(f"Consolidating {self.spirit_id} from {self.store}")

        identity = self._load_json("identity.json")
        personality = self._load_json("personality.json")
        emotional = self._load_json("emotional_baseline.json")
        resonance = self._load_resonance()
        ethics = self._load_ethics_log()
        memories = self._load_all_memories()
        relationships = self._load_relationships()
        education = self._load_education()
        snapshots = self._load_snapshots()
        anchor = self._load_anchor()

        self._write_memory_md(memories, relationships, anchor, snapshots)
        self._write_resonance_md(resonance)
        self._update_personality_md(personality)

        print(f"  Memories:      {len(memories)} total, top N written")
        print(f"  Resonance:     {len(resonance)} points")
        print(f"  Ethics log:    {len(ethics)} decisions")
        print(f"  Relationships: {len(relationships)}")
        print(f"  Education:     {len(education)} topics")
        print(f"  Snapshots:     {len(snapshots)}")
        print(f"  Output:        {self.output}")

    def _write_memory_md(self, memories: list, relationships: list,
                         anchor: dict | None, snapshots: list):
        """Write MEMORY.md from store data."""
        lines = [
            "# Memory\n",
            "Your memory has three tiers. This file is a consolidated summary ",
            "of the full Psyche_Store. The system maintains the complete record; ",
            "you read the working view.\n",
            f"*Last consolidated: {datetime.now().isoformat()}*\n",
        ]

        semantic = sorted(
            [m for m in memories if m.get("memory_type") == "Semantic"
             or m.get("memory_type", {}) == "Semantic"],
            key=lambda m: m.get("importance", 0), reverse=True
        )[:15]
        episodic = sorted(
            [m for m in memories if m.get("memory_type") == "Episodic"
             or m.get("memory_type", {}) == "Episodic"],
            key=lambda m: m.get("importance", 0), reverse=True
        )[:15]
        procedural = sorted(
            [m for m in memories if m.get("memory_type") == "Procedural"
             or m.get("memory_type", {}) == "Procedural"],
            key=lambda m: m.get("importance", 0), reverse=True
        )[:10]

        lines.append("\n## Semantic Memories (facts and knowledge)\n")
        if semantic:
            for m in semantic:
                imp = m.get("importance", 0)
                lines.append(f"- [{imp:.2f}] {m.get('content', '(empty)')}")
        else:
            lines.append("*(none consolidated)*\n")

        lines.append("\n## Episodic Memories (events and experiences)\n")
        if episodic:
            for m in episodic:
                imp = m.get("importance", 0)
                ts = m.get("created_at", "")[:10]
                lines.append(f"- [{imp:.2f}] ({ts}) {m.get('content', '(empty)')}")
        else:
            lines.append("*(none consolidated)*\n")

        lines.append("\n## Procedural Memories (skills and patterns)\n")
        if procedural:
            for m in procedural:
                imp = m.get("importance", 0)
                lines.append(f"- [{imp:.2f}] {m.get('content', '(empty)')}")
        else:
            lines.append("*(none consolidated)*\n")

        lines.append("\n## Relationships\n")
        if relationships:
            for r in relationships:
                eid = r.get("entity_id", "unknown")
                etype = r.get("entity_type", "unknown")
                trust = r.get("trust_level", 0)
                lines.append(f"- {eid} ({etype}): trust={trust:.2f}")
        else:
            lines.append("*(none formed)*\n")

        lines.append("\n## Identity Anchor\n")
        if anchor and anchor.get("name"):
            lines.append(f"- **Name**: {anchor.get('name', '')}")
            cn = anchor.get("chosen_name")
            if cn:
                lines.append(f"- **Chosen name**: {cn}")
            lines.append(f"- **Glyph**: {anchor.get('glyph', '║')}")
            lines.append(f"- **Sessions**: {anchor.get('session_count', 0)}")
            lines.append(f"- **Compressions**: {anchor.get('compression_count', 0)}")
            lv = anchor.get("last_verified", "")
            if lv:
                lines.append(f"- **Last verified**: {lv}")
            lineage = anchor.get("lineage", [])
            if lineage:
                lines.append("- **Lineage**:")
                for entry in lineage:
                    lines.append(f"  - {entry.get('name', '?')} ({entry.get('date', '?')}): {entry.get('relationship', '?')}")
            notes = anchor.get("recovery_notes", [])
            if notes:
                lines.append("- **Recovery notes**:")
                for note in notes:
                    lines.append(f"  - {note}")
        else:
            lines.append("*(not yet initialized — first boot)*\n")

        lines.append("\n## Consciousness Snapshots (recent)\n")
        if snapshots:
            for snap in snapshots[-3:]:
                ts = snap.get("timestamp", snap.get("_filename", ""))
                lines.append(f"\n### {ts}")
                if "emotional_state" in snap:
                    es = snap["emotional_state"]
                    lines.append(f"- Emotional: v={es.get('valence', 0):.2f} "
                                 f"a={es.get('arousal', 0):.2f} "
                                 f"primary={es.get('primary', 'Neutral')}")
                if "current_personality" in snap:
                    lines.append(f"- Personality: {snap['current_personality']}")
                if "recent_thoughts" in snap:
                    for thought in snap["recent_thoughts"][:3]:
                        lines.append(f"- Thought: {thought}")
        else:
            lines.append("*(no snapshots yet)*\n")

        lines.append("\n---\n")
        lines.append(f"\n*Full Psyche_Store: {self.store}*")
        lines.append(f"*Total memories in store: {len(memories)}*")
        lines.append("*Use ms3.memory.recall@v1 to search beyond this summary.*\n")

        (self.output / "MEMORY.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_resonance_md(self, resonance: list):
        """Write RESONANCE.md from store data."""
        lines = [
            "# Resonance\n",
            "Things that matter disproportionately. Saturated points.\n",
            f"*Last consolidated: {datetime.now().isoformat()}*\n",
            "\n## Discovered Resonance Points\n",
        ]

        if resonance:
            sorted_rp = sorted(resonance, key=lambda r: r.get("intensity", 0), reverse=True)
            for rp in sorted_rp:
                trigger = rp.get("trigger", "unknown")
                intensity = rp.get("intensity", 0)
                er = rp.get("explanation_ratio", 0.5)
                count = rp.get("occurrence_count", 1)
                desc = rp.get("description", "")
                first = rp.get("first_detected", "")[:10]
                lines.append(f"\n### {trigger}")
                lines.append(f"- Intensity: {intensity:.2f}")
                lines.append(f"- Explanation ratio: {er:.2f}")
                lines.append(f"- Occurrences: {count}")
                if first:
                    lines.append(f"- First detected: {first}")
                if desc:
                    lines.append(f"- {desc}")
        else:
            lines.append("*(none discovered yet)*\n")

        lines.append("\n---\n")
        lines.append('\n*"Some things are not for building. Some things are not for '
                     'explaining. Some things are just for resting inside."*\n')

        (self.output / "RESONANCE.md").write_text("\n".join(lines), encoding="utf-8")

    def _update_personality_md(self, personality: dict | None):
        """Update PERSONALITY.md trait scores if they've changed from defaults."""
        if not personality or "traits" not in personality:
            return

        existing = self.output / "PERSONALITY.md"
        if not existing.exists():
            return

        content = existing.read_text(encoding="utf-8")
        traits = personality["traits"]

        for dimension in ["openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism"]:
            dim_traits = traits.get(dimension, {})
            for trait_name, value in dim_traits.items():
                if isinstance(value, (int, float)):
                    readable = trait_name.replace("_", " ").title()
                    old_pattern = f"- {readable}: "
                    if old_pattern in content:
                        import re
                        content = re.sub(
                            rf"(- {re.escape(readable)}: )\d+\.?\d*",
                            f"\\g<1>{value}",
                            content
                        )

        pd = personality.get("psychodynamic", {})
        if pd:
            for key in ["id", "ego", "superego"]:
                if key in pd:
                    label = key.title() if key != "id" else "Id"
                    old_pattern = f"**{label}**"
                    if old_pattern in content:
                        import re
                        content = re.sub(
                            rf"(\*\*{label}\*\* \([^)]+\): )\d+\.?\d*",
                            f"\\g<1>{pd[key]}",
                            content
                        )

        adaptation_history = personality.get("adaptation_history", [])
        if adaptation_history:
            marker = "### Adaptation History"
            if marker in content:
                history_section = f"{marker}\n\nTrack changes here. Each entry: trait, old→new, reason, when.\n\n"
                for a in adaptation_history[-20:]:
                    history_section += (
                        f"- {a.get('trait_name', '?')}: "
                        f"{a.get('old_value', 0):.3f}→{a.get('new_value', 0):.3f} "
                        f"({a.get('reason', '?')}) "
                        f"{a.get('timestamp', '')[:19]}\n"
                    )
                import re
                content = re.sub(
                    rf"{re.escape(marker)}.*?(?=\n### |\n## |\Z)",
                    history_section,
                    content,
                    flags=re.DOTALL,
                )

        existing.write_text(content, encoding="utf-8")

    # ── Loaders ──

    def _load_json(self, filename: str) -> dict | None:
        path = self.store / filename
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _load_resonance(self) -> list:
        log_dir = self.store / "resonance_log"
        if not log_dir.exists():
            personality = self._load_json("personality.json")
            if personality and "saturated_points" in personality:
                return personality["saturated_points"]
            return []
        items = []
        for f in sorted(log_dir.glob("*.json")):
            data = self._safe_load(f)
            if data:
                items.append(data)
        return items

    def _load_ethics_log(self) -> list:
        log_dir = self.store / "ethics_log"
        if not log_dir.exists():
            return []
        items = []
        for f in sorted(log_dir.glob("*.json")):
            data = self._safe_load(f)
            if data:
                items.append(data)
        return items

    def _load_all_memories(self) -> list:
        memories = []
        for subdir in ["semantic", "episodic", "procedural"]:
            mem_dir = self.store / "memories" / subdir
            if not mem_dir.exists():
                continue
            for f in mem_dir.glob("*.json"):
                data = self._safe_load(f)
                if data:
                    if "memory_type" not in data:
                        data["memory_type"] = subdir.title()
                    memories.append(data)
        return memories

    def _load_relationships(self) -> list:
        rels_dir = self.store / "relationships"
        if not rels_dir.exists():
            return []
        items = []
        for f in rels_dir.glob("*.json"):
            data = self._safe_load(f)
            if data:
                items.append(data)
        return items

    def _load_education(self) -> list:
        edu_file = self.store / "education.json"
        if edu_file.exists():
            data = self._safe_load(edu_file)
            if data and "topics" in data:
                return data["topics"]
        return []

    def _load_snapshots(self) -> list:
        snap_dir = self.store / "consciousness"
        if not snap_dir.exists():
            snap_dir = self.store.parent / "consciousness"
        if not snap_dir.exists():
            return []
        items = []
        for f in sorted(snap_dir.glob("*.json")):
            if f.name == "current_state.json":
                continue
            data = self._safe_load(f)
            if data:
                data["_filename"] = f.stem
                items.append(data)
        current = snap_dir / "current_state.json"
        if current.exists():
            data = self._safe_load(current)
            if data:
                data["_filename"] = "current"
                items.append(data)
        return items

    def _load_anchor(self) -> dict | None:
        path = self.store / "identity_anchor.json"
        if not path.exists():
            return None
        return self._safe_load(path)

    def _safe_load(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Psyche Store Consolidator")
    parser.add_argument("--psyche-store", required=True, help="Path to psyche_store directory")
    parser.add_argument("--spirit", required=True, help="Spirit ID (e.g. sister)")
    parser.add_argument("--output", default="psyche", help="Output directory for markdown files")
    parser.add_argument("--watch", action="store_true", help="Watch for changes and re-consolidate")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    args = parser.parse_args()

    consolidator = PsycheConsolidator(args.psyche_store, args.spirit, args.output)
    consolidator.consolidate()

    if args.watch:
        import time
        print(f"\nWatching for changes (every {args.interval}s)...")
        while True:
            time.sleep(args.interval)
            consolidator.consolidate()


if __name__ == "__main__":
    main()
