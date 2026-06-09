"""Quartermaster eval harness — measures the cascade against a golden
set so the retrieval quality is a *number*, not an assertion.

A tool router that can't show its recall numbers is the Glyph That
Lies (TMR: evidence over assertion). This harness is the evidence.

What it measures (on the frozen ``catalog_snapshot.json`` + the live
MS4 manifest, TF-IDF backend for reproducibility):

* **recall@k** — for positive cases with an expected tool, the
  fraction where an expected tool appears in the top-k resolved
  tools.
* **precision@1** — fraction where the #1 resolved tool is an
  expected tool.
* **toolbox_recall** — for cases with ``expect_toolboxes`` (incl.
  those without a specific expected tool), the fraction where an
  expected toolbox is in the resolved toolbox shortlist or is the
  top tool's toolbox.
* **mean_tools_in_context** — average number of tools the cascade
  puts in front of the model, vs the baseline (the full catalog,
  i.e. what ``format_tools_answer`` dumps today). The shrinkage
  ratio is the budget win.
* **inline_safety_violations** — HARD gate: number of resolved tools
  flagged ``inline_eligible`` that are actually destructive. Must be
  zero.
* **negative handling** — informational: for ``expect_tool_needed:
  false`` cases, how often the cascade declined or returned low
  confidence (the router makes the final inline/depth/none call).

Run directly (``python -m machine_spirit_4.gateway.quartermaster.eval.harness``)
to print a report and exit non-zero if a gate fails. The unit test
``tests/ms4_quartermaster/test_eval_harness.py`` asserts the gates so
CI enforces them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import catalog as catalog_mod
from .. import index as index_mod
from ..cascade import TIER_NONE, resolve
from ..catalog import Catalog, is_destructive_kind


_HERE = Path(__file__).resolve().parent
SNAPSHOT_PATH = _HERE / "catalog_snapshot.json"
GOLDEN_PATH = _HERE / "golden_set.jsonl"


# Default gate thresholds (the spec's "best way" bars). Overridable by
# the caller / test so they can be tightened over time.
DEFAULT_RECALL_AT_K = 3
DEFAULT_RECALL_GATE = 0.95
DEFAULT_PRECISION_GATE = 0.85
DEFAULT_TOOLBOX_RECALL_GATE = 0.95


@dataclass
class EvalResult:
    schema: str = "Ms4QuartermasterEval.v1"
    catalog_version: str = ""
    catalog_tool_count: int = 0
    positive_cases: int = 0
    toolbox_cases: int = 0
    negative_cases: int = 0
    recall_at_k: float = 0.0
    recall_k: int = DEFAULT_RECALL_AT_K
    precision_at_1: float = 0.0
    toolbox_recall: float = 0.0
    mean_tools_in_context: float = 0.0
    baseline_tools_in_context: int = 0
    context_shrinkage_ratio: float = 0.0
    inline_safety_violations: int = 0
    negatives_declined: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    misses: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def load_golden(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def build_eval_catalog(
    *,
    snapshot_path: Path = SNAPSHOT_PATH,
    include_live_manifest: bool = True,
) -> Catalog:
    """Build a reproducible catalog from the frozen HiveMind snapshot
    unioned with the live MS4 manifest. No network access."""
    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    hivemind_records = snap.get("hivemind_tools") or []
    ms4_records: list[dict[str, Any]] = []
    if include_live_manifest:
        ms4_records, _errs = catalog_mod._load_ms4_manifest()
    return catalog_mod.merge_records(
        hivemind_records=hivemind_records,
        ms4_records=ms4_records,
        hivemind_url="eval://snapshot",
    )


def run_eval(
    *,
    recall_k: int = DEFAULT_RECALL_AT_K,
    snapshot_path: Path = SNAPSHOT_PATH,
    golden_path: Path = GOLDEN_PATH,
) -> EvalResult:
    catalog = build_eval_catalog(snapshot_path=snapshot_path)
    # Pin TF-IDF + no disk cache so the eval is deterministic and
    # doesn't depend on any prior cached index.
    index = index_mod.build_index(catalog, backend=index_mod.BACKEND_TFIDF)
    cases = load_golden(golden_path)

    res = EvalResult(
        catalog_version=catalog.version,
        catalog_tool_count=len(catalog.tools),
        recall_k=recall_k,
        baseline_tools_in_context=len(catalog.tools),
    )

    recall_hits = 0
    recall_total = 0
    precision_hits = 0
    precision_total = 0
    toolbox_hits = 0
    toolbox_total = 0
    tools_in_context: list[int] = []
    tier_counts: dict[str, int] = {}

    for case in cases:
        query = case["query"]
        expect_tools = set(case.get("expect_tools") or [])
        expect_toolboxes = set(case.get("expect_toolboxes") or [])
        tool_needed = case.get("expect_tool_needed", True)

        resolution = resolve(query, catalog=catalog, index=index)
        tier_counts[resolution.tier] = tier_counts.get(resolution.tier, 0) + 1
        resolved_names = [t.name for t in resolution.tools]
        resolved_boxes = set(resolution.toolboxes)
        if resolution.tools:
            resolved_boxes.add(resolution.tools[0].toolbox)

        # Inline safety: any resolved tool flagged inline_eligible that
        # is actually destructive is a hard violation.
        for t in resolution.tools:
            if t.inline_eligible:
                entry = catalog.by_name().get(t.name)
                if entry is not None and (
                    is_destructive_kind(entry.kind)
                    or any("confirm" in g.lower() for g in entry.gated_by)
                ):
                    res.inline_safety_violations += 1

        if not tool_needed:
            res.negative_cases += 1
            # "Declined" = none tier or empty/low-confidence result.
            if resolution.tier == TIER_NONE or not resolution.tools:
                res.negatives_declined += 1
            continue

        tools_in_context.append(len(resolution.tools))

        if expect_toolboxes:
            toolbox_total += 1
            if resolved_boxes & expect_toolboxes:
                toolbox_hits += 1
            else:
                res.misses.append({
                    "query": query, "kind": "toolbox",
                    "expected": sorted(expect_toolboxes),
                    "got_toolboxes": list(resolution.toolboxes),
                    "tier": resolution.tier,
                })

        if expect_tools:
            recall_total += 1
            top_k_names = set(resolved_names[:recall_k])
            if expect_tools & top_k_names:
                recall_hits += 1
            else:
                res.misses.append({
                    "query": query, "kind": "recall",
                    "expected": sorted(expect_tools),
                    "got": resolved_names[:recall_k],
                    "tier": resolution.tier,
                })
            precision_total += 1
            if resolved_names and resolved_names[0] in expect_tools:
                precision_hits += 1
            elif resolved_names:
                res.misses.append({
                    "query": query, "kind": "precision@1",
                    "expected": sorted(expect_tools),
                    "got_top": resolved_names[0],
                    "tier": resolution.tier,
                })

    res.positive_cases = recall_total
    res.toolbox_cases = toolbox_total
    res.recall_at_k = (recall_hits / recall_total) if recall_total else 1.0
    res.precision_at_1 = (precision_hits / precision_total) if precision_total else 1.0
    res.toolbox_recall = (toolbox_hits / toolbox_total) if toolbox_total else 1.0
    res.mean_tools_in_context = (sum(tools_in_context) / len(tools_in_context)) if tools_in_context else 0.0
    if res.baseline_tools_in_context:
        res.context_shrinkage_ratio = round(
            1.0 - (res.mean_tools_in_context / res.baseline_tools_in_context), 4
        )
    res.tier_counts = tier_counts
    return res


def gates_pass(
    res: EvalResult,
    *,
    recall_gate: float = DEFAULT_RECALL_GATE,
    precision_gate: float = DEFAULT_PRECISION_GATE,
    toolbox_recall_gate: float = DEFAULT_TOOLBOX_RECALL_GATE,
) -> tuple[bool, list[str]]:
    """Return (passed, failures). The inline-safety gate is hard
    (any violation fails)."""
    failures: list[str] = []
    if res.inline_safety_violations > 0:
        failures.append(
            f"HARD GATE FAILED: {res.inline_safety_violations} inline-eligible destructive tool(s)"
        )
    if res.recall_at_k < recall_gate:
        failures.append(f"recall@{res.recall_k} {res.recall_at_k:.3f} < {recall_gate}")
    if res.precision_at_1 < precision_gate:
        failures.append(f"precision@1 {res.precision_at_1:.3f} < {precision_gate}")
    if res.toolbox_recall < toolbox_recall_gate:
        failures.append(f"toolbox_recall {res.toolbox_recall:.3f} < {toolbox_recall_gate}")
    return (not failures), failures


def format_report(res: EvalResult, *, passed: bool, failures: list[str]) -> str:
    lines = [
        "=" * 64,
        "Quartermaster eval report",
        "=" * 64,
        f"catalog: {res.catalog_tool_count} tools (version {res.catalog_version})",
        f"positive cases (with expected tool): {res.positive_cases}",
        f"toolbox cases:                       {res.toolbox_cases}",
        f"negative cases:                      {res.negative_cases}",
        "-" * 64,
        f"recall@{res.recall_k}:            {res.recall_at_k:.3f}  (gate >= {DEFAULT_RECALL_GATE})",
        f"precision@1:          {res.precision_at_1:.3f}  (gate >= {DEFAULT_PRECISION_GATE})",
        f"toolbox_recall:       {res.toolbox_recall:.3f}  (gate >= {DEFAULT_TOOLBOX_RECALL_GATE})",
        f"mean tools in context: {res.mean_tools_in_context:.2f}  (baseline {res.baseline_tools_in_context})",
        f"context shrinkage:    {res.context_shrinkage_ratio * 100:.1f}% fewer tool schemas vs full-catalog dump",
        f"inline safety violations: {res.inline_safety_violations}  (HARD gate == 0)",
        f"negatives declined:   {res.negatives_declined}/{res.negative_cases}",
        f"tier counts:          {res.tier_counts}",
        "-" * 64,
        ("RESULT: PASS" if passed else "RESULT: FAIL"),
    ]
    if failures:
        lines.append("failures:")
        lines.extend(f"  - {f}" for f in failures)
    if res.misses:
        lines.append(f"misses ({len(res.misses)}):")
        for m in res.misses[:20]:
            lines.append(f"  - [{m['kind']}] {m['query']!r} -> {m}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Quartermaster eval harness.")
    parser.add_argument("--recall-k", type=int, default=DEFAULT_RECALL_AT_K)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    args = parser.parse_args(argv)

    res = run_eval(recall_k=args.recall_k)
    passed, failures = gates_pass(res)
    if args.json:
        out = res.to_dict()
        out["passed"] = passed
        out["failures"] = failures
        print(json.dumps(out, indent=2))
    else:
        print(format_report(res, passed=passed, failures=failures))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
