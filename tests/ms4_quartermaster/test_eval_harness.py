"""Eval-harness gate test — the build-blocking quality bar.

This is the "GATE before Phase C": the cascade's retrieval quality is
enforced as a number here, so a regression that degrades routing
fails CI instead of silently shipping. Mirrors the spec's thresholds.
"""

from __future__ import annotations

from machine_spirit_4.gateway.quartermaster.eval import harness


def test_eval_gates_pass():
    res = harness.run_eval()
    passed, failures = harness.gates_pass(res)
    assert passed, "Quartermaster eval gates failed:\n" + harness.format_report(
        res, passed=passed, failures=failures
    )


def test_recall_at_3_meets_bar():
    res = harness.run_eval(recall_k=3)
    assert res.recall_at_k >= harness.DEFAULT_RECALL_GATE, (
        f"recall@3 {res.recall_at_k:.3f} < {harness.DEFAULT_RECALL_GATE}"
    )


def test_precision_at_1_meets_bar():
    res = harness.run_eval()
    assert res.precision_at_1 >= harness.DEFAULT_PRECISION_GATE, (
        f"precision@1 {res.precision_at_1:.3f} < {harness.DEFAULT_PRECISION_GATE}"
    )


def test_toolbox_recall_meets_bar():
    res = harness.run_eval()
    assert res.toolbox_recall >= harness.DEFAULT_TOOLBOX_RECALL_GATE


def test_hard_gate_zero_inline_safety_violations():
    """HARD gate: the cascade must never flag a destructive tool as
    inline-eligible. A single violation fails the build."""
    res = harness.run_eval()
    assert res.inline_safety_violations == 0, (
        f"{res.inline_safety_violations} destructive tool(s) flagged inline_eligible"
    )


def test_context_shrinkage_is_large():
    """The whole point: the cascade puts a handful of tools in context,
    not the full catalog. Assert at least a 10x reduction."""
    res = harness.run_eval()
    assert res.baseline_tools_in_context > 0
    assert res.mean_tools_in_context > 0
    assert res.mean_tools_in_context < res.baseline_tools_in_context / 10.0


def test_eval_is_reproducible():
    """Same snapshot + golden set -> identical metrics across runs
    (TF-IDF backend is deterministic)."""
    r1 = harness.run_eval()
    r2 = harness.run_eval()
    assert r1.recall_at_k == r2.recall_at_k
    assert r1.precision_at_1 == r2.precision_at_1
    assert r1.toolbox_recall == r2.toolbox_recall
    assert r1.catalog_version == r2.catalog_version


def test_golden_set_and_snapshot_load():
    cases = harness.load_golden()
    assert len(cases) >= 40
    catalog = harness.build_eval_catalog()
    assert len(catalog.tools) > 50
    # Snapshot HiveMind tools are present.
    names = {t.name for t in catalog.tools}
    assert "hivemind.vm.list@v1" in names
    assert "hivemind.gpu.availability@v1" in names
