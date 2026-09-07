"""UI contract: Hermes banner follows backend operator_state, not a CSS hide."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "machine_spirit_4" / "web" / "index.html"
SERVER = ROOT / "machine_spirit_4" / "gateway" / "server.py"


def test_hermes_banner_uses_operator_state_and_keeps_failed_audit_hooks():
    html = HTML.read_text(encoding="utf-8")
    assert 'id="hermesUpdateBtn" type="button" disabled' in html
    assert "function renderHermesBanner(info)" in html
    assert "operator_state" in html
    assert "installed_relation" in html
    # Failed presentation remains for actionable failures.
    assert "Last Hermes update failed" in html
    assert "Failed target:" in html
    # Current/newer no-op must not be a CSS-only hide of last.status===failed.
    assert "info.operator_state === 'newer'" in html or "info.operator_state === 'current'" in html
    assert "last && last.status === 'failed'" in html
    # The failed branch must be gated by operator_state so a reconciled
    # current/newer snapshot does not keep the stale failure banner.
    failed_idx = html.index("Last Hermes update failed")
    banner_fn = html[html.index("function renderHermesBanner"): html.index("async function refreshHermesVersion")]
    assert "operator_state" in banner_fn
    assert "newer" in banner_fn and "current" in banner_fn
    # Keep the raw failure copy for when operator_state is still failed.
    assert failed_idx > 0
    # Unsigned-latest block is the primary headline; failed history is audit.
    assert "operator_state === 'blocked'" in banner_fn
    assert banner_fn.index("operator_state === 'blocked'") < banner_fn.index(
        "Last Hermes update failed"
    )
    assert "No signed Hermes update" in banner_fn
    assert "No signed update" in banner_fn
    assert banner_fn.index("completedVersionChange") < banner_fn.index(
        "info.operator_state === 'newer'"
    )


def test_hermes_banner_does_not_claim_update_when_newer_than_discovered_latest():
    html = HTML.read_text(encoding="utf-8")
    banner_fn = html[html.index("function renderHermesBanner"): html.index("async function refreshHermesVersion")]
    assert "operator_state" in banner_fn
    assert "No signed update" in banner_fn or "up to date" in html
    # Button contracts remain wired.
    assert "function triggerHermesUpdate(targetVersion)" in html
    assert "/api/v1/hermes/update" in html


def test_hermes_blocked_banner_names_unsigned_and_unknown_signatures_truthfully():
    html = HTML.read_text(encoding="utf-8")
    banner_fn = html[html.index("function renderHermesBanner"): html.index("async function refreshHermesVersion")]
    assert "blockedReason === 'official_tag_unsigned'" in banner_fn
    assert "blockedReason === 'official_tag_signature_unknown'" in banner_fn
    assert "? 'is unsigned'" in banner_fn
    assert "? 'signature status unknown'" in banner_fn
    assert "whose signature status is unknown" in banner_fn


def test_hermes_banner_offers_unsigned_newest_when_release_policy_allows():
    """allow_unsigned: enabled Update button with explicit unsigned copy;
    require_signed keeps today's disabled 'No signed update' state."""
    html = HTML.read_text(encoding="utf-8")
    banner_fn = html[html.index("function renderHermesBanner"): html.index("async function refreshHermesVersion")]
    assert "info.release_signature_policy === 'allow_unsigned'" in html
    assert "function hermesPolicyAllowsUnsigned(info)" in html
    assert "hermesUnsignedOfferCopy(info)" in banner_fn
    assert (
        "is unsigned; policy allows (HERMES_RELEASE_SIGNATURE_POLICY=allow_unsigned)"
        in html
    )
    assert "newest release ${tag}" in html
    # Strict-policy presentation is untouched.
    assert "No signed update" in banner_fn
    assert "F4 refuses unsigned official tags" in banner_fn
    assert "'Retry signed update'" in html
    # Pin dialog follows the same policy instead of contradicting the button.
    assert "unsigned - policy allows" in html
    assert "unsigned - not offered" in html


def test_gateway_version_route_still_serves_version_info():
    server = SERVER.read_text(encoding="utf-8")
    assert "hermes_admin.version_info()" in server
    assert "hermes_admin.initialize_state()" in server
    assert "hermes_admin.recover_interrupted_update()" in server
    assert "reconcile_durable_terminal_state" in server
