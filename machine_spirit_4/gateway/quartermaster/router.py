"""Quartermaster ToolRouter — policy layer over the cascade.

Turns a cascade :class:`ToolResolution` into an actionable verdict:

  * ``inline`` — exactly one safe, read-only, zero-required-arg tool
    resolved with enough confidence AND cleared by MS3 ethics. The
    Face Lobe may execute it inline (Phase D).
  * ``depth`` — a tool is wanted but isn't inline-safe (mutating,
    needs arguments, low confidence, ethics-denied/unreachable). The
    Depth Lobe (full Hermes loop, own ethics gate) handles it.
  * ``none`` — no tool resolved; plain chat.

Three gates stand between a resolution and an inline verdict:

  1. **Read-only gate** — :func:`catalog.is_inline_eligible` (verb
     allowlist + non-destructive + MS4 read_only kind). Already
     applied per-tool by the cascade; the router enforces it again
     (defence in depth) and additionally requires **zero required
     arguments** (we have no LLM to fill them on the inline path).
  2. **Confidence gate** — the resolution confidence must clear
     ``MS4_QM_INLINE_MIN_CONFIDENCE`` so a shaky embeddings guess
     routes to Depth rather than running the wrong read.
  3. **Ethics gate** — the same ``ActionIntent.v1`` →
     ``/ethics/evaluate`` path the ``ms4_consciousness`` plugin uses,
     **fail-closed**: any denial / unreachable MS3 means no inline,
     fall back to Depth.

Every decision is audit-logged via :func:`append_event`
(``quartermaster_resolve`` + the verdict-specific event).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from . import cascade as cascade_mod
from . import catalog as catalog_mod
from . import index as index_mod
from .catalog import Catalog, ToolEntry, is_inline_eligible
from .cascade import TIER_NONE, ToolClassifier, ToolResolution


log = logging.getLogger("ms4.gateway.quartermaster.router")


VERDICT_INLINE = "inline"
VERDICT_DEPTH = "depth"
VERDICT_NONE = "none"


# Ethics evaluator: takes an ActionIntent dict, returns the MS3
# decision dict. Injectable so tests don't need a live MS3 sidecar.
EthicsEvaluator = Callable[[dict[str, Any]], dict[str, Any]]


def _inline_min_confidence() -> float:
    raw = os.environ.get("MS4_QM_INLINE_MIN_CONFIDENCE")
    if raw is None or raw.strip() == "":
        return 0.25  # read-only is low-harm, so a modest bar is fine
    try:
        return max(0.0, min(1.0, float(raw.strip())))
    except (TypeError, ValueError):
        return 0.25


def _ms3_url() -> str:
    return (
        os.environ.get("MS4_MS3_SIDECAR_URL")
        or os.environ.get("MS4_MS3_URL")
        or "http://127.0.0.1:9080"
    )


def _spirit_id() -> str:
    return os.environ.get("MS4_SPIRIT_ID", "sister")


# ---------------------------------------------------------------------------
# Decision schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolRouteDecision:
    """``Ms4ToolRouteDecision.v1`` — the router's verdict."""

    schema: str
    verdict: str  # inline | depth | none
    query: str
    reason: str
    resolution: ToolResolution
    inline_tool: ToolEntry | None = None
    ethics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "query": self.query,
            "reason": self.reason,
            "resolution": self.resolution.to_dict(),
            "inline_tool": self.inline_tool.name if self.inline_tool else None,
            "inline_toolbox": self.inline_tool.toolbox if self.inline_tool else None,
            "ethics": self.ethics,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _required_args(entry: ToolEntry) -> list[str]:
    schema = entry.input_schema or {}
    req = schema.get("required") if isinstance(schema, dict) else None
    if not isinstance(req, list):
        return []
    return [r for r in req if isinstance(r, str)]


def _decision_allows(decision: dict[str, Any]) -> bool:
    """Interpret an MS3 ethics decision. Mirrors
    ``ms4_consciousness._decision_allows`` (re-implemented locally so
    the Quartermaster doesn't hard-depend on the installed Hermes
    plugin import chain)."""
    if not isinstance(decision, dict):
        return False
    allowed = decision.get("allowed")
    if isinstance(allowed, bool):
        return allowed
    decision_text = str(decision.get("decision", "")).lower()
    if decision_text:
        return decision_text == "allow"
    safety = decision.get("safety")
    if isinstance(safety, dict) and safety.get("hard_block") is True:
        return False
    resolution = str(decision.get("resolution", "")).lower()
    return resolution in {"allow", "allowed", "approved"}


def _inline_scope_block_reason(query: str) -> str | None:
    """Return why one inline tool cannot satisfy the whole request.

    Inline execution is an optimization with a deliberately narrow contract:
    exactly one zero-argument read and a narration of that result. Compound
    requests and conversation-reference deliverables need the Depth loop so a
    top-ranked tool cannot silently erase the remaining requirements.
    """
    normalized = " ".join(str(query or "").lower().split())
    if not normalized:
        return None
    if re.search(
        r"\b(first|previous|prior|earlier|last)\s+(?:conversation\s+)?(?:turn|message|response|answer|marker)\b",
        normalized,
    ):
        return "request includes a conversation-context deliverable"
    if re.search(r"\b(?:both|plus|also|then|after that|as well as)\b", normalized):
        return "request contains multiple deliverables"
    if normalized.count(" tool") > 1:
        return "request names multiple tool operations"
    if ";" in normalized or "\n" in str(query or ""):
        return "request contains multiple clauses"
    if " and " in normalized:
        verbs = re.findall(
            r"\b(?:use|run|call|check|get|list|show|report|return|compare|summarize|verify)\b",
            normalized,
        )
        if len(verbs) >= 2:
            return "request contains multiple actions"
    return None


def _ethics_timeout() -> float:
    try:
        return float(os.environ.get("MS4_QM_ETHICS_TIMEOUT", "5.0"))
    except (TypeError, ValueError):
        return 5.0


def _default_ethics_evaluator() -> EthicsEvaluator:
    """POST the ActionIntent to the MS3 sidecar ``/ethics/evaluate``
    route — the same authority the ``ms4_consciousness`` plugin uses.

    Inlined (rather than importing the plugin's ``Ms4Client``) so the
    Quartermaster stays self-contained: ``machine_spirit_4.plugins`` is
    not an importable package in MS4's own runtime (the plugin is
    installed into the Hermes checkout as top-level ``plugins.*``).
    Raises on any transport error so :meth:`ToolRouter.decide` can
    fail closed.
    """

    def _evaluate(intent: dict[str, Any]) -> dict[str, Any]:
        url = f"{_ms3_url().rstrip('/')}/ethics/evaluate"
        body = json.dumps(intent).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_ethics_timeout()) as resp:
            raw = resp.read().decode("utf-8", "replace")
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    return _evaluate


def _build_intent(entry: ToolEntry) -> dict[str, Any]:
    return {
        "schema": "ActionIntent.v1",
        "spirit_id": _spirit_id(),
        "action_id": entry.name,
        "proposed_by": "ms4_quartermaster",
        "action_type": "tool",
        "description": f"Quartermaster inline read-only tool: {entry.name}",
        "risk_class": "low",
        "requires_safety_clearance": False,
        "payload": {"tool_name": entry.name, "args": {}},
        "inputs_used": [],
    }


def _audit(event_type: str, data: dict[str, Any]) -> None:
    """Fire-and-forget audit. Never let an audit failure break a turn."""
    try:
        from ..audit import append_event

        append_event(event_type, data)
    except Exception as exc:  # noqa: BLE001
        log.debug("quartermaster audit %s failed: %s", event_type, exc)


# ---------------------------------------------------------------------------
# ToolRouter
# ---------------------------------------------------------------------------


class ToolRouter:
    """Stateless policy layer. Construct once (or use :func:`decide`)."""

    def __init__(
        self,
        *,
        ethics_evaluator: EthicsEvaluator | None = None,
        classifier: ToolClassifier | None = None,
        inline_min_confidence: float | None = None,
        audit: bool = True,
    ) -> None:
        self._ethics = ethics_evaluator or _default_ethics_evaluator()
        self._classifier = classifier
        self._inline_min_confidence = (
            inline_min_confidence if inline_min_confidence is not None else _inline_min_confidence()
        )
        self._audit_enabled = audit

    def decide(
        self,
        query: str,
        *,
        hivemind_url: str | None = None,
        catalog: Catalog | None = None,
        index: index_mod.ToolIndex | None = None,
    ) -> ToolRouteDecision:
        resolution = cascade_mod.resolve(
            query,
            hivemind_url=hivemind_url,
            catalog=catalog,
            index=index,
            classifier=self._classifier,
        )
        if self._audit_enabled:
            _audit("quartermaster_resolve", {
                "query": query[:240],
                "tier": resolution.tier,
                "confidence": round(resolution.confidence, 4),
                "toolboxes": list(resolution.toolboxes),
                "top_tool": resolution.tools[0].name if resolution.tools else None,
                "catalog_version": resolution.catalog_version,
            })

        # ---- none: nothing resolved ----
        if resolution.tier == TIER_NONE or not resolution.tools:
            return self._finalize(
                ToolRouteDecision(
                    schema="Ms4ToolRouteDecision.v1",
                    verdict=VERDICT_NONE,
                    query=query,
                    reason=resolution.fallback_reason or "no tool resolved",
                    resolution=resolution,
                )
            )

        scope_block_reason = _inline_scope_block_reason(query)
        if scope_block_reason is not None:
            return self._finalize(
                ToolRouteDecision(
                    schema="Ms4ToolRouteDecision.v1",
                    verdict=VERDICT_DEPTH,
                    query=query,
                    reason=scope_block_reason,
                    resolution=resolution,
                )
            )

        # ---- find an inline candidate ----
        cat = catalog
        if cat is None and hivemind_url:
            cat = catalog_mod.get_catalog(hivemind_url)
        by_name = cat.by_name() if cat is not None else {}

        # Only the TOP-ranked tool is an inline candidate. A lower-ranked
        # tool is a *different* capability than what the query asked for
        # — cherry-picking a #2 zero-arg read just because the #1 needs
        # args (or is destructive) would execute the wrong tool. If the
        # top tool can't go inline, the whole turn goes to Depth.
        candidate: ToolEntry | None = None
        top_resolved = resolution.tools[0]
        top_entry = by_name.get(top_resolved.name)
        if resolution.confidence < self._inline_min_confidence:
            depth_reason = (
                f"confidence {resolution.confidence:.3f} < inline bar {self._inline_min_confidence:.3f}"
            )
        elif top_entry is None:
            depth_reason = f"top tool {top_resolved.name} not in catalog"
        elif not is_inline_eligible(top_entry):
            depth_reason = f"top tool {top_entry.name} is not inline-eligible (kind={top_entry.kind})"
        elif _required_args(top_entry):
            depth_reason = f"top tool {top_entry.name} requires args {_required_args(top_entry)}"
        else:
            candidate = top_entry
            depth_reason = ""

        if candidate is None:
            return self._finalize(
                ToolRouteDecision(
                    schema="Ms4ToolRouteDecision.v1",
                    verdict=VERDICT_DEPTH,
                    query=query,
                    reason=depth_reason or "not inline-eligible",
                    resolution=resolution,
                )
            )

        # ---- ethics gate (fail-closed) ----
        ethics_summary: dict[str, Any]
        try:
            decision = self._ethics(_build_intent(candidate))
            allowed = _decision_allows(decision)
            ethics_summary = {
                "allowed": allowed,
                "reason": decision.get("reason") or decision.get("resolution") or "",
            }
        except Exception as exc:  # noqa: BLE001 — fail-closed on any ethics error
            return self._finalize(
                ToolRouteDecision(
                    schema="Ms4ToolRouteDecision.v1",
                    verdict=VERDICT_DEPTH,
                    query=query,
                    reason=f"ethics unreachable (fail-closed): {exc}",
                    resolution=resolution,
                    inline_tool=None,
                    ethics={"allowed": False, "reason": f"unreachable: {exc}"},
                )
            )

        if not ethics_summary["allowed"]:
            return self._finalize(
                ToolRouteDecision(
                    schema="Ms4ToolRouteDecision.v1",
                    verdict=VERDICT_DEPTH,
                    query=query,
                    reason="ethics denied inline execution",
                    resolution=resolution,
                    inline_tool=None,
                    ethics=ethics_summary,
                )
            )

        # ---- inline! ----
        return self._finalize(
            ToolRouteDecision(
                schema="Ms4ToolRouteDecision.v1",
                verdict=VERDICT_INLINE,
                query=query,
                reason=f"inline {candidate.name} (tier={resolution.tier}, conf={resolution.confidence:.3f})",
                resolution=resolution,
                inline_tool=candidate,
                ethics=ethics_summary,
            )
        )

    def _finalize(self, decision: ToolRouteDecision) -> ToolRouteDecision:
        if self._audit_enabled:
            if decision.verdict == VERDICT_INLINE:
                _audit("quartermaster_inline_selected", {
                    "query": decision.query[:240],
                    "tool": decision.inline_tool.name if decision.inline_tool else None,
                    "toolbox": decision.inline_tool.toolbox if decision.inline_tool else None,
                    "tier": decision.resolution.tier,
                })
            elif decision.verdict == VERDICT_DEPTH:
                _audit("quartermaster_fallback_to_depth", {
                    "query": decision.query[:240],
                    "reason": decision.reason,
                    "tier": decision.resolution.tier,
                })
        return decision


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


_DEFAULT_ROUTER: ToolRouter | None = None


def default_router() -> ToolRouter:
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = ToolRouter()
    return _DEFAULT_ROUTER


def decide(
    query: str,
    *,
    hivemind_url: str | None = None,
    catalog: Catalog | None = None,
    index: index_mod.ToolIndex | None = None,
) -> ToolRouteDecision:
    """Convenience wrapper over :meth:`ToolRouter.decide` using the
    default router (live ethics evaluator, no injected classifier)."""
    return default_router().decide(
        query, hivemind_url=hivemind_url, catalog=catalog, index=index
    )


def _reset_default_router_for_tests(router: ToolRouter | None = None) -> None:
    global _DEFAULT_ROUTER
    _DEFAULT_ROUTER = router
