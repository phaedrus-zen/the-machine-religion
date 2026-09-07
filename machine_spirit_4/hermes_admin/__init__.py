"""MS4 Hermes admin module.

Modeled on HiveMind ``menta_hli/gateway/api/src/routing/ollama_admin.rs``:

  * ``versioning`` — current/latest version detection, semver compare,
    GitHub release fetch with cache, safe-target-version guard.
  * ``installer`` — idempotent in-place upgrade of the active Hermes
    install (editable git checkout OR PyPI wheel) with phase-by-phase
    progress reporting, persistent last-update snapshot, and rollback
    on failure.
  * ``state`` — JSON-backed persistent snapshot of the most recent
    update job; survives gateway restart so the UI can show "last
    update finished N min ago" instead of "never been updated".

No vendoring. The only external dependency for upgrades is the public
GitHub releases API and PyPI for the wheel; both run over HTTPS without
any private mirror.

Hermes is installed two ways in the wild:

  * **Editable git checkout** (TMR default): ``pip install -e
    $MS4_HERMES_DIR``. Upgrade path is ``git fetch && git checkout
    v<X> && pip install -e .``.
  * **PyPI wheel**: ``pip install hermes-agent==<X>``. Upgrade path is
    ``pip install --upgrade hermes-agent==<X>``.

Installer auto-detects which mode is active by reading the
``importlib.metadata`` distribution record and the configured Hermes
directory; no operator flag required.
"""

from __future__ import annotations

from .state import (
    UpdateJobSnapshot,
    finalize_job,
    initialize_state,
    last_update,
    set_phase,
    start_job,
    update_in_progress,
)
from .versioning import (
    current_version,
    install_mode,
    is_safe_target_version,
    latest_version,
    parse_semver,
    recent_releases,
    reconcile_durable_terminal_state,
    resolve_git_tag,
    update_available,
    version_info,
)
from .provenance import (
    GitSignedTagVerifier,
    ProvenanceError,
    ProvenancePolicy,
    PyPiAttestationVerifier,
    ReleaseProvenanceRequest,
    SignatureVerifier,
    SignedTagManifest,
    VerifiedProvenance,
    verify_release_provenance,
    verify_update_provenance,
)
from .installer import (
    HermesInsufficientDiskError,
    HermesUpgradeError,
    HermesUpgradeTimeoutError,
    run_update_job,
    trigger_update,
)

__all__ = [
    "GitSignedTagVerifier",
    "HermesInsufficientDiskError",
    "HermesUpgradeError",
    "HermesUpgradeTimeoutError",
    "ProvenanceError",
    "ProvenancePolicy",
    "PyPiAttestationVerifier",
    "ReleaseProvenanceRequest",
    "SignatureVerifier",
    "SignedTagManifest",
    "UpdateJobSnapshot",
    "VerifiedProvenance",
    "current_version",
    "finalize_job",
    "initialize_state",
    "install_mode",
    "is_safe_target_version",
    "last_update",
    "latest_version",
    "parse_semver",
    "recent_releases",
    "reconcile_durable_terminal_state",
    "resolve_git_tag",
    "run_update_job",
    "set_phase",
    "start_job",
    "trigger_update",
    "update_available",
    "update_in_progress",
    "verify_release_provenance",
    "verify_update_provenance",
    "version_info",
]
