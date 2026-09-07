"""F4 supply-chain provenance gate for the Hermes updater.

Fail-closed, cryptographic verification that MUST run BEFORE any working-tree
mutation or install. It binds the signed release tag + package version and
commit-addressed tree to an AUTHORIZED SIGNER, while independently enforcing the
checkout-origin allowlist, using git's NATIVE SSH signed-tag verification
(``git verify-tag`` + a pinned OpenSSH ``allowed_signers`` policy).

Why this mechanism:

* It is already available in the toolchain the editable updater depends on
  (git >= 2.34 with ``gpg.format=ssh``). No new dependency is added and NO
  signature mathematics is hand-rolled here -- git/ssh-keygen do the crypto;
  this module enforces POLICY on git's structured output.
* The actual upstream signed annotated-tag shape is bound directly: the signed
  tag object's ``tag`` and ``object`` headers bind the release alias and target
  commit, while its ``Hermes Agent v<version> (<tag-version>)`` subject binds the
  package version to that alias. Because the signature covers the whole tag
  object, tampering any of those values invalidates verification.

Trust boundaries and fail-closed posture:

* Absent policy (no ``allowed_signers`` file, or no origin allowlist) is
  REFUSED. There is no "no policy => allow" path.
* An unsigned tag, an annotated-but-unsigned tag, or a tag signed by a key
  outside the ``allowed_signers`` file is REFUSED (git ``verify-tag`` exits
  non-zero in every one of those cases).
* A signed release whose tag, package version, or commit disagrees with the
  request is REFUSED. The checkout origin is independently allowlisted before
  any release content is trusted.
* A release older than the installed floor is REFUSED (downgrade / replay).
* PyPI wheel provenance (PEP 740 attestations / Sigstore) is an EXPLICIT,
  UNSHIPPED integration dependency. The pypi path therefore fails closed: no
  attestation verifier is configured here, so pypi provenance is REFUSED rather
  than downgraded to transport/hash-only trust. See the runbook (F4 section)
  and DEPENDENCY_POLICY_NOTE.md for the integration contract.

This module is import-standalone (stdlib + the sibling ``versioning`` semver
parser only); it does NOT import ``installer`` (the installer imports the gate),
so there is no import cycle and the gate is unit-testable in isolation.
"""

from __future__ import annotations

import os
import platform as _platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from . import versioning

__all__ = [
    "ProvenanceError",
    "ReleaseProvenanceRequest",
    "ProvenancePolicy",
    "SignedTagManifest",
    "VerifiedProvenance",
    "SignatureVerifier",
    "GitSignedTagVerifier",
    "PyPiAttestationVerifier",
    "verify_editable_preflight",
    "verify_release_provenance",
    "verify_update_provenance",
]


class ProvenanceError(Exception):
    """Raised when release provenance cannot be verified (fail-closed).

    Deliberately NOT a subclass of ``installer.HermesUpgradeError`` (that would
    require importing the installer and create a cycle). The installer's gate
    seam catches this and re-raises a ``HermesUpgradeError`` so the persisted
    job finalizes as ``failed`` on the normal path.
    """


# --------------------------------------------------------------------------- #
# Normalization helpers (structured, deterministic, injectable for tests).
# --------------------------------------------------------------------------- #
_ARCH_ALIASES = {
    "amd64": "x86_64",
    "x64": "x86_64",
    "x86-64": "x86_64",
    "arm64": "aarch64",
    "arm64e": "aarch64",
}


def normalize_platform(name: str | None = None) -> str:
    """Canonical, comparable platform token (defaults to the live platform)."""
    value = sys.platform if name is None else name
    return value.strip().lower()


def normalize_arch(machine: str | None = None) -> str:
    """Canonical CPU-architecture token (defaults to the live machine)."""
    value = _platform.machine() if machine is None else machine
    token = value.strip().lower()
    return _ARCH_ALIASES.get(token, token)


def normalize_origin(url: str | None) -> str:
    """Canonicalize an origin for allowlist / manifest comparison.

    Filesystem-path remotes (used by the hermetic tests) and https URLs both
    normalize by trimming, unifying separators, dropping a trailing slash and
    an optional ``.git`` suffix, and case-folding (git hosts and Windows paths
    are case-insensitive). This is intentionally conservative -- an unknown
    origin fails the allowlist rather than being coerced to match.
    """
    if not url:
        return ""
    text = url.strip().replace("\\", "/").rstrip("/")
    if text.casefold().endswith(".git"):
        text = text[: -len(".git")]
    return text.casefold()


# --------------------------------------------------------------------------- #
# Data model.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReleaseProvenanceRequest:
    """What the updater is about to trust, expressed for verification."""

    release_name: str
    version: str
    platform: str
    arch: str
    origin: str
    source_kind: str  # "git" | "pypi"
    tag: str | None = None
    expected_commit: str | None = None
    artifact_digest: str | None = None


@dataclass(frozen=True)
class SignedTagManifest:
    """Release bindings parsed from the SIGNED upstream annotated tag object.

    ``tag`` and ``commit`` come from signed tag-object headers; ``name`` and
    ``version`` come from the signed upstream release subject. Optional legacy
    claims remain representable for deterministic compatibility tests, but the
    real upstream parser does not invent claims absent from the signed object.
    """

    name: str
    version: str
    tag: str
    commit: str
    platform: str | None = None
    arch: str | None = None
    origin: str | None = None


@dataclass(frozen=True)
class VerifiedProvenance:
    """Result of a successful signature verification (pre-binding)."""

    signer_principal: str
    key_type: str
    key_fingerprint: str
    manifest: SignedTagManifest
    commit: str


@dataclass(frozen=True)
class ProvenancePolicy:
    """Operator-provisioned trust policy. Fail-closed unless configured."""

    source_kind: str
    allowed_signers_file: Path | None = None
    allowed_signers_text: str | None = None
    allowed_origins: frozenset[str] = frozenset()
    allowed_principals: frozenset[str] = frozenset()
    allowed_fingerprints: frozenset[str] = frozenset()
    allowed_platforms: frozenset[str] = frozenset()
    allowed_arches: frozenset[str] = frozenset()
    min_version: str | None = None
    trusted_ssh_keygen: Path | None = None
    pypi_attestation_configured: bool = False

    def is_configured(self) -> bool:
        """True only when enough trust anchors exist to verify at all.

        * git source: an existing non-empty ``allowed_signers`` file OR inline
          pinned public policy, plus a non-empty origin allowlist.
        * pypi: an attestation verifier must be explicitly configured (it is
          not shipped), so this is False by default -> pypi fails closed.
        """
        if self.source_kind == "git":
            file_ok = (
                self.allowed_signers_file is not None
                and self.allowed_signers_file.is_file()
                and self.allowed_signers_file.stat().st_size > 0
            )
            inline_ok = bool(
                self.allowed_signers_text and self.allowed_signers_text.strip()
            )
            signers_ok = file_ok or inline_ok
            return bool(signers_ok and self.allowed_origins)
        if self.source_kind == "pypi":
            return bool(self.pypi_attestation_configured and self.allowed_origins)
        return False

    @classmethod
    def from_environ(
        cls,
        environ: dict[str, str],
        *,
        source_kind: str,
        current_version: str | None = None,
    ) -> "ProvenancePolicy":
        """Build a policy from ``MS4_HERMES_PROVENANCE_*`` environment keys.

        The anti-downgrade floor is the MAX of the installed version and any
        explicit ``MS4_HERMES_PROVENANCE_MIN_VERSION`` so a release is never
        accepted below what is already installed (replay/downgrade defense),
        even if the operator forgets to set an explicit floor.
        """
        signers_raw = environ.get("MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS", "").strip()
        signers_file = Path(signers_raw).expanduser() if signers_raw else None
        signers_text = (
            environ.get("MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS_INLINE") or ""
        ).strip() or None
        origins = _split_env(environ.get("MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS"))
        principals = _split_env(environ.get("MS4_HERMES_PROVENANCE_ALLOWED_PRINCIPALS"))
        fingerprints = _split_env(
            environ.get("MS4_HERMES_PROVENANCE_ALLOWED_FINGERPRINTS")
        )
        platforms = _split_env(environ.get("MS4_HERMES_PROVENANCE_PLATFORMS"))
        arches = _split_env(environ.get("MS4_HERMES_PROVENANCE_ARCHES"))
        env_floor = (environ.get("MS4_HERMES_PROVENANCE_MIN_VERSION") or "").strip() or None
        floor = _max_version(current_version, env_floor)
        ssh_keygen_raw = (
            environ.get("MS4_HERMES_PROVENANCE_SSH_KEYGEN") or ""
        ).strip()
        ssh_keygen = Path(ssh_keygen_raw).expanduser() if ssh_keygen_raw else None
        pypi_attest = _env_flag(environ.get("MS4_HERMES_PROVENANCE_PYPI_ATTESTATION"))
        return cls(
            source_kind=source_kind,
            allowed_signers_file=signers_file,
            allowed_signers_text=signers_text,
            allowed_origins=frozenset(normalize_origin(o) for o in origins),
            allowed_principals=frozenset(principals),
            allowed_fingerprints=frozenset(fingerprints),
            allowed_platforms=frozenset(normalize_platform(p) for p in platforms),
            allowed_arches=frozenset(normalize_arch(a) for a in arches),
            min_version=floor,
            trusted_ssh_keygen=ssh_keygen,
            pypi_attestation_configured=pypi_attest,
        )


def _split_env(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [tok for tok in re.split(r"[,\s]+", raw.strip()) if tok]


def _env_flag(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


_PRERELEASE_VERSION_RE = re.compile(r"^\s*v?\d+(?:\.\d+){0,2}-")


def _reject_prerelease(value: str | None, label: str) -> None:
    if value and _PRERELEASE_VERSION_RE.match(value):
        raise ProvenanceError(
            f"{label} {value!r} is a prerelease; provenance updates require stable versions"
        )


def _max_version(a: str | None, b: str | None) -> str | None:
    """Return the greater valid floor; reject malformed non-empty floors."""
    parsed = []
    for label, value in (("current version", a), ("configured minimum version", b)):
        if not value:
            continue
        _reject_prerelease(value, label)
        parsed_value = versioning.parse_semver(value)
        if parsed_value is None:
            raise ProvenanceError(f"{label} {value!r} is not parseable")
        parsed.append((parsed_value, value))
    if not parsed:
        return None
    parsed.sort(key=lambda pv: pv[0])
    return parsed[-1][1]


# --------------------------------------------------------------------------- #
# Signature verifiers.
# --------------------------------------------------------------------------- #
class SignatureVerifier(Protocol):
    def verify(self, request: ReleaseProvenanceRequest) -> VerifiedProvenance:
        ...


_GOOD_SIG_RE = re.compile(
    r'signature for (?P<principal>\S+) with (?P<keytype>\S+) key (?P<fpr>SHA256:\S+)'
)
_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_GIT_REPOSITORY_ENV_VARS = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    }
)
_UPSTREAM_SUBJECT_RE = re.compile(
    r"^Hermes Agent v(?P<version>[0-9A-Za-z][0-9A-Za-z._+-]{0,63}) "
    r"\((?P<tag_version>[0-9]+(?:\.[0-9]+){2,3})\)$"
)


def _default_git_runner(argv: list[str]) -> subprocess.CompletedProcess:
    """Shell-free, non-interactive git invocation (argv array, stdin closed)."""
    env = os.environ.copy()
    for key in tuple(env):
        if (
            key in _GIT_REPOSITORY_ENV_VARS
            or key in {"GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"}
            or re.fullmatch(r"GIT_CONFIG_(?:KEY|VALUE)_\d+", key)
        ):
            env.pop(key, None)
    if sys.platform == "win32":
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "core.longpaths"
        env["GIT_CONFIG_VALUE_0"] = "true"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["LANGUAGE"] = "C"
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        env=env,
    )


def parse_upstream_tag_object(tag_object_text: str) -> SignedTagManifest | None:
    """Parse bindings from Hermes' actual signed annotated-tag format."""
    headers_text, separator, body = tag_object_text.partition("\n\n")
    if not separator:
        return None
    fields: dict[str, str] = {}
    for line in headers_text.splitlines():
        key, separator, value = line.partition(" ")
        if key in {"object", "type", "tag"}:
            if not separator or key in fields:
                return None
            fields[key] = value.strip()
    commit = fields.get("object", "")
    tag = fields.get("tag", "")
    if fields.get("type") != "commit" or not _COMMIT_RE.fullmatch(commit) or not tag:
        return None
    subject = body.splitlines()[0].strip() if body.splitlines() else ""
    match = _UPSTREAM_SUBJECT_RE.fullmatch(subject)
    if match is None or tag != f"v{match.group('tag_version')}":
        return None
    return SignedTagManifest(
        name="hermes-agent",
        version=match.group("version"),
        tag=tag,
        commit=commit.lower(),
    )


def parse_verify_output(stderr_text: str) -> tuple[str, str, str] | None:
    """Extract ``(principal, key_type, fingerprint)`` from git verify output.

    git writes the human-readable verification line to STDERR (confirmed by the
    hermetic spike). A successful, principal-matched verification looks like:
    ``Good "git" signature for you@example with ED25519 key SHA256:...``.
    """
    match = _GOOD_SIG_RE.search(stderr_text or "")
    if not match:
        return None
    return match.group("principal"), match.group("keytype"), match.group("fpr")


@dataclass
class GitSignedTagVerifier:
    """Verify an SSH-signed git tag via ``git verify-tag`` + allowed_signers."""

    allowed_signers_file: Path | None
    directory: str
    allowed_signers_text: str | None = None
    trusted_ssh_keygen: Path | None = None
    runner: Callable[[list[str]], subprocess.CompletedProcess] = _default_git_runner

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return self.runner(["git", "-C", self.directory, *args])

    def read_origin(self) -> str:
        proc = self._git("remote", "get-url", "origin")
        if proc.returncode != 0:
            raise ProvenanceError("could not read the checkout's origin remote URL")
        url = (proc.stdout or "").strip()
        if not url:
            raise ProvenanceError("checkout origin remote URL is empty")
        return url

    def require_trusted_ssh_keygen(self) -> Path:
        program = self.trusted_ssh_keygen
        if program is None or not program.is_absolute() or not program.is_file():
            raise ProvenanceError(
                "trusted ssh-keygen program is not configured as an existing absolute file"
            )
        return program

    def assert_no_replace_refs(self) -> None:
        proc = self._git("for-each-ref", "--format=%(refname)", "refs/replace/")
        if proc.returncode != 0:
            raise ProvenanceError("could not inspect repository replace refs")
        refs = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if refs:
            raise ProvenanceError(
                f"repository contains forbidden replace refs: {', '.join(refs[:4])}"
            )

    def resolve_tag_object(self, tag: str) -> str:
        proc = self._git(
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"refs/tags/{tag}^{{tag}}",
        )
        tag_object = (proc.stdout or "").strip()
        if proc.returncode != 0 or not _COMMIT_RE.fullmatch(tag_object):
            raise ProvenanceError(f"could not resolve annotated tag object for {tag!r}")
        return tag_object.lower()

    def verify(self, request: ReleaseProvenanceRequest) -> VerifiedProvenance:
        tag = request.tag
        if not tag:
            raise ProvenanceError("editable provenance requires a resolved release tag")
        if self.allowed_signers_file is None and not self.allowed_signers_text:
            raise ProvenanceError("no allowed_signers policy file configured (fail-closed)")
        trusted_ssh_keygen = self.require_trusted_ssh_keygen()
        self.assert_no_replace_refs()
        tag_object = self.resolve_tag_object(tag)
        # 1) signature + authorized-signer check. Non-zero exit covers unsigned
        #    tags, annotated-but-unsigned tags, and keys outside allowed_signers.
        if self.allowed_signers_text:
            with tempfile.TemporaryDirectory(prefix="ms4-hermes-signers-") as temp_dir:
                signers_file = Path(temp_dir) / "allowed_signers"
                signers_file.write_text(
                    self.allowed_signers_text.rstrip() + "\n", encoding="utf-8"
                )
                verified = self._git(
                    "-c", "gpg.format=ssh",
                    "-c", f"gpg.ssh.program={trusted_ssh_keygen}",
                    "-c", f"gpg.ssh.allowedSignersFile={signers_file}",
                    "verify-tag", tag_object,
                )
        else:
            verified = self._git(
                "-c", "gpg.format=ssh",
                "-c", f"gpg.ssh.program={trusted_ssh_keygen}",
                "-c", f"gpg.ssh.allowedSignersFile={self.allowed_signers_file}",
                "verify-tag", tag_object,
            )
        if verified.returncode != 0:
            detail = (verified.stderr or "").strip().splitlines()
            tail = detail[-1] if detail else "no detail"
            raise ProvenanceError(
                f"tag {tag!r} did not verify against the authorized signers: {tail[:200]}"
            )
        ident = parse_verify_output(verified.stderr or "")
        if ident is None:
            raise ProvenanceError(
                f"tag {tag!r} verified but no authorized signer principal was reported"
            )
        principal, keytype, fingerprint = ident
        # 2) actual upstream release bindings carried by the signed tag object.
        raw = self._git("cat-file", "tag", tag_object)
        if raw.returncode != 0:
            raise ProvenanceError(f"could not read signed tag object {tag_object!r}")
        manifest = parse_upstream_tag_object(raw.stdout or "")
        if manifest is None:
            raise ProvenanceError(
                f"signed tag {tag!r} is missing well-formed upstream release metadata"
            )
        # 3) independent tag -> commit resolution (artifact-digest anchor).
        resolved = self._git(
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{tag_object}^{{commit}}",
        )
        commit = (resolved.stdout or "").strip()
        if resolved.returncode != 0 or not _COMMIT_RE.fullmatch(commit):
            raise ProvenanceError(f"could not resolve signed tag {tag!r} to a commit id")
        return VerifiedProvenance(
            signer_principal=principal,
            key_type=keytype,
            key_fingerprint=fingerprint,
            manifest=manifest,
            commit=commit,
        )


@dataclass
class PyPiAttestationVerifier:
    """Fail-closed placeholder for PyPI wheel provenance (PEP 740 / Sigstore).

    No attestation verifier ships in this candidate. ``verify`` always refuses,
    documenting the explicit integration dependency rather than silently
    downgrading pypi trust to transport/hash-only. A future integration wires a
    real Sigstore/PEP-740 verifier here (and flips
    ``ProvenancePolicy.pypi_attestation_configured`` on).
    """

    def verify(self, request: ReleaseProvenanceRequest) -> VerifiedProvenance:
        raise ProvenanceError(
            "pypi attestation verifier is not configured (fail-closed): PyPI wheel "
            "provenance (PEP 740 / Sigstore) is an unshipped integration dependency"
        )


# --------------------------------------------------------------------------- #
# Core gate + binding.
# --------------------------------------------------------------------------- #
def _bind(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ProvenanceError(
            f"{label} mismatch: signed={actual!r} does not match requested={expected!r}"
        )


def _bind_token(claim: str, want: str, label: str) -> None:
    """Bind a manifest token that may be the wildcard ``any`` (portable source)."""
    if claim != want and claim.strip().lower() != "any":
        raise ProvenanceError(
            f"{label} mismatch: release claims {claim!r}, target is {want!r}"
        )


def verify_release_provenance(
    request: ReleaseProvenanceRequest,
    policy: ProvenancePolicy,
    verifier: SignatureVerifier | None,
) -> VerifiedProvenance:
    """Fail-closed provenance decision. Raises ``ProvenanceError`` on refusal.

    Order matters: policy/identity/freshness checks run first, THEN the
    cryptographic signature check, THEN the signed tag object is bound back to the
    request. Every path that cannot prove authenticity raises.
    """
    if request.source_kind != policy.source_kind:
        raise ProvenanceError(
            f"source kind mismatch: request={request.source_kind!r}, "
            f"policy={policy.source_kind!r}"
        )
    if policy.source_kind == "git" and (
        request.expected_commit is None
        or not _COMMIT_RE.fullmatch(request.expected_commit)
    ):
        raise ProvenanceError(
            "git provenance requires a valid 40/64-hex expected commit"
        )
    _reject_prerelease(request.version, "release version")
    _reject_prerelease(policy.min_version, "configured minimum version")
    if not policy.is_configured():
        raise ProvenanceError(
            "no provenance policy configured (fail-closed); refusing to trust the "
            f"{request.source_kind} release"
        )
    if verifier is None:
        raise ProvenanceError("no signature verifier available (fail-closed)")

    # Origin policy -- reject origin confusion (a release from an un-allowlisted
    # source) BEFORE trusting any of its content.
    if normalize_origin(request.origin) not in policy.allowed_origins:
        raise ProvenanceError(
            f"release origin {request.origin!r} is not in the configured origin allowlist"
        )

    # Freshness -- reject downgrade / replay of an older signed release.
    req_ver = versioning.parse_semver(request.version)
    if req_ver is None:
        raise ProvenanceError(f"release version {request.version!r} is not parseable")
    if policy.min_version is not None:
        floor = versioning.parse_semver(policy.min_version)
        if floor is None:
            raise ProvenanceError(
                f"configured minimum version {policy.min_version!r} is not parseable"
            )
        if req_ver < floor:
            raise ProvenanceError(
                f"release version {request.version} is below the floor "
                f"{policy.min_version} (downgrade/replay refused)"
            )

    # Optional platform/arch allowlists (host is permitted to take this release).
    if policy.allowed_platforms and request.platform not in policy.allowed_platforms:
        raise ProvenanceError(f"platform {request.platform!r} is not permitted by policy")
    if policy.allowed_arches and request.arch not in policy.allowed_arches:
        raise ProvenanceError(f"architecture {request.arch!r} is not permitted by policy")

    # Cryptographic signature + signer identity (raises on unsigned/unknown key).
    verified = verifier.verify(request)

    # Bind the signed upstream tag object to the request. The tag alias,
    # package version, and target commit are covered by the signature.
    manifest = verified.manifest
    _bind(manifest.name, request.release_name, "release name")
    _bind(manifest.version, request.version, "release version")
    _bind(manifest.tag, request.tag or "", "release tag")
    if manifest.platform is not None:
        _bind_token(manifest.platform, request.platform, "platform")
    if manifest.arch is not None:
        _bind_token(manifest.arch, request.arch, "architecture")
    if manifest.origin is not None:
        _bind(normalize_origin(manifest.origin), normalize_origin(request.origin), "origin")
    if policy.source_kind == "git":
        expected_commit = request.expected_commit or ""
        _bind(manifest.commit.lower(), expected_commit.lower(), "artifact digest (commit)")
        _bind(verified.commit.lower(), expected_commit.lower(), "resolved commit")

    # Signer identity allowlist (beyond the allowed_signers file, an optional
    # explicit principal allowlist) -- reject an unexpected authorized signer.
    if policy.allowed_principals and verified.signer_principal not in policy.allowed_principals:
        raise ProvenanceError(
            f"signer {verified.signer_principal!r} is not in the authorized principal allowlist"
        )
    if (
        policy.allowed_fingerprints
        and verified.key_fingerprint not in policy.allowed_fingerprints
    ):
        raise ProvenanceError(
            f"signer fingerprint {verified.key_fingerprint!r} is not authorized"
        )
    return verified


def verify_editable_preflight(
    *,
    directory: str,
    current_version: str | None,
    environ: dict[str, str],
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> str:
    """Validate pinned policy, replace-ref absence, and origin before fetch."""
    policy = ProvenancePolicy.from_environ(
        environ,
        source_kind="git",
        current_version=current_version,
    )
    if not policy.is_configured():
        raise ProvenanceError(
            "no provenance policy configured (fail-closed); refusing pre-fetch access"
        )
    verifier = GitSignedTagVerifier(
        allowed_signers_file=policy.allowed_signers_file,
        directory=directory,
        allowed_signers_text=policy.allowed_signers_text,
        trusted_ssh_keygen=policy.trusted_ssh_keygen,
        runner=runner or _default_git_runner,
    )
    verifier.require_trusted_ssh_keygen()
    verifier.assert_no_replace_refs()
    origin = verifier.read_origin()
    if normalize_origin(origin) not in policy.allowed_origins:
        raise ProvenanceError(
            f"release origin {origin!r} is not in the configured origin allowlist"
        )
    return origin


def verify_update_provenance(
    *,
    mode: str,
    release_name: str,
    version: str,
    tag: str | None,
    directory: str | None,
    expected_commit: str | None,
    current_version: str | None,
    environ: dict[str, str],
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> VerifiedProvenance:
    """High-level gate used by the installer. Fail-closed by construction.

    Builds the policy from the environment and the appropriate verifier for the
    install mode, assembles the request (reading origin + binding the fetched
    commit as the artifact digest for editable installs), and delegates to
    :func:`verify_release_provenance`.
    """
    _reject_prerelease(version, "release version")
    if mode == "editable":
        if expected_commit is None or not _COMMIT_RE.fullmatch(expected_commit):
            raise ProvenanceError(
                "editable provenance requires a valid 40/64-hex expected commit"
            )
        policy = ProvenancePolicy.from_environ(
            environ, source_kind="git", current_version=current_version
        )
        if directory is None:
            raise ProvenanceError("editable provenance requires a checkout directory")
        verifier = GitSignedTagVerifier(
            allowed_signers_file=policy.allowed_signers_file,
            directory=directory,
            allowed_signers_text=policy.allowed_signers_text,
            trusted_ssh_keygen=policy.trusted_ssh_keygen,
            runner=runner or _default_git_runner,
        )
        origin = verifier.read_origin()
        request = ReleaseProvenanceRequest(
            release_name=release_name,
            version=version,
            platform=normalize_platform(),
            arch=normalize_arch(),
            origin=origin,
            source_kind="git",
            tag=tag,
            expected_commit=expected_commit,
            artifact_digest=expected_commit,
        )
        return verify_release_provenance(request, policy, verifier)

    if mode == "pypi":
        policy = ProvenancePolicy.from_environ(
            environ, source_kind="pypi", current_version=current_version
        )
        request = ReleaseProvenanceRequest(
            release_name=release_name,
            version=version,
            platform=normalize_platform(),
            arch=normalize_arch(),
            origin="https://pypi.org/project/hermes-agent/",
            source_kind="pypi",
            tag=None,
            expected_commit=None,
            artifact_digest=None,
        )
        return verify_release_provenance(request, policy, PyPiAttestationVerifier())

    raise ProvenanceError(f"unknown install mode for provenance: {mode!r}")
