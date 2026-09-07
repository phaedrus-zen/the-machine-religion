"""F4 supply-chain provenance -- focused red/green suite (candidate author).

Hermetic, offline, self-contained. Three layers:

* Layer A (deterministic, no subprocess): the policy + binding engine
  (:func:`provenance.verify_release_provenance`) driven by a STUB verifier, plus
  the structured parsers/normalizers and the fail-closed pypi path. Every
  refusal DIRECTION is a separate, named test so a mutant that drops one check
  is killed by exactly one test.
* Layer B (real git + real ssh): :class:`provenance.GitSignedTagVerifier`
  against genuinely SSH-signed tags -- GOOD accepts; unsigned / wrong-key /
  annotated-unsigned / manifest-less / no-policy all REFUSE. Skips (never
  fails) if git or ssh-keygen is unavailable on the host.
* Layer C (installer integration, ``@pytest.mark.real_provenance`` so the
  autouse neutralizer does NOT stub the seam): the real gate delegated through
  ``installer._verify_release_provenance``, and the security-critical
  VERIFICATION-BEFORE-MUTATION ordering proven end-to-end through
  ``run_update_job`` (a refusal leaves HEAD untouched and never reaches the
  detached checkout).
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from machine_spirit_4.hermes_admin import installer, provenance, state, versioning
from machine_spirit_4.scripts import runtime_common

GIT = shutil.which("git")
SSH_KEYGEN = shutil.which("ssh-keygen")
requires_git = pytest.mark.skipif(GIT is None, reason="git not on PATH")
requires_ssh = pytest.mark.skipif(
    GIT is None or SSH_KEYGEN is None, reason="git or ssh-keygen not on PATH"
)
UPSTREAM_TAG = "v2026.7.7.2"


@pytest.fixture(autouse=True)
def _pin_signed_only_release_policy(monkeypatch):
    """This module proves the F4 signed-only gate. Pin the strict policy so the
    default ``allow_unsigned`` release policy (operator decision 2026-09-07)
    cannot turn a refusal proof into a pass."""
    monkeypatch.setenv(
        versioning.RELEASE_SIGNATURE_POLICY_ENV,
        versioning.RELEASE_SIGNATURE_POLICY_REQUIRE_SIGNED,
    )
UPSTREAM_TAG_VERSION = "2026.7.7.2"
UPSTREAM_VERSION = "0.18.2"
UPSTREAM_COMMIT = "9de9c25f620ff7f1ce0fd5457d596052d5159596"
UPSTREAM_ORIGIN = "https://github.com/NousResearch/hermes-agent.git"
UPSTREAM_SIGNER_PRINCIPAL = "127238744+teknium1@users.noreply.github.com"
UPSTREAM_SIGNER_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIPpWPAE2WMbZ0fAZ8xsqiTIJqA28qDBfGru8kPrpNyUb"
)
UPSTREAM_SIGNER_FINGERPRINT = "SHA256:x9xNOpeJhoEAY2gWhmWHZROC3QF3VjOEbmNo9vQ8y2A"
UPSTREAM_ALLOWED_SIGNERS = (
    f'{UPSTREAM_SIGNER_PRINCIPAL} namespaces="git" {UPSTREAM_SIGNER_KEY}'
)


# --------------------------------------------------------------------------- #
# Layer A helpers -- deterministic engine exercise via a stub verifier.
# --------------------------------------------------------------------------- #
class _StubVerifier:
    """Records whether it was invoked so we can prove PRE-signature refusals
    short-circuit before any crypto is attempted (ordering inside the gate)."""

    def __init__(self, verified: provenance.VerifiedProvenance) -> None:
        self.verified = verified
        self.calls = 0

    def verify(self, request: provenance.ReleaseProvenanceRequest) -> provenance.VerifiedProvenance:
        self.calls += 1
        return self.verified


def _base(tmp_path: Path):
    """A fully-consistent (policy, request, stub) triple -- the GREEN baseline.

    Individual RED tests mutate exactly one dimension via ``dataclasses.replace``.
    """
    origin = "https://github.com/acme/hermes-agent"
    signers = tmp_path / "allowed_signers"
    signers.write_text("release-bot ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    policy = provenance.ProvenancePolicy(
        source_kind="git",
        allowed_signers_file=signers,
        allowed_origins=frozenset({provenance.normalize_origin(origin)}),
    )
    request = provenance.ReleaseProvenanceRequest(
        release_name="hermes-agent",
        version="1.4.0",
        platform="linux",
        arch="x86_64",
        origin=origin,
        source_kind="git",
        tag="v1.4.0",
        expected_commit="a" * 40,
    )
    manifest = provenance.SignedTagManifest(
        name="hermes-agent",
        version="1.4.0",
        tag="v1.4.0",
        commit="a" * 40,
        platform="any",
        arch="any",
        origin=origin,
    )
    verified = provenance.VerifiedProvenance(
        signer_principal="release-bot",
        key_type="ED25519",
        key_fingerprint="SHA256:placeholder",
        manifest=manifest,
        commit="a" * 40,
    )
    return policy, request, _StubVerifier(verified)


def _restub(stub: _StubVerifier, **manifest_overrides) -> _StubVerifier:
    """Return a fresh stub whose manifest (and optionally resolved commit) is
    a mutated copy of ``stub``'s verified result."""
    commit = manifest_overrides.pop("_resolved_commit", stub.verified.commit)
    manifest = dataclasses.replace(stub.verified.manifest, **manifest_overrides)
    verified = dataclasses.replace(stub.verified, manifest=manifest, commit=commit)
    return _StubVerifier(verified)


def test_a_green_accepts_consistent_release(tmp_path):
    policy, request, stub = _base(tmp_path)
    result = provenance.verify_release_provenance(request, policy, stub)
    assert result.signer_principal == "release-bot"
    assert result.manifest.version == "1.4.0"
    assert stub.calls == 1


def test_a_unconfigured_policy_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    empty = dataclasses.replace(policy, allowed_origins=frozenset())
    with pytest.raises(provenance.ProvenanceError, match="no provenance policy configured"):
        provenance.verify_release_provenance(request, empty, stub)
    assert stub.calls == 0


def test_a_missing_signers_file_is_unconfigured(tmp_path):
    policy, request, stub = _base(tmp_path)
    gone = dataclasses.replace(policy, allowed_signers_file=tmp_path / "nope")
    with pytest.raises(provenance.ProvenanceError, match="no provenance policy configured"):
        provenance.verify_release_provenance(request, gone, stub)
    assert stub.calls == 0


def test_a_none_verifier_refuses(tmp_path):
    policy, request, _stub = _base(tmp_path)
    with pytest.raises(provenance.ProvenanceError, match="no signature verifier"):
        provenance.verify_release_provenance(request, policy, None)


def test_a_source_kind_mismatch_refuses_before_verifier(tmp_path):
    policy, request, stub = _base(tmp_path)
    mislabeled = dataclasses.replace(
        request,
        source_kind="pypi",
        expected_commit=None,
    )
    with pytest.raises(provenance.ProvenanceError, match="source kind"):
        provenance.verify_release_provenance(mislabeled, policy, stub)
    assert stub.calls == 0


def test_a_origin_confusion_refuses_before_signature(tmp_path):
    policy, request, stub = _base(tmp_path)
    evil = dataclasses.replace(request, origin="https://evil.example/hermes-agent")
    with pytest.raises(provenance.ProvenanceError, match="origin"):
        provenance.verify_release_provenance(evil, policy, stub)
    assert stub.calls == 0  # refused before any crypto


def test_a_downgrade_replay_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    floored = dataclasses.replace(policy, min_version="2.0.0")
    with pytest.raises(provenance.ProvenanceError, match="downgrade"):
        provenance.verify_release_provenance(request, floored, stub)
    assert stub.calls == 0


def test_a_stable_to_prerelease_refuses_before_signature(tmp_path):
    policy, request, stub = _base(tmp_path)
    stable_floor = dataclasses.replace(policy, min_version="1.4.0")
    prerelease = dataclasses.replace(request, version="1.4.0-rc1")
    with pytest.raises(provenance.ProvenanceError, match="prerelease"):
        provenance.verify_release_provenance(prerelease, stable_floor, stub)
    assert stub.calls == 0


@pytest.mark.parametrize(
    ("current_version", "configured_floor"),
    [
        ("1.4.0-rc2", None),
        (None, "1.4.0-rc1"),
    ],
)
def test_a_prerelease_current_or_floor_refuses(current_version, configured_floor):
    env = {}
    if configured_floor is not None:
        env["MS4_HERMES_PROVENANCE_MIN_VERSION"] = configured_floor
    with pytest.raises(provenance.ProvenanceError, match="prerelease"):
        provenance.ProvenancePolicy.from_environ(
            env,
            source_kind="git",
            current_version=current_version,
        )


def test_a_rc2_to_rc1_refuses_before_git():
    def forbidden_git(_argv):
        raise AssertionError("git ran before prerelease refusal")

    with pytest.raises(provenance.ProvenanceError, match="prerelease"):
        provenance.verify_update_provenance(
            mode="editable",
            release_name="hermes-agent",
            version="1.4.0-rc1",
            tag=UPSTREAM_TAG,
            directory="unused",
            expected_commit="a" * 40,
            current_version="1.4.0-rc2",
            environ={},
            runner=forbidden_git,
        )


def test_a_unparseable_version_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    junk = dataclasses.replace(request, version="not-a-semver")
    with pytest.raises(provenance.ProvenanceError, match="parseable"):
        provenance.verify_release_provenance(junk, policy, stub)


def test_a_platform_not_permitted_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    restricted = dataclasses.replace(policy, allowed_platforms=frozenset({"win32"}))
    with pytest.raises(provenance.ProvenanceError, match="platform"):
        provenance.verify_release_provenance(request, restricted, stub)
    assert stub.calls == 0


def test_a_arch_not_permitted_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    restricted = dataclasses.replace(policy, allowed_arches=frozenset({"aarch64"}))
    with pytest.raises(provenance.ProvenanceError, match="architecture"):
        provenance.verify_release_provenance(request, restricted, stub)
    assert stub.calls == 0


def test_a_bind_name_mismatch_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    bad = _restub(stub, name="totally-other-agent")
    with pytest.raises(provenance.ProvenanceError, match="release name"):
        provenance.verify_release_provenance(request, policy, bad)
    assert bad.calls == 1  # signature ran; binding caught it


def test_a_bind_version_mismatch_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    bad = _restub(stub, version="9.9.9")
    with pytest.raises(provenance.ProvenanceError, match="release version"):
        provenance.verify_release_provenance(request, policy, bad)


def test_a_bind_tag_mismatch_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    bad = _restub(stub, tag="v9.9.9")
    with pytest.raises(provenance.ProvenanceError, match="release tag"):
        provenance.verify_release_provenance(request, policy, bad)


def test_a_bind_platform_mismatch_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    bad = _restub(stub, platform="win32")  # concrete, non-"any" -> must match
    with pytest.raises(provenance.ProvenanceError, match="platform mismatch"):
        provenance.verify_release_provenance(request, policy, bad)


def test_a_bind_arch_mismatch_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    bad = _restub(stub, arch="aarch64")
    with pytest.raises(provenance.ProvenanceError, match="architecture mismatch"):
        provenance.verify_release_provenance(request, policy, bad)


def test_a_bind_origin_mismatch_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    bad = _restub(stub, origin="https://github.com/acme/other-repo")
    with pytest.raises(provenance.ProvenanceError, match="origin mismatch"):
        provenance.verify_release_provenance(request, policy, bad)


def test_a_bind_commit_digest_mismatch_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    bad = _restub(stub, commit="b" * 40, _resolved_commit="b" * 40)
    with pytest.raises(provenance.ProvenanceError, match="artifact digest"):
        provenance.verify_release_provenance(request, policy, bad)


def test_a_resolved_commit_mismatch_refuses(tmp_path):
    """Manifest commit matches the request, but the independently-resolved tag
    commit does not -- tag ref moved under a re-used manifest. Must refuse."""
    policy, request, stub = _base(tmp_path)
    bad = _restub(stub, _resolved_commit="c" * 40)
    with pytest.raises(provenance.ProvenanceError, match="resolved commit"):
        provenance.verify_release_provenance(request, policy, bad)


def test_a_unauthorized_principal_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    restricted = dataclasses.replace(policy, allowed_principals=frozenset({"trusted-bot"}))
    with pytest.raises(provenance.ProvenanceError, match="authorized principal"):
        provenance.verify_release_provenance(request, restricted, stub)
    assert stub.calls == 1  # identity checked after a real signature


def test_a_wrong_pinned_fingerprint_refuses(tmp_path):
    policy, request, stub = _base(tmp_path)
    pinned = dataclasses.replace(
        policy,
        allowed_fingerprints=frozenset({"SHA256:not-the-signer"}),
    )
    with pytest.raises(provenance.ProvenanceError, match="fingerprint"):
        provenance.verify_release_provenance(request, pinned, stub)
    assert stub.calls == 1


def test_a_portable_any_platform_and_arch_accepted(tmp_path):
    """A manifest that claims ``any`` binds to any concrete host platform/arch."""
    policy, request, stub = _base(tmp_path)
    req = dataclasses.replace(request, platform="darwin", arch="aarch64")
    result = provenance.verify_release_provenance(req, policy, stub)
    assert result.manifest.platform == "any"


# --------------------------------------------------------------------------- #
# Layer A -- structured parsers / normalizers (no ad hoc string parsing).
# --------------------------------------------------------------------------- #
def test_a_parse_upstream_tag_object_good():
    text = (
        f"object {UPSTREAM_COMMIT}\n"
        "type commit\n"
        f"tag {UPSTREAM_TAG}\n"
        "tagger teknium1 <127238744+teknium1@users.noreply.github.com> 1783458000 -0700\n"
        "\n"
        f"Hermes Agent v{UPSTREAM_VERSION} ({UPSTREAM_TAG_VERSION})\n\n"
        "Official upstream release notes.\n"
        "-----BEGIN SSH SIGNATURE-----\nU1NIU0lH...\n-----END SSH SIGNATURE-----\n"
    )
    manifest = provenance.parse_upstream_tag_object(text)
    assert manifest is not None
    assert manifest.name == "hermes-agent"
    assert manifest.version == UPSTREAM_VERSION
    assert manifest.tag == UPSTREAM_TAG
    assert manifest.commit == UPSTREAM_COMMIT


def test_a_parse_upstream_tag_object_missing_headers_is_none():
    assert provenance.parse_upstream_tag_object("no tag object here") is None


def test_a_parse_upstream_tag_object_tag_message_mismatch_is_none():
    text = (
        f"object {UPSTREAM_COMMIT}\n"
        "type commit\n"
        f"tag {UPSTREAM_TAG}\n\n"
        f"Hermes Agent v{UPSTREAM_VERSION} (2026.7.7.1)\n"
    )
    assert provenance.parse_upstream_tag_object(text) is None


def test_a_parse_verify_output_good():
    line = (
        'Good "git" signature for hermes-signer@example.invalid with '
        "ED25519 key SHA256:98xsKws7nlI9t1K5+a2ev6o69IfCaelnVSE/HBnXBa0"
    )
    parsed = provenance.parse_verify_output(line)
    assert parsed == (
        "hermes-signer@example.invalid",
        "ED25519",
        "SHA256:98xsKws7nlI9t1K5+a2ev6o69IfCaelnVSE/HBnXBa0",
    )


def test_a_parse_verify_output_no_principal_is_none():
    # "No principal matched." shape -> not a trusted, principal-bound line.
    assert provenance.parse_verify_output("Good \"git\" signature with ED25519 key SHA256:x") is None
    assert provenance.parse_verify_output("") is None


def test_a_normalizers():
    assert provenance.normalize_arch("amd64") == "x86_64"
    assert provenance.normalize_arch("ARM64") == "aarch64"
    assert provenance.normalize_arch("x86_64") == "x86_64"
    assert provenance.normalize_platform("  Win32 ") == "win32"
    assert provenance.normalize_origin("https://github.com/A/B.git/") == "https://github.com/a/b"
    assert provenance.normalize_origin("C:\\Repos\\Hermes\\") == "c:/repos/hermes"
    assert provenance.normalize_origin(None) == ""


def test_a_runtime_env_pins_official_upstream_policy(monkeypatch):
    monkeypatch.setenv("MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS", "wrong-file")
    monkeypatch.setenv("MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS_INLINE", "wrong-key")
    monkeypatch.setenv("MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS", "https://evil.invalid/repo")
    env = runtime_common.ms4_env()
    assert "MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS" not in env
    assert env["MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS_INLINE"] == UPSTREAM_ALLOWED_SIGNERS
    assert env["MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS"] == UPSTREAM_ORIGIN
    assert env["MS4_HERMES_PROVENANCE_ALLOWED_PRINCIPALS"] == UPSTREAM_SIGNER_PRINCIPAL
    assert env["MS4_HERMES_PROVENANCE_ALLOWED_FINGERPRINTS"] == UPSTREAM_SIGNER_FINGERPRINT
    expected_ssh_keygen = (
        r"C:\Windows\System32\OpenSSH\ssh-keygen.exe"
        if runtime_common.sys.platform == "win32"
        else "/usr/bin/ssh-keygen"
    )
    assert env["MS4_HERMES_PROVENANCE_SSH_KEYGEN"] == expected_ssh_keygen


@pytest.mark.parametrize(
    ("current_version", "configured_floor", "expected"),
    [
        ("not-a-version", None, "current version"),
        ("1.4.0", "not-a-version", "configured minimum version"),
    ],
)
def test_a_malformed_downgrade_floors_refuse(
    tmp_path,
    current_version,
    configured_floor,
    expected,
):
    signers = tmp_path / "allowed_signers"
    signers.write_text("release-bot ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    env = {
        "MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS": str(signers),
        "MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS": UPSTREAM_ORIGIN,
    }
    if configured_floor is not None:
        env["MS4_HERMES_PROVENANCE_MIN_VERSION"] = configured_floor
    with pytest.raises(provenance.ProvenanceError, match=expected):
        provenance.ProvenancePolicy.from_environ(
            env,
            source_kind="git",
            current_version=current_version,
        )


@pytest.mark.parametrize("expected_commit", [None, "not-a-commit", "a" * 39])
def test_a_editable_expected_commit_is_required_before_git(expected_commit):
    def forbidden_git(_argv):
        raise AssertionError("git ran before expected_commit validation")

    with pytest.raises(provenance.ProvenanceError, match="expected commit"):
        provenance.verify_update_provenance(
            mode="editable",
            release_name="hermes-agent",
            version=UPSTREAM_VERSION,
            tag=UPSTREAM_TAG,
            directory="unused",
            expected_commit=expected_commit,
            current_version="0.16.0",
            environ={},
            runner=forbidden_git,
        )


@pytest.mark.parametrize("expected_commit", [None, "not-a-commit", "a" * 39])
def test_a_core_git_expected_commit_is_required_before_verifier(
    tmp_path,
    expected_commit,
):
    policy, request, stub = _base(tmp_path)
    invalid = dataclasses.replace(request, expected_commit=expected_commit)
    with pytest.raises(provenance.ProvenanceError, match="expected commit"):
        provenance.verify_release_provenance(invalid, policy, stub)
    assert stub.calls == 0


@pytest.mark.parametrize(
    ("platform_name", "expects_trusted_longpaths"),
    [("win32", True), ("linux", False)],
)
def test_a_git_runner_hardens_environment_and_locale(
    monkeypatch,
    platform_name,
    expects_trusted_longpaths,
):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "gpg.ssh.program")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "fake-verifier")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'gpg.ssh.program'='fake-verifier'")
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setattr(provenance.sys, "platform", platform_name)
    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    provenance._default_git_runner(["git", "version"])

    env = captured["env"]
    assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert env["LC_ALL"] == "C"
    assert env["LANG"] == "C"
    assert env["LANGUAGE"] == "C"
    if expects_trusted_longpaths:
        assert env["GIT_CONFIG_COUNT"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == "core.longpaths"
        assert env["GIT_CONFIG_VALUE_0"] == "true"
    else:
        assert "GIT_CONFIG_COUNT" not in env
        assert "GIT_CONFIG_KEY_0" not in env
        assert "GIT_CONFIG_VALUE_0" not in env
    assert "GIT_CONFIG_PARAMETERS" not in env
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_a_installer_git_environment_disables_replacements_and_attacker_config(monkeypatch):
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.repositoryFormatVersion")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "999")
    env = installer._command_env(["git", "checkout", "a" * 40])
    assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
    if sys.platform == "win32":
        assert env["GIT_CONFIG_COUNT"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == "core.longpaths"
        assert env["GIT_CONFIG_VALUE_0"] == "true"
    else:
        assert "GIT_CONFIG_COUNT" not in env
        assert "GIT_CONFIG_KEY_0" not in env
        assert "GIT_CONFIG_VALUE_0" not in env


def test_a_verifier_pins_immutable_tag_object_for_all_security_reads(tmp_path):
    tag_object = "d" * 40
    commit = "a" * 40
    trusted_ssh_keygen = tmp_path / "ssh-keygen"
    trusted_ssh_keygen.write_text("trusted", encoding="utf-8")
    signers = tmp_path / "allowed_signers"
    signers.write_text("release-bot ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    calls = []
    raw_tag = (
        f"object {commit}\n"
        "type commit\n"
        f"tag {UPSTREAM_TAG}\n"
        "tagger release-bot <release@example.invalid> 1783458000 +0000\n\n"
        f"Hermes Agent v{UPSTREAM_VERSION} ({UPSTREAM_TAG_VERSION})\n\n"
        "release\n-----BEGIN SSH SIGNATURE-----\nU1NI\n-----END SSH SIGNATURE-----\n"
    )

    def runner(argv):
        calls.append(argv)
        args = argv[3:]
        if args[0] == "for-each-ref":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if args[:3] == ["rev-parse", "--verify", "--end-of-options"]:
            if args[-1] == f"refs/tags/{UPSTREAM_TAG}^{{tag}}":
                return subprocess.CompletedProcess(argv, 0, stdout=f"{tag_object}\n", stderr="")
            assert args[-1] == f"{tag_object}^{{commit}}"
            return subprocess.CompletedProcess(argv, 0, stdout=f"{commit}\n", stderr="")
        if "verify-tag" in args:
            assert args[-1] == tag_object
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="",
                stderr=(
                    'Good "git" signature for release-bot with ED25519 key '
                    "SHA256:placeholder\n"
                ),
            )
        if args[:2] == ["cat-file", "tag"]:
            assert args[-1] == tag_object
            return subprocess.CompletedProcess(argv, 0, stdout=raw_tag, stderr="")
        raise AssertionError(f"unexpected git command: {argv}")

    verifier = provenance.GitSignedTagVerifier(
        allowed_signers_file=signers,
        directory=str(tmp_path),
        trusted_ssh_keygen=trusted_ssh_keygen,
        runner=runner,
    )
    verified = verifier.verify(_request(UPSTREAM_TAG, commit))
    assert verified.commit == commit
    assert any(call[-1] == f"refs/tags/{UPSTREAM_TAG}^{{tag}}" for call in calls)
    assert all(
        call[-1] != UPSTREAM_TAG
        for call in calls
        if "verify-tag" in call or "cat-file" in call
    )


# --------------------------------------------------------------------------- #
# Layer A -- fail-closed pypi + unknown mode.
# --------------------------------------------------------------------------- #
def test_a_pypi_unconfigured_fails_closed():
    with pytest.raises(provenance.ProvenanceError, match="no provenance policy configured"):
        provenance.verify_update_provenance(
            mode="pypi", release_name="hermes-agent", version="1.4.0", tag=None,
            directory=None, expected_commit=None, current_version="1.3.0", environ={},
        )


def test_a_pypi_even_when_flagged_has_no_verifier():
    """Operator flips the attestation flag + allowlists pypi, but no verifier
    ships -> still fails closed (explicit unshipped integration dependency)."""
    env = {
        "MS4_HERMES_PROVENANCE_PYPI_ATTESTATION": "true",
        "MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS": "https://pypi.org/project/hermes-agent/",
    }
    with pytest.raises(provenance.ProvenanceError, match="attestation verifier is not configured"):
        provenance.verify_update_provenance(
            mode="pypi", release_name="hermes-agent", version="1.4.0", tag=None,
            directory=None, expected_commit=None, current_version="1.3.0", environ=env,
        )


def test_a_unknown_mode_refuses():
    with pytest.raises(provenance.ProvenanceError, match="unknown install mode"):
        provenance.verify_update_provenance(
            mode="carrier-pigeon", release_name="hermes-agent", version="1.4.0", tag=None,
            directory=None, expected_commit=None, current_version=None, environ={},
        )


# --------------------------------------------------------------------------- #
# Layer B helpers -- real git + real ssh signed tags.
# --------------------------------------------------------------------------- #
# Subcommands that WRITE to a repository. Before any of these runs through
# ``_g`` we assert the target is a real, isolated child repo -- see the F4
# parent-git-escape repair (R2): a ``git init`` that silently failed must never
# let a subsequent ``git -C child`` discover and mutate an enclosing PARENT
# checkout (the shipped bug committed the dirty parent as "seed" by "F4 Test").
_MUTATING = frozenset({
    "config", "add", "commit", "checkout", "reset", "clean", "restore",
    "tag", "update-ref", "commit-tree", "rm", "mv", "branch", "merge",
    "cherry-pick", "revert", "am", "apply", "stash", "push", "replace",
})


def _assert_isolated_repo(path, *, bare=False):
    """Fail closed unless ``path`` is its OWN git repo (never a walk-up parent).

    Reads (read-only) ``git -C <path> rev-parse --absolute-git-dir`` and requires
    it to resolve to ``<path>`` (bare) or ``<path>/.git`` (non-bare). If a nested
    ``git init`` silently failed, ``<path>`` has no valid ``.git`` and git would
    discover an enclosing parent repo instead; that mismatch -- or a hard
    rev-parse failure -- raises BEFORE any config/add/commit can touch a parent.
    """
    proc = subprocess.run(
        [GIT, "-C", str(path), "rev-parse", "--absolute-git-dir"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"refusing git write: no isolated repo at {path} "
            f"(rev-parse failed rc={proc.returncode}: {proc.stderr.strip()!r})"
        )
    expected = os.path.normcase(os.path.realpath(
        str(path) if bare else os.path.join(str(path), ".git")))
    got = os.path.normcase(os.path.realpath(proc.stdout.strip()))
    if got != expected:
        raise AssertionError(
            f"refusing git write: {path} resolves to git dir {got!r}, not the "
            f"isolated child {expected!r} (would escape to a parent repo)"
        )


def _g(repo, *args, check=True):
    if args and args[0] in _MUTATING:
        _assert_isolated_repo(repo)  # fail-closed BEFORE any parent could be mutated
    proc = subprocess.run(
        [GIT, "-C", str(repo), *args],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        pytest.fail(f"git -C {repo} {args!r} failed: {proc.stderr}")
    return proc


def _init(path, *, bare=False):
    args = [GIT, "init", "-q"] + (["--bare"] if bare else []) + [str(path)]
    proc = subprocess.run(
        args, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:  # never swallow a failed init (Windows MAX_PATH, etc.)
        raise AssertionError(
            f"git init failed rc={proc.returncode} at {path}: {proc.stderr.strip()!r}"
        )
    _assert_isolated_repo(path, bare=bare)  # prove the child .git is real + isolated
    if not bare:
        _g(path, "config", "user.name", "F4 Test")
        _g(path, "config", "user.email", "f4@example.invalid")
        _g(path, "config", "commit.gpgsign", "false")


def _keygen(path, comment):
    proc = subprocess.run(
        [SSH_KEYGEN, "-t", "ed25519", "-N", "", "-C", comment, "-f", str(path)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"available ssh-keygen failed: {proc.stderr}")
    return path


def _allowed_signers(pub_path, principal, dest):
    parts = Path(pub_path).read_text(encoding="utf-8").split()
    dest.write_text(f"{principal} {parts[0]} {parts[1]}\n", encoding="utf-8")
    return dest


def _upstream_message(*, version, tag_version=UPSTREAM_TAG_VERSION):
    return (
        f"Hermes Agent v{version} ({tag_version})\n\n"
        "Official upstream release notes.\n"
    )


def _sign_tag(repo, tag, commit, privkey, message):
    proc = subprocess.run(
        [GIT, "-C", str(repo), "-c", "gpg.format=ssh", "-c", f"user.signingKey={privkey}",
         "tag", "-s", "-m", message, tag, commit],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"available ssh tag signing failed: {proc.stderr}")
    return proc


def _commit_repo(tmp_path, name="repo"):
    repo = tmp_path / name
    _init(repo)
    (repo / "f.txt").write_text("hello\n", encoding="utf-8")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "seed")
    commit = _g(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, commit


def _verifier(repo, signers):
    return provenance.GitSignedTagVerifier(
        allowed_signers_file=signers,
        directory=str(repo),
        trusted_ssh_keygen=Path(SSH_KEYGEN),
    )


def _request(tag, commit, origin="https://github.com/acme/hermes-agent", version=UPSTREAM_VERSION):
    return provenance.ReleaseProvenanceRequest(
        release_name="hermes-agent", version=version, platform="any", arch="any",
        origin=origin, source_kind="git", tag=tag, expected_commit=commit,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows long-path regression")
@requires_ssh
def test_a_windows_git_runner_verifies_deep_packed_signed_tag_with_trusted_longpaths(
    tmp_path,
    monkeypatch,
):
    repo, commit = _commit_repo(tmp_path, "shallow-source")
    private_key = _keygen(tmp_path / "signer", "hermes-deep-pack-signer")
    signers = _allowed_signers(
        f"{private_key}.pub",
        "hermes-deep-pack-signer@example.invalid",
        tmp_path / "allowed_signers",
    )
    _sign_tag(
        repo,
        UPSTREAM_TAG,
        commit,
        private_key,
        _upstream_message(version=UPSTREAM_VERSION),
    )
    tag_object = _g(repo, "rev-parse", f"{UPSTREAM_TAG}^{{tag}}").stdout.strip()
    assert _g(repo, "cat-file", "-t", tag_object).stdout.strip() == "tag"

    _assert_isolated_repo(repo)
    _g(repo, "repack", "-a", "-d")
    _assert_isolated_repo(repo)
    _g(repo, "prune-packed")
    assert not (repo / ".git" / "objects" / tag_object[:2] / tag_object[2:]).exists()
    assert list((repo / ".git" / "objects" / "pack").glob("*.pack"))

    fixed_length = len(str(tmp_path)) + len("\\deep-\\repo")
    deep_component = "deep-" + ("x" * max(16, 220 - fixed_length))
    deep_repo = tmp_path / deep_component / "repo"
    deep_repo.parent.mkdir()
    repo.rename(deep_repo)
    pack_paths = list((deep_repo / ".git" / "objects" / "pack").glob("*.pack"))
    assert pack_paths
    pack_path_length = max(len(str(path)) for path in pack_paths)
    assert pack_path_length > 260

    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.repositoryFormatVersion")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "999")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'gpg.ssh.program'='attacker'")
    verified = provenance._default_git_runner(
        [
            GIT,
            "-C",
            str(deep_repo),
            "-c",
            "gpg.format=ssh",
            "-c",
            f"gpg.ssh.program={SSH_KEYGEN}",
            "-c",
            f"gpg.ssh.allowedSignersFile={signers}",
            "verify-tag",
            tag_object,
        ]
    )
    assert verified.returncode == 0, (
        f"deep packed signed tag did not verify at path length {pack_path_length}: "
        f"{verified.stderr}"
    )


@requires_ssh
def test_b_good_signature_accepts(tmp_path):
    repo, commit = _commit_repo(tmp_path)
    priv = _keygen(tmp_path / "signer", "hermes-signer")
    principal = "hermes-signer@example.invalid"
    signers = _allowed_signers(f"{priv}.pub", principal, tmp_path / "allowed_signers")
    msg = _upstream_message(version=UPSTREAM_VERSION)
    _sign_tag(repo, UPSTREAM_TAG, commit, priv, msg)
    verified = _verifier(repo, signers).verify(_request(UPSTREAM_TAG, commit))
    assert verified.signer_principal == principal
    assert verified.key_type == "ED25519"
    assert verified.commit == commit
    assert verified.manifest.commit == commit
    assert verified.manifest.name == "hermes-agent"
    assert verified.manifest.version == UPSTREAM_VERSION
    assert verified.manifest.tag == UPSTREAM_TAG


@requires_ssh
def test_b_replace_refs_refuse_before_release_verification(tmp_path):
    repo, original = _commit_repo(tmp_path)
    _g(repo, "remote", "add", "origin", UPSTREAM_ORIGIN)
    (repo / "f.txt").write_text("replacement\n", encoding="utf-8")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "replacement")
    replacement = _g(repo, "rev-parse", "HEAD").stdout.strip()
    _g(repo, "replace", original, replacement)
    assert _g(repo, "show", f"{original}:f.txt").stdout == "replacement\n"
    hardened_env = os.environ.copy()
    hardened_env["GIT_NO_REPLACE_OBJECTS"] = "1"
    hardened = subprocess.run(
        [GIT, "-C", str(repo), "show", f"{original}:f.txt"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        env=hardened_env,
    )
    assert hardened.returncode == 0
    assert hardened.stdout == "hello\n"
    signers = tmp_path / "allowed_signers"
    signers.write_text("release-bot ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    env = {
        "MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS": str(signers),
        "MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS": UPSTREAM_ORIGIN,
        "MS4_HERMES_PROVENANCE_SSH_KEYGEN": str(SSH_KEYGEN),
    }
    with pytest.raises(provenance.ProvenanceError, match="replace refs"):
        provenance.verify_editable_preflight(
            directory=str(repo),
            current_version="0.16.0",
            environ=env,
        )


@requires_ssh
def test_b_repo_gpg_program_cannot_override_trusted_verifier(tmp_path):
    repo, commit = _commit_repo(tmp_path)
    priv = _keygen(tmp_path / "signer", "hermes-signer")
    principal = "hermes-signer@example.invalid"
    signers = _allowed_signers(f"{priv}.pub", principal, tmp_path / "allowed_signers")
    _sign_tag(repo, UPSTREAM_TAG, commit, priv, _upstream_message(version=UPSTREAM_VERSION))
    _g(repo, "config", "gpg.ssh.program", str(tmp_path / "fake-ssh-keygen"))
    verified = _verifier(repo, signers).verify(_request(UPSTREAM_TAG, commit))
    assert verified.commit == commit


def test_b_missing_trusted_ssh_keygen_refuses_before_git(tmp_path):
    signers = tmp_path / "allowed_signers"
    signers.write_text("release-bot ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")

    def forbidden_git(_argv):
        raise AssertionError("git ran without a trusted ssh-keygen")

    verifier = provenance.GitSignedTagVerifier(
        allowed_signers_file=signers,
        directory=str(tmp_path),
        runner=forbidden_git,
    )
    with pytest.raises(provenance.ProvenanceError, match="trusted ssh-keygen"):
        verifier.verify(_request(UPSTREAM_TAG, "a" * 40))


@requires_ssh
def test_b_unsigned_lightweight_tag_refuses(tmp_path):
    repo, commit = _commit_repo(tmp_path)
    signers = tmp_path / "allowed_signers"
    signers.write_text("p ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    _g(repo, "tag", "light", commit)
    with pytest.raises(
        provenance.ProvenanceError,
        match="annotated tag object|did not verify",
    ):
        _verifier(repo, signers).verify(_request("light", commit))


@requires_ssh
def test_b_wrong_key_refuses(tmp_path):
    repo, commit = _commit_repo(tmp_path)
    signer = _keygen(tmp_path / "signer", "hermes-signer")
    attacker = _keygen(tmp_path / "attacker", "attacker")
    principal = "hermes-signer@example.invalid"
    signers = _allowed_signers(f"{signer}.pub", principal, tmp_path / "allowed_signers")
    msg = _upstream_message(version=UPSTREAM_VERSION)
    _sign_tag(repo, UPSTREAM_TAG, commit, attacker, msg)  # signed by the WRONG key
    with pytest.raises(provenance.ProvenanceError, match="did not verify"):
        _verifier(repo, signers).verify(_request(UPSTREAM_TAG, commit))


@requires_ssh
def test_b_annotated_unsigned_refuses(tmp_path):
    repo, commit = _commit_repo(tmp_path)
    signers = tmp_path / "allowed_signers"
    signers.write_text("p ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    _g(repo, "tag", "-a", "-m", "annotated but not signed", "annot", commit)
    with pytest.raises(provenance.ProvenanceError, match="did not verify"):
        _verifier(repo, signers).verify(_request("annot", commit))


@requires_ssh
def test_b_signed_but_malformed_upstream_message_refuses(tmp_path):
    repo, commit = _commit_repo(tmp_path)
    priv = _keygen(tmp_path / "signer", "hermes-signer")
    principal = "hermes-signer@example.invalid"
    signers = _allowed_signers(f"{priv}.pub", principal, tmp_path / "allowed_signers")
    _sign_tag(repo, UPSTREAM_TAG, commit, priv, "plain message with no release binding")
    with pytest.raises(provenance.ProvenanceError, match="upstream release metadata"):
        _verifier(repo, signers).verify(_request(UPSTREAM_TAG, commit))


@requires_ssh
def test_b_no_allowed_signers_policy_refuses(tmp_path):
    repo, commit = _commit_repo(tmp_path)
    priv = _keygen(tmp_path / "signer", "hermes-signer")
    msg = _upstream_message(version=UPSTREAM_VERSION)
    _sign_tag(repo, UPSTREAM_TAG, commit, priv, msg)
    verifier = provenance.GitSignedTagVerifier(allowed_signers_file=None, directory=str(repo))
    with pytest.raises(provenance.ProvenanceError, match="allowed_signers"):
        verifier.verify(_request(UPSTREAM_TAG, commit))


@requires_ssh
def test_b_malformed_inline_allowed_signers_refuses(tmp_path):
    repo, commit = _commit_repo(tmp_path)
    priv = _keygen(tmp_path / "signer", "hermes-signer")
    _sign_tag(repo, UPSTREAM_TAG, commit, priv, _upstream_message(version=UPSTREAM_VERSION))
    verifier = provenance.GitSignedTagVerifier(
        allowed_signers_file=None,
        allowed_signers_text="not an allowed_signers policy",
        directory=str(repo),
        trusted_ssh_keygen=Path(SSH_KEYGEN),
    )
    with pytest.raises(provenance.ProvenanceError, match="did not verify"):
        verifier.verify(_request(UPSTREAM_TAG, commit))


# --------------------------------------------------------------------------- #
# Layer C helpers -- installer integration + run_update_job ordering.
# --------------------------------------------------------------------------- #
def _bare_remote_and_seed(tmp_path):
    remote = tmp_path / "remote_bare"
    seed = tmp_path / "seed"
    _init(remote, bare=True)
    _init(seed)
    _g(seed, "remote", "add", "origin", str(remote))
    (seed / "s.txt").write_text("seed\n", encoding="utf-8")
    _g(seed, "add", "-A")
    _g(seed, "commit", "-q", "-m", "seed-commit")
    commit = _g(seed, "rev-parse", "HEAD").stdout.strip()
    return remote, seed, commit


def _target_of(tmp_path, remote):
    target = tmp_path / "target"
    _init(target)
    (target / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _g(target, "add", "tracked.txt")
    _g(target, "commit", "-q", "-m", "baseline")
    baseline = _g(target, "rev-parse", "HEAD").stdout.strip()
    _g(target, "remote", "add", "origin", str(remote))
    return target, baseline


def _plugin_src(tmp_path):
    src = tmp_path / "plugin_src"
    (src / "sub").mkdir(parents=True)
    (src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")
    return src


def _wire_versioning(monkeypatch, directory, version, tag):
    versioning._clear_caches_for_test()
    monkeypatch.setattr(installer, "_require_valid_git_bash_candidate", lambda: None)
    monkeypatch.setattr(
        installer.versioning, "install_mode",
        lambda: {"mode": "editable", "version": "0.13.0", "directory": directory},
    )
    monkeypatch.setattr(
        installer.versioning, "latest_version",
        lambda force_refresh=False: versioning.CachedLatest(
            version, "2026-01-01T00:00:00Z", tag, "https://example/r", 0.0
        ),
    )
    monkeypatch.setattr(installer.versioning, "recent_releases", lambda force_refresh=False: [])
    monkeypatch.setattr(installer.versioning, "resolve_git_tag", lambda target: tag)


def _mock_install(monkeypatch):
    monkeypatch.setattr(installer, "_pip_install_editable", lambda *a, **k: None)
    monkeypatch.setattr(installer, "_reinstall_editable", lambda *a, **k: None)
    monkeypatch.setattr(
        installer, "_validate",
        lambda python, *, expected_version, editable_directory=None, expected_commit=None: expected_version,
    )


def _set_policy_env(monkeypatch, *, signers, origin):
    monkeypatch.delenv("MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS", raising=False)
    monkeypatch.setenv(
        "MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS_INLINE",
        Path(signers).read_text(encoding="utf-8").strip(),
    )
    monkeypatch.setenv("MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS", str(origin))
    monkeypatch.setenv("MS4_HERMES_PROVENANCE_SSH_KEYGEN", str(SSH_KEYGEN))


@requires_ssh
@pytest.mark.real_provenance
@pytest.mark.parametrize("entrypoint", ["trigger", "run_update_job"])
def test_c_wrong_origin_refuses_before_release_lookup_or_job(
    tmp_path,
    monkeypatch,
    entrypoint,
):
    repo, _commit = _commit_repo(tmp_path, name="checkout")
    _g(repo, "remote", "add", "origin", "https://evil.invalid/hermes-agent.git")
    monkeypatch.setattr(installer, "_require_valid_git_bash_candidate", lambda: None)
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {"mode": "editable", "version": "0.13.0", "directory": repo},
    )
    signers = tmp_path / "allowed_signers"
    signers.write_text("release-bot ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    _set_policy_env(
        monkeypatch,
        signers=signers,
        origin="https://github.com/acme/allowed-hermes-agent.git",
    )
    reached = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            reached.append(name)
            raise AssertionError(f"{name} ran before origin preflight")

        return fail

    monkeypatch.setattr(installer.versioning, "latest_version", forbidden("latest release lookup"))
    monkeypatch.setattr(installer.versioning, "recent_releases", forbidden("recent release lookup"))
    monkeypatch.setattr(installer.state, "start_job", forbidden("start_job"))
    monkeypatch.setattr(installer, "_git_fetch_resolve", forbidden("fetch"))
    state._reset_for_tests(tmp_path / f"{entrypoint}-snap.json")
    if entrypoint == "trigger":
        invoke = lambda: installer.trigger_update(target_version="0.14.0")
    else:
        invoke = lambda: installer.run_update_job(
            target_version="0.14.0",
            plugin_src=_plugin_src(tmp_path),
            venv_python=tmp_path / "py",
        )
    with pytest.raises(installer.HermesUpgradeError, match="origin"):
        invoke()
    assert reached == []
    assert state.last_update() is None


@requires_ssh
@pytest.mark.real_provenance
def test_c_vetted_origin_url_remains_fetch_target_across_remote_flip(
    tmp_path,
    monkeypatch,
):
    repo, commit = _commit_repo(tmp_path, name="checkout")
    allowed_origin = str(tmp_path / "allowed-remote.git")
    evil_origin = str(tmp_path / "evil-remote.git")
    _g(repo, "remote", "add", "origin", allowed_origin)
    _wire_versioning(monkeypatch, repo, "0.14.0", UPSTREAM_TAG)
    _mock_install(monkeypatch)
    signers = tmp_path / "allowed_signers"
    signers.write_text("release-bot ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    _set_policy_env(monkeypatch, signers=signers, origin=allowed_origin)
    fetch_targets = []
    post_checks = []

    def fake_fetch(directory, target_tag, remote_url="origin"):
        fetch_targets.append(remote_url)
        assert remote_url == allowed_origin
        command = installer._git_fetch_command(
            directory,
            target_tag,
            remote_url=remote_url,
        )
        assert allowed_origin in command
        assert evil_origin not in command
        _g(repo, "remote", "set-url", "origin", evil_origin)
        _g(repo, "remote", "set-url", "origin", allowed_origin)
        return commit

    def fake_post_fetch_provenance(**_kwargs):
        post_checks.append(_g(repo, "remote", "get-url", "origin").stdout.strip())

    monkeypatch.setattr(installer, "_git_fetch_resolve", fake_fetch)
    monkeypatch.setattr(installer, "_verify_release_provenance", fake_post_fetch_provenance)
    monkeypatch.setattr(installer, "_git_detach_checkout", lambda *_args: None)
    monkeypatch.setattr(installer, "_restamp_managed_plugin", lambda *_args: None)

    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    result = installer.run_update_job(
        target_version="0.14.0",
        plugin_src=_plugin_src(tmp_path),
        venv_python=tmp_path / "py",
        vetted_origin=allowed_origin,
    )
    assert result == {"to_version": "0.14.0"}
    assert fetch_targets == [allowed_origin]
    assert post_checks == [allowed_origin]


@requires_ssh
@pytest.mark.real_provenance
def test_c_seam_accepts_real_signed_tag(tmp_path, monkeypatch):
    """installer._verify_release_provenance delegates to the real gate and
    returns cleanly for a genuinely signed, correctly-bound release."""
    repo, commit = _commit_repo(tmp_path, name="checkout")
    origin = "https://github.com/acme/hermes-agent"
    _g(repo, "remote", "add", "origin", origin)
    priv = _keygen(tmp_path / "signer", "hermes-signer")
    principal = "hermes-signer@example.invalid"
    signers = _allowed_signers(f"{priv}.pub", principal, tmp_path / "allowed_signers")
    msg = _upstream_message(version="0.14.0")
    _sign_tag(repo, UPSTREAM_TAG, commit, priv, msg)
    _set_policy_env(monkeypatch, signers=signers, origin=origin)

    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    # Must not raise.
    installer._verify_release_provenance(
        mode="editable", target="0.14.0", tag=UPSTREAM_TAG, directory=repo,
        expected_commit=commit, current_version="0.13.0",
    )
    assert state.last_update().phase == "verifying_provenance"


@requires_ssh
@pytest.mark.real_provenance
def test_c_seam_refuses_unsigned_as_upgrade_error(tmp_path, monkeypatch):
    repo, commit = _commit_repo(tmp_path, name="checkout")
    origin = "https://github.com/acme/hermes-agent"
    _g(repo, "remote", "add", "origin", origin)
    signers = tmp_path / "allowed_signers"
    signers.write_text("p ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    _g(repo, "tag", "light", commit)
    _set_policy_env(monkeypatch, signers=signers, origin=origin)

    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    with pytest.raises(installer.HermesUpgradeError, match="provenance"):
        installer._verify_release_provenance(
            mode="editable", target="0.14.0", tag="light", directory=repo,
            expected_commit=commit, current_version="0.13.0",
        )


@requires_ssh
@pytest.mark.real_provenance
def test_c_run_update_job_verifies_before_any_mutation(tmp_path, monkeypatch):
    """THE ordering guarantee: an unsigned release is refused by the gate BEFORE
    the first working-tree mutation. HEAD is untouched, the detached checkout is
    never entered, and the phase trail reaches ``verifying_provenance`` but never
    ``checking_out``."""
    remote, _seed, _commit = _bare_remote_and_seed(tmp_path)
    _g(_seed, "tag", UPSTREAM_TAG, _commit)  # UNSIGNED lightweight
    _g(_seed, "push", "origin", f"refs/tags/{UPSTREAM_TAG}:refs/tags/{UPSTREAM_TAG}")
    target, baseline = _target_of(tmp_path, remote)

    _wire_versioning(monkeypatch, target, "0.14.0", UPSTREAM_TAG)
    _mock_install(monkeypatch)
    signers = tmp_path / "allowed_signers"
    signers.write_text("p ssh-ed25519 AAAAplaceholder\n", encoding="utf-8")
    _set_policy_env(monkeypatch, signers=signers, origin=str(remote))

    detach_calls = {"n": 0}
    real_detach = installer._git_detach_checkout

    def _spy_detach(directory, commit):
        detach_calls["n"] += 1
        return real_detach(directory, commit)

    monkeypatch.setattr(installer, "_git_detach_checkout", _spy_detach)

    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    with pytest.raises(installer.HermesUpgradeError, match="provenance"):
        installer.run_update_job(
            target_version="0.14.0", plugin_src=_plugin_src(tmp_path), venv_python=tmp_path / "py", vetted_origin=str(remote)
        )

    assert detach_calls["n"] == 0  # NO mutation attempted
    assert _g(target, "rev-parse", "HEAD").stdout.strip() == baseline  # HEAD untouched
    final = state.last_update()
    assert final.status == "failed"
    phases = [entry["phase"] for entry in final.progress]
    assert "verifying_provenance" in phases
    assert "checking_out" not in phases


@requires_ssh
@pytest.mark.real_provenance
def test_c_run_update_job_accepts_signed_release_end_to_end(tmp_path, monkeypatch):
    """A correctly signed + bound release passes the gate and the full editable
    pipeline completes, with provenance verified BEFORE the checkout phase."""
    remote, seed, commit = _bare_remote_and_seed(tmp_path)
    priv = _keygen(tmp_path / "signer", "hermes-signer")
    principal = "hermes-signer@example.invalid"
    signers = _allowed_signers(f"{priv}.pub", principal, tmp_path / "allowed_signers")
    msg = _upstream_message(version="0.14.0")
    _sign_tag(seed, UPSTREAM_TAG, commit, priv, msg)
    _g(seed, "push", "origin", f"refs/tags/{UPSTREAM_TAG}:refs/tags/{UPSTREAM_TAG}")
    target, _baseline = _target_of(tmp_path, remote)

    _wire_versioning(monkeypatch, target, "0.14.0", UPSTREAM_TAG)
    _mock_install(monkeypatch)
    _set_policy_env(monkeypatch, signers=signers, origin=str(remote))

    state._reset_for_tests(tmp_path / "snap.json")
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    result = installer.run_update_job(
        target_version="0.14.0", plugin_src=_plugin_src(tmp_path), venv_python=tmp_path / "py", vetted_origin=str(remote)
    )
    assert result == {"to_version": "0.14.0"}
    final = state.last_update()
    assert final.status == "success"
    phases = [entry["phase"] for entry in final.progress]
    assert "verifying_provenance" in phases
    assert phases.index("verifying_provenance") < phases.index("checking_out")
    assert _g(target, "rev-parse", "HEAD").stdout.strip() == commit  # landed on the signed commit
