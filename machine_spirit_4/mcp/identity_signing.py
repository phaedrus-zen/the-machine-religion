"""MS4 -> HiveMind -> PsyKyo identity envelope signing.

PsyKyo's caller-trust gate (PsyKyo<->HiveMind integration amendment,
landing in Wave A) can be satisfied two ways:

  1. forward the legacy ``PSYKYO_EXECUTE_SAFE_AUTO`` actuation token
     verbatim (works today, but couples MS4 to PsyKyo's exact token
     value and offers no agent attribution); OR
  2. send a signed ``agent_identity`` envelope proving the caller is
     the MS4 agent the operator pre-registered.

This module implements option 2 (preferred for v1+). The envelope
shape is fixed by the amendment so PsyKyo's verifier (subagent C1)
and HiveMind's MCP gateway forwarder (subagent C2) stay in lockstep:

    {
      "agent": "ms4",
      "identity_proof": "<base64(HMAC-SHA256(agent + nonce + issued_at))>",
      "parent_action_authorization": "<MS4-side evidence string>",
      "issued_at": "<ISO8601 UTC>",
      "nonce": "<32-char random hex>",
      "envelope_version": 1
    }

The shared HMAC secret lives in env var
``MS4_HIVEMIND_IDENTITY_SECRET`` on the MS4 side; the matching
``PSYKYO_NATIVE_AGENT_MS4_SECRET`` is set on the PsyKyo side via
PsyKyo's native_agents.json. The operator is responsible for rolling
both at once.

Failure modes are intentionally non-fatal upstream: when the secret
isn't set or signing fails for any reason, the caller in
``gateway.hivemind_tools._call_tool`` logs the failure and omits the
envelope, letting the HiveMind MCP gateway fall back to whatever
legacy auth path it has (e.g. the actuation token) without MS4
needing to know.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import unittest
from datetime import datetime, timezone

AGENT_ID = "ms4"
SECRET_ENV_VAR = "MS4_HIVEMIND_IDENTITY_SECRET"
ENVELOPE_VERSION = 1


class IdentityConfigError(RuntimeError):
    """Raised when identity signing is requested but the shared
    secret is not configured (env var unset or empty)."""


_cached_secret: bytes | None = None


def _reset_secret_cache() -> None:
    """Test hook: clear the cached secret so the next call re-reads
    the env. Production callers don't need this -- the secret is
    read once per process lifetime, matching how operator secrets
    are normally rotated (set env -> restart process)."""
    global _cached_secret
    _cached_secret = None


def get_identity_secret() -> bytes:
    """Return the configured shared secret as UTF-8 bytes.

    Cached after first successful read so we don't re-stat the env
    block on every MCP call. Raises :class:`IdentityConfigError`
    when the env var is unset or empty.
    """
    global _cached_secret
    if _cached_secret is not None:
        return _cached_secret
    raw = os.environ.get(SECRET_ENV_VAR, "")
    if not raw:
        raise IdentityConfigError(
            f"identity signing requested but {SECRET_ENV_VAR} is unset or empty"
        )
    _cached_secret = raw.encode("utf-8")
    return _cached_secret


def is_secret_configured() -> bool:
    """True iff the shared-secret env var is set + non-empty.

    Reads the env var directly (does NOT consult the cache) so
    callers can detect operator-side env changes without restarting
    MS4. Caching is only for the secret bytes returned by
    :func:`get_identity_secret`.
    """
    return bool(os.environ.get(SECRET_ENV_VAR, ""))


def generate_nonce() -> str:
    """Return a 32-character random hex string (16 bytes of entropy)."""
    return secrets.token_hex(16)


def sign_envelope(
    agent_id: str = AGENT_ID,
    parent_action_authorization: str = "",
    *,
    secret: bytes | None = None,
    issued_at_override: str | None = None,
    nonce_override: str | None = None,
) -> dict:
    """Build a signed identity envelope for an outbound MCP call.

    The signing payload is the exact byte-concatenation
    ``agent_id + nonce + issued_at`` (UTF-8, no separators) to
    match PsyKyo's verifier byte-for-byte.

    Parameters
    ----------
    agent_id:
        Logical agent identifier. Defaults to :data:`AGENT_ID`.
        Override only if MS4 ever signs on behalf of another agent.
    parent_action_authorization:
        Free-form string the verifier can correlate with MS4's
        ethics decision log entry for the parent action. Typical
        shape today: ``"ms4_call_tool:<tool_name>"``.
    secret:
        Override the env-derived secret (test hook + room for a
        future secret-rotation pipeline). Defaults to
        :func:`get_identity_secret`.
    issued_at_override / nonce_override:
        Test hooks for deterministic signing. Production callers
        leave both unset so every envelope gets a fresh ISO8601
        timestamp + random nonce.

    Returns
    -------
    dict
        Envelope ready to drop into JSON-RPC
        ``params.agent_identity``.
    """
    if secret is None:
        secret = get_identity_secret()
    nonce = nonce_override if nonce_override is not None else generate_nonce()
    if issued_at_override is not None:
        issued_at = issued_at_override
    else:
        issued_at = datetime.now(timezone.utc).isoformat()
    payload = (agent_id + nonce + issued_at).encode("utf-8")
    digest = hmac.new(secret, payload, hashlib.sha256).digest()
    identity_proof = base64.b64encode(digest).decode("ascii")
    return {
        "agent": agent_id,
        "identity_proof": identity_proof,
        "parent_action_authorization": parent_action_authorization,
        "issued_at": issued_at,
        "nonce": nonce,
        "envelope_version": ENVELOPE_VERSION,
    }


_NONCE_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


class _SignEnvelopeTests(unittest.TestCase):
    """Inline tests so the module ships its own regression harness.
    Run with ``python identity_signing.py``."""

    def setUp(self) -> None:
        self._saved_env = os.environ.pop(SECRET_ENV_VAR, None)
        _reset_secret_cache()

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop(SECRET_ENV_VAR, None)
        else:
            os.environ[SECRET_ENV_VAR] = self._saved_env
        _reset_secret_cache()

    def test_sign_envelope_returns_correct_shape_when_secret_configured(self) -> None:
        os.environ[SECRET_ENV_VAR] = "unit_test_secret_value"
        env = sign_envelope(parent_action_authorization="unit_test:shape")
        self.assertEqual(env["agent"], "ms4")
        self.assertEqual(env["envelope_version"], 1)
        self.assertEqual(env["parent_action_authorization"], "unit_test:shape")
        self.assertIn("identity_proof", env)
        self.assertIn("issued_at", env)
        self.assertIn("nonce", env)
        self.assertEqual(len(env["nonce"]), 32)
        self.assertIsNotNone(_NONCE_HEX_RE.match(env["nonce"]))
        # base64-encoded HMAC-SHA256 digest is 44 chars (32 bytes
        # -> 44 with one '=' padding char).
        self.assertEqual(len(env["identity_proof"]), 44)
        self.assertTrue(env["identity_proof"].endswith("="))

    def test_sign_envelope_raises_when_secret_unset(self) -> None:
        with self.assertRaises(IdentityConfigError):
            sign_envelope()

    def test_sign_envelope_signature_is_deterministic_for_same_inputs(self) -> None:
        os.environ[SECRET_ENV_VAR] = "deterministic_secret"
        fixed_nonce = "a" * 32
        fixed_ts = "2026-05-26T20:00:00+00:00"
        env_a = sign_envelope(
            parent_action_authorization="det:test",
            issued_at_override=fixed_ts,
            nonce_override=fixed_nonce,
        )
        env_b = sign_envelope(
            parent_action_authorization="det:test",
            issued_at_override=fixed_ts,
            nonce_override=fixed_nonce,
        )
        self.assertEqual(env_a["identity_proof"], env_b["identity_proof"])
        self.assertEqual(env_a["nonce"], env_b["nonce"])
        self.assertEqual(env_a["issued_at"], env_b["issued_at"])

    def test_sign_envelope_signature_differs_for_different_nonce(self) -> None:
        os.environ[SECRET_ENV_VAR] = "differing_nonce_secret"
        fixed_ts = "2026-05-26T20:00:00+00:00"
        env_a = sign_envelope(
            issued_at_override=fixed_ts,
            nonce_override="a" * 32,
        )
        env_b = sign_envelope(
            issued_at_override=fixed_ts,
            nonce_override="b" * 32,
        )
        self.assertNotEqual(env_a["identity_proof"], env_b["identity_proof"])
        self.assertNotEqual(env_a["nonce"], env_b["nonce"])

    def test_generate_nonce_produces_unique_32_char_hex(self) -> None:
        nonces = {generate_nonce() for _ in range(100)}
        self.assertEqual(len(nonces), 100, "expected 100 unique nonces")
        for n in nonces:
            self.assertEqual(len(n), 32, f"nonce wrong length: {n!r}")
            self.assertIsNotNone(_NONCE_HEX_RE.match(n), f"nonce not hex: {n!r}")


if __name__ == "__main__":
    unittest.main()
