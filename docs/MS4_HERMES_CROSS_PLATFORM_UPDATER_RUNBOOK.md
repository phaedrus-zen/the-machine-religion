# MS4 Hermes Cross-Platform Updater — Operator Runbook

Scope: the `machine_spirit_4/hermes_admin` updater (`versioning.py`,
`installer.py`, `state.py`, `provenance.py`). This runbook covers how the
updater picks a checkout directory across OSes, what it owns vs. what it will
never touch, the rollback states it can land in, the **F4 supply-chain
provenance gate** (§9), the **mandatory external service restart**, and the
live-service gate that is deliberately *not* automated yet (F5).

This document is a candidate authored under isolated review. It describes the
R1 candidate behavior in this tree; it is not a claim that any live host has
been updated.

---

## 0. TL;DR for the operator

1. On **Linux/macOS/Jetson**, set `MS4_HERMES_DIR` to the real Hermes checkout
   before triggering an update. Do not rely on the default on POSIX.
2. The updater owns exactly one subtree inside the checkout:
   `plugins/ms4_consciousness/`. Everything else in the checkout is yours and
   is never deleted or reset.
3. If anything outside that subtree is dirty (tracked *or* untracked), the
   editable update **refuses to start**. Clean it up yourself — the updater
   will not.
4. A successful update changes files on disk. The **running** Hermes/MS4
   process keeps the old code until you **restart the external service**
   (F5). The updater never restarts services.
5. If an editable update fails after it began mutating, it rolls back to the
   exact prior commit + install + managed plugin. A *rollback* failure is
   fatal and visible (`rollback_failed`) and needs manual recovery.
6. **F4 supply-chain gate (§9):** the fetched release must pass provenance
   verification **before** the first working-tree mutation. For editable
   installs you MUST provision an `allowed_signers` policy file and an origin
   allowlist (`MS4_HERMES_PROVENANCE_*`), or every update fails closed. The
   PyPI path currently fails closed by design (attestation verifier not yet
   shipped). No policy configured means no update — this is intentional.
7. **Durable terminal reconcile:** on startup, an ownerless persisted `running`
   job is failed only after the updater lock is acquired. Startup and every
   read-only version refresh then reconcile installed ≥ selected latest as a
   current/newer no-op that supersedes a stale failure banner without deleting
   audit. If the newest
   publication is unsigned, the first-class control selects the newest earlier
   signed stable release that is still newer than installed. With no newer
   signed candidate, the state is `blocked: official_tag_unsigned` (or
   `official_tag_signature_unknown`) and Update is disabled; any prior failed
   job remains audit only. Unknown versions fail closed. Explicit downgrade
   targets are refused. Do not treat a CSS hide of `last.status=failed` as the
   fix.
8. **Windows Git-Bash gate:** a current/newer or retry-current no-op is
   **zero-external**: classify from local `install_mode()` plus already-
   persisted/cached latest only (TTL-expired cache still counts). That path
   never Git Bash, HTTP, checkout, subprocess, or starts an update job.
   Missing local latest cannot prove no-op: fail closed into the mutating
   path, validate sealed Git-Bash first, and only then permit release
   discovery. A mutating update seals, hashes, and smokes
   `C:\Program Files\Git\bin\bash.exe` **before** origin, disk, release
   discovery, job start, or fetch. `install_mode()` is a local metadata
   read, not a supply-chain effect.

---

## 1. Install modes

`versioning.install_mode()` detects how `hermes-agent` is installed in MS4's
contained interpreter:

| mode       | meaning                                             | update mechanism |
|------------|-----------------------------------------------------|------------------|
| `editable` | `pip install -e` against a local git checkout       | git fetch tag → detached checkout → re-stamp managed plugin → `pip install -e .` → validate |
| `pypi`     | ordinary wheel install                              | `pip install --upgrade hermes-agent==<X>` → validate |
| `missing`  | not installed                                       | refused in preflight |
| `unknown`  | metadata present but neither shape matches          | refused |

The **editable** path is the one with git lifecycle, the F2 clean-check, and
the F3 rollback. The **pypi** path relies on pip's own wheel replacement and
has no git rollback (see §6 for pypi recovery).

---

## 2. Preparation

Before triggering an editable update:

1. **Confirm the checkout directory** the updater will use (see §3). On POSIX,
   set `MS4_HERMES_DIR` explicitly.
2. **Confirm the checkout is clean** except for the managed plugin:
   ```
   git -C <checkout> -c core.quotePath=false status --porcelain=v1 --untracked-files=all
   ```
   Any line whose path is **not** strictly inside `plugins/ms4_consciousness/`
   will cause the update to refuse (§4/§5). Commit, stash, or remove your own
   changes first. The updater will not do this for you.
3. **Confirm the contained interpreter exists.** The default is
   `machine_spirit_4/.venv/Scripts/python.exe` on Windows and
   `machine_spirit_4/.venv/bin/python` on POSIX (`_default_venv_python`).
4. **Confirm the managed plugin source-of-truth exists** at
   `machine_spirit_4/plugins/hermes/ms4_consciousness` (`DEFAULT_PLUGIN_SRC`).
   This lives outside the Hermes checkout so a tag bump can never delete it.

---

## 3. `MS4_HERMES_DIR` and per-platform defaults (F6)

The checkout directory is resolved by `versioning._default_hermes_dir(...)`,
wrapped by `versioning.hermes_dir()`, and mirrored for subprocess/env
propagation by `runtime_common.hermes_dir()`. Resolution order is
deterministic and identical to the unit tests in
`tests/ms4_hermes_admin/test_versioning_r1.py` and
`tests/ms4_hermes_admin/test_installer.py`:

1. **`MS4_HERMES_DIR` wins on every platform.** It is `expanduser()`-expanded.
   This is the **recommended** control for all non-Windows deployments.
2. Otherwise, if a managed-checkout marker exists at
   `machine_spirit_4/runtime/hermes_active_checkout.json` (schema
   `Ms4HermesActiveCheckout.v1`, published by a successful managed migration),
   `runtime_common.hermes_dir()` returns its `directory` field. This is how
   subprocess spawners and `ms4_env()` pick up the isolated checkout after
   the updater migrates away from a dirty operator tree without requiring a
   manual `MS4_HERMES_DIR` override.
3. Otherwise an OS-appropriate default is used — POSIX **never** silently
   reuses the Windows `~/Documents` shape:

   | `sys.platform`            | default checkout dir |
   |---------------------------|----------------------|
   | `win32`                   | `<home>/Documents/hermes-agent` (preserves the existing deployed Windows layout) |
   | `darwin`                  | `<home>/Library/Application Support/hermes-agent` |
   | anything else (Linux x86_64/ARM64, Jetson/Thor `aarch64`, *BSD) | `$XDG_DATA_HOME/hermes-agent` if `XDG_DATA_HOME` is set, else `<home>/.local/share/hermes-agent` |

**Why POSIX should always set `MS4_HERMES_DIR`:** the POSIX default is an XDG
data location, which is correct for a fresh install but is almost certainly
*not* where an existing operator already cloned Hermes. If your Hermes checkout
lives somewhere else (a repo dir, `/opt`, a home subdir), set `MS4_HERMES_DIR`
so the updater operates on the real checkout instead of an empty default path.

Examples:

```bash
# Linux / Jetson / macOS
export MS4_HERMES_DIR="/opt/hermes-agent"        # or wherever the checkout is
```

```powershell
# Windows (only needed to override the ~/Documents default)
$env:MS4_HERMES_DIR = "D:\hermes-agent"
```

---

## 4. Managed plugin ownership

The updater **owns exactly one path** inside the Hermes checkout:

```
<checkout>/plugins/ms4_consciousness/
```

Facts:

- It is **generated/stamped by MS4**, copied from the source-of-truth at
  `machine_spirit_4/plugins/hermes/ms4_consciousness` after every checkout
  swap (`_sync_plugin` / `_restamp_managed_plugin`).
- It is **untracked** in the Hermes repo and intentionally **not** gitignored,
  so a naive clean-check would self-block every second update. That is exactly
  the F2 bug this candidate fixes.
- The clean-check (`_ensure_clean_checkout`) ignores dirty paths **only** when
  it can prove **every** reported path is *strictly inside* this subtree
  (`_is_within_managed_plugin`) **and** the on-disk subtree is a real directory
  physically contained in the checkout (`_managed_plugin_is_safely_contained` —
  resolves POSIX symlinks and Windows junctions and rejects any escape).
- Removal during a re-stamp is confined to a *freshly validated* managed
  subtree. The recursive delete can never reach a user path, a sibling like
  `plugins/ms4_consciousness_backup/`, a case variant, a traversal, or a
  redirected symlink/junction target.

**Everything else in the checkout is yours.** Tracked modifications, unrelated
untracked files, backup relics (`*.orig`, `*.bak`), tool files — all of these
keep the update **refused** and are **never** deleted or reset.

---

## 5. Clean-check refusal (F2) — what trips it

The editable update refuses (before any fetch/checkout/plugin mutation) if the
checkout has any change outside the managed subtree. Refused cases, all covered
by hermetic tests:

- an unrelated **untracked** file anywhere outside the subtree;
- a **tracked modification** to any file (including files inside the subtree's
  *parent* `plugins/` but not the subtree itself);
- a **path-prefix collision** sibling (`plugins/ms4_consciousness_backup/...`);
- a **symlink/junction escape** where `plugins/ms4_consciousness` redirects
  outside the checkout;
- a **case variation** of the subtree name (`plugins/MS4_Consciousness/...`);
- a **traversal** path (`plugins/ms4_consciousness/../evil`);
- a **backup relic** left next to real files;
- any **unparseable or C-quoted** status path (fail closed).

Only genuine, strictly-contained managed-plugin changes are ignored so a repeat
update can pass the clean-check after the plugin was stamped by the prior run.

Resolution: inspect the `git status` output from §2.2, then commit/stash/remove
the offending paths **yourself**. The updater deliberately does not.

---

## 6. Rollback states (F3)

An editable update captures the exact clean pre-update state **before** the
first working-tree mutation (`_capture_editable_prestate`): the `HEAD` commit,
the checkout **mode** (`branch` vs `detached`), and the branch name if any.

The first mutation is the detached checkout of the release commit. From that
point on, a failure in **checkout, plugin re-stamp, pip install, or
validation** triggers `_rollback_editable`, which:

1. Restores the pre-update commit **and mode** (`_restore_editable_commit`) —
   switching back to the original branch if it was on one, else re-detaching
   onto the exact commit.
2. Re-stamps **only** the managed plugin (`_restamp_managed_plugin`).
3. Reinstalls the editable package (`_reinstall_editable`) so recorded
   distribution metadata matches the restored source.

It uses **only** whole-ref/commit `git checkout`. It never runs
`git reset --hard`, `git clean`, `git restore`, or `git checkout -- <path>`, so
pre-existing untracked and non-managed changes are never discarded.

### Phase → meaning

Phases are persisted to the state snapshot (`state.PHASES`, §7):

| phase             | meaning |
|-------------------|---------|
| `queued`          | job accepted |
| `preflight`       | safety checks (target-version guard, local install_mode; Git Bash seal/hash/smoke only on a mutating update, before origin/disk/release/fetch) |
| `fetching_release`| resolving latest / recent release metadata |
| `fetching_remote` | `git fetch` of the one release tag ref (mutates only `.git`) |
| `verifying_provenance` | **F4 gate**: signer/key + bound identity/version/platform/arch/origin/commit + freshness verified on the fetched signed tag, **before** any working-tree mutation (§9). A refusal here needs no rollback. |
| `checking_out`    | detached checkout of the release commit (**first working-tree mutation**) |
| `syncing_plugin`  | re-stamping `plugins/ms4_consciousness/` |
| `pip_installing`  | `pip install -e .` (editable) or `--upgrade` (pypi) |
| `validating`      | import + version + HEAD-match check |
| `rolling_back`    | restoring the captured pre-update state |
| `rolled_back`     | **update failed but prior state was fully restored** |
| `rollback_failed` | **update failed AND rollback failed — manual recovery required** |
| `done`            | success |
| `error`           | terminal failure (see `error` field) |

### Reading the outcome

- `rolled_back`: the update did **not** apply; you are back on the prior
  commit, prior editable install, and prior managed plugin. The raised error
  embeds the **primary** failure reason. Safe to investigate and retry.
- `rollback_failed`: **fatal.** The raised error embeds **both** the primary
  failure and the rollback failure. The checkout may be on the release commit
  or partially restored. Do not retry blindly — inspect the checkout `HEAD`,
  the managed plugin, and `pip show hermes-agent`, then restore by hand. This
  state is intentionally loud and never silently swallowed.

### pypi-mode recovery

The `pypi` path has no git rollback. pip replaces the wheel atomically, so a
failed upgrade generally leaves the prior version installed. If a pypi upgrade
leaves the environment wrong, recover manually:

```
<venv-python> -m pip install --no-input "hermes-agent==<prior-version>"
```

---

## 7. State snapshot & crash recovery

The job snapshot is persisted (atomically, via temp file + `os.replace`) to:

```
machine_spirit_4/runtime/hermes_update_state.json
```

It lives under the MS4 runtime, **never** inside the Hermes checkout. On gateway
startup, MS4 hydrates it and probes the cross-process updater lock. A snapshot
still marked `running` is flipped to `failed` / `error` only when that lock is
free; a job still owned by another process stays running. Version GETs remain
read-only with respect to lock ownership. The failure tells the operator to
re-check the version and re-trigger, so a crashed job does not survive a cold
gateway start as a permanently disabled button.

---

## 8. External service restart requirement (F5 handoff)

**A successful update only changes files on disk.** The already-running
Hermes/MS4 process continues executing the previously-imported code until it is
restarted. The updater **does not** restart any service, task, or process, and
must not (that is an operator/live-service action, out of scope for this
candidate).

After a `done` result:

1. Restart the external MS4/Hermes service by the host's normal mechanism
   (systemd unit, Windows scheduled task/service, launchd, or the operator's
   supervisor) — **manually or via the owning control plane, not from this
   updater**.
2. Re-read the version and confirm it matches the intended target.

This end-to-end "update on disk → restart → serve new version" flow against a
**live** service is the **F5 live-service gate**. It is not exercised by CI
(which injects the `pip install` and validation boundaries) and must be proven
on a real host as a hardware/live gate.

---

## 9. Remaining gates NOT covered here

### F4 — supply-chain provenance (candidate — implemented in this tree)

The updater already constrained the target version and git tag with strict
allowlists (`is_safe_target_version`, `_require_safe_git_tag`), fetched exactly
one tag ref, verified the tag resolves to a well-formed commit and that `HEAD`
lands on it, and verified the installed version string. Those are integrity /
injection defenses — they are **not** authenticity. This F4 candidate adds the
missing authenticity/provenance layer in `provenance.py`, gated **before** the
first working-tree mutation (the new `verifying_provenance` phase, §6).

**Mechanism (editable / git source).** git-native **SSH signed-tag
verification** — no new dependency and no hand-rolled signature crypto:

- The release tag is an **SSH-signed** annotated tag (`git tag -s` with
  `gpg.format=ssh`). Verification runs
  `git -c gpg.format=ssh -c gpg.ssh.allowedSignersFile=<file> verify-tag <tag>`;
  git/ssh do the cryptography and return non-zero for an unsigned tag, an
  annotated-but-unsigned tag, or a key outside the `allowed_signers` file.
- The signed tag **message** carries a structured provenance manifest between
  `-----BEGIN HERMES PROVENANCE-----` / `-----END HERMES PROVENANCE-----` with
  `name`, `version`, `platform`, `arch`, `origin`, `commit`. Because the
  signature covers the whole tag object, tampering **any** manifest field or
  the target commit invalidates the signature.
- After a good signature, the gate **binds** every manifest field back to what
  the updater is about to install: release name, version, platform, arch
  (`any` = portable), origin (normalized), and the artifact digest (the
  tag→commit id, cross-checked against the independently fetched commit). It
  also enforces an **anti-downgrade/replay floor** (max of the installed
  version and any configured minimum) and optional platform/arch/principal
  allowlists.

**Fail-closed posture (no policy ⇒ no update).** If the `allowed_signers`
file is missing/empty, or no origin allowlist is set, provenance is **refused**
— there is no "no policy ⇒ allow" path. Origin confusion, downgrade/replay,
unsigned/unknown-key, manifest-less or malformed metadata, digest mismatch,
and platform/arch mismatch all refuse. A refusal is surfaced as
`HermesUpgradeError` and, because it runs before the first mutation, needs no
rollback (proven end-to-end: an unsigned release never reaches
`checking_out`).

**Operator policy — environment (editable).** Provision these before an
editable update (secrets are file paths / public data only; the private
signing key never touches the host being updated):

| env var | meaning |
|---------|---------|
| `MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS` | path to an OpenSSH `allowed_signers` file listing the authorized release principals + public keys (**required**) |
| `MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS` | comma/space list of allowed origin remotes, normalized (**required**) |
| `MS4_HERMES_PROVENANCE_ALLOWED_PRINCIPALS` | optional extra allowlist of signer principals (beyond the signers file) |
| `MS4_HERMES_PROVENANCE_PLATFORMS` | optional allowed `sys.platform` tokens |
| `MS4_HERMES_PROVENANCE_ARCHES` | optional allowed CPU arch tokens (`amd64`→`x86_64`, `arm64`→`aarch64`) |
| `MS4_HERMES_PROVENANCE_MIN_VERSION` | optional explicit anti-downgrade floor (the effective floor is the max of this and the installed version) |
| `MS4_HERMES_PROVENANCE_PYPI_ATTESTATION` | pypi attestation opt-in flag; **has no effect until a verifier ships** (see below) |

**PyPI wheel provenance — explicit UNSHIPPED integration dependency.** The
`pypi` path **fails closed**: no PEP 740 / Sigstore attestation verifier ships
in this candidate, so pypi provenance is **refused** rather than downgraded to
transport/hash-only trust. Closing the pypi arm means wiring a real attestation
verifier into `PyPiAttestationVerifier` and flipping the policy on. See
`DEPENDENCY_POLICY_NOTE.md` for the integration contract.

**What is proven vs. what still requires a live gate.** The gate LOGIC and its
every refusal direction are proven hermetically with **real** `git verify-tag`
+ `ssh-keygen` SSH signatures in `tests/ms4_hermes_admin/test_provenance_r1.py`
(and every property is mutation-tested). Oracle/REST `update_available` now
requires the official annotated tag object to carry signature bytes; unsigned
published tags are labeled and not offered by Retry/Pin. That is tag-object
inspection, not a substitute for F4 signer verification at install time. What
is NOT closed here: (1) **upstream publication of a new authorized signed
tag** after `v2026.8.3` / `0.20.0`; (2) the **PyPI attestation verifier**
(unshipped dependency above); (3) a REAL signed live update against GitHub +
the external MS4 restart, which remains the **F5 live-service gate**.

### Hardware / architecture gate (open)

CI (`.github/workflows/ms4-hermes-admin.yml`) runs the complete
`tests/ms4_hermes_admin` product gate from a basetemp under the runner's external
temporary directory on **real** GitHub-hosted Windows, Linux **x86_64**, and
macOS **arm64** runners, driving the real `git` binary. The suite aborts if that
basetemp is inside any parent Git root. That is real OS coverage, but:

- **Linux `aarch64` (Jetson/Thor/ARM64 servers)** is a **hardware execution
  gate** — it must run on a real aarch64 host (the workflow's
  `hardware-execution-gates` job, manual dispatch, self-hosted ARM64 runner).
  A green x86_64 matrix is **not** a substitute.
- A **real updater run** (live GitHub fetch + real wheel install + external
  restart) is the F5 gate above; CI injects those boundaries and does not
  perform a live update.

CI string simulation is **not** physical cross-platform proof for these two
axes.

---

## 10. Quick reference

| thing | value |
|-------|-------|
| Managed subtree (owned) | `<checkout>/plugins/ms4_consciousness/` |
| Managed plugin source-of-truth | `machine_spirit_4/plugins/hermes/ms4_consciousness` |
| Checkout dir override | `MS4_HERMES_DIR` (wins on every OS) |
| Managed checkout marker | `machine_spirit_4/runtime/hermes_active_checkout.json` (used by `runtime_common.hermes_dir()` when env unset) |
| Managed checkout path | `machine_spirit_4/runtime/hermes-managed/` |
| Windows default | `~/Documents/hermes-agent` |
| macOS default | `~/Library/Application Support/hermes-agent` |
| POSIX default | `$XDG_DATA_HOME/hermes-agent` or `~/.local/share/hermes-agent` |
| Contained interpreter | `machine_spirit_4/.venv/{Scripts/python.exe \| bin/python}` |
| State snapshot | `machine_spirit_4/runtime/hermes_update_state.json` |
| Terminal reconcile | installed ≥ latest supersedes stale `failed` presentation; audit kept in `progress` + `superseded_failure` |
| Operator state | `GET /api/v1/hermes/version` fields `installed_relation`, `operator_state` |
| Success phase | `done` |
| Recoverable failure | `rolled_back` (prior state restored) |
| Fatal failure | `rollback_failed` (manual recovery) |
| Never emitted | `git reset --hard`, `git clean`, `git restore`, `git checkout -- <path>` |
| F4 provenance phase | `verifying_provenance` (before first mutation) |
| F4 required env (editable) | `MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS`, `MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS` |
| F4 verify verbs (read-only) | `git verify-tag`, `git cat-file tag`, `git rev-parse`, `git remote get-url` |
| F4 pypi path | fail-closed (attestation verifier unshipped — see §9) |
| Restart after update | required, manual/control-plane (F5) |
| Open gates | F4 live signing-key/attestation provisioning + F5 (live-service) + Linux aarch64 hardware (gate logic implemented + hermetically tested in this candidate) |
