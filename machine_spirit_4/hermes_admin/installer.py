"""Idempotent Hermes upgrader.

Mirrors HiveMind ``ollama_admin::run_update_job``. The shape is the
same — phase transitions, persistent snapshot, blocking subprocess
launches off the request thread, recovery on failure — but the actual
install commands run through the operator's normal package manager:

  * ``editable``: fetch the resolved release tag, resolve it to a commit,
    then detached-checkout that commit and run ``pip install -e .``
    inside MS4's contained ``.venv``. Re-stamps the
    ``ms4_consciousness`` plugin from the MS4 source-of-truth after
    checkout so a Hermes tag bump never loses it.
  * ``pypi``:     ``pip install --upgrade hermes-agent==<X>`` inside
    MS4's contained ``.venv``. No git involvement.

There is no vendored Hermes anywhere in TMR. The install commands
pull straight from GitHub (for git operations) and PyPI (for the
wheel), the same paths a fresh operator would use by hand.

Dirty-checkout isolation (F2): when the active editable install points
at an external/operator checkout with any dirt, that checkout becomes a
read-only origin/provenance source and the update runs in
``runtime/hermes-managed``. Managed-path allowances apply only after the
active install points at that exact contained checkout. No user-owned
file is auto-deleted, reset, checked out, imported, or built.

Resource preflight (F9): an editable update is dominated by one long network
step (``git fetch``) whose peak cost is disk, not time -- git streams the
incoming pack to ``tmp_pack_*`` and then indexes it alongside. On a nearly
full volume that fetch does not fail, it crawls, so the only symptom used to
be an opaque ``Command timed out after 600s`` ten minutes later, plus an
orphaned partial pack that made the next attempt worse. Free space is now
measured (and reported) before the job starts and again immediately before the
fetch; a timeout is now reported as a *timeout* distinct from a command
failure; and a step's wall-clock budget is enforced by the updater itself,
because ``subprocess.run(timeout=...)`` provably does not bound its own
runtime once grandchildren inherit the captured pipes.

Rollback (F3): in-place managed updates retain the exact prior HEAD and
checkout mode. First migration failures retain the dirty source bytes
without reinstalling from them; a post-install failure is fatal/manual
recovery rather than executing dirty source. Neither path uses ``git
reset --hard``, ``git clean``, ``git restore``, or ``git checkout --
<path>``. PyPI updates rely on pip's own wheel replacement.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import ntpath
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import BinaryIO, Callable

from . import provenance, state, versioning

_LOG = logging.getLogger(__name__)
_HEX_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_PRERELEASE_TARGET_RE = re.compile(r"^\s*v?\d+(?:\.\d+){0,2}-")


MS4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLUGIN_SRC = MS4_ROOT / "plugins" / "hermes" / "ms4_consciousness"

# F2: MS4-generated paths the clean-checkout guard may ignore. Git reports
# paths POSIX-style (forward slashes) on every platform, so containment is
# proven against these forward-slash segments.
MANAGED_PLUGIN_PARENT = "plugins"
MANAGED_PLUGIN_NAME = "ms4_consciousness"
MANAGED_PLUGIN_RELPOSIX = f"{MANAGED_PLUGIN_PARENT}/{MANAGED_PLUGIN_NAME}"
MANAGED_CHECKOUT_PARENT = "runtime"
MANAGED_CHECKOUT_NAME = "hermes-managed"
ACTIVE_CHECKOUT_MARKER_NAME = "hermes_active_checkout.json"
UPDATE_LOCK_NAME = "hermes_update.lock"
CANONICAL_HERMES_SOURCE_ORIGIN = (
    "https://github.com/NousResearch/hermes-agent.git"
)
_MANAGED_BACKUP_ROOT_RE = re.compile(r"^\.ms4-upgrade-backup-\d{4}-\d{2}-\d{2}$")
_MANAGED_BACKUP_ROOT_FILES = frozenset(
    {"test_environment_bash_paths.py", "test_ms4_consciousness_plugin.py"}
)


def _default_venv_python(
    ms4_root: Path | PurePath,
    *,
    platform_name: str = sys.platform,
) -> Path | PurePath:
    """Return the contained interpreter path for the target platform."""
    if platform_name == "win32":
        return ms4_root / ".venv" / "Scripts" / "python.exe"
    return ms4_root / ".venv" / "bin" / "python"


def _default_managed_checkout(ms4_root: Path | PurePath) -> Path | PurePath:
    """Return MS4's isolated editable Hermes checkout on every platform."""
    return ms4_root / MANAGED_CHECKOUT_PARENT / MANAGED_CHECKOUT_NAME


def _active_checkout_marker(ms4_root: Path | PurePath) -> Path | PurePath:
    return ms4_root / MANAGED_CHECKOUT_PARENT / ACTIVE_CHECKOUT_MARKER_NAME


DEFAULT_VENV_PYTHON = _default_venv_python(MS4_ROOT)


class HermesUpgradeError(RuntimeError):
    """Raised when an upgrade step fails. Surfaced to the operator."""


class HermesDirtyCheckoutError(HermesUpgradeError):
    """The active editable checkout contains operator-owned changes."""


class HermesUpdateLockedError(HermesUpgradeError):
    """Another process owns the whole Hermes update transaction."""


class HermesInstallMismatchError(HermesUpgradeError):
    """Installed editable identity does not match the vetted release."""


class HermesUpgradeTimeoutError(HermesUpgradeError):
    """A step exhausted its wall-clock budget.

    Distinct from a command *failure*: a timeout means the updater stopped
    waiting, NOT that git/pip reported an error. The command may have been
    making slow forward progress (see the throughput note attached to the
    message), so the operator's remedy is different -- fix the starved
    resource or raise the budget, rather than debug a git error.
    """


class HermesInsufficientDiskError(HermesUpgradeError):
    """The checkout volume lacks the free space an update provably needs."""


def _as_upgrade_error(exc: Exception, context: str) -> HermesUpgradeError:
    if isinstance(exc, HermesUpgradeError):
        return exc
    return HermesUpgradeError(f"{context}: {type(exc).__name__}: {exc}")


@dataclass
class _WholeJobLock:
    path: Path
    handle: BinaryIO
    released: bool = False

    @classmethod
    def acquire(cls, path: Path) -> "_WholeJobLock":
        handle: BinaryIO | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+b", buffering=0)
            if os.name == "nt":
                import msvcrt

                if path.stat().st_size == 0:
                    handle.write(b"\0")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            if handle is not None:
                handle.close()
            busy_errnos = {errno.EACCES, errno.EAGAIN}
            if hasattr(errno, "EDEADLK"):
                busy_errnos.add(errno.EDEADLK)
            if exc.errno in busy_errnos:
                raise HermesUpdateLockedError(
                    "Another process already owns the Hermes update lock"
                ) from exc
            raise HermesUpgradeError(
                f"Could not acquire Hermes update lock at {path}: {exc}"
            ) from exc
        assert handle is not None
        return cls(path=path, handle=handle)

    def release(self) -> None:
        if self.released:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.released = True
            self.handle.close()

    def __enter__(self) -> "_WholeJobLock":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def _update_lock_path() -> Path:
    """Keep tests isolated by colocating the lock with the active state path."""
    return state._STORE.path.parent / UPDATE_LOCK_NAME


_LOCAL_UPDATE_GUARD = threading.Lock()
_LOCAL_UPDATE_JOB_ID: str | None = None


def _set_local_update_job(job_id: str) -> None:
    global _LOCAL_UPDATE_JOB_ID
    with _LOCAL_UPDATE_GUARD:
        _LOCAL_UPDATE_JOB_ID = job_id


def _clear_local_update_job(job_id: str) -> None:
    global _LOCAL_UPDATE_JOB_ID
    with _LOCAL_UPDATE_GUARD:
        if _LOCAL_UPDATE_JOB_ID == job_id:
            _LOCAL_UPDATE_JOB_ID = None


def _local_update_snapshot() -> state.UpdateJobSnapshot | None:
    snapshot = state.last_update(refresh=True)
    with _LOCAL_UPDATE_GUARD:
        if (
            snapshot is not None
            and snapshot.status == "running"
            and snapshot.job_id == _LOCAL_UPDATE_JOB_ID
        ):
            return snapshot
    return None


def recover_interrupted_update() -> state.UpdateJobSnapshot | None:
    """Fail an ownerless durable running job after a cold process start.

    A live updater proves ownership with both the process-local job id and the
    cross-process lock.  If neither exists, the persisted ``running`` snapshot
    came from an interrupted process and must not keep the UI disabled forever.
    """
    snapshot = state.last_update(refresh=True)
    if snapshot is None or snapshot.status != "running":
        return snapshot
    if _local_update_snapshot() is not None:
        return snapshot
    try:
        job_lock = _WholeJobLock.acquire(_update_lock_path())
    except HermesUpdateLockedError:
        return snapshot
    try:
        return state.fail_interrupted_job()
    finally:
        job_lock.release()


_GIT_TAG_RE = re.compile(r"^v?[0-9A-Za-z][0-9A-Za-z._-]{0,63}$")
_GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_GIT_BASH_PATH_ENV = "HERMES_GIT_BASH_PATH"
_WINDOWS_GIT_BASH_PATH = r"C:\Program Files\Git\bin\bash.exe"
_WINDOWS_GIT_BASH_SHA256 = "c79fa1cd106bd3e2321f87e580930142ba9309d17b968fa5e3e71dedbbdc36aa"
_GIT_BASH_SMOKE_TIMEOUT_SECONDS = 10
_GIT_BASH_UNSAFE_ENV_NAMES = frozenset(
    {"bash_env", "env", "bashopts", "shellopts", "ps4", "bash_xtracefd"}
)
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


def _sanitized_git_bash_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in tuple(env):
        folded = key.casefold()
        if folded in _GIT_BASH_UNSAFE_ENV_NAMES or folded.startswith("bash_func_"):
            env.pop(key, None)
    return env


def _open_windows_read_seal(path: str) -> BinaryIO:
    """Open ``path`` for hashing while denying concurrent write/delete access."""
    if sys.platform != "win32":
        raise OSError("Win32 read sealing is available only on Windows")

    import ctypes
    import msvcrt
    from ctypes import wintypes

    generic_read = 0x80000000
    file_share_read = 0x00000001
    open_existing = 3
    file_attribute_normal = 0x00000080
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        path,
        generic_read,
        file_share_read,
        None,
        open_existing,
        file_attribute_normal,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error), path)

    try:
        descriptor = msvcrt.open_osfhandle(
            int(handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        close_handle(handle)
        raise
    try:
        return os.fdopen(descriptor, "rb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def _require_valid_git_bash_candidate() -> str | None:
    if sys.platform != "win32":
        return os.environ.get(_GIT_BASH_PATH_ENV)
    candidate = os.environ.get(_GIT_BASH_PATH_ENV)
    if not candidate:
        raise HermesUpgradeError(
            f"{_GIT_BASH_PATH_ENV} is required on Windows and must be "
            f"{_WINDOWS_GIT_BASH_PATH}"
        )
    if ntpath.normcase(ntpath.normpath(candidate)) != ntpath.normcase(
        ntpath.normpath(_WINDOWS_GIT_BASH_PATH)
    ):
        raise HermesUpgradeError(
            f"{_GIT_BASH_PATH_ENV} must be {_WINDOWS_GIT_BASH_PATH}; got {candidate}"
        )
    canonical_path = Path(_WINDOWS_GIT_BASH_PATH)
    try:
        valid = canonical_path.is_file()
    except OSError:
        valid = False
    if not valid:
        raise HermesUpgradeError(
            f"{_GIT_BASH_PATH_ENV} does not name an existing file: "
            f"{_WINDOWS_GIT_BASH_PATH}"
        )

    try:
        with _open_windows_read_seal(_WINDOWS_GIT_BASH_PATH) as handle:
            digest = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            actual_digest = digest.hexdigest().lower()
            if actual_digest != _WINDOWS_GIT_BASH_SHA256:
                raise HermesUpgradeError(
                    f"Git Bash SHA-256 mismatch for {_WINDOWS_GIT_BASH_PATH}: "
                    f"expected {_WINDOWS_GIT_BASH_SHA256}, got {actual_digest}"
                )

            smoke_argv = [
                _WINDOWS_GIT_BASH_PATH,
                "--noprofile",
                "--norc",
                "-c",
                "printf GIT_BASH_OK",
            ]
            try:
                smoke = subprocess.run(
                    smoke_argv,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="strict",
                    timeout=_GIT_BASH_SMOKE_TIMEOUT_SECONDS,
                    check=False,
                    shell=False,
                    env=_sanitized_git_bash_env(),
                )
            except subprocess.TimeoutExpired as exc:
                raise HermesUpgradeError(
                    "Git Bash smoke test timed out after "
                    f"{_GIT_BASH_SMOKE_TIMEOUT_SECONDS}s"
                ) from exc
            except UnicodeError as exc:
                raise HermesUpgradeError(
                    f"Git Bash smoke test decode failure: {exc}"
                ) from exc
            except OSError as exc:
                raise HermesUpgradeError(
                    f"Git Bash smoke test execution failed: {exc}"
                ) from exc
            if smoke.returncode != 0:
                raise HermesUpgradeError(
                    f"Git Bash smoke test exit={smoke.returncode}: "
                    f"{(smoke.stderr or '')[:600]}"
                )
            if smoke.stdout != "GIT_BASH_OK":
                raise HermesUpgradeError(
                    "Git Bash smoke test stdout mismatch: expected 'GIT_BASH_OK', "
                    f"got {smoke.stdout!r}"
                )
    except (OSError, ValueError) as exc:
        raise HermesUpgradeError(
            f"Could not open/read/hash sealed Git Bash {_WINDOWS_GIT_BASH_PATH}: {exc}"
        ) from exc
    return _WINDOWS_GIT_BASH_PATH


def _require_safe_git_tag(target_tag: str) -> str:
    invalid = (
        not _GIT_TAG_RE.fullmatch(target_tag)
        or target_tag.startswith(("-", "."))
        or target_tag.endswith(".")
        or target_tag.casefold().endswith(".lock")
        or ".." in target_tag
    )
    if invalid:
        raise HermesUpgradeError(f"unsafe resolved git tag: {target_tag!r}")
    return target_tag


def _display_command(cmd: list[str], *, platform_name: str = sys.platform) -> str:
    """Render argv using the quoting rules of the platform that runs it."""
    if platform_name == "win32":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def _command_env(cmd: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    if Path(cmd[0]).name.casefold() in {"git", "git.exe"}:
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
        # The updater runs in a daemon thread with no operator attached.
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GCM_INTERACTIVE"] = "Never"
        env["GIT_NO_REPLACE_OBJECTS"] = "1"
        env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


# --------------------------------------------------------------------------- #
# Free-space preflight.
#
# A Hermes editable update is a git fetch: git streams the incoming pack to
# ``.git/objects/pack/tmp_pack_XXXXXX``, then ``index-pack`` writes the final
# ``.pack`` + ``.idx`` alongside it. Peak usage is therefore roughly the
# incoming pack counted twice, on the volume that holds the checkout.
#
# When that volume is effectively full the fetch does not fail cleanly -- it
# crawls. The 2026-06-30 job (job 1c178865) wrote 78.6 MB of tmp_pack in 630 s
# (~128 KB/s) against 249 MB free and was killed by the wall-clock budget, and
# an earlier attempt the same morning left another 47.5 MB orphan behind. Both
# surfaced only as an opaque "Command timed out after 600s", ten minutes late.
#
# So: measure before committing to the fetch, and refuse fast with the numbers.
# --------------------------------------------------------------------------- #

MIN_FREE_BYTES_ENV = "MS4_HERMES_MIN_FREE_BYTES"
#: Absolute floor. Even a tiny repository needs working room for pack indexing,
#: pip's build/temp trees, and the venv rewrite that follows the checkout.
ABSOLUTE_MIN_FREE_BYTES = 1 * 1024**3
#: Proportional term: half the already-materialized pack bytes. An incremental
#: fetch is normally far smaller than the whole repository, but the tmp pack and
#: the indexed pack coexist, so a fraction of repository size is the honest
#: scale for "how big can this get".
PACK_HEADROOM_DIVISOR = 2
_GIT_TMP_PACK_PREFIX = "tmp_pack_"


def _free_bytes(path: Path | PurePath) -> int:
    """Free bytes on the volume holding ``path`` (nearest existing ancestor)."""
    candidate = Path(path)
    while True:
        try:
            return shutil.disk_usage(str(candidate)).free
        except (OSError, ValueError):
            parent = candidate.parent
            if parent == candidate:
                raise
            candidate = parent


def _pack_dir_usage(directory: Path | PurePath) -> tuple[int, int, list[str]]:
    """Return ``(pack_bytes, orphan_bytes, orphan_names)`` for a checkout.

    ``orphan_*`` describe leftover ``tmp_pack_*`` files: partial downloads from
    fetches that were killed before ``index-pack`` finished. They are dead
    weight that git only reclaims on ``git gc``, and every further timeout adds
    another one -- so they are surfaced to the operator rather than silently
    ignored. Nothing here deletes them; the checkout may be operator-owned.
    """
    pack_dir = Path(directory) / ".git" / "objects" / "pack"
    pack_bytes = 0
    orphan_bytes = 0
    orphans: list[str] = []
    try:
        entries = sorted(pack_dir.iterdir(), key=lambda item: item.name)
    except OSError:
        return (0, 0, [])
    for entry in entries:
        try:
            if not entry.is_file():
                continue
            size = entry.stat().st_size
        except OSError:
            continue
        if entry.name.startswith(_GIT_TMP_PACK_PREFIX):
            orphan_bytes += size
            orphans.append(entry.name)
        elif entry.name.endswith(".pack"):
            pack_bytes += size
    return (pack_bytes, orphan_bytes, orphans)


def _configured_min_free_bytes(environ: dict[str, str] | None = None) -> int | None:
    """Operator override for the free-space floor, or ``None`` when unset.

    Rejects junk fail-closed rather than silently falling back, so a typo in
    the deployment environment cannot quietly disable the guard.
    """
    source = os.environ if environ is None else environ
    raw = (source.get(MIN_FREE_BYTES_ENV) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise HermesUpgradeError(
            f"{MIN_FREE_BYTES_ENV} must be a base-10 byte count; got {raw!r}"
        ) from exc
    if value < 0:
        raise HermesUpgradeError(
            f"{MIN_FREE_BYTES_ENV} must not be negative; got {raw!r}"
        )
    return value


def _required_free_bytes(
    pack_bytes: int,
    *,
    reference_pack_bytes: int = 0,
    environ: dict[str, str] | None = None,
) -> int:
    """Bytes the update must see free before it is allowed to start.

    Two regimes, because they cost very differently:

    * **Incremental** (the target already has packs): only the new objects
      arrive. Half the existing pack size is a generous ceiling for that.
    * **Fresh** (the target has no packs -- the F2 dirty-checkout migration
      creates an empty ``runtime/hermes-managed`` and fetches into it): the
      whole history arrives as one pack and is then indexed *alongside* the
      download, so peak usage is about twice the repository size.
      ``reference_pack_bytes`` carries the known size of the source checkout,
      since the empty target cannot reveal it.
    """
    override = _configured_min_free_bytes(environ)
    if override is not None:
        return override
    if pack_bytes <= 0 and reference_pack_bytes > 0:
        needed = 2 * reference_pack_bytes
    else:
        needed = max(pack_bytes, reference_pack_bytes) // PACK_HEADROOM_DIVISOR
    return max(ABSOLUTE_MIN_FREE_BYTES, needed)


def _format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value} B"


def _preflight_disk_space(
    directory: Path | PurePath,
    *,
    reference_pack_bytes: int = 0,
    environ: dict[str, str] | None = None,
    record: bool = True,
) -> dict[str, int]:
    """Refuse fast when ``directory``'s volume cannot hold the update.

    Raises :class:`HermesInsufficientDiskError` with the actual numbers and the
    override knob, instead of letting the fetch crawl into a wall-clock
    timeout. Returns the measurements so callers can log them.
    """
    directory = Path(directory)
    pack_bytes, orphan_bytes, orphans = _pack_dir_usage(directory)
    try:
        free = _free_bytes(directory)
    except OSError as exc:
        raise HermesUpgradeError(
            f"could not measure free space for the Hermes checkout {directory}: {exc}"
        ) from exc
    required = _required_free_bytes(
        pack_bytes,
        reference_pack_bytes=reference_pack_bytes,
        environ=environ,
    )
    measurements = {
        "free_bytes": free,
        "required_bytes": required,
        "pack_bytes": pack_bytes,
        "orphan_pack_bytes": orphan_bytes,
    }
    if record:
        note = (
            f"disk preflight {directory}: free={_format_bytes(free)}, "
            f"required={_format_bytes(required)}, "
            f"packs={_format_bytes(pack_bytes)}"
        )
        if orphan_bytes:
            note += (
                f", reclaimable partial-fetch garbage={_format_bytes(orphan_bytes)}"
                f" ({len(orphans)} file(s))"
            )
        state.append_progress(note)
    if free >= required:
        return measurements

    remedy = (
        f"free at least {_format_bytes(required - free)} more on that volume, "
        f"or set {MIN_FREE_BYTES_ENV} to an explicit byte count if this floor "
        f"is wrong for that host"
    )
    if orphan_bytes:
        remedy = (
            f"run 'git -C {directory} gc --prune=now' to reclaim "
            f"{_format_bytes(orphan_bytes)} of abandoned partial-fetch packs "
            f"({', '.join(orphans[:4])}), and/or " + remedy
        )
    raise HermesInsufficientDiskError(
        f"insufficient free space for a Hermes update on {directory}: "
        f"{_format_bytes(free)} free, {_format_bytes(required)} required "
        f"(repository packs {_format_bytes(pack_bytes)}). "
        f"A git fetch writes an incoming pack and then indexes it, so a nearly "
        f"full volume makes the fetch crawl until it hits the wall-clock budget "
        f"instead of failing cleanly. Remedy: {remedy}."
    )


#: Grace beyond a step's own budget before the updater stops waiting on the
#: interpreter's cleanup and declares the step timed out on its own authority.
POST_TIMEOUT_GRACE_SECONDS = 15


def _run_bounded(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """``subprocess.run`` whose wall-clock bound is enforced on the CALLER.

    ``subprocess.run(timeout=...)`` does NOT bound its own runtime. It kills
    only the direct child, then blocks in ``communicate()`` until every process
    that inherited the captured stdout/stderr pipes exits. ``git fetch`` spawns
    ``git-remote-https`` / ``index-pack`` grandchildren that inherit exactly
    those handles, so the "timeout" lasts as long as the slowest grandchild.
    Measured on this host: a 3 s budget against a 45 s pipe-holding grandchild
    raised ``TimeoutExpired`` after 45.1 s. Job 1c178865 shows the same shape --
    a 600 s budget, a job that finalized at 635.6 s, and a ``tmp_pack`` that
    kept growing for ~30 s after git was killed.

    So the wait happens on a worker thread and the deadline is enforced here.
    Past ``timeout + POST_TIMEOUT_GRACE_SECONDS`` the updater stops waiting and
    reports the timeout, instead of the job hanging for an unbounded time in a
    phase the operator can neither observe nor cancel.

    ``subprocess.run`` is deliberately retained as the execution primitive: it
    is the injection seam the rest of this suite drives, and keeping it means
    argv/env/stdin policy is asserted against the exact call that runs.
    """
    outcome: dict[str, object] = {}

    def _call() -> None:
        try:
            outcome["result"] = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=text,
                timeout=timeout,
                check=False,
                env=_command_env(cmd),
            )
        except BaseException as exc:  # re-raised on the calling thread
            outcome["error"] = exc

    worker = threading.Thread(
        target=_call,
        daemon=True,
        name="hermes-bounded-exec",
    )
    worker.start()
    worker.join(timeout + POST_TIMEOUT_GRACE_SECONDS)
    if worker.is_alive():
        # The child (or an orphaned grandchild holding its pipes) outlived even
        # the grace window. Abandon the daemon thread rather than wedge the
        # update job on it; nothing downstream consumes a partial result.
        raise subprocess.TimeoutExpired(cmd, timeout)
    error = outcome.get("error")
    if error is not None:
        raise error  # type: ignore[misc]
    result = outcome.get("result")
    if result is None:
        raise HermesUpgradeError(
            f"command produced no result and no error: {_display_command(cmd)}"
        )
    return result  # type: ignore[return-value]


def _timeout_error(
    display: str,
    *,
    timeout: int,
    elapsed: float,
    cwd_hint: Path | PurePath | None = None,
) -> HermesUpgradeTimeoutError:
    """Build a timeout message that says *timed out*, not *failed*."""
    detail = (
        f"Command TIMED OUT (did not fail) after {timeout}s "
        f"[wall clock {elapsed:.1f}s]: {display}"
    )
    if cwd_hint is not None:
        try:
            free = _free_bytes(cwd_hint)
        except OSError:
            free = -1
        if free >= 0:
            detail += f"; free space on that volume is now {_format_bytes(free)}"
            required = _required_free_bytes(
                _pack_dir_usage(cwd_hint)[0],
            )
            if free < required:
                detail += (
                    f" (below the {_format_bytes(required)} an update needs -- "
                    f"the command was almost certainly starved for disk, "
                    f"not hung)"
                )
    detail += (
        ". No partial result was used and no working-tree mutation is implied "
        "by a timeout in this phase. Re-running the update is safe."
    )
    return HermesUpgradeTimeoutError(detail)


def _git_directory_hint(cmd: list[str]) -> Path | None:
    """Recover the ``git -C <dir>`` operand so timeouts can report its volume."""
    if not cmd or Path(cmd[0]).name.casefold() not in {"git", "git.exe"}:
        return None
    for index, token in enumerate(cmd[1:-1], start=1):
        if token == "-C":
            try:
                return Path(cmd[index + 1])
            except (IndexError, TypeError, ValueError):
                return None
    return None


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    """Blocking subprocess wrapper. Stdout/stderr are captured so the
    failure message contains the actual error text the operator needs
    to see in the UI, not a generic non-zero exit code.
    """
    display = _display_command(cmd)
    state.append_progress(f"$ {display}")
    started = time.monotonic()
    try:
        result = _run_bounded(
            cmd,
            cwd=cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise _timeout_error(
            display,
            timeout=timeout,
            elapsed=time.monotonic() - started,
            cwd_hint=_git_directory_hint(cmd) or cwd,
        ) from exc
    except FileNotFoundError as exc:
        raise HermesUpgradeError(f"Command not found ({cmd[0]}): {exc}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        message = stderr or stdout or f"exit {result.returncode}"
        raise HermesUpgradeError(f"{cmd[0]} exit={result.returncode}: {message[:600]}")
    if result.stdout:
        state.append_progress(result.stdout.strip().splitlines()[-1][:200])
    return result


def _run_capture(cmd: list[str], *, cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """Like :func:`_run` but returns the ``CompletedProcess`` even on a
    non-zero exit, for git *queries* where a non-zero code is a meaningful
    answer rather than an error (e.g. ``symbolic-ref`` exits 1 on a detached
    HEAD). Still argv-based, shell-free, stdin-closed, and bounded; only a
    timeout or missing executable raises.
    """
    display = _display_command(cmd)
    state.append_progress(f"$ {display}")
    started = time.monotonic()
    try:
        return _run_bounded(cmd, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise _timeout_error(
            display,
            timeout=timeout,
            elapsed=time.monotonic() - started,
            cwd_hint=_git_directory_hint(cmd) or cwd,
        ) from exc
    except FileNotFoundError as exc:
        raise HermesUpgradeError(f"Command not found ({cmd[0]}): {exc}") from exc


def _run_capture_bytes(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Binary query runner; preserves CR/LF and rejects decoding ambiguity."""
    display = _display_command(cmd)
    state.append_progress(f"$ {display}")
    started = time.monotonic()
    try:
        return _run_bounded(cmd, cwd=cwd, timeout=timeout, text=False)
    except subprocess.TimeoutExpired as exc:
        raise _timeout_error(
            display,
            timeout=timeout,
            elapsed=time.monotonic() - started,
            cwd_hint=_git_directory_hint(cmd) or cwd,
        ) from exc
    except FileNotFoundError as exc:
        raise HermesUpgradeError(f"Command not found ({cmd[0]}): {exc}") from exc
    except OSError as exc:
        raise HermesUpgradeError(
            f"Command execution failed ({cmd[0]}): {exc}"
        ) from exc


def _normalized_real_path(path: Path | PurePath) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _same_path(left: Path | PurePath, right: Path | PurePath) -> bool:
    try:
        return _normalized_real_path(left) == _normalized_real_path(right)
    except OSError:
        return False


def _is_exact_managed_checkout(directory: Path | PurePath) -> bool:
    expected = Path(_default_managed_checkout(MS4_ROOT))
    if os.path.normcase(os.path.abspath(str(directory))) != os.path.normcase(
        os.path.abspath(str(expected))
    ):
        return False
    try:
        managed = Path(directory)
        _assert_managed_checkout_path(managed)
        _assert_managed_git_dir(managed)
        _assert_git_toplevel(managed)
    except HermesUpgradeError:
        return False
    return True


def _assert_managed_checkout_path(directory: Path) -> None:
    """Prove ``directory`` is exactly MS4's contained managed checkout.

    The lexical equality rejects arbitrary injected destinations. Real-path
    equality additionally rejects a redirected ``runtime`` parent or
    ``hermes-managed`` symlink/junction on Windows and POSIX.
    """
    runtime_root = MS4_ROOT / MANAGED_CHECKOUT_PARENT
    expected = Path(_default_managed_checkout(MS4_ROOT))
    try:
        if os.path.normcase(os.path.abspath(str(directory))) != os.path.normcase(
            os.path.abspath(str(expected))
        ):
            raise HermesUpgradeError(
                f"managed Hermes checkout must be exactly {expected}; got {directory}"
            )
        for label, path in (
            ("MS4 root", MS4_ROOT),
            ("runtime root", runtime_root),
            ("managed checkout", directory),
        ):
            if os.path.lexists(path) and (
                not Path(path).is_dir()
                or _path_is_reparse_point(Path(path))
            ):
                raise HermesUpgradeError(
                    f"managed Hermes {label} is not a real non-reparse directory: {path}"
                )
        ms4_real = _normalized_real_path(MS4_ROOT)
        runtime_real = _normalized_real_path(runtime_root)
        expected_runtime_real = os.path.normcase(
            os.path.join(ms4_real, MANAGED_CHECKOUT_PARENT)
        )
        managed_real = _normalized_real_path(directory)
        expected_managed_real = os.path.normcase(
            os.path.join(expected_runtime_real, MANAGED_CHECKOUT_NAME)
        )
        if (
            os.path.islink(runtime_root)
            or os.path.islink(directory)
            or runtime_real != expected_runtime_real
            or managed_real != expected_managed_real
        ):
            raise HermesUpgradeError(
                "managed Hermes checkout escapes the MS4 runtime directory "
                f"(possible symlink/junction): {directory}"
            )
    except HermesUpgradeError:
        raise
    except (OSError, ValueError) as exc:
        raise HermesUpgradeError(
            f"could not prove managed Hermes checkout containment: {directory}"
        ) from exc


def _git_fetch_command(
    directory: Path | PurePath,
    target_tag: str,
    *,
    remote_url: str = "origin",
) -> list[str]:
    """Build a bounded fetch for the one release ref the updater needs."""
    _require_safe_git_tag(target_tag)
    tag_ref = f"refs/tags/{target_tag}"
    return [
        "git",
        "-C",
        str(directory),
        "-c",
        "credential.interactive=never",
        "-c",
        "maintenance.auto=false",
        "fetch",
        "--no-tags",
        remote_url,
        f"{tag_ref}:{tag_ref}",
    ]


def _status_paths(line: str) -> list[str] | None:
    """Extract the repo-relative path(s) from one ``--porcelain=v1`` line.

    Returns ``None`` (fail closed) for anything we will not parse with
    certainty: malformed lines, or C-quoted paths (``"..."`` -- git only
    quotes paths containing control/odd characters; the managed plugin never
    does, so a quoted path is treated as suspicious). Rename/copy lines
    (``R``/``C``) yield BOTH the origin and destination so a move that
    straddles the managed boundary is judged on both endpoints. Status is run
    with ``core.quotePath=false`` so ordinary non-ASCII names are literal.
    """
    if len(line) < 4:
        return None
    status = line[:2]
    path_field = line[3:]
    if status and status[0] in ("R", "C") and " -> " in path_field:
        tokens = path_field.split(" -> ")
    else:
        tokens = [path_field]
    out: list[str] = []
    for token in tokens:
        if not token or token.startswith('"'):
            return None
        out.append(token)
    return out


def _is_within_managed_plugin(path: str) -> bool:
    """True iff ``path`` is STRICTLY inside ``plugins/ms4_consciousness/``.

    Git always emits forward-slash, repo-relative paths. Case-sensitive on
    purpose (a case variant is NOT the managed subtree and must fail closed).
    Rejects backslashes, empty/``.``/``..`` segments (traversal), path-prefix
    siblings (``plugins/ms4_consciousness_backup/...``), and the bare
    directory itself (requires at least one child segment).
    """
    if not path or "\\" in path:
        return False
    segments = path.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        return False
    return (
        len(segments) >= 3
        and segments[0] == MANAGED_PLUGIN_PARENT
        and segments[1] == MANAGED_PLUGIN_NAME
    )


def _managed_backup_root(path: str) -> str | None:
    """Return the dated root for an exact legacy MS4 backup artifact."""
    if not path or "\\" in path:
        return None
    segments = path.split("/")
    if len(segments) < 2 or any(seg in ("", ".", "..") for seg in segments):
        return None
    root = segments[0]
    if not _MANAGED_BACKUP_ROOT_RE.fullmatch(root):
        return None
    payload = segments[1:]
    if len(payload) == 1 and payload[0] in _MANAGED_BACKUP_ROOT_FILES:
        return root
    if len(payload) >= 2 and payload[0] == MANAGED_PLUGIN_NAME:
        return root
    return None


def _managed_backup_path_is_safely_contained(directory: Path, path: str) -> bool:
    """Prove a legacy backup artifact resolves inside its dated checkout root."""
    root = _managed_backup_root(path)
    if root is None:
        return False
    try:
        base_real = os.path.normcase(os.path.realpath(str(directory)))
        backup_dir = os.path.join(str(directory), root)
        backup_real = os.path.normcase(os.path.realpath(backup_dir))
        expected_backup = os.path.normcase(os.path.join(base_real, root))
        candidate = os.path.join(str(directory), *path.split("/"))
        candidate_real = os.path.normcase(os.path.realpath(candidate))
        return (
            not os.path.islink(backup_dir)
            and os.path.isdir(backup_real)
            and backup_real == expected_backup
            and os.path.commonpath((backup_real, candidate_real)) == backup_real
        )
    except (OSError, ValueError):
        return False


def _managed_plugin_is_safely_contained(directory: Path) -> bool:
    """Prove ``<directory>/plugins/ms4_consciousness`` is a real directory
    physically contained in the checkout -- no symlink/junction escape.

    ``os.path.realpath`` resolves both POSIX symlinks and Windows junctions,
    so a redirected subtree resolves to a location other than the expected
    one and is rejected. ``os.path.islink`` is an additional guard for POSIX
    symlinks. Fails closed on any OS error.
    """
    try:
        base_real = os.path.realpath(str(directory))
        plugins_dir = os.path.join(str(directory), MANAGED_PLUGIN_PARENT)
        plugin_dir = os.path.join(plugins_dir, MANAGED_PLUGIN_NAME)
        if os.path.islink(plugins_dir) or os.path.islink(plugin_dir):
            return False
        expected = os.path.normcase(
            os.path.join(base_real, MANAGED_PLUGIN_PARENT, MANAGED_PLUGIN_NAME)
        )
        real_plugin = os.path.normcase(os.path.realpath(plugin_dir))
        return real_plugin == expected and os.path.isdir(real_plugin)
    except OSError:
        return False


def _ensure_no_replace_refs(directory: Path) -> None:
    result = _run(
        [
            "git",
            "-C",
            str(directory),
            "for-each-ref",
            "--format=%(refname)",
            "refs/replace/",
        ],
        timeout=30,
    )
    refs = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if refs:
        raise HermesUpgradeError(
            f"Hermes checkout contains forbidden replace refs: {', '.join(refs[:4])}"
        )


def _dirty_checkout_lines(directory: Path) -> list[str]:
    _ensure_no_replace_refs(directory)
    result = _run(
        [
            "git",
            "-C",
            str(directory),
            "-c",
            "core.quotePath=false",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        timeout=30,
    )
    return [line for line in (result.stdout or "").splitlines() if line.strip()]


def _ensure_pristine_external_checkout(directory: Path) -> None:
    """External/operator checkouts migrate on any tracked or untracked dirt."""
    dirty = _dirty_checkout_lines(directory)
    if not dirty:
        return
    preview = "; ".join(dirty[:8])
    suffix = f"; and {len(dirty) - 8} more" if len(dirty) > 8 else ""
    raise HermesDirtyCheckoutError(
        "External Hermes checkout has tracked or untracked changes; "
        "using the isolated MS4-managed checkout instead of mutating it: "
        f"{preview}{suffix}"
    )


def _ensure_clean_checkout(directory: Path) -> None:
    """Refuse an editable update against a dirty checkout (F2).

    Ignored changes are limited to the safely contained managed plugin and
    untracked legacy MS4 backup artifacts described in the module docstring.
    Anything else fails closed -- no user-owned file is deleted or reset.
    """
    dirty = _dirty_checkout_lines(directory)
    if not dirty:
        return

    offending: list[str] = []
    managed_plugin_dirty = False
    for line in dirty:
        paths = _status_paths(line)
        if paths is None:
            offending.append(line)
        elif all(_is_within_managed_plugin(path) for path in paths):
            managed_plugin_dirty = True
        elif line[:2] == "??" and all(
            _managed_backup_path_is_safely_contained(directory, path) for path in paths
        ):
            continue
        else:
            offending.append(line)

    if (
        not offending
        and managed_plugin_dirty
        and not _managed_plugin_is_safely_contained(directory)
    ):
        # Every dirty path claims to be inside the managed subtree, but the
        # subtree is not a real contained directory (symlink/junction escape,
        # missing, or not a directory). Fail closed on all of them.
        offending = list(dirty)

    if not offending:
        return

    preview = "; ".join(offending[:8])
    suffix = f"; and {len(offending) - 8} more" if len(offending) > 8 else ""
    raise HermesDirtyCheckoutError(
        "Hermes checkout has tracked or untracked changes outside the MS4-managed "
        f"updater paths ({MANAGED_PLUGIN_RELPOSIX}/ and dated legacy backups); "
        "refusing update before "
        f"fetch/checkout/plugin mutation: {preview}{suffix}"
    )


def _git_head(directory: Path) -> str:
    result = _run(
        ["git", "-C", str(directory), "rev-parse", "--verify", "HEAD^{commit}"],
        timeout=30,
    )
    commit = (result.stdout or "").strip()
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise HermesUpgradeError(f"Git returned an invalid HEAD commit id: {commit!r}")
    return commit.lower()


def _assert_editable_git_dir(directory: Path) -> None:
    """Filesystem-only preconditions for an editable checkout (no subprocess)."""
    if not directory.exists():
        raise HermesUpgradeError(f"Hermes checkout not found at {directory}")
    if not (directory / ".git").exists():
        raise HermesUpgradeError(f"{directory} is not a git checkout; cannot tag-pin in editable mode")


def _assert_git_toplevel(directory: Path) -> None:
    result = _run(
        ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
        timeout=30,
    )
    top = (result.stdout or "").strip()
    if not top or not _same_path(Path(top), directory):
        raise HermesUpgradeError(
            f"refusing managed checkout whose git top-level is not itself: {directory}"
        )


def _assert_managed_git_dir(directory: Path) -> None:
    """Require a real ``<managed>/.git`` directory contained in the target."""
    git_dir = directory / ".git"
    commondir = git_dir / "commondir"
    config_path = git_dir / "config"
    worktree_config = git_dir / "config.worktree"
    expected = os.path.normcase(
        os.path.join(_normalized_real_path(directory), ".git")
    )
    if (
        not git_dir.is_dir()
        or _path_is_reparse_point(git_dir)
        or _normalized_real_path(git_dir) != expected
    ):
        raise HermesUpgradeError(
            f"managed Hermes checkout requires a real contained .git directory: {git_dir}"
        )
    if os.path.lexists(commondir):
        raise HermesUpgradeError(
            f"managed Hermes checkout forbids alternate Git common dirs: {commondir}"
        )
    result = _run(
        ["git", "-C", str(directory), "rev-parse", "--absolute-git-dir"],
        timeout=30,
    )
    reported = (result.stdout or "").strip()
    if (
        not reported
        or _path_is_reparse_point(Path(reported))
        or _normalized_real_path(Path(reported)) != expected
    ):
        raise HermesUpgradeError(
            f"managed Hermes checkout gitdir escapes containment: {reported!r}"
        )
    common = _run(
        [
            "git",
            "-C",
            str(directory),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        timeout=30,
    )
    reported_common = (common.stdout or "").strip()
    if (
        not reported_common
        or _path_is_reparse_point(Path(reported_common))
        or _normalized_real_path(Path(reported_common)) != expected
    ):
        raise HermesUpgradeError(
            f"managed Hermes checkout common dir escapes containment: {reported_common!r}"
        )
    expected_config = os.path.normcase(os.path.join(expected, "config"))
    if (
        not config_path.is_file()
        or _path_is_reparse_point(config_path)
        or _normalized_real_path(config_path) != expected_config
    ):
        raise HermesUpgradeError(
            f"managed Hermes checkout requires a real contained config: {config_path}"
        )
    config = _run(
        [
            "git",
            "-C",
            str(directory),
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "config",
        ],
        timeout=30,
    )
    reported_config = (config.stdout or "").strip()
    if (
        not reported_config
        or _path_is_reparse_point(Path(reported_config))
        or _normalized_real_path(Path(reported_config)) != expected_config
    ):
        raise HermesUpgradeError(
            f"managed Hermes checkout config escapes containment: {reported_config!r}"
        )
    reported_worktree_config_result = _run(
        [
            "git",
            "-C",
            str(directory),
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "config.worktree",
        ],
        timeout=30,
    )
    expected_worktree_config = os.path.normcase(
        os.path.join(expected, "config.worktree")
    )
    reported_worktree_config = (
        reported_worktree_config_result.stdout or ""
    ).strip()
    if (
        not reported_worktree_config
        or _normalized_real_path(Path(reported_worktree_config))
        != expected_worktree_config
    ):
        raise HermesUpgradeError(
            "managed Hermes worktree config path escapes containment: "
            f"{reported_worktree_config!r}"
        )
    if os.path.lexists(worktree_config):
        raise HermesUpgradeError(
            f"managed Hermes checkout forbids worktree config: {worktree_config}"
        )
    worktree_extension = _run_capture(
        [
            "git",
            "config",
            "--file",
            str(config_path),
            "--no-includes",
            "--get-all",
            "extensions.worktreeConfig",
        ],
        timeout=30,
    )
    if worktree_extension.returncode == 0:
        raise HermesUpgradeError(
            "managed Hermes checkout forbids extensions.worktreeConfig"
        )
    if (
        worktree_extension.returncode != 1
        or (worktree_extension.stdout or "").strip()
        or (worktree_extension.stderr or "").strip()
    ):
        detail = (
            worktree_extension.stderr
            or worktree_extension.stdout
            or f"exit {worktree_extension.returncode}"
        ).strip()
        raise HermesUpgradeError(
            f"could not classify managed Hermes worktree config: {detail[:300]}"
        )


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _path_is_reparse_point(path: Path) -> bool:
    """True when ``path`` is a symlink/junction/reparse point (fail-closed on error)."""
    try:
        if os.path.islink(path):
            return True
        if sys.platform == "win32":
            attrs = getattr(os.lstat(path), "st_file_attributes", 0)
            if attrs & _FILE_ATTRIBUTE_REPARSE_POINT:
                return True
        return False
    except OSError:
        return True


def _is_unborn_checkout(directory: Path) -> bool:
    """Return True only for Git's unambiguous, symbolic unborn-HEAD state."""
    head = _run_capture(
        [
            "git",
            "-C",
            str(directory),
            "rev-parse",
            "--verify",
            "--quiet",
            "HEAD^{commit}",
        ],
        timeout=30,
    )
    if head.returncode == 0:
        return False
    if (
        head.returncode != 1
        or (head.stdout or "").strip()
        or (head.stderr or "").strip()
    ):
        return False

    symbolic = _run_capture(
        ["git", "-C", str(directory), "symbolic-ref", "--quiet", "HEAD"],
        timeout=30,
    )
    head_ref = (symbolic.stdout or "").strip()
    if (
        symbolic.returncode != 0
        or (symbolic.stderr or "").strip()
        or not head_ref.startswith("refs/heads/")
    ):
        return False
    valid_ref = _run_capture(
        ["git", "check-ref-format", head_ref],
        timeout=30,
    )
    if (
        valid_ref.returncode != 0
        or (valid_ref.stdout or "").strip()
        or (valid_ref.stderr or "").strip()
    ):
        return False
    branch = _run_capture(
        ["git", "-C", str(directory), "show-ref", "--verify", "--quiet", head_ref],
        timeout=30,
    )
    return (
        branch.returncode == 1
        and not (branch.stdout or "").strip()
        and not (branch.stderr or "").strip()
    )


def _managed_checkout_has_only_git_entry(directory: Path) -> bool:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return False
    return len(entries) == 1 and entries[0].name == ".git"


def _path_identity(path: Path) -> tuple[int, ...]:
    info = os.lstat(path)
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
        int(getattr(info, "st_file_attributes", 0)),
    )


def _directory_object_identity(path: Path) -> tuple[int, ...]:
    """Bind a directory object without treating sibling churn as replacement."""
    info = os.lstat(path)
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(getattr(info, "st_file_attributes", 0)),
    )


def _contained_git_control_file_bytes(git_dir: Path, name: str) -> bytes | None:
    path = git_dir / name
    try:
        expected = os.path.normcase(
            os.path.join(_normalized_real_path(git_dir), name)
        )
        if (
            _path_is_reparse_point(path)
            or not path.is_file()
            or _normalized_real_path(path) != expected
        ):
            return None
        return path.read_bytes()
    except (OSError, ValueError):
        return None


def _config_values(result: subprocess.CompletedProcess, context: str) -> list[str]:
    output = result.stdout or b""
    stderr = result.stderr or b""
    if (
        result.returncode != 0
        or stderr.strip()
        or not output.endswith(b"\0")
    ):
        detail_bytes = stderr or output
        detail = (
            detail_bytes.decode("utf-8", errors="backslashreplace")
            if detail_bytes
            else f"exit {result.returncode}"
        ).strip()
        raise HermesUpgradeError(f"{context} is absent or ambiguous: {detail[:300]}")
    raw_values = output[:-1].split(b"\0")
    if not raw_values or any(not value for value in raw_values):
        raise HermesUpgradeError(f"{context} contains an empty value")
    try:
        values = [value.decode("utf-8", errors="strict") for value in raw_values]
    except UnicodeDecodeError as exc:
        raise HermesUpgradeError(f"{context} is not valid UTF-8") from exc
    for value in values:
        if value != value.strip() or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in value
        ):
            raise HermesUpgradeError(
                f"{context} contains control or surrounding whitespace"
            )
    return values


def _require_unrewritten_git_url(
    directory: Path,
    source_origin: str,
    *,
    repository_exists: bool = True,
) -> None:
    """Prove effective Git config leaves the vetted source URL byte-exact."""
    repository_args = (
        ["-C", str(directory)]
        if repository_exists
        else [f"--git-dir={directory / '.git'}"]
    )
    resolved = _run_capture(
        [
            "git",
            *repository_args,
            "ls-remote",
            "--get-url",
            source_origin,
        ],
        timeout=30,
    )
    urls = (resolved.stdout or "").splitlines()
    if (
        resolved.returncode != 0
        or (resolved.stderr or "").strip()
        or urls != [source_origin]
    ):
        detail = (
            resolved.stderr
            or resolved.stdout
            or f"exit {resolved.returncode}"
        ).strip()
        raise HermesUpgradeError(
            "effective Git config rewrites or ambiguously resolves the vetted "
            f"Hermes source URL: {detail[:400]}"
        )


def _unambiguous_local_origin(directory: Path) -> str:
    """Return the one raw contained local origin, rejecting other config sources."""
    git_dir = directory / ".git"
    config_path = git_dir / "config"
    config_bytes = _contained_git_control_file_bytes(git_dir, "config")
    if config_bytes is None:
        raise HermesUpgradeError(
            f"managed Hermes checkout config is not a contained regular file: {config_path}"
        )
    if b"\0" in config_bytes:
        raise HermesUpgradeError("managed Hermes checkout config contains NUL bytes")
    includes = _run_capture(
        [
            "git",
            "config",
            "--file",
            str(config_path),
            "--no-includes",
            "--null",
            "--get-regexp",
            r"^include(if\..*)?\.path$",
        ],
        timeout=30,
    )
    if includes.returncode == 0:
        raise HermesUpgradeError(
            "managed Hermes checkout local config contains include/includeIf directives"
        )
    if (
        includes.returncode != 1
        or (includes.stdout or "").strip()
        or (includes.stderr or "").strip()
    ):
        detail = (
            includes.stderr
            or includes.stdout
            or f"exit {includes.returncode}"
        ).strip()
        raise HermesUpgradeError(
            f"could not classify managed Hermes config includes: {detail[:300]}"
        )

    local = _run_capture_bytes(
        [
            "git",
            "config",
            "--file",
            str(config_path),
            "--no-includes",
            "--null",
            "--get-all",
            "remote.origin.url",
        ],
        timeout=30,
    )
    local_values = _config_values(local, "managed Hermes local origin")
    if len(local_values) != 1:
        raise HermesUpgradeError(
            "managed Hermes local origin must have exactly one URL value"
        )

    effective = _run_capture_bytes(
        [
            "git",
            "-C",
            str(directory),
            "config",
            "--null",
            "--get-all",
            "remote.origin.url",
        ],
        timeout=30,
    )
    effective_values = _config_values(effective, "managed Hermes effective origin")
    if effective_values != local_values:
        raise HermesUpgradeError(
            "managed Hermes effective origin has additional or external config sources"
        )
    return local_values[0]


@dataclass(frozen=True)
class _OriginRepairGuardState:
    identities: tuple[tuple[int, ...], ...]
    head_bytes: bytes
    config_bytes: bytes
    origin: str


def _origin_repair_guard_state(
    directory: Path,
) -> _OriginRepairGuardState | None:
    """Capture all stale-origin repair guards without mutating repository state."""
    runtime_root = MS4_ROOT / MANAGED_CHECKOUT_PARENT
    git_dir = directory / ".git"
    head_path = git_dir / "HEAD"
    config_path = git_dir / "config"
    try:
        _assert_managed_checkout_path(directory)
        for path in (runtime_root, directory, git_dir):
            if not path.is_dir() or _path_is_reparse_point(path):
                return None
        _assert_managed_git_dir(directory)
        _assert_git_toplevel(directory)
        if not _managed_checkout_has_only_git_entry(directory):
            return None
        if not _is_unborn_checkout(directory):
            return None
        head_bytes = _contained_git_control_file_bytes(git_dir, "HEAD")
        config_bytes = _contained_git_control_file_bytes(git_dir, "config")
        if head_bytes is None or config_bytes is None:
            return None
        origin = _unambiguous_local_origin(directory)
        # Progress persistence atomically replaces a sibling state file under
        # runtime_root, legitimately changing that directory's timestamps.
        # Bind the directory object itself while keeping full identities for
        # every managed-checkout path that the repair can affect.
        identities = (_directory_object_identity(runtime_root),) + tuple(
            _path_identity(path)
            for path in (
                directory,
                git_dir,
                head_path,
                config_path,
            )
        )
    except (HermesUpgradeError, OSError, UnicodeError, ValueError):
        return None
    return _OriginRepairGuardState(
        identities=identities,
        head_bytes=head_bytes,
        config_bytes=config_bytes,
        origin=origin,
    )


def _repair_empty_unborn_managed_origin(directory: Path, source_origin: str) -> bool:
    """Reset origin on an empty unborn managed checkout when all guards pass.

    Permitted only for the updater-owned managed path in the exact state where
    the directory contains nothing outside ``.git`` and has no valid HEAD.
    Returns True when origin was repaired; False without mutation otherwise.
    """
    before = _origin_repair_guard_state(directory)
    if before is None:
        return False
    if before.origin == source_origin:
        return False
    # Repeat every observable guard under the updater lock immediately before
    # mutation. This deliberately does not claim atomicity against an arbitrary
    # uncooperative external writer racing after this final snapshot.
    after = _origin_repair_guard_state(directory)
    if after is None or after != before:
        return False
    _run(
        [
            "git",
            "config",
            "--file",
            str(directory / ".git" / "config"),
            "--fixed-value",
            "--replace-all",
            "remote.origin.url",
            source_origin,
            before.origin,
        ],
        timeout=30,
    )
    post = _origin_repair_guard_state(directory)
    if post is None or post.origin != source_origin:
        raise HermesUpgradeError(
            "managed Hermes origin repair postcondition did not match vetted source"
        )
    return True


def _prepare_managed_checkout(directory: Path, source_origin: str) -> None:
    """Create or validate MS4's isolated checkout without touching the source."""
    created = not directory.exists()
    if created:
        # Resolve system/global URL rewrites without discovering any enclosing
        # repository and before creating the managed parent or .git directory.
        _require_unrewritten_git_url(
            directory,
            source_origin,
            repository_exists=False,
        )
    _assert_managed_checkout_path(directory)
    if directory.exists():
        _assert_editable_git_dir(directory)
        _assert_managed_git_dir(directory)
        _assert_git_toplevel(directory)
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after creating the parent to close a symlink/junction race.
        _assert_managed_checkout_path(directory)
        _run(["git", "init", str(directory)], timeout=120)
        _assert_editable_git_dir(directory)
        _assert_managed_checkout_path(directory)
        _assert_managed_git_dir(directory)
        _assert_git_toplevel(directory)

    _require_unrewritten_git_url(directory, source_origin)
    if created:
        _run(
            ["git", "-C", str(directory), "remote", "add", "origin", source_origin],
            timeout=30,
        )

    managed_origin = _unambiguous_local_origin(directory)
    if managed_origin != source_origin:
        if created:
            raise HermesUpgradeError(
                "new managed Hermes checkout origin does not match initialization source"
            )
        if not _repair_empty_unborn_managed_origin(directory, source_origin):
            raise HermesUpgradeError(
                "managed Hermes checkout origin does not match the vetted source origin"
            )
        managed_origin = _unambiguous_local_origin(directory)
        if managed_origin != source_origin:
            raise HermesUpgradeError(
                "managed Hermes checkout origin repair did not persist vetted source"
            )
    _ensure_clean_checkout(directory)


@dataclass(frozen=True)
class _MarkerPreState:
    path: Path
    contents: bytes | None


def _capture_active_marker_prestate() -> _MarkerPreState:
    marker = Path(_active_checkout_marker(MS4_ROOT))
    try:
        contents = marker.read_bytes() if marker.exists() else None
    except OSError as exc:
        raise HermesUpgradeError(
            f"could not capture prior managed checkout marker: {exc}"
        ) from exc
    return _MarkerPreState(path=marker, contents=contents)


def _restore_active_marker(pre: _MarkerPreState) -> None:
    if pre.contents is None:
        try:
            pre.path.unlink(missing_ok=True)
        except OSError as exc:
            raise HermesUpgradeError(
                f"could not remove newly-published managed checkout marker: {exc}"
            ) from exc
        return
    pre.path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="hermes_active_checkout_restore_",
        suffix=".json",
        dir=str(pre.path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(pre.contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, pre.path)
    except Exception as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise HermesUpgradeError(
            f"could not restore prior managed checkout marker: {exc}"
        ) from exc


def _publish_active_checkout(
    directory: Path,
    *,
    version: str,
    commit: str,
) -> None:
    """Atomically publish the managed source selected on the next restart."""
    _assert_managed_checkout_path(directory)
    _assert_managed_git_dir(directory)
    _assert_git_toplevel(directory)
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise HermesUpgradeError(
            f"refusing to publish invalid managed checkout commit: {commit!r}"
        )
    if _git_head(directory) != commit.lower():
        raise HermesUpgradeError(
            "refusing to publish a managed checkout marker whose commit "
            "does not match HEAD"
        )
    marker = Path(_active_checkout_marker(MS4_ROOT))
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "Ms4HermesActiveCheckout.v1",
        "directory": str(directory),
        "version": version,
        "commit": commit.lower(),
    }
    fd, temporary = tempfile.mkstemp(
        prefix="hermes_active_checkout_",
        suffix=".json",
        dir=str(marker.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    except Exception as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise HermesUpgradeError(
            f"could not publish managed Hermes checkout marker: {exc}"
        ) from exc


def _fetch_source_url(directory: Path, remote_url: str) -> str:
    if remote_url != "origin":
        return remote_url
    configured = _run_capture_bytes(
        [
            "git",
            "-C",
            str(directory),
            "config",
            "--null",
            "--get-all",
            "remote.origin.url",
        ],
        timeout=30,
    )
    values = _config_values(configured, "Hermes fetch origin")
    if len(values) != 1:
        raise HermesUpgradeError(
            "Hermes fetch origin must have exactly one URL value"
        )
    return values[0]


def _git_fetch_resolve(
    directory: Path,
    target_tag: str,
    *,
    remote_url: str = "origin",
) -> str:
    """Fetch exactly the release tag ref and resolve it to a commit id.

    This mutates only ``.git`` (a new tag ref + objects); it does NOT touch
    the working tree, HEAD, or any branch the operator is on, so it is safe
    to run before capturing the rollback pre-state and never itself needs to
    be rolled back.
    """
    _require_unrewritten_git_url(
        directory,
        _fetch_source_url(directory, remote_url),
    )
    state.set_phase("fetching_remote")
    # Last gate before the only long-running network step. The checkout may
    # have been created since the job-level preflight and free space is a
    # moving target on a shared box, so re-measure against the exact directory
    # the pack will land in. Callers that know they are fetching a full history
    # into an empty target size that case explicitly beforehand (see
    # _run_update_job_locked); this is the unconditional backstop.
    _preflight_disk_space(directory)
    _run(
        _git_fetch_command(directory, target_tag, remote_url=remote_url),
        timeout=600,
    )
    tag_ref = f"refs/tags/{target_tag}^{{commit}}"
    resolved = _run(
        ["git", "-C", str(directory), "rev-parse", "--verify", tag_ref],
        timeout=30,
    )
    commit = (resolved.stdout or "").strip()
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise HermesUpgradeError(
            f"Fetched git tag {target_tag!r} resolved to an invalid commit id: {commit!r}"
        )
    return commit.lower()


def _git_detach_checkout(directory: Path, commit: str) -> None:
    """Detached-checkout ``commit`` and verify HEAD landed on it.

    This is the FIRST working-tree mutation of an editable update. Any
    failure here (or after) is what the F3 rollback restores.
    """
    state.set_phase("checking_out")
    _run(
        [
            "git",
            "-C",
            str(directory),
            "-c",
            "advice.detachedHead=false",
            "checkout",
            "--detach",
            commit,
        ],
        timeout=120,
    )
    head = _git_head(directory)
    if head != commit:
        raise HermesUpgradeError(
            f"Hermes checkout HEAD {head} does not match fetched release commit {commit}"
        )


def _git_fetch_checkout(directory: Path, target_tag: str) -> str:
    """Clean-gate, fetch, resolve, and detached-checkout a release tag.

    Kept as a single composed entry point (its external behavior is
    unchanged) so the adversarial real-git tests keep exercising the exact
    fetch/checkout contract. :func:`run_update_job` calls the same pieces
    directly so it can capture the rollback pre-state between the clean gate
    and the first mutation.
    """
    _require_safe_git_tag(target_tag)
    _assert_editable_git_dir(directory)
    _ensure_clean_checkout(directory)
    commit = _git_fetch_resolve(directory, target_tag)
    _git_detach_checkout(directory, commit)
    return commit


@dataclass(frozen=True)
class _EditablePreState:
    """Exact clean pre-update checkout state captured BEFORE any mutation."""

    directory: Path
    commit: str
    mode: str  # "branch" | "detached"
    branch: str | None


def _git_current_branch(directory: Path) -> str | None:
    """Return the current branch name, or ``None`` when HEAD is detached.

    Uses ``symbolic-ref --quiet --short HEAD`` which exits non-zero (no error
    text) on a detached HEAD -- hence :func:`_run_capture`, not :func:`_run`.
    """
    result = _run_capture(
        ["git", "-C", str(directory), "symbolic-ref", "--quiet", "--short", "HEAD"],
        timeout=30,
    )
    if result.returncode != 0:
        return None
    branch = (result.stdout or "").strip()
    return branch or None


def _capture_editable_prestate(directory: Path) -> _EditablePreState:
    """Capture the exact pre-update HEAD commit and checkout mode (F3).

    Must run AFTER the clean-check passes and BEFORE the first working-tree
    mutation (the detached checkout). Read-only.
    """
    commit = _git_head(directory)
    branch = _git_current_branch(directory)
    mode = "branch" if branch is not None else "detached"
    return _EditablePreState(directory=directory, commit=commit, mode=mode, branch=branch)


def _restore_editable_commit(pre: _EditablePreState) -> None:
    """Return the checkout to the exact pre-update commit AND mode (F3).

    A branch pre-state is restored by switching back to that branch (its ref
    was never moved -- a detached checkout does not move branches), preserving
    "on a branch". A detached pre-state is restored by re-detaching onto the
    exact commit. Uses ONLY ``git checkout`` of a whole ref/commit -- never
    ``git reset --hard``, ``git clean``, ``git restore``, or
    ``git checkout -- <path>`` -- so untracked and pre-existing non-managed
    changes are never discarded.
    """
    directory = pre.directory
    if pre.mode == "branch" and pre.branch and not pre.branch.startswith("-"):
        _run(
            [
                "git",
                "-C",
                str(directory),
                "-c",
                "advice.detachedHead=false",
                "checkout",
                pre.branch,
            ],
            timeout=120,
        )
    else:
        _run(
            [
                "git",
                "-C",
                str(directory),
                "-c",
                "advice.detachedHead=false",
                "checkout",
                "--detach",
                pre.commit,
            ],
            timeout=120,
        )
    head = _git_head(directory)
    if head != pre.commit:
        raise HermesUpgradeError(
            f"rollback restored HEAD {head}, expected pre-update commit {pre.commit}"
        )


def _restamp_managed_plugin(plugin_src: Path, directory: Path) -> None:
    """Idempotently re-stamp ONLY the managed plugin subtree.

    Used both by the forward path (so a repeat update whose tree still carries
    the previously-stamped, untracked plugin does not self-block) and by
    rollback. If a managed subtree is already present it is removed and
    recopied -- but ONLY after :func:`_managed_plugin_is_safely_contained`
    proves it is a real directory physically inside the checkout (no
    symlink/junction escape). The recursive delete is therefore confined to a
    freshly validated managed subtree and can never reach a user/OS path.
    """
    plugin_dst = directory / MANAGED_PLUGIN_PARENT / MANAGED_PLUGIN_NAME
    if plugin_dst.exists() or plugin_dst.is_symlink():
        if not _managed_plugin_is_safely_contained(directory):
            raise HermesUpgradeError(
                "refusing to remove a managed plugin destination that is not a real "
                f"contained directory (possible symlink/junction escape): {plugin_dst}"
            )
        shutil.rmtree(plugin_dst)
    _sync_plugin(plugin_src, plugin_dst)


def _reinstall_editable(python: Path, directory: Path) -> None:
    """Reinstall the editable package from ``directory`` during rollback so
    the recorded distribution metadata matches the restored source tree."""
    _run([str(python), "-m", "pip", "install", "--no-input", "-e", str(directory)], timeout=900)


def _rollback_editable(
    pre: _EditablePreState,
    plugin_src: Path,
    venv_python: Path,
    primary: HermesUpgradeError,
    marker_prestate: _MarkerPreState | None = None,
) -> None:
    """Restore the exact pre-update editable state, then re-raise (F3).

    Records the primary and rollback outcomes as distinct progress phases.
    Always raises: on a successful rollback it raises a HermesUpgradeError
    embedding the verbatim primary error; on a failed rollback it raises one
    embedding BOTH errors and leaves a ``rollback_failed`` phase so the
    operator sees that manual recovery is required (fatal + visible).
    """
    phase_error: Exception | None = None
    try:
        state.set_phase("rolling_back", note=f"restore {pre.commit[:12]} ({pre.mode})")
    except Exception as exc:
        phase_error = exc
    try:
        _restore_editable_commit(pre)
        _restamp_managed_plugin(plugin_src, pre.directory)
        _reinstall_editable(venv_python, pre.directory)
        if marker_prestate is not None:
            _restore_active_marker(marker_prestate)
    except Exception as failure:
        rollback_error = _as_upgrade_error(failure, "rollback failed")
        try:
            state.set_phase("rollback_failed", note=str(rollback_error)[:200])
        except Exception:
            pass
        raise HermesUpgradeError(
            "Hermes update failed AND rollback ALSO failed; manual recovery required. "
            f"primary error: {primary} || rollback error: {rollback_error}"
        ) from rollback_error
    try:
        state.set_phase(
            "rolled_back",
            note="restored prior commit, editable install, and managed plugin",
        )
    except Exception as exc:
        phase_error = phase_error or exc
    if phase_error is not None:
        raise HermesUpgradeError(
            "Hermes update failed; source/install rollback succeeded, but durable "
            f"rollback state could not be recorded: {phase_error}. "
            f"primary error: {primary}"
        ) from phase_error
    raise HermesUpgradeError(
        f"Hermes update failed; rolled back to the prior state. primary error: {primary}"
    ) from primary


def _rollback_managed_migration(
    source_directory: Path,
    primary: HermesUpgradeError,
    *,
    install_attempted: bool,
) -> None:
    """Fail safely without ever invoking build/install code from dirty source."""
    state_errors: list[Exception] = []
    try:
        state.set_phase(
            "rolling_back",
            note=f"preserve operator checkout at {source_directory}",
        )
    except Exception as exc:
        state_errors.append(exc)
    if install_attempted:
        note = (
            "managed install was attempted; operator checkout was not touched "
            "and automatic reinstall from dirty source is forbidden"
        )
        try:
            state.set_phase("rollback_failed", note=note)
        except Exception as exc:
            state_errors.append(exc)
        state_suffix = (
            "; rollback state persistence also failed: "
            + " | ".join(str(error) for error in state_errors)
            if state_errors
            else ""
        )
        raise HermesUpgradeError(
            "Hermes managed-checkout update failed after install activation; "
            "manual recovery required, but the operator checkout was not touched. "
            f"primary error: {primary}{state_suffix}"
        ) from primary
    try:
        state.set_phase(
            "rolled_back",
            note="operator checkout and prior editable install were never touched",
        )
    except Exception as exc:
        state_errors.append(exc)
    state_suffix = (
        "; rollback state persistence also failed: "
        + " | ".join(str(error) for error in state_errors)
        if state_errors
        else ""
    )
    raise HermesUpgradeError(
        "Hermes managed-checkout update failed before install activation; "
        f"prior editable install retained. primary error: {primary}{state_suffix}"
    ) from primary


def _sync_plugin(plugin_src: Path, plugin_dst: Path) -> None:
    """Re-stamp ``ms4_consciousness`` plugin from MS4's source-of-truth.

    Modeled on Ollama's "re-warm" step — after a Hermes checkout swap,
    we restore our plugin into the new tree before the editable install
    re-registers it. The source lives outside the Hermes checkout so a
    tag bump never deletes it.
    """
    if not plugin_src.exists():
        raise HermesUpgradeError(f"MS4 plugin source missing: {plugin_src}")
    if plugin_dst.exists():
        raise HermesUpgradeError(
            "Hermes plugin destination already exists; refusing recursive replacement: "
            f"{plugin_dst}"
        )
    state.set_phase("syncing_plugin", note=f"sync {plugin_src} -> {plugin_dst}")
    plugin_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(plugin_src, plugin_dst)


def _pip_install_editable(python: Path, directory: Path) -> None:
    state.set_phase("pip_installing")
    _run([str(python), "-m", "pip", "install", "--no-input", "-e", str(directory)], timeout=900)


def _pip_install_wheel(python: Path, target_version: str) -> None:
    state.set_phase("pip_installing")
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--upgrade",
            f"hermes-agent=={target_version}",
        ],
        timeout=900,
    )


def _validate(
    python: Path,
    *,
    expected_version: str,
    editable_directory: Path | None = None,
    expected_commit: str | None = None,
    mismatch_error: type[HermesUpgradeError] = HermesUpgradeError,
) -> str:
    state.set_phase("validating")
    result = _run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m, plugins.ms4_consciousness as p; "
            "print(m.version('hermes-agent'))",
        ],
        timeout=60,
    )
    version = (result.stdout or "").strip().splitlines()[-1] if result.stdout else ""
    if not version:
        raise mismatch_error("Validation succeeded but version output was empty")
    if version != expected_version:
        raise mismatch_error(
            f"Validation found installed version {version}, expected {expected_version}"
        )
    if (editable_directory is None) != (expected_commit is None):
        raise mismatch_error(
            "Editable validation requires both directory and expected commit"
        )
    if editable_directory is not None and expected_commit is not None:
        head = _git_head(editable_directory)
        if head != expected_commit.lower():
            raise mismatch_error(
                f"Hermes checkout HEAD {head} does not match fetched release commit "
                f"{expected_commit.lower()}"
            )
    return version


def _validate_managed_install(
    python: Path,
    *,
    expected_version: str,
    editable_directory: Path,
    expected_commit: str,
) -> str:
    """Prove the installed editable distribution and imports use ``directory``."""
    version = _validate(
        python,
        expected_version=expected_version,
        editable_directory=editable_directory,
        expected_commit=expected_commit,
        mismatch_error=HermesInstallMismatchError,
    )
    probe = """
import importlib.metadata as metadata
import importlib.util
import json

records = []
for distribution in metadata.distributions(name="hermes-agent"):
    raw = distribution.read_text("direct_url.json")
    direct_url = None
    if raw:
        try:
            direct_url = json.loads(raw)
        except json.JSONDecodeError:
            direct_url = None
    records.append({"version": distribution.version, "direct_url": direct_url})

def origin(name):
    spec = importlib.util.find_spec(name)
    return spec.origin if spec is not None else None

print(json.dumps({
    "distributions": records,
    "plugin_origin": origin("plugins.ms4_consciousness"),
    "run_agent_origin": origin("run_agent"),
}))
"""
    result = _run([str(python), "-c", probe], timeout=60)
    raw_payload = (result.stdout or "").strip().splitlines()
    try:
        payload = json.loads(raw_payload[-1]) if raw_payload else None
    except (json.JSONDecodeError, TypeError) as exc:
        raise HermesInstallMismatchError(
            "Managed install validation returned invalid metadata JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise HermesInstallMismatchError(
            "Managed install validation returned no metadata object"
        )

    matched_direct_url = False
    for record in payload.get("distributions") or []:
        if not isinstance(record, dict) or record.get("version") != expected_version:
            continue
        direct_url = record.get("direct_url")
        if not isinstance(direct_url, dict):
            continue
        dir_info = direct_url.get("dir_info")
        url = direct_url.get("url")
        if not isinstance(dir_info, dict) or not dir_info.get("editable"):
            continue
        if not isinstance(url, str):
            continue
        source = versioning._file_url_to_path(url)
        if (
            source is not None
            and _is_exact_managed_checkout(editable_directory)
            and os.path.normcase(os.path.abspath(str(source)))
            == os.path.normcase(os.path.abspath(str(editable_directory)))
        ):
            matched_direct_url = True
            break
    if not matched_direct_url:
        raise HermesInstallMismatchError(
            "Installed Hermes editable direct_url does not match the managed checkout"
        )

    managed_real = _normalized_real_path(editable_directory)
    for label in ("plugin_origin", "run_agent_origin"):
        origin = payload.get(label)
        if not isinstance(origin, str) or not origin:
            raise HermesInstallMismatchError(
                f"Managed Hermes validation missing {label}"
            )
        try:
            origin_real = _normalized_real_path(Path(origin))
            if os.path.commonpath((managed_real, origin_real)) != managed_real:
                raise HermesInstallMismatchError(
                    f"Managed Hermes {label} resolves outside the managed checkout: "
                    f"{origin}"
                )
        except ValueError as exc:
            raise HermesInstallMismatchError(
                f"Managed Hermes {label} is on a different path root: {origin}"
            ) from exc
    return version


def _verify_release_provenance(
    *,
    mode: str,
    target: str,
    tag: str | None,
    directory: Path | None,
    expected_commit: str | None,
    current_version: str | None,
) -> None:
    """F4 supply-chain gate -- MUST run before any working-tree mutation/install.

    Delegates to the standalone :mod:`provenance` engine: git-native SSH
    signed-tag verification for editable installs (``git verify-tag`` against an
    operator-provisioned allowed_signers policy, binding release identity /
    version / platform / arch / origin / commit to an authorized signer, with a
    downgrade/replay floor), and a fail-closed PyPI attestation integration
    point for wheels. A refusal is surfaced as :class:`HermesUpgradeError` so the
    persisted job finalizes as failed on the normal path; because this runs
    before the first mutation, a refusal needs no rollback.

    This is a deliberate, patchable seam. Hermetic suites that are not
    exercising F4 replace it through the tests' autouse fixture -- exactly as
    they already inject pip/validate -- so the F2/F3/F6/F8 behavior they assert
    is unchanged. The real gate (accept + every refusal direction) is proven
    against real SSH signatures in ``tests/ms4_hermes_admin/test_provenance_r1.py``.

    Under ``HERMES_RELEASE_SIGNATURE_POLICY=allow_unsigned`` (the default) the
    cryptographic signer check is the ONLY step skipped; see
    :func:`_verify_unsigned_release_bindings` for what still runs.
    """
    if (
        versioning.release_signature_policy()
        == versioning.RELEASE_SIGNATURE_POLICY_ALLOW_UNSIGNED
    ):
        _verify_unsigned_release_bindings(
            mode=mode,
            target=target,
            tag=tag,
            directory=directory,
            expected_commit=expected_commit,
            current_version=current_version,
        )
        return
    state.set_phase("verifying_provenance")
    try:
        provenance.verify_update_provenance(
            mode=mode,
            release_name="hermes-agent",
            version=target,
            tag=tag,
            directory=str(directory) if directory is not None else None,
            expected_commit=expected_commit,
            current_version=current_version,
            environ=dict(os.environ),
        )
    except provenance.ProvenanceError as exc:
        raise HermesUpgradeError(f"F4 provenance verification failed: {exc}") from exc


def _verify_unsigned_release_bindings(
    *,
    mode: str,
    target: str,
    tag: str | None,
    directory: Path | None,
    expected_commit: str | None,
    current_version: str | None,
) -> None:
    """``allow_unsigned`` counterpart of the F4 gate; runs in the same slot.

    Everything that does not depend on a tag signature still runs: stable
    (non-prerelease) target, the anti-downgrade floor, the origin / platform /
    arch allowlists whenever the operator configured them, no replace refs,
    and the fetched tag resolving to the exact commit the checkout is about to
    activate (plus the upstream tag-object bindings when the annotated tag
    carries them). Only the cryptographic signer verification is skipped, and
    that is logged at WARNING and stamped into the job audit trail. Wheel
    version equality and rollback are enforced by the caller as before.
    """
    label = tag or f"hermes-agent=={target}"
    warning = f"signature policy allow_unsigned: installing unsigned tag {label}"
    state.set_phase("verifying_provenance", note=warning)
    _LOG.warning(warning)
    if mode not in {"editable", "pypi"}:
        raise HermesUpgradeError(f"unknown install mode for provenance: {mode!r}")
    if _PRERELEASE_TARGET_RE.match(target):
        raise HermesUpgradeError(
            f"release version {target!r} is a prerelease; updates require stable versions"
        )
    target_parsed = versioning.parse_semver(target)
    if target_parsed is None:
        raise HermesUpgradeError(f"release version {target!r} is not parseable")
    try:
        policy = provenance.ProvenancePolicy.from_environ(
            dict(os.environ),
            source_kind="git" if mode == "editable" else "pypi",
            current_version=current_version,
        )
    except provenance.ProvenanceError as exc:
        raise HermesUpgradeError(f"release policy is malformed: {exc}") from exc
    if policy.min_version is not None:
        floor = versioning.parse_semver(policy.min_version)
        if floor is None or target_parsed < floor:
            raise HermesUpgradeError(
                f"release version {target} is below the floor {policy.min_version} "
                "(downgrade/replay refused)"
            )
    host_platform = provenance.normalize_platform()
    host_arch = provenance.normalize_arch()
    if policy.allowed_platforms and host_platform not in policy.allowed_platforms:
        raise HermesUpgradeError(f"platform {host_platform!r} is not permitted by policy")
    if policy.allowed_arches and host_arch not in policy.allowed_arches:
        raise HermesUpgradeError(f"architecture {host_arch!r} is not permitted by policy")
    if mode == "pypi":
        state.append_progress(
            "unsigned wheel: attestation skipped by policy; pip resolves the exact "
            "pinned version and post-install validation binds it"
        )
        return
    if not tag:
        raise HermesUpgradeError("editable update requires a resolved release tag")
    if directory is None:
        raise HermesUpgradeError("editable update requires a checkout directory")
    if expected_commit is None or not _HEX_COMMIT_RE.fullmatch(expected_commit):
        raise HermesUpgradeError("editable update requires a valid 40/64-hex expected commit")
    verifier = provenance.GitSignedTagVerifier(
        allowed_signers_file=None,
        directory=str(directory),
    )
    try:
        origin = verifier.read_origin()
        if (
            policy.allowed_origins
            and provenance.normalize_origin(origin) not in policy.allowed_origins
        ):
            raise provenance.ProvenanceError(
                f"release origin {origin!r} is not in the configured origin allowlist"
            )
        verifier.assert_no_replace_refs()
    except provenance.ProvenanceError as exc:
        raise HermesUpgradeError(f"unsigned release binding failed: {exc}") from exc
    resolved = verifier.runner(
        [
            "git", "-C", str(directory),
            "rev-parse", "--verify", "--end-of-options",
            f"refs/tags/{tag}^{{commit}}",
        ]
    )
    resolved_commit = (resolved.stdout or "").strip()
    if resolved.returncode != 0 or not _HEX_COMMIT_RE.fullmatch(resolved_commit):
        raise HermesUpgradeError(f"could not resolve tag {tag!r} to a commit id")
    if resolved_commit.lower() != expected_commit.lower():
        raise HermesUpgradeError(
            f"tag {tag!r} resolves to {resolved_commit}, not the fetched commit "
            f"{expected_commit}"
        )
    try:
        tag_object = verifier.resolve_tag_object(tag)
    except provenance.ProvenanceError:
        state.append_progress(
            f"unsigned tag {tag} is lightweight; bound by commit {resolved_commit} only"
        )
        return
    raw = verifier.runner(["git", "-C", str(directory), "cat-file", "tag", tag_object])
    if raw.returncode != 0:
        raise HermesUpgradeError(f"could not read tag object {tag_object!r}")
    manifest = provenance.parse_upstream_tag_object(raw.stdout or "")
    if manifest is None:
        note = (
            f"unsigned tag {tag} does not carry upstream release metadata; "
            f"bound by commit {resolved_commit} only"
        )
        state.append_progress(note)
        _LOG.warning(note)
        return
    for actual, expected, what in (
        (manifest.name, "hermes-agent", "release name"),
        (manifest.version, target, "release version"),
        (manifest.tag, tag, "release tag"),
        (manifest.commit.lower(), expected_commit.lower(), "tag object commit"),
    ):
        if actual != expected:
            raise HermesUpgradeError(
                f"unsigned tag {tag!r} {what} mismatch: tag says {actual!r}, "
                f"release says {expected!r}"
            )
    state.append_progress(
        f"unsigned tag {tag} bound to commit {resolved_commit} and version {target}"
    )


def _verify_release_origin_preflight(
    *,
    directory: Path,
    current_version: str | None,
    record_phase: bool = True,
) -> str:
    """Fail before fetch when policy, replace refs, or checkout origin are unsafe."""
    if record_phase:
        state.set_phase("verifying_origin")
    if (
        versioning.release_signature_policy()
        == versioning.RELEASE_SIGNATURE_POLICY_ALLOW_UNSIGNED
    ):
        # Signature-independent subset: replace refs, readable origin, and the
        # origin allowlist when configured. The allowed_signers / trusted
        # ssh-keygen requirements only serve signature verification.
        try:
            policy = provenance.ProvenancePolicy.from_environ(
                dict(os.environ),
                source_kind="git",
                current_version=current_version,
            )
            verifier = provenance.GitSignedTagVerifier(
                allowed_signers_file=None,
                directory=str(directory),
            )
            verifier.assert_no_replace_refs()
            origin = verifier.read_origin()
            if (
                policy.allowed_origins
                and provenance.normalize_origin(origin) not in policy.allowed_origins
            ):
                raise provenance.ProvenanceError(
                    f"release origin {origin!r} is not in the configured origin allowlist"
                )
        except provenance.ProvenanceError as exc:
            raise HermesUpgradeError(f"F4 origin preflight failed: {exc}") from exc
        return origin
    try:
        return provenance.verify_editable_preflight(
            directory=str(directory),
            current_version=current_version,
            environ=dict(os.environ),
        )
    except provenance.ProvenanceError as exc:
        raise HermesUpgradeError(f"F4 origin preflight failed: {exc}") from exc


def _resolve_update_target(
    current_version: str | None,
    *,
    force_refresh: bool = False,
) -> versioning.CachedLatest | None:
    """Release the primary Update control installs, per the signature policy."""
    return versioning.latest_offerable_version(
        current_version if isinstance(current_version, str) else None,
        force_refresh=force_refresh,
    )


def _no_newer_release_error() -> HermesUpgradeError:
    if (
        versioning.release_signature_policy()
        == versioning.RELEASE_SIGNATURE_POLICY_REQUIRE_SIGNED
    ):
        return HermesUpgradeError("No newer signed Hermes release is available")
    return HermesUpgradeError("No newer Hermes release is available")


def _require_canonical_production_origin(origin: str) -> None:
    if origin != CANONICAL_HERMES_SOURCE_ORIGIN:
        raise HermesUpgradeError(
            "production Hermes source origin must use exact canonical bytes "
            f"{CANONICAL_HERMES_SOURCE_ORIGIN!r}; got {origin!r}"
        )


def run_update_job(
    *,
    target_version: str | None,
    plugin_src: Path = DEFAULT_PLUGIN_SRC,
    venv_python: Path = DEFAULT_VENV_PYTHON,
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    vetted_origin: str | None = None,
    _job_lock: _WholeJobLock | None = None,
) -> dict[str, str | None]:
    """Run one whole update while holding the cross-process OS lock."""
    job_lock = _job_lock or _WholeJobLock.acquire(_update_lock_path())
    try:
        return _run_update_job_locked(
            target_version=target_version,
            plugin_src=plugin_src,
            venv_python=venv_python,
            runner=runner,
            vetted_origin=vetted_origin,
        )
    finally:
        job_lock.release()


def _run_update_job_locked(
    *,
    target_version: str | None,
    plugin_src: Path = DEFAULT_PLUGIN_SRC,
    venv_python: Path = DEFAULT_VENV_PYTHON,
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    vetted_origin: str | None = None,
) -> dict[str, str | None]:
    """Run an upgrade synchronously. Used by tests and by the async
    spawn path in :func:`trigger_update`.

    Returns ``{"to_version": "0.14.0"}`` on success. Raises
    :class:`HermesUpgradeError` on failure (and finalizes the
    persisted job snapshot with ``status="failed"``).
    """
    try:
        state.set_phase("preflight")
        if target_version is not None and (
            not versioning.is_safe_target_version(target_version)
            or target_version.startswith(("-", "."))
        ):
            raise HermesUpgradeError(f"refused unsafe target_version: {target_version!r}")
        mode = versioning.install_mode()
        if mode.get("mode") == "missing":
            raise HermesUpgradeError("Hermes is not installed in the contained venv")
        # Mutating path only: Git Bash before origin/disk/release/fetch.
        # Local install_mode already ran so a missing install can refuse
        # without launching the host binary.
        _require_valid_git_bash_candidate()
        editable_directory: Path | None = None
        fetch_origin: str | None = None
        if mode.get("mode") == "editable":
            directory = mode.get("directory") or versioning.hermes_dir()
            editable_directory = Path(directory)
            _assert_editable_git_dir(editable_directory)
            # Fail in milliseconds on a starved volume rather than discovering
            # it 600 s later as an opaque fetch timeout (job 1c178865).
            _preflight_disk_space(editable_directory)
            current_origin = _verify_release_origin_preflight(
                directory=editable_directory,
                current_version=mode.get("version"),
            )
            if vetted_origin is None:
                _require_canonical_production_origin(current_origin)
                fetch_origin = CANONICAL_HERMES_SOURCE_ORIGIN
            elif current_origin != vetted_origin:
                raise HermesUpgradeError(
                    "Hermes checkout origin changed after the exact vetted "
                    "preflight; refusing update"
                )
            else:
                fetch_origin = vetted_origin

        state.set_phase("fetching_release")
        if target_version is None:
            latest = _resolve_update_target(mode.get("version"), force_refresh=True)
        else:
            latest = versioning.latest_version(force_refresh=True)
            # Refresh the recent releases too so resolve_git_tag can pair the
            # operator-picked wheel version to its actual git tag.
            versioning.recent_releases(force_refresh=True)
        if not latest and not target_version:
            raise _no_newer_release_error()
        target = target_version or (latest.version if latest else "")
        if (
            not target
            or not versioning.is_safe_target_version(target)
            or target.startswith(("-", "."))
        ):
            raise HermesUpgradeError(f"resolved target_version is unsafe or empty: {target!r}")

        if mode.get("mode") == "pypi" and target == mode.get("version"):
            state.set_phase(
                "validating",
                note=f"Hermes {target} is already installed; no changes required",
            )
            installed_version = _validate(
                venv_python,
                expected_version=target,
            )
            state.finalize_job(to_version=installed_version)
            return {"to_version": installed_version}

        if mode["mode"] == "editable":
            tag = versioning.resolve_git_tag(target) or (target if target.startswith("v") else f"v{target}")
            _require_safe_git_tag(tag)
            assert editable_directory is not None
            assert fetch_origin is not None
            source_directory = editable_directory
            migration_source: Path | None = None
            managed_directory = Path(_default_managed_checkout(MS4_ROOT))
            source_is_managed = _is_exact_managed_checkout(source_directory)
            if not source_is_managed and _same_path(
                source_directory,
                managed_directory,
            ):
                raise HermesUpgradeError(
                    "Hermes checkout is a real-path alias of the managed target; "
                    "refusing mutation unless the path is lexically exact"
                )
            try:
                if source_is_managed:
                    managed_origin = _unambiguous_local_origin(source_directory)
                    if managed_origin != fetch_origin:
                        raise HermesUpgradeError(
                            "managed Hermes raw local origin does not exactly match "
                            "the vetted fetch origin"
                        )
                    _ensure_clean_checkout(source_directory)
                else:
                    _ensure_pristine_external_checkout(source_directory)
            except HermesDirtyCheckoutError:
                if source_is_managed:
                    raise
                _prepare_managed_checkout(managed_directory, fetch_origin)
                editable_directory = managed_directory
                migration_source = source_directory
            # Re-check in the exact repository context consumed by the explicit
            # URL fetch. ``ls-remote --get-url`` resolves config without network.
            _require_unrewritten_git_url(editable_directory, fetch_origin)
            # Pre-activation gates + rollback pre-state capture. Migration may
            # have initialized the contained target, but the operator checkout
            # and active install are still untouched here.
            pre_state = (
                _capture_editable_prestate(editable_directory)
                if migration_source is None
                else None
            )
            marker_prestate = (
                _capture_active_marker_prestate()
                if migration_source is None
                and _is_exact_managed_checkout(editable_directory)
                else None
            )
            # Size the free-space guard for the fetch that is about to run. On
            # the F2 migration path the target is a brand-new EMPTY checkout,
            # so its own pack bytes are 0 and the whole history is about to
            # arrive; take the scale from the operator source it replaces.
            _preflight_disk_space(
                editable_directory,
                reference_pack_bytes=_pack_dir_usage(source_directory)[0],
            )
            expected_commit = _git_fetch_resolve(
                editable_directory,
                tag,
                remote_url=fetch_origin,
            )
            # F4 supply-chain provenance gate. The fetch above only added a tag
            # ref + objects to .git (no working tree/HEAD/branch change), so the
            # signed tag object is now present to verify. Verify signer/key,
            # bound identity/version/platform/arch/origin/commit, and freshness
            # BEFORE the first working-tree mutation below. A refusal here has
            # nothing to roll back.
            _verify_release_provenance(
                mode="editable",
                target=target,
                tag=tag,
                directory=editable_directory,
                expected_commit=expected_commit,
                current_version=mode.get("version"),
            )
            if target == mode.get("version") and migration_source is None:
                current_head = _git_head(editable_directory)
                if current_head == expected_commit:
                    try:
                        if _is_exact_managed_checkout(editable_directory):
                            installed_version = _validate_managed_install(
                                venv_python,
                                expected_version=target,
                                editable_directory=editable_directory,
                                expected_commit=expected_commit,
                            )
                        else:
                            installed_version = _validate(
                                venv_python,
                                expected_version=target,
                                editable_directory=editable_directory,
                                expected_commit=expected_commit,
                                mismatch_error=HermesInstallMismatchError,
                            )
                    except HermesInstallMismatchError as mismatch:
                        state.append_progress(
                            "Same-version editable validation did not match the "
                            f"vetted release; continuing safe correction: {mismatch}"
                        )
                    else:
                        if _is_exact_managed_checkout(editable_directory):
                            _publish_active_checkout(
                                editable_directory,
                                version=installed_version,
                                commit=expected_commit,
                            )
                        state.finalize_job(to_version=installed_version)
                        return {"to_version": installed_version}
            # First mutation begins at the detached checkout. From here on any
            # failure (checkout, plugin stamp, pip, validation) triggers a full
            # restore of the captured pre-state (F3).
            install_attempted = False
            try:
                _git_detach_checkout(editable_directory, expected_commit)
                _ensure_clean_checkout(editable_directory)
                _restamp_managed_plugin(plugin_src, editable_directory)
                install_attempted = True
                _pip_install_editable(venv_python, editable_directory)
                if _is_exact_managed_checkout(editable_directory):
                    installed_version = _validate_managed_install(
                        venv_python,
                        expected_version=target,
                        editable_directory=editable_directory,
                        expected_commit=expected_commit,
                    )
                    _publish_active_checkout(
                        editable_directory,
                        version=installed_version,
                        commit=expected_commit,
                    )
                else:
                    installed_version = _validate(
                        venv_python,
                        expected_version=target,
                        editable_directory=editable_directory,
                        expected_commit=expected_commit,
                    )
                state.finalize_job(to_version=installed_version)
                return {"to_version": installed_version}
            except Exception as failure:
                primary = _as_upgrade_error(
                    failure,
                    "editable update failed after mutation began",
                )
                if migration_source is not None:
                    _rollback_managed_migration(
                        migration_source,
                        primary,
                        install_attempted=install_attempted,
                    )
                assert pre_state is not None
                _rollback_editable(
                    pre_state,
                    plugin_src,
                    venv_python,
                    primary,
                    marker_prestate,
                )
                raise  # unreachable: rollback helpers always raise
        elif mode["mode"] == "pypi":
            # F4: pypi wheel provenance is a fail-closed integration dependency
            # (PEP 740 / Sigstore attestation). Verification must precede the
            # install; absent a configured attestation verifier this refuses
            # rather than trusting transport/hash only.
            _verify_release_provenance(
                mode="pypi",
                target=target,
                tag=None,
                directory=None,
                expected_commit=None,
                current_version=mode.get("version"),
            )
            _pip_install_wheel(venv_python, target)
            installed_version = _validate(venv_python, expected_version=target)
        else:
            raise HermesUpgradeError(f"Unknown Hermes install mode: {mode['mode']!r}")

        state.finalize_job(to_version=installed_version)
        return {"to_version": installed_version}
    except HermesUpgradeError as exc:
        try:
            state.finalize_job(error=str(exc))
        except Exception as persistence_error:
            raise HermesUpgradeError(
                f"{exc}; additionally could not persist terminal failure state: "
                f"{persistence_error}"
            ) from persistence_error
        raise
    except Exception as failure:
        primary = _as_upgrade_error(failure, "unexpected Hermes update failure")
        try:
            state.finalize_job(error=str(primary))
        except Exception as persistence_error:
            raise HermesUpgradeError(
                f"{primary}; additionally could not persist terminal failure state: "
                f"{persistence_error}"
            ) from persistence_error
        raise primary from failure


def trigger_update(
    *,
    target_version: str | None = None,
    request_user: str | None = None,
    plugin_src: Path = DEFAULT_PLUGIN_SRC,
    venv_python: Path = DEFAULT_VENV_PYTHON,
) -> state.UpdateJobSnapshot:
    """Spawn a background upgrade. Returns the running job snapshot
    immediately. Idempotent — refuses to start a second job while one
    is already running.
    """
    state.initialize_state()
    local = _local_update_snapshot()
    if local is not None:
        return local
    if target_version is not None and (
        not versioning.is_safe_target_version(target_version)
        or target_version.startswith(("-", "."))
    ):
        raise HermesUpgradeError(f"refused unsafe target_version: {target_version!r}")

    try:
        job_lock = _WholeJobLock.acquire(_update_lock_path())
    except HermesUpdateLockedError:
        state.last_update(refresh=True)
        raise
    try:
        # Canonical order (permanent):
        # 1. Local classification only: last_update + install_mode, and the
        #    already-persisted/cached latest release if present (TTL does
        #    not matter). A current/newer, retry-current, or explicit target
        #    equal to the safely parsed installed version is a durable no-op
        #    here with zero Git-Bash, HTTP, checkout, subprocess, or
        #    update-job work. Missing local latest is unknown: fail closed
        #    into the mutating path unless the explicit target equals
        #    installed.
        # 2. A genuinely mutating update then seals/hashes/smokes Git Bash
        #    before the first candidate-dependent effect (origin, disk,
        #    release discovery/network, job start, fetch).
        existing = state.last_update(refresh=True)
        retrying_terminal_failure = (
            existing is not None and existing.status == "failed"
        )
        if existing is not None and existing.status == "running":
            state.fail_interrupted_job()
        mode = versioning.install_mode()
        current_version = mode.get("version")
        if (
            target_version is None
            and retrying_terminal_failure
            and existing is not None
            and existing.install_mode == "editable"
            and mode.get("mode") == "editable"
            and isinstance(current_version, str)
            and versioning.is_safe_target_version(current_version)
            and not current_version.startswith(("-", "."))
            and existing.to_version == current_version
        ):
            snap = state.publish_current_noop_success(
                expected_job_id=existing.job_id,
                current_version=current_version,
                request_user=request_user,
            )
            job_lock.release()
            return snap
        current_parsed = versioning.parse_semver(
            current_version if isinstance(current_version, str) else None
        )
        if target_version is not None:
            target_parsed = versioning.parse_semver(target_version)
            if (
                current_parsed is not None
                and target_parsed is not None
                and target_parsed < current_parsed
            ):
                raise HermesUpgradeError(
                    f"refusing downgrade from installed {current_version} to {target_version}"
                )
            if (
                current_parsed is not None
                and target_parsed is not None
                and target_parsed == current_parsed
                and isinstance(current_version, str)
                and versioning.is_safe_target_version(current_version)
                and not current_version.startswith(("-", "."))
            ):
                # Pin dialog posts latest-signed == installed. That is a
                # reinstall of the already-current version, not an upgrade.
                existing_now = state.last_update(refresh=True)
                if existing_now is not None and existing_now.status == "failed":
                    snap = state.reconcile_stale_failure_for_current_or_newer(
                        current_version=current_version,
                        latest_version=current_version,
                        relation="current",
                        request_user=request_user,
                    )
                    job_lock.release()
                    return snap or existing_now
                if existing_now is None:
                    snap = state.start_job(
                        from_version=current_version,
                        to_version=current_version,
                        install_mode=mode.get("mode"),
                        request_user=request_user,
                    )
                    state.append_progress(
                        f"Verified current no-op: installed {current_version} "
                        "equals the requested pin; skipped Git/install"
                    )
                    state.finalize_job(to_version=current_version)
                    snap = state.last_update(refresh=True) or snap
                    job_lock.release()
                    return snap
                job_lock.release()
                if existing_now is not None:
                    return existing_now
        if (
            target_version is None
            and isinstance(current_version, str)
            and versioning.is_safe_target_version(current_version)
            and not current_version.startswith(("-", "."))
        ):
            discovered = versioning.cached_latest_version()
            latest_str = discovered.version if discovered else None
            relation = versioning.installed_relation(current_version, latest_str)
            if relation in {"current", "newer"} and isinstance(latest_str, str):
                existing_now = state.last_update(refresh=True)
                if existing_now is not None and existing_now.status == "failed":
                    snap = state.reconcile_stale_failure_for_current_or_newer(
                        current_version=current_version,
                        latest_version=latest_str,
                        relation=relation,
                        request_user=request_user,
                    )
                    job_lock.release()
                    return snap or existing_now
                if existing_now is not None and existing_now.status == "success":
                    job_lock.release()
                    return existing_now
        _require_valid_git_bash_candidate()
        vetted_origin: str | None = None
        if mode.get("mode") == "editable":
            directory = mode.get("directory") or versioning.hermes_dir()
            editable_directory = Path(directory)
            _assert_editable_git_dir(editable_directory)
            # Refuse synchronously so the operator sees the disk problem in the
            # trigger response, instead of a background job that starts, looks
            # healthy, and dies later. record=False: no job snapshot owns this
            # progress yet (the previous job's would be wrong to append to).
            _preflight_disk_space(editable_directory, record=False)
            vetted_origin = _verify_release_origin_preflight(
                directory=editable_directory,
                current_version=mode.get("version"),
                record_phase=False,
            )
            _require_canonical_production_origin(vetted_origin)
        from_version = mode.get("version")
        resolved_target = target_version
        if resolved_target is None:
            latest = _resolve_update_target(from_version)
            if latest is None:
                raise _no_newer_release_error()
            resolved_target = latest.version
        snap = state.start_job(
            from_version=from_version,
            to_version=resolved_target,
            install_mode=mode.get("mode"),
            request_user=request_user,
        )
        _set_local_update_job(snap.job_id)
    except Exception as exc:
        job_lock.release()
        if isinstance(exc, HermesUpgradeError):
            raise
        raise HermesUpgradeError(
            f"could not prepare Hermes update job: {exc}"
        ) from exc

    def worker() -> None:
        try:
            run_update_job(
                target_version=resolved_target,
                plugin_src=plugin_src,
                venv_python=venv_python,
                vetted_origin=vetted_origin,
                _job_lock=job_lock,
            )
        except HermesUpgradeError:
            pass
        finally:
            _clear_local_update_job(snap.job_id)

    try:
        thread = threading.Thread(
            target=worker,
            daemon=True,
            name=f"hermes-upgrade-{snap.job_id[:8]}",
        )
        thread.start()
    except Exception as exc:
        try:
            state.finalize_job(error=f"could not start Hermes update worker: {exc}")
        finally:
            _clear_local_update_job(snap.job_id)
            job_lock.release()
        raise HermesUpgradeError(f"could not start Hermes update worker: {exc}") from exc
    return snap
