from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat as stat_module
import struct
import subprocess
import sys
import tempfile
import time
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).with_name("oracle_browser_runtime.mjs")
INDEX_HTML = ROOT / "machine_spirit_4" / "web" / "index.html"
IS_WINDOWS = os.name == "nt"

# R6: load the Win32 DLLs ONCE at module scope. Reloading them per spawn churned
# the process handle count and made the handle-leak negative flaky; a single
# cached handle makes the leak test measure only real Job/process handle leaks.
if IS_WINDOWS:
    from ctypes import wintypes as _wt

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _NTDLL = ctypes.WinDLL("ntdll", use_last_error=True)
    _KERNEL32.CreateJobObjectW.restype = ctypes.c_void_p
    _KERNEL32.QueryInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]

    # R7: handle-relative ownership. No-follow directory/file handles, file-ID +
    # link-count + reparse detection via the HANDLE, ADS enumeration, and
    # handle-relative non-replacing rename (SetFileInformationByHandle).
    _INVALID_HANDLE = ctypes.c_void_p(-1).value
    _GENERIC_READ = 0x80000000
    _DELETE = 0x00010000
    _FILE_SHARE_READ = 0x1
    _FILE_SHARE_WRITE = 0x2
    # NOTE: FILE_SHARE_DELETE (0x4) is intentionally OMITTED: while we hold the
    # handle, no other actor can rename or delete the pinned object (closes ABA).
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000       # required to open a directory handle
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000     # no-follow
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    _FileRenameInfo = 3

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", _wt.DWORD),
            ("ftCreationTime", _wt.FILETIME),
            ("ftLastAccessTime", _wt.FILETIME),
            ("ftLastWriteTime", _wt.FILETIME),
            ("dwVolumeSerialNumber", _wt.DWORD),
            ("nFileSizeHigh", _wt.DWORD),
            ("nFileSizeLow", _wt.DWORD),
            ("nNumberOfLinks", _wt.DWORD),
            ("nFileIndexHigh", _wt.DWORD),
            ("nFileIndexLow", _wt.DWORD),
        ]

    _KERNEL32.CreateFileW.restype = ctypes.c_void_p
    _KERNEL32.CreateFileW.argtypes = [
        _wt.LPCWSTR, _wt.DWORD, _wt.DWORD, ctypes.c_void_p,
        _wt.DWORD, _wt.DWORD, ctypes.c_void_p]
    _KERNEL32.GetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _KERNEL32.SetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, _wt.DWORD]
    _KERNEL32.FindFirstStreamW.restype = ctypes.c_void_p
    _KERNEL32.FindFirstStreamW.argtypes = [_wt.LPCWSTR, ctypes.c_int, ctypes.c_void_p, _wt.DWORD]
    _KERNEL32.FindNextStreamW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _KERNEL32.FindClose.argtypes = [ctypes.c_void_p]
    _KERNEL32.CloseHandle.argtypes = [ctypes.c_void_p]


# ---------------------------------------------------------------------------
# Finding 10: a bounded process-tree cleanup boundary. On Windows the Node
# process (and therefore its Chrome descendant, spawned after assignment) is
# placed in a kill-on-close Job Object; on timeout we TerminateJobObject and the
# whole tree dies. Elsewhere Node runs in its own process group and a timeout
# SIGKILLs the group. Either way no descendant can outlive the run.
# ---------------------------------------------------------------------------

PROFILE_PREFIX = "ms4-oracle-browser-"
STAGING_PREFIX = ".staging-"


def _dir_identity(path):
    """R6: no-follow device+inode identity. On Windows Python's os.lstat populates
    st_dev/st_ino from the volume serial + 64-bit file id (GetFileInformationByHandle)
    and st_reparse_tag for reparse points; on POSIX it is the device/inode. Returns
    None if the path is absent."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return {
        "dev": st.st_dev,
        "ino": st.st_ino,
        "is_symlink": stat_module.S_ISLNK(st.st_mode),
        "is_reparse": bool(getattr(st, "st_reparse_tag", 0)) if IS_WINDOWS else False,
    }


def _components_contained(child, parent):
    """R6: component-aware containment (NOT string prefixing). `C:\\Temp-sibling\\x`
    is NOT contained under `C:\\Temp` even though the string prefix matches.
    Case-insensitive on Windows; requires a strict subdirectory."""
    def norm(p):
        parts = Path(os.path.abspath(p)).parts
        return tuple(c.lower() for c in parts) if IS_WINDOWS else tuple(parts)
    c = norm(child)
    p = norm(parent)
    return len(c) > len(p) and c[:len(p)] == p


class _RunResult:
    def __init__(self, returncode, stdout, stderr, timed_out):
        # stdout/stderr are RAW BYTES so a byte-for-byte comparison against the
        # UTF-8 persisted JSON is exact (locale text decoding would corrupt the
        # non-ASCII em-dash/ellipsis/emoji in the payload).
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stdout_text = (stdout or b"").decode("utf-8", "replace")
        self.stderr_text = (stderr or b"").decode("utf-8", "replace")
        self.timed_out = timed_out
        # R6 process-primitive: proof the owned tree reached zero active processes.
        self.tree_death = None
        # R6 transaction (populated by _run_browser_transaction):
        self.profile_dir = None
        self.profile_residue = None
        self.staging_dir = None
        self.publish_dir = None          # set ONLY when the wrapper committed evidence
        self.committed = False
        self.cleanup_attestation = None
        self.cleanup_error = None        # cleanup/transaction failure detail (retained)
        self.primary_error = None        # harness/process error detail (retained)

    @property
    def profile_residue_removed(self):
        return bool(self.profile_residue and self.profile_residue.get("removed"))


def _win_pinned_rmtree(p, captured_identity):
    """R8 P1-02: delete a directory HANDLE-RELATIVELY. Open it no-follow with
    FILE_SHARE_READ only (deny write/delete/rename share) so a concurrent
    rename-aside swap is refused with a sharing violation WHILE we hold it; verify
    the file index via the handle equals the captured identity (a swap that beat us
    to the pin is caught); then delete the (pinned) contents and mark the directory
    delete-on-close. Returns (ok, error)."""
    h = _KERNEL32.CreateFileW(
        str(p), _GENERIC_READ | _DELETE, _FILE_SHARE_READ, None, _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT, None)
    if not h or h == _INVALID_HANDLE:
        return False, f"could not pin dir for delete (WinError {ctypes.get_last_error()})"
    try:
        info = _BY_HANDLE_FILE_INFORMATION()
        if not _KERNEL32.GetFileInformationByHandle(ctypes.c_void_p(h), ctypes.byref(info)):
            return False, "GetFileInformationByHandle failed on delete pin"
        if info.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            return False, "refusing: pinned dir is a reparse point"
        hino = (info.nFileIndexHigh << 32) | info.nFileIndexLow
        if captured_identity and hino != int(captured_identity.get("ino", -1)):
            return False, "refusing: pinned dir file index != captured identity (swap)"
        for child in list(os.scandir(p)):  # read share permits our own enumeration
            if child.is_dir(follow_symlinks=False):
                shutil.rmtree(child.path, ignore_errors=False)
            else:
                os.remove(child.path)
        # FILE_DISPOSITION_INFO (class 4): delete-on-close of the pinned dir.
        cbuf = ctypes.create_string_buffer(struct.pack("<I", 1), 4)
        if not _KERNEL32.SetFileInformationByHandle(ctypes.c_void_p(h), 4, cbuf, 4):
            return False, f"delete-on-close failed (WinError {ctypes.get_last_error()})"
        return True, ""
    except OSError as exc:
        return False, f"pinned delete error: {exc}"
    finally:
        _KERNEL32.CloseHandle(ctypes.c_void_p(h))  # close triggers the delete-on-close


def _remove_owned_dir(path, *, expected_parent, prefix, captured_identity):
    """R8 (RevB P1-02, RevA P1-4): HANDLE-RELATIVE identity-bound removal. Refuses
    anything not component-contained under ``expected_parent``, wrong-prefixed, a
    symlink/reparse point, whose identity has drifted, or whose identity cannot be
    read (non-null required). On Windows the delete is done through a pinned handle
    that a concurrent swap cannot beat; POSIX uses the identity-rechecked path.
    Uses no-follow ``lexists`` (a dangling reparse is NOT 'absent'). CHECKED
    result; caller fails closed on residue."""
    if not path:
        return {"removed": True, "dir": None, "attempts": 0, "remaining": [], "verified": True}
    p = os.path.abspath(path)
    name = os.path.basename(p)
    if not _components_contained(p, expected_parent) or not name.startswith(prefix):
        return {"removed": False, "dir": p, "attempts": 0, "remaining": [p],
                "error": "refusing: not component-contained under owned parent / wrong prefix"}
    last_err = None
    for attempt in range(1, 11):
        if not os.path.lexists(p):  # no-follow: a dangling reparse is NOT absent
            return {"removed": True, "dir": p, "attempts": attempt, "remaining": [], "verified": True}
        ident = _dir_identity(p)
        if ident is None:  # R8: a failed identity read is NOT permission to delete
            return {"removed": False, "dir": p, "attempts": attempt, "remaining": [p],
                    "error": "refusing: could not capture no-follow identity"}
        if ident["is_symlink"] or ident["is_reparse"]:
            return {"removed": False, "dir": p, "attempts": attempt, "remaining": [p],
                    "error": "refusing: symlink/reparse point"}
        if captured_identity and (ident["dev"] != captured_identity["dev"] or ident["ino"] != captured_identity["ino"]):
            return {"removed": False, "dir": p, "attempts": attempt, "remaining": [p],
                    "error": "refusing: directory identity changed since creation (replacement/foreign)"}
        if IS_WINDOWS:
            ok, err = _win_pinned_rmtree(p, captured_identity)
            if not ok:
                last_err = err
                # a refusal (swap/reparse/identity) is terminal, not a retry.
                if err and err.startswith("refusing:"):
                    return {"removed": False, "dir": p, "attempts": attempt, "remaining": [p], "error": err}
        else:
            try:
                shutil.rmtree(p)
            except OSError as exc:
                last_err = str(exc)
        if not os.path.lexists(p):
            return {"removed": True, "dir": p, "attempts": attempt, "remaining": [], "verified": True}
        time.sleep(0.3)
    remaining = [str(f) for f in Path(p).rglob("*")] if os.path.lexists(p) else []
    return {"removed": not os.path.lexists(p), "dir": p, "attempts": 10, "remaining": remaining, "error": last_err}


def _remove_owned_profile(profile_dir, captured_identity=None):
    """R6 profile teardown: identity-bound removal of an exact ms4-oracle-browser-*
    profile under the system temp root."""
    return _remove_owned_dir(profile_dir, expected_parent=tempfile.gettempdir(),
                             prefix=PROFILE_PREFIX, captured_identity=captured_identity)


def _run_node_in_tree(args, *, cwd, env, timeout):
    """R6 PROCESS PRIMITIVE (no filesystem ownership). Runs Node in a handle-bound
    Job Object (Windows) or its own process group (POSIX) with handle-complete
    teardown and a proven tree-death (active-process-count zero) result. The
    filesystem transaction (profile/staging/publish) is owned by
    ``_run_browser_transaction`` / the direct entrypoint, never here."""
    if IS_WINDOWS:
        return _run_node_windows_job(args, cwd=cwd, env=env, timeout=timeout)
    return _run_node_posix_group(args, cwd=cwd, env=env, timeout=timeout)


def _run_node_windows_job(args, *, cwd, env, timeout):
    kernel32 = _KERNEL32

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_void_p),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", ctypes.c_uint32),
            ("TotalProcesses", ctypes.c_uint32),
            ("ActiveProcesses", ctypes.c_uint32),
            ("TotalTerminatedProcesses", ctypes.c_uint32),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9
    JobObjectBasicAccountingInformation = 1
    CREATE_SUSPENDED = 0x00000004
    ntdll = _NTDLL

    def _active_count(job):
        info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        ok = kernel32.QueryInformationJobObject(
            ctypes.c_void_p(job), JobObjectBasicAccountingInformation,
            ctypes.byref(info), ctypes.sizeof(info), None)
        return int(info.ActiveProcesses) if ok else None

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    # R6: enter try/finally IMMEDIATELY after CreateJobObject so the ONLY job
    # handle is ALWAYS closed (kill-on-close reaps any assigned tree) on EVERY
    # branch: Set/Popen/Assign/Resume failure, timeout, cancellation, assertion.
    proc = None
    timed_out = False
    tree_death = {"waited": False, "active_processes": None, "zero": False}
    try:
        if env.get("MS4_TEST_FORCE_CANCEL") == "1":
            raise KeyboardInterrupt("test hook: external cancellation before spawn")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            ctypes.c_void_p(job), JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if env.get("MS4_TEST_FORCE_JOB_SETUP_FAIL") == "1":
            raise OSError("test hook: forced Job configuration failure")
        # B-03: create Node SUSPENDED, assign it while it cannot yet spawn Chrome,
        # FAIL CLOSED on any Job API error, THEN resume (closes the pre-assignment
        # race). The retained Popen handle is our process identity — we NEVER
        # reacquire the tree by PID.
        proc = subprocess.Popen(
            args, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=CREATE_SUSPENDED,
        )
        # A3: the fault hooks SKIP the real call so the child is genuinely
        # unassigned / unresumed, exercising the actual failure-state cleanup
        # (the suspended child must still be reaped) rather than post-success.
        if env.get("MS4_TEST_FORCE_ASSIGN_FAIL") == "1":
            assigned = 0
        else:
            assigned = kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(int(proc._handle)))
        if not assigned:
            raise OSError(f"AssignProcessToJobObject failed (fail-closed): WinError {ctypes.get_last_error()}")
        if env.get("MS4_TEST_FORCE_RESUME_FAIL") == "1":
            status = 0xC0000001
        else:
            status = ntdll.NtResumeProcess(ctypes.c_void_p(int(proc._handle)))
        if status != 0:
            raise OSError(f"NtResumeProcess failed (fail-closed): NTSTATUS 0x{status & 0xFFFFFFFF:08X}")
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kernel32.TerminateJobObject(ctypes.c_void_p(job), 1)
            try:
                stdout, stderr = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
        # R6 tree-death proof: terminate any straggler descendants still in the
        # job, then WAIT (bounded) for the Job's ACTIVE process count to reach
        # zero. Node exiting is not proof the renderer/GPU/utility children died;
        # the Job accounting is (handle-bound, immune to PID reuse).
        kernel32.TerminateJobObject(ctypes.c_void_p(job), 1)
        deadline = time.monotonic() + 15
        ac = _active_count(job)
        tree_death["waited"] = True
        while ac not in (0, None) and time.monotonic() < deadline:
            time.sleep(0.1)
            ac = _active_count(job)
        tree_death["active_processes"] = ac
        tree_death["zero"] = (ac == 0)
        if env.get("MS4_TEST_FORCE_TREE_DEATH_UNKNOWN") == "1":
            # Simulate a failed QueryInformationJobObject: death is UNKNOWN, not zero.
            tree_death = {"waited": True, "active_processes": None, "zero": False}
        result = _RunResult(proc.returncode, stdout, stderr, timed_out)
        result.tree_death = tree_death
        return result
    except BaseException:
        # R6: reap any created process on ANY setup/spawn/assign/resume error or
        # external cancellation/KeyboardInterrupt; the finally then closes the job
        # (kill-on-close reaps the tree). No handle is leaked on any branch — the
        # retained process handle is explicitly closed too (a suspended, never-
        # resumed child would otherwise keep its handle until GC).
        if proc is not None:
            try:
                kernel32.TerminateJobObject(ctypes.c_void_p(job), 1)
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.communicate(timeout=10)
            except Exception:
                pass
            try:
                proc._handle.Close()  # release the retained process handle deterministically
            except Exception:
                pass
        raise
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(job))


def _run_node_posix_group(args, *, cwd, env, timeout):
    proc = subprocess.Popen(
        args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    timed_out = False
    tree_death = {"waited": False, "active_processes": None, "zero": False}
    pgid_error = None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError as exc:
        # R8 P1-04: a failed getpgid is NOT "empty" — it is UNKNOWN. Only an
        # authoritative killpg(pgid, 0) -> ESRCH proves the group is empty.
        pgid = None
        pgid_error = exc

    def _group_state():
        """R8 P1-04: 'empty' only on authoritative ESRCH; 'alive' on success;
        'unknown' on EPERM / any query failure / unknown pgid (never 'empty')."""
        if pgid is None:
            return "unknown"
        try:
            os.killpg(pgid, 0)
            return "alive"
        except ProcessLookupError:
            return "empty"
        except OSError:
            return "unknown"  # EPERM / EINVAL / etc. — group state cannot be proven

    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            stdout, stderr = proc.communicate()
        # R6 tree-death proof: SIGKILL the whole group, then wait (bounded) for
        # the group to be empty (killpg(pgid, 0) raises when no member remains).
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        deadline = time.monotonic() + 15
        tree_death["waited"] = True
        while time.monotonic() < deadline and _group_state() == "alive":
            time.sleep(0.1)
        state = _group_state()
        tree_death["state"] = state
        tree_death["pgid_error"] = None if pgid_error is None else f"{type(pgid_error).__name__}: {pgid_error}"
        tree_death["zero"] = (state == "empty")  # ONLY authoritative ESRCH
        tree_death["active_processes"] = 0 if state == "empty" else None
        result = _RunResult(proc.returncode, stdout, stderr, timed_out)
        result.tree_death = tree_death
        return result
    except BaseException:
        # R6: reap the whole group on external cancellation/assertion, then re-raise.
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
        try:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=10)
        except Exception:
            pass
        raise
    finally:
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass


def _png_ok(path):
    try:
        return Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _validate_staging_graph(staging_dir, *, variant, run_id, stdout_bytes, index_sha):
    """A4: validate the COMPLETE inner hash graph inside staging BEFORE any commit.
    Requires the strict inner seal (schema/runId/variant/nonce/digests + exact
    artifact hash set), the publish-manifest bound by hash, a byte-identical
    result JSON == the wrapper-captured stdout, PNG signatures, and the
    served-bytes/source digests. Returns (ok, errors, parsed_result)."""
    errors = []
    sdir = Path(staging_dir)
    seal_p = sdir / "SEAL.json"
    if not seal_p.is_file():
        return False, ["missing inner SEAL.json"], None
    try:
        seal = json.loads(seal_p.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        return False, [f"unparseable inner SEAL.json: {exc}"], None
    if seal.get("schema") != "OracleR6InnerSeal.v1" or seal.get("sealed") is not True:
        errors.append("inner seal schema/sealed invalid")
    if seal.get("runId") != run_id or seal.get("variant") != variant:
        errors.append("inner seal runId/variant mismatch")
    if not seal.get("fixtureNonce"):
        errors.append("inner seal missing fixtureNonce")
    for key in (
        "servedHtmlSha256",
        "harnessSha256",
        "voiceInputSessionSha256",
        "proofAudioSha256",
        "nodeExecSha256",
        "chromeExecSha256",
    ):
        if not seal.get(key):
            errors.append(f"inner seal missing digest {key}")
    artifacts = seal.get("artifacts") or {}
    if not artifacts:
        errors.append("inner seal has an empty artifact set")
    result_name = f"oracle_browser_runtime__{variant}.json"
    for req in (result_name, f"oracle_final_state__{variant}.png"):
        if req not in artifacts:
            errors.append(f"inner seal missing required artifact {req}")
    for name, meta in artifacts.items():
        f = sdir / name
        if not f.is_file():
            errors.append(f"sealed artifact missing on disk: {name}")
            continue
        raw = f.read_bytes()
        if _sha256(raw) != meta.get("sha256") or len(raw) != meta.get("bytes"):
            errors.append(f"sealed artifact hash/size mismatch: {name}")
    pm_p = sdir / "publish_manifest.json"
    if not pm_p.is_file():
        errors.append("missing publish_manifest.json")
    else:
        pm_raw = pm_p.read_bytes()
        if _sha256(pm_raw) != seal.get("publishManifestSha256"):
            errors.append("publish_manifest hash not bound by inner seal")
        try:
            pm = json.loads(pm_raw.decode("utf-8"))
            if pm.get("runId") != run_id or pm.get("variant") != variant:
                errors.append("publish_manifest runId/variant mismatch")
            if (pm.get("artifacts") or {}) != artifacts:
                errors.append("publish_manifest artifact set != seal artifact set")
        except ValueError as exc:
            errors.append(f"unparseable publish_manifest: {exc}")
    for png in sdir.glob("*.png"):
        if not _png_ok(png):
            errors.append(f"invalid PNG signature: {png.name}")
    parsed = None
    result_p = sdir / result_name
    if not result_p.is_file():
        errors.append(f"missing result artifact {result_name}")
    else:
        rb = result_p.read_bytes()
        if rb != (stdout_bytes or b""):
            errors.append("staged result JSON != wrapper-captured stdout (byte-for-byte)")
        try:
            parsed = json.loads(rb.decode("utf-8"))
            ident = parsed.get("identity") or {}
            if parsed.get("ok") is not True:
                errors.append("result ok != true")
            if ident.get("runId") != run_id or ident.get("variant") != variant:
                errors.append("result identity runId/variant mismatch")
            if ident.get("acceptanceTarget") is not True:
                errors.append("result acceptanceTarget != true")
            if index_sha is not None and ident.get("servedHtmlSha256") != index_sha:
                errors.append("served-bytes digest != on-disk index.html")
            if ident.get("harnessSha256") != seal.get("harnessSha256"):
                errors.append("result harness digest != seal harness digest")
        except ValueError as exc:
            errors.append(f"unparseable result JSON: {exc}")
    return (not errors), errors, parsed


def _win_has_ads(path) -> bool:
    """R7: True if the file carries any NTFS alternate data stream beyond the
    default `::$DATA`."""
    class _W32_FIND_STREAM_DATA(ctypes.Structure):
        _fields_ = [("StreamSize", ctypes.c_longlong), ("cStreamName", ctypes.c_wchar * 296)]
    data = _W32_FIND_STREAM_DATA()
    h = _KERNEL32.FindFirstStreamW(str(path), 0, ctypes.byref(data), 0)  # FindStreamInfoStandard
    if not h or h == _INVALID_HANDLE:
        return False
    try:
        while True:
            name = data.cStreamName or ""
            if name and name != "::$DATA":
                return True
            if not _KERNEL32.FindNextStreamW(ctypes.c_void_p(h), ctypes.byref(data)):
                break
    finally:
        _KERNEL32.FindClose(ctypes.c_void_p(h))
    return False


def _artifact_is_safe(path) -> tuple[bool, str]:
    """R7 (P1-03/P2-2 A+B): reject a staged artifact that is a symlink/reparse
    point, a hardlink (link_count>1), or carries an NTFS alternate data stream.
    All checks are no-follow (lstat)."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        return False, f"unstattable ({exc})"
    if stat_module.S_ISLNK(st.st_mode):
        return False, "symlink"
    if getattr(st, "st_file_attributes", 0) & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
        return False, "reparse point"
    if getattr(st, "st_nlink", 1) > 1:
        return False, f"hardlink (link_count={st.st_nlink})"
    if IS_WINDOWS and _win_has_ads(path):
        return False, "NTFS alternate data stream (ADS)"
    return True, ""


_FILE_SHARE_DELETE = 0x4


def _win_open_no_write_share(path):
    """R8 P1-01: open a file for read while DENYING write sharing, so no concurrent
    actor can MODIFY it while we hold the handle. FILE_SHARE_DELETE is permitted so
    that our own handle-relative rename of the parent staging directory (a move,
    which needs delete/rename access on the contained objects) still succeeds; a
    write attempt (the reproduced mutation) is refused with a sharing violation.
    Returns the handle int, or None on failure."""
    h = _KERNEL32.CreateFileW(
        str(path), _GENERIC_READ, _FILE_SHARE_READ | _FILE_SHARE_DELETE,  # NO write share
        None, _OPEN_EXISTING, _FILE_FLAG_OPEN_REPARSE_POINT, None)
    if not h or h == _INVALID_HANDLE:
        return None
    return h


def _win_handle_commit(staging_dir, publish_dir, evidence_dir, staging_identity) -> tuple[bool, str]:
    """R7 P1-3: commit HANDLE-RELATIVELY. Pin staging with a no-follow,
    no-FILE_SHARE_DELETE handle (so no actor can swap/delete it while held),
    verify via the handle it is not a reparse point and its file index matches
    the captured identity, then rename it into the publish parent via
    SetFileInformationByHandle(FileRenameInfo, ReplaceIfExists=FALSE)."""
    sh = _KERNEL32.CreateFileW(
        str(staging_dir), _GENERIC_READ | _DELETE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE, None, _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT, None)
    if not sh or sh == _INVALID_HANDLE:
        return False, f"could not pin staging handle: WinError {ctypes.get_last_error()}"
    try:
        info = _BY_HANDLE_FILE_INFORMATION()
        if not _KERNEL32.GetFileInformationByHandle(ctypes.c_void_p(sh), ctypes.byref(info)):
            return False, "GetFileInformationByHandle failed on pinned staging"
        if info.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            return False, "pinned staging is a reparse point"
        handle_ino = (info.nFileIndexHigh << 32) | info.nFileIndexLow
        if staging_identity and handle_ino != int(staging_identity.get("ino", -1)):
            return False, "pinned staging file index != captured identity (swap detected)"
        # FILE_RENAME_INFO with RootDirectory=NULL + a fully-qualified FileName is
        # the robust form: it renames the object referenced by the PINNED handle
        # (no pathname re-resolution), non-replacing (ReplaceIfExists=0).
        target = os.path.abspath(str(publish_dir))
        name_u16 = target.encode("utf-16-le") + b"\x00\x00"
        # header: ReplaceIfExists(DWORD=0), pad(DWORD), RootDirectory(HANDLE=0),
        # FileNameLength(DWORD, bytes EXCLUDING the NUL) ; then FileName (UTF-16LE + NUL)
        header = struct.pack("<IIQI", 0, 0, 0, len(name_u16) - 2)
        buf = header + name_u16
        cbuf = ctypes.create_string_buffer(buf, len(buf))
        if not _KERNEL32.SetFileInformationByHandle(ctypes.c_void_p(sh), _FileRenameInfo, cbuf, len(buf)):
            return False, f"handle-relative rename failed (WinError {ctypes.get_last_error()})"
        return True, ""
    finally:
        _KERNEL32.CloseHandle(ctypes.c_void_p(sh))


def _posix_handle_commit(staging_dir, publish_dir, evidence_dir, staging_identity) -> tuple[bool, str]:
    """POSIX handle-relative commit via retained parent dir fd + O_NOFOLLOW.
    NOTE: renameat2(RENAME_NOREPLACE) is not exposed by Python's os, so a
    non-replacing guarantee here needs a ctypes renameat2 on Linux; the lexists
    precheck narrows but does not fully close the POSIX replace race (P2-1)."""
    pfd = os.open(evidence_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            sfd = os.open(staging_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            return False, f"could not open staging no-follow: {exc}"
        try:
            st = os.fstat(sfd)
            if staging_identity and st.st_ino != int(staging_identity.get("ino", -1)):
                return False, "pinned staging inode != captured identity (swap detected)"
        finally:
            os.close(sfd)
        os.rename(os.path.basename(staging_dir), os.path.basename(publish_dir),
                  src_dir_fd=pfd, dst_dir_fd=pfd)
        return True, ""
    except OSError as exc:
        return False, f"handle-relative rename failed: {exc}"
    finally:
        os.close(pfd)


def _handle_commit(staging_dir, publish_dir, *, evidence_dir, staging_identity, variant,
                   run_id, stdout_bytes, index_sha, force_publish_collision) -> tuple[bool, str]:
    """R7: the ONE atomic acceptance operation. Refuse a foreign publish target,
    reject unsafe artifacts (reparse/hardlink/ADS), RE-VALIDATE the complete graph
    immediately before the switch, then rename handle-relatively. Nothing fallible
    runs after the rename."""
    if force_publish_collision:
        try:
            os.mkdir(publish_dir)
            (Path(publish_dir) / "FOREIGN.txt").write_text("do-not-touch", encoding="utf-8")
        except OSError:
            pass
    if os.path.lexists(publish_dir):
        return False, f"publish target already exists (foreign; refusing to delete): {publish_dir}"
    # R8 P1-08: EXACT member set — reject any staged file outside the sealed set +
    # the known wrapper sidecars; require the mandatory hive-cells screenshot.
    seal_p = Path(staging_dir) / "SEAL.json"
    try:
        sealed_names = set((json.loads(seal_p.read_text("utf-8")).get("artifacts") or {}).keys())
    except (OSError, ValueError) as exc:
        return False, f"unreadable/invalid staging seal: {exc}"
    allowed = sealed_names | {"SEAL.json", "publish_manifest.json", "cleanup_attestation.json", "run_result.json"}
    members = sorted(Path(staging_dir).iterdir())
    for f in members:
        if f.name not in allowed:
            return False, f"unsealed extra staged member (refusing commit): {f.name}"
        if f.is_file():
            safe, reason = _artifact_is_safe(str(f))
            if not safe:
                return False, f"unsafe staged artifact {f.name}: {reason}"
    if variant == "canonical" and "oracle_hive_cells__canonical.png" not in sealed_names:
        return False, "canonical acceptance graph missing mandatory hive-cells screenshot"

    def _validate():
        ok, errs, _p = _validate_staging_graph(
            staging_dir, variant=variant, run_id=run_id, stdout_bytes=stdout_bytes, index_sha=index_sha)
        return ok, errs

    if not IS_WINDOWS:
        ok, errs = _validate()
        if not ok:
            return False, "staging graph changed between validation and commit (ABA): " + "; ".join(errs)
        return _posix_handle_commit(staging_dir, publish_dir, evidence_dir, staging_identity)

    def _pin_and_validate(target_dir):
        """Open no-write-share handles on every file (deny concurrent WRITES;
        FILE_SHARE_DELETE lets our own dir move proceed), then validate the graph
        while the writes are locked out. Returns (ok, errs, handles) with handles
        held OPEN for the caller to close."""
        held = []
        for f in sorted(Path(target_dir).iterdir()):
            if f.is_file():
                h = _win_open_no_write_share(str(f))
                if h is None:
                    for hh in held:
                        _KERNEL32.CloseHandle(ctypes.c_void_p(hh))
                    return False, [f"could not pin staged file no-write-share: {f.name}"], []
                held.append(h)
        vok, verrs, _p = _validate_staging_graph(
            target_dir, variant=variant, run_id=run_id, stdout_bytes=stdout_bytes, index_sha=index_sha)
        return vok, verrs, held

    # R8 P1-01: validate with WRITES locked out; then handle-relatively rename;
    # then RE-VALIDATE the PUBLISHED dir with writes locked out again and FAIL
    # CLOSED (identity-bound removal, no accepted evidence) on any post-rename
    # mismatch. A concurrent mutation is either blocked by the lock or detected
    # and rejected — it can never yield accepted evidence.
    ok, errs, pins = _pin_and_validate(staging_dir)
    try:
        if not ok:
            return False, "staging graph changed between validation and commit (ABA): " + "; ".join(errs)
    finally:
        for h in pins:
            try:
                _KERNEL32.CloseHandle(ctypes.c_void_p(h))
            except Exception:
                pass
    try:
        committed_ok, commit_err = _win_handle_commit(staging_dir, publish_dir, evidence_dir, staging_identity)
    except OSError as exc:
        committed_ok, commit_err = False, f"commit raised: {exc}"
    if not committed_ok:
        return False, commit_err
    # Post-rename verification under fresh no-write pins: the published graph MUST
    # still match the seal. Any drift => remove the published dir (identity-bound)
    # and fail closed, so a mutation between validation and rename is never accepted.
    pok, perrs, ppins = _pin_and_validate(publish_dir)
    for h in ppins:
        try:
            _KERNEL32.CloseHandle(ctypes.c_void_p(h))
        except Exception:
            pass
    if not pok:
        pub_id = _dir_identity(publish_dir)
        _remove_owned_dir(publish_dir, expected_parent=str(evidence_dir),
                          prefix=f"{variant}__", captured_identity=pub_id)
        return False, "published graph mutated after validation (rejected, removed): " + "; ".join(perrs)
    return True, ""


def _mutate_sealed_png_for_test(staging_dir, variant) -> None:
    """Test hook (P1-3 ABA): corrupt a sealed PNG AFTER graph validation and just
    BEFORE the commit re-validation, to prove the re-validation-under-pin catches
    a post-validation mutation and fails the commit closed."""
    png = Path(staging_dir) / f"oracle_final_state__{variant}.png"
    if png.is_file():
        b = bytearray(png.read_bytes())
        if b:
            b[-1] ^= 0xFF
            png.write_bytes(bytes(b))


def _run_browser_transaction(variant, *, evidence_dir, node, chrome, repo_root=None,
                             extra_env=None, timeout=120, force_commit_fail=False,
                             force_profile_delete_fail=False, force_publish_collision=False,
                             force_test_mutate_before_rename=False):
    """R6 OWNER of the profile/staging/publish transaction. The child writes ONLY
    private staging; this owner (1) proves the owned tree reached zero active
    processes, (2) deletes the identity-bound profile, (3) ATOMICALLY commits
    staging -> the accepted `<variant>__<runId>` publish dir (never deleting a
    pre-existing/foreign target), and (4) writes a post-cleanup attestation. ANY
    cleanup failure => nonzero returncode, retained primary+cleanup errors, and NO
    accepted publish directory (so no outer seal can bless it). The direct CLI and
    pytest paths both use this single state machine."""
    run_id = str(uuid.uuid4())
    temp_root = os.path.abspath(tempfile.gettempdir())
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = os.path.join(temp_root, f"{PROFILE_PREFIX}{run_id}")
    staging_dir = str(evidence_dir / f"{STAGING_PREFIX}{variant}__{run_id}")
    publish_dir = str(evidence_dir / f"{variant}__{run_id}")
    # A2: refuse (never delete) a pre-existing owned path — a collision at a fresh
    # runId path is a foreign object. Create fresh and capture NON-NULL identity.
    for pre in (profile_dir, staging_dir, publish_dir):
        if os.path.lexists(pre):
            raise RuntimeError(f"refusing: run path already exists (foreign): {pre}")
    os.mkdir(profile_dir)
    os.mkdir(staging_dir)
    profile_identity = _dir_identity(profile_dir)
    staging_identity = _dir_identity(staging_dir)
    if not profile_identity or not staging_identity:
        raise RuntimeError("refusing: could not capture non-null owned-directory identity at creation")

    env = _base_env(chrome, variant, evidence_dir)
    env["MS4_BROWSER_RUN_ID"] = run_id
    env["MS4_BROWSER_PROFILE_DIR"] = profile_dir
    env["MS4_BROWSER_STAGING_DIR"] = staging_dir
    if repo_root is not None:
        env["MS4_BROWSER_REPO_ROOT"] = str(repo_root)
    if extra_env:
        env.update(extra_env)

    primary = None
    try:
        result = _run_node_in_tree([node, str(HARNESS)], cwd=str(ROOT), env=env, timeout=timeout)
    except (KeyboardInterrupt, SystemExit):
        # R7 P1-4 + R8 P2-01: external cancellation MUST propagate with NO accepted
        # evidence. Clean the owned profile/staging by identity (never a publish
        # dir) and SURFACE any cleanup failure as a structured residue marker
        # before re-raising — the "guaranteed cleanup" claim is never silently
        # false (a locked owned profile cannot be force-deleted, but it is recorded).
        prof_res = _remove_owned_profile(profile_dir, profile_identity)
        stg_res = {"removed": True}
        if os.path.lexists(staging_dir):
            stg_res = _remove_owned_dir(staging_dir, expected_parent=str(evidence_dir),
                                        prefix=STAGING_PREFIX, captured_identity=staging_identity)
        if not prof_res.get("removed") or not stg_res.get("removed"):
            try:
                Path(evidence_dir).mkdir(parents=True, exist_ok=True)
                (Path(evidence_dir) / f".cancellation_residue__{run_id}.json").write_text(
                    json.dumps({"schema": "OracleR8CancellationResidue.v1", "runId": run_id,
                                "profile_residue": prof_res, "staging_residue": stg_res}, indent=2) + "\n",
                    encoding="utf-8")
            except OSError:
                pass
        raise
    except Exception as exc:  # spawn/setup error: preserved; NOT swallowed as success
        primary = f"{type(exc).__name__}: {exc}"
        result = _RunResult(1, b"", b"", False)
        # R8 P1-03: an exception that did NOT return an authoritative zero-active
        # query is UNKNOWN death, never fabricated as zero. Deletion is withheld
        # and no commit occurs; the belt still best-effort removes an empty profile
        # (a real leftover live tree would lock it and surface as residue).
        result.tree_death = {"zero": False, "active_processes": None, "unknown": True, "afterError": True}
    result.profile_dir = profile_dir
    result.staging_dir = staging_dir
    result.primary_error = primary

    cleanup_errors = []
    tree_death = result.tree_death or {"zero": False, "active_processes": None, "waited": False}
    # R7 P1-1/P1-4: deletion AND commit require PROVEN tree death (zero is True).
    # An UNKNOWN death (e.g. QueryInformationJobObject failed => active_processes
    # None, zero False) is NOT safe and must fail closed — never treated as zero.
    if tree_death.get("zero") is not True:
        cleanup_errors.append(f"tree death NOT proven (zero={tree_death.get('zero')}, active={tree_death.get('active_processes')})")
        result.profile_residue = {"removed": False, "dir": profile_dir,
                                  "error": "tree death unproven; profile delete withheld (death-before-delete)"}
    elif force_profile_delete_fail:
        result.profile_residue = {"removed": False, "dir": profile_dir, "attempts": 10,
                                  "remaining": [profile_dir], "error": "test hook: forced profile-delete failure"}
    else:
        result.profile_residue = _remove_owned_profile(profile_dir, profile_identity)
    if not result.profile_residue.get("removed"):
        cleanup_errors.append(f"profile not removed: {result.profile_residue.get('error') or result.profile_residue.get('remaining')}")

    # ---- A1/R7: validate the COMPLETE graph in staging, write the attestation
    # into staging, then make the identity-bound non-replacing rename the FINAL
    # op. Commit requires PROVEN tree death. ----
    committed = False
    index_sha = _sha256(INDEX_HTML.read_bytes())
    if result.returncode == 0 and not result.timed_out and tree_death.get("zero") is True and not cleanup_errors:
        # A2: verify staging identity BEFORE reading the seal / building the graph.
        cur = _dir_identity(staging_dir)
        if not cur or cur.get("is_symlink") or cur.get("is_reparse"):
            cleanup_errors.append("staging identity unreadable or reparse before commit")
        elif cur["dev"] != staging_identity["dev"] or cur["ino"] != staging_identity["ino"]:
            cleanup_errors.append("staging identity changed before commit (replacement)")
        else:
            graph_ok, graph_errors, _parsed = _validate_staging_graph(
                staging_dir, variant=variant, run_id=run_id,
                stdout_bytes=result.stdout or b"", index_sha=index_sha)
            if not graph_ok:
                cleanup_errors.append("staging graph invalid: " + "; ".join(graph_errors))

    attestation = {
        "schema": "OracleR6CleanupAttestation.v1",
        "variant": variant, "runId": run_id,
        "treeDeath": tree_death,
        "profile": {"dir": profile_dir, "identity": profile_identity,
                    "residue": result.profile_residue, "absent": not os.path.lexists(profile_dir)},
        "harness": {"returncode": result.returncode, "timedOut": result.timed_out},
        "primaryError": primary,
    }
    result.cleanup_attestation = attestation

    if result.returncode == 0 and not result.timed_out and tree_death.get("zero") is True and not cleanup_errors:
        # Write the attestation AND a TYPED run-result sidecar INTO staging (part of
        # the committed set) BEFORE the rename; both are bound by the OUTER seal
        # graph. The typed sidecar (P1-02 B) carries exact int/bool exit + stream
        # hashes so the binder never parses exit state by substring, and lives IN
        # the generation (P1-4 A) rather than as an external replacing file.
        try:
            (Path(staging_dir) / "cleanup_attestation.json").write_text(
                json.dumps(attestation, indent=2) + "\n", encoding="utf-8")
            direct_exit = {
                "schema": "OracleR7RunResult.v1",
                "runId": run_id, "variant": variant,
                # R8 P1-07: bind the ABSOLUTE evidence root as a location challenge.
                # A byte-for-byte copy of this graph to another root carries the OLD
                # evidenceRoot and is rejected by the binder/validator.
                "evidenceRoot": os.path.abspath(str(evidence_dir)),
                "publishBasename": os.path.basename(publish_dir),
                "returncode": int(result.returncode),      # exact int (0 to reach here)
                "timedOut": bool(result.timed_out),          # exact bool
                "treeDeathZero": tree_death.get("zero") is True,
                "stdoutSha256": _sha256(result.stdout or b""),
                "stderrSha256": _sha256(result.stderr or b""),
                "stdoutBytes": len(result.stdout or b""),
                "stderrBytes": len(result.stderr or b""),
            }
            (Path(staging_dir) / "run_result.json").write_text(
                json.dumps(direct_exit, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            cleanup_errors.append(f"attestation/sidecar write failed: {exc}")
        if not cleanup_errors:
            if force_commit_fail:
                cleanup_errors.append("test hook: forced final-commit failure")
            else:
                # R7 P1-3: perform the commit HANDLE-RELATIVELY. `_handle_commit`
                # opens a no-follow, no-share-delete handle on staging (pinning it
                # against swap), re-verifies identity + link-count + no reparse/ADS,
                # RE-VALIDATES the full graph under that pin immediately before the
                # switch, then renames via the handle (SetFileInformationByHandle on
                # Windows / renameat under a retained parent fd on POSIX). Any
                # mutation between validation and rename fails closed.
                if force_test_mutate_before_rename:
                    _mutate_sealed_png_for_test(staging_dir, variant)
                ok, err = _handle_commit(
                    staging_dir, publish_dir, evidence_dir=str(evidence_dir),
                    staging_identity=staging_identity, variant=variant, run_id=run_id,
                    stdout_bytes=result.stdout or b"", index_sha=index_sha,
                    force_publish_collision=force_publish_collision)
                if ok:
                    committed = True
                else:
                    cleanup_errors.append(err)

    # Not committed => discard staging (identity-bound); never touch a foreign publish.
    if not committed and os.path.lexists(staging_dir):
        result.staging_residue = _remove_owned_dir(
            staging_dir, expected_parent=str(evidence_dir), prefix=STAGING_PREFIX,
            captured_identity=staging_identity)
        if not result.staging_residue.get("removed"):
            cleanup_errors.append(f"staging not removed: {result.staging_residue.get('error')}")

    if cleanup_errors:
        result.cleanup_error = "; ".join(cleanup_errors)
        result.committed = False
        result.publish_dir = None
        if result.returncode == 0:
            result.returncode = 70  # A4: a nonzero effective exit; direct exit 0 => committed
    else:
        result.committed = committed
        result.publish_dir = publish_dir if committed else None

    # Belt (R8 P1-4, Reviewer A): profile deletion is DEATH-BEFORE-DELETE. Only
    # delete when the tree is PROVEN dead (zero is True); when death is unknown the
    # profile is LEFT and its residue stays surfaced (never a silent delete that
    # contradicts the death-before-delete attestation). Staging is not a process
    # holder, so failed-run staging is still swept.
    if tree_death.get("zero") is True and os.path.lexists(profile_dir):
        _remove_owned_profile(profile_dir, profile_identity)
    if not result.committed and os.path.lexists(staging_dir):
        _remove_owned_dir(staging_dir, expected_parent=str(evidence_dir),
                          prefix=STAGING_PREFIX, captured_identity=staging_identity)
    return result


def _surviving_browser_pids(marker: str) -> list[int]:
    """PIDs of chrome/edge processes whose command line references `marker`
    (e.g. a temp profile dir). Used to prove process-tree cleanup."""
    if IS_WINDOWS:
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe' OR Name='msedge.exe'\" | "
            "Where-Object { $_.CommandLine -like '*" + marker + "*' } | "
            "Select-Object -ExpandProperty ProcessId"
        )
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    out = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def _chrome_path() -> Path | None:
    configured = os.environ.get("MS4_CHROME_PATH")
    candidates = [
        Path(configured) if configured else None,
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/chromium"),
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    return next((path for path in candidates if path and path.is_file()), None)


def _require_browser() -> tuple[str, Path]:
    node = shutil.which("node")
    chrome = _chrome_path()
    if not node or not chrome:
        pytest.skip("Node and Chrome/Edge are required for the Oracle browser runtime test")
    return node, chrome


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _base_env(chrome: Path, variant: str, evidence_dir: Path) -> dict:
    env = os.environ.copy()
    env["MS4_CHROME_PATH"] = str(chrome)
    env["MS4_BROWSER_VARIANT"] = variant
    env["MS4_BROWSER_EVIDENCE_DIR"] = str(evidence_dir)  # finding 11: always retain artifacts
    env.pop("MS4_BROWSER_TARGET_URL", None)
    env.pop("MS4_BROWSER_TARGET_PINNED", None)
    env.pop("MS4_BROWSER_ALLOW_EXTERNAL_TARGET", None)
    env.pop("MS4_BROWSER_TEST_HANG", None)
    env.pop("MS4_BROWSER_TEST_CLEANUP_FAIL", None)
    env.pop("MS4_BROWSER_TEST_SEAL_FAIL", None)
    env.pop("MS4_BROWSER_TEST_TASKKILL_FAIL", None)
    env.pop("MS4_BROWSER_REPO_ROOT", None)
    # R6: the transaction owner sets these per run; never inherit them.
    env.pop("MS4_BROWSER_PROFILE_DIR", None)
    env.pop("MS4_BROWSER_STAGING_DIR", None)
    env.pop("MS4_BROWSER_RUN_ID", None)
    env.pop("MS4_TEST_FORCE_JOB_SETUP_FAIL", None)
    env.pop("MS4_TEST_FORCE_ASSIGN_FAIL", None)
    env.pop("MS4_TEST_FORCE_RESUME_FAIL", None)
    env.pop("MS4_TEST_FORCE_CANCEL", None)
    env.pop("MS4_TEST_FORCE_TREE_DEATH_UNKNOWN", None)
    return env


def _run_variant(variant: str, *, evidence_dir: Path, node: str, chrome: Path, repo_root: Path | None = None) -> dict:
    completed = _run_browser_transaction(
        variant, evidence_dir=Path(evidence_dir), node=node, chrome=chrome, repo_root=repo_root)
    # R6: acceptance == the WRAPPER-owned transaction committed. Cleanup failure,
    # unproven tree death, or a nonzero/timed-out harness means NO accepted run.
    assert completed.cleanup_error is None, f"transaction cleanup failed: {completed.cleanup_error}"
    assert completed.returncode == 0 and not completed.timed_out, (
        f"harness variant={variant} failed ({completed.returncode}) timed_out={completed.timed_out} "
        f"primary={completed.primary_error}\nstdout tail:\n{completed.stdout_text[-3000:]}\n"
        f"stderr tail:\n{completed.stderr_text[-3000:]}"
    )
    assert completed.committed is True and completed.publish_dir, "wrapper must have committed accepted evidence"
    assert completed.tree_death and completed.tree_death.get("zero") is True, \
        f"tree death (Job active-count zero) not proven: {completed.tree_death}"

    result = json.loads(completed.stdout_text)
    assert result["ok"] is True
    identity = result["identity"]
    assert identity["variant"] == variant
    assert identity["acceptanceTarget"] is True
    assert identity["pageUrl"] == identity["loopbackUrl"]
    assert identity.get("profileOwnedByHarness") is False and identity.get("wrapperDriven") is True, \
        "acceptance runs are wrapper-driven; the child must not self-publish/clean"

    run_id = identity["runId"]
    publish_dir = Path(completed.publish_dir)
    assert publish_dir.is_dir() and publish_dir.parent == Path(evidence_dir)
    assert publish_dir.name == f"{variant}__{run_id}", "committed dir must be <variant>__<runId>"
    assert list(Path(evidence_dir).glob(f".staging-{variant}__{run_id}")) == [], \
        "this run's staging must be atomically committed, not stranded"

    # R6: run-owned profile removed (identity-bound), no run-owned survivor.
    assert completed.profile_residue and completed.profile_residue.get("removed") is True, \
        f"run-owned profile must be fully removed: {completed.profile_residue}"
    assert not Path(completed.profile_dir).exists(), "run-owned profile dir must not survive"
    assert _surviving_browser_pids(Path(completed.profile_dir).name) == [], "no run-owned browser process may survive"

    persisted = (publish_dir / f"oracle_browser_runtime__{variant}.json").read_bytes()
    assert persisted == completed.stdout, "persisted JSON must equal stdout byte-for-byte"
    for name in (
        f"oracle_browser_runtime__{variant}.json",
        f"oracle_final_state__{variant}.png",
        "publish_manifest.json",
        "SEAL.json",
        "cleanup_attestation.json",
    ):
        assert (publish_dir / name).is_file(), f"missing required artifact {name}"
    seal = json.loads((publish_dir / "SEAL.json").read_text("utf-8"))
    assert seal["sealed"] is True and seal["runId"] == run_id and seal["variant"] == variant
    manifest = json.loads((publish_dir / "publish_manifest.json").read_text("utf-8"))
    assert manifest["runId"] == run_id and manifest["variant"] == variant
    assert manifest.get("artifacts"), "publish_manifest must list a nonempty artifact set"
    for name, meta in manifest["artifacts"].items():
        raw = (publish_dir / name).read_bytes()
        assert _sha256(raw) == meta["sha256"], f"publish_manifest hash mismatch for {name}"
        assert len(raw) == meta["bytes"], f"publish_manifest byte count mismatch for {name}"
    # PNG signature on every retained PNG.
    for png in publish_dir.glob("*.png"):
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{png.name} is not a valid PNG"
    # R6 cleanup attestation (written INTO staging before the final rename, so it
    # proves tree death + profile absence; the committed dir's existence at the
    # publish path is itself the proof the atomic rename was the last op).
    att = json.loads((publish_dir / "cleanup_attestation.json").read_text("utf-8"))
    assert att["schema"] == "OracleR6CleanupAttestation.v1"
    assert att["treeDeath"]["zero"] is True, "attestation must prove tree death"
    assert att["profile"]["absent"] is True, "attestation must prove profile absence"
    assert att["runId"] == run_id and att["variant"] == variant

    for key in (
        "harnessSha256",
        "voiceInputSessionSha256",
        "proofAudioSha256",
        "nodeExecSha256",
        "chromeExecSha256",
        "servedHtmlSha256",
    ):
        assert identity.get(key), f"identity missing runtime digest {key}"
    assert manifest.get("harnessSha256") == identity["harnessSha256"]
    on_disk = _sha256(INDEX_HTML.read_bytes())
    assert identity["servedHtmlSha256"] == on_disk
    assert identity["fixtureIdentity"]["servedHtmlSha256"] == on_disk
    return result


def test_voice_readiness_latched_output_failure_requires_newer_success(tmp_path: Path) -> None:
    """ASR health cannot mask fail-closed output; a newer success clears it."""
    node, chrome = _require_browser()
    result = _run_variant("canonical", evidence_dir=tmp_path, node=node, chrome=chrome)
    state = result["voiceReadinessState"]

    assert state["latched"]["terminalKind"] == "error_fail_closed"
    assert state["latched"]["plainReadySurfaceIds"] == []
    assert "input ready" in state["latched"]["topText"].lower()
    assert "output blocked" in state["latched"]["topText"].lower()
    assert state["latched"]["liveRole"] == "status"
    assert state["latched"]["liveMode"] == "polite"
    assert state["latched"]["liveAtomic"] == "true"
    assert state["recovered"]["turnId"] > state["latched"]["failedTurnId"]
    assert state["recovered"]["terminalKind"] == "done"
    assert state["recovered"]["topText"].lower() == "voice ready"


def test_oracle_hive_cells_and_barge_in_in_real_browser(tmp_path: Path) -> None:
    """Canonical variant: all findings 1-6 runtime/ownership regressions."""
    node, chrome = _require_browser()
    result = _run_variant("canonical", evidence_dir=tmp_path, node=node, chrome=chrome)

    # Selected Face-model admission is behavioral, not a static source check:
    # an early turn waits for catalog restoration + exact warm, A cannot
    # authorize B, explicit-to-Auto waits for a model-free server tombstone,
    # failures have zero inference/TTS/history/reflex effects, and retries
    # succeed without a page reload.
    fp = result["facePrewarm"]
    assert fp["beforeCatalog"] == {"voiceCalls": 0, "prewarmCalls": 0}
    assert fp["beforeCatalogWarm"]["voiceCalls"] == 0
    assert "model=restored%3A4b" in fp["catalogVoiceUrl"]
    assert fp["catalogObservedTurnId"] is not None
    assert fp["catalogObservedTerminal"] == "blocked_zero_audio"
    assert fp["voicesAfterSupersede"] == fp["voicesBeforeRace"]
    assert "model=race-b%3A4b" in fp["raceRetryUrl"]
    assert fp["after503"]["voiceCalls"] == fp["failBaseline"]["voiceCalls"]
    assert fp["after503"]["synthCalls"] == fp["failBaseline"]["synthCalls"]
    assert fp["after503"]["reflexCalls"] == fp["failBaseline"]["reflexCalls"]
    assert fp["after503"]["historyChildren"] == fp["failBaseline"]["historyChildren"]
    assert fp["afterMismatchVoiceCalls"] == fp["failBaseline"]["voiceCalls"]
    assert fp["afterLatencyFailure"] == fp["failBaseline"]
    assert "model=failure%3A4b" in fp["recoveryUrl"]
    assert fp["retainedAfterCatalogFailure"] == "saved-offline:4b"
    assert "model=saved-offline%3A4b" in fp["savedCatalogFailureUrl"]
    assert fp["pendingAutoBeforeReceipt"] == fp["pendingAutoBaseline"]
    assert "model=" not in fp["pendingAutoUrl"]
    assert fp["programmaticAutoBeforeReceipt"] == fp["programmaticAutoBaseline"]
    assert "model=" not in fp["programmaticAutoUrl"]
    assert fp["afterAutoFailure"] == fp["autoFailureBaseline"]
    assert fp["afterAutoMismatch"] == fp["autoFailureBaseline"]
    assert fp["afterAutoStall"] == fp["autoStallBaseline"]
    assert "model=" not in fp["autoStallRecoveryUrl"]
    assert fp["bStateAfterStaleAuto"] == {
        "model": "race-b:4b",
        "status": "ready",
        "generation": fp["bGenerationBeforeStaleAuto"],
    }
    assert "model=race-b%3A4b" in fp["bAfterStaleAutoUrl"]
    assert fp["prewarmsBeforeInitialAuto"] == len(fp["prewarmCalls"])
    assert "model=" not in fp["freshAutoUrl"]
    invalidations = [
        call for call in fp["prewarmCalls"] if call["operation"] == "invalidate"
    ]
    assert invalidations
    assert all("model" not in call for call in invalidations)

    # Finding 1: stalled fetch AND stalled read hit a bounded deadline; recovery works.
    sd = result["streamDeadline"]
    assert sd["fetchCase"]["terminalKind"] == "error_timeout"
    assert sd["fetchCase"]["controllerCleared"] is True
    assert sd["readCase"]["terminalKind"] == "error_timeout"
    assert sd["recoveryOnsetMs"] is not None
    # Finding 2: delayed barge ack ownership.
    assert result["bargeAck"]["ackAfterPlayAckFalseAndNull"] == 0
    assert result["bargeAck"]["ackAfterCancelAndOwn"] == 1
    # Finding 3: silence and mixed error are degraded, not done.
    assert result["audibleVerdict"]["silent"]["terminalKind"] == "degraded_silent"
    assert result["audibleVerdict"]["silent"]["onsetMs"] is None
    assert result["audibleVerdict"]["mixed"]["terminalKind"] == "degraded_audio_error"
    assert result["audibleVerdict"]["durationGated"]["terminalKind"] == "degraded_audio_error"
    assert result["audibleVerdict"]["durationGated"]["presence"] == "blocked"
    # Finding 4: full-duplex enable/disable ownership.
    fd = result["fullDuplexRace"]
    assert fd["staleCase"]["enabled"] is False and fd["staleCase"]["pollTimer"] is False
    assert fd["winner"]["enabled"] is True and fd["loserStreamEnded"] is True and fd["winnerStreamLive"] is True
    # Finding 5: monotonic snapshot ownership.
    assert result["snapshotOwnership"]["lastTurnIdAfterARelease"] == result["snapshotOwnership"]["turnIdB"]
    # Existing contracts still hold.
    assert result["ownershipRace"]["presenceAfterStale"] == "speaking"
    assert result["abort"]["synthFetchCount"] == 1
    assert result["abort"]["synthSignalAborted"] is True
    assert result["abort"]["synthReaderCancelled"] is True
    assert result["abort"]["synthResult"] is False
    assert result["vad"]["offset"]["requestsAfterHangover"] == 0
    assert result["vad"]["offset"]["requests"] == 1
    assert result["vad"]["adaptive"]["adaptiveLong"] < result["vad"]["adaptive"]["base"]

    # Bounded experience polish: every truthful stage retains balanced eyes,
    # status color stays inside intentional semantic cells, optional generated
    # sound is accessibility-safe, and a barge-in owns/cancels pending cues.
    polish = result["experiencePolish"]
    assert polish["distinctSignatures"] == 7
    assert polish["defaultAmbience"] is True
    assert polish["defaultAmbienceToggle"] is True
    assert polish["reducedMotionOptionalSound"] is False
    assert polish["reflexScheduled"] is True
    assert polish["missingReflexFallback"] == "__experience_reflex__"
    assert polish["reflexFailureTelemetry"] == {
        "scheduled": False,
        "onsetCallbacks": 0,
        "telemetry": {
            "reflexRequestedId": "__missing_experience_reflex__",
            "reflexSelectedId": "__experience_reflex__",
            "reflexScheduledId": None,
            "reflexPlayedId": None,
            "reflexFallbackSelected": True,
            "reflexFallbackUsed": False,
        },
    }
    assert polish["reflexSuccessTelemetry"] == {
        "reflexRequestedId": "__missing_experience_reflex__",
        "reflexSelectedId": "__experience_reflex__",
        "reflexScheduledId": "__experience_reflex__",
        "reflexPlayedId": "__experience_reflex__",
        "reflexFallbackSelected": True,
        "reflexFallbackUsed": True,
    }
    assert polish["reflexQueuedPresence"] != "reflex"
    assert polish["reflexAudibleState"] == {
        "presence": "reflex",
        "audioReactive": "true",
        "audible": True,
        "onsetAnalyserRetained": True,
        "onsetCallbacks": 1,
    }
    for state, geometry in polish["stateGeometry"].items():
        assert geometry["presence"] == state
        assert geometry["pupils"] == 2
        assert geometry["leftEyeCells"] == geometry["rightEyeCells"] > 0
        assert geometry["leakingStatusCells"] == 0
    assert polish["cue"]["played"] is True
    assert polish["cue"]["semantic"] == "cue"
    assert polish["cue"]["peakGain"] <= 0.03
    assert polish["cueSnapshot"]["speechLikeSources"] == 0
    assert polish["cueStoppedMs"] <= 250
    assert abs(polish["mouthDeltaDuringCue"]) < 0.02
    assert polish["mutedCue"]["reason"] == "optional_sound_disabled"
    assert polish["suspendedCue"]["reason"] == "audio_context_suspended"
    assert polish["sinkFailureCue"]["reason"] == "sink_failure"
    assert polish["layouts"]["mobile390x844"]["viewport"] == {"width": 390, "height": 844}
    for layout in polish["layouts"].values():
        assert layout["horizontalOverflowPx"] == 0
        assert layout["stageInsideViewportWidth"] is True
        assert layout["mirrorInsideStage"] is True
        assert layout["mirrorConsoleOverlap"] <= 0.5
        assert layout["consoleActionOverlap"] <= 0.5
        assert layout["stageControlsOverlap"] <= 0.5

    principal_failures = []
    lifecycle = polish["audioGraphLifecycle"]
    for path in (
        "speechStop",
        "auxiliarySpeechFailure",
        "gainBackedAuxiliarySpeechFailure",
        "generatedSpeechFailure",
        "reflexStartFailure",
        "announcementStartFailure",
        "eggStartFailure",
        "eggConstructionFailure",
        "cueCancellation",
        "ambienceStop",
        "ambienceStopFailure",
        "ambienceReplacement",
        "ambienceFailure",
        "ambienceOscillatorFailure",
        "repeatedCycles",
    ):
        verdict = lifecycle[path]
        if not verdict["allDisconnected"]:
            principal_failures.append(
                f"{path} disconnected {verdict['disconnected']}/{verdict['created']} "
                f"real nodes ({verdict['types']})"
            )
    if lifecycle["logicalState"] != {
        "activeSources": 0,
        "speechLikeSources": 0,
        "ambienceActive": False,
    }:
        principal_failures.append(
            f"logical optional-audio state not drained: {lifecycle['logicalState']}"
        )
    if polish["muteControl"] != {
        "reachable": True,
        "persisted": "false",
        "cueReason": "optional_sound_disabled",
    }:
        principal_failures.append(
            f"production mute control is not reachable/persisted: {polish['muteControl']}"
        )
    reduced_visual = polish["reducedMotionVisual"]
    if set(reduced_visual.values()) != {"none"}:
        principal_failures.append(
            f"reduced-motion leaves perpetual animation active: {reduced_visual}"
        )
    if polish.get("mobileArtifact") != {"width": 390, "height": 844}:
        principal_failures.append(
            f"390x844 artifact has nonliteral PNG dimensions: {polish.get('mobileArtifact')}"
        )
    final_state = polish["finalEvidenceState"]
    if (
        final_state["presence"] != "idle"
        or final_state["turnPhase"] != "idle"
        or final_state["audioReactive"] != "false"
        or final_state["subtitle"] != "Ready."
        or final_state["activeSpeechSources"] != 0
        or final_state["optionalAudio"] != {
            "activeSources": 0,
            "speechLikeSources": 0,
            "ambienceActive": False,
        }
    ):
        principal_failures.append(
            f"final evidence state is contradictory/not normalized: {final_state}"
        )
    assert not principal_failures, "Principal v5 regressions:\n- " + "\n- ".join(principal_failures)

    # 2026-07-05 PTT terminal-state regression (faithful real SSE, extended in the
    # harness's own mixed/multiSource cases): a late non-silent onset recorded
    # AFTER the degraded_audio_error terminal must NOT re-enter speaking — the
    # stage stays blocked, owned sources fully drain, and the onset monitor is
    # cleared. The healthy silent-then-audible control still reaches speaking then
    # done after full drain.
    mixed = result["audibleVerdict"]["mixed"]
    assert mixed["onsetMs"] is not None, "the late onset must be recorded in retained diagnostics"
    assert mixed["reenteredSpeaking"] is False, "a late onset must never re-enter speaking after the terminal"
    assert mixed["presence"] == "blocked" and mixed["turnPhase"] == "blocked"
    assert mixed["ownedSources"] == 0 and mixed["onsetMonitorActive"] is False
    assert mixed["completionMs"] is None and mixed["clientCompletionMs"] is None
    healthy = result["audibleVerdict"]["multiSource"]
    assert healthy["becameSpeaking"] is True and healthy["terminalKind"] == "done"
    assert healthy["ownedSources"] == 0
    assert isinstance(healthy["completionMs"], (int, float))
    assert healthy["completionMs"] >= healthy["onsetMs"]
    assert healthy["clientCompletionMs"] == healthy["completionMs"]
    assert healthy["completionMs"] != healthy["onsetMs"]

    drained_turn = result["retained"]["drainedState"]["lastVoiceTurn"]
    assert isinstance(drained_turn["telemetry"]["audibleCompletionMs"], (int, float))
    assert (
        drained_turn["telemetry"]["audibleCompletionMs"]
        >= drained_turn["telemetry"]["firstNonSilentOnsetMs"]
    )
    assert (
        drained_turn["clientMetrics"]["client_full_utterance_completion_ms"]
        == drained_turn["telemetry"]["audibleCompletionMs"]
    )
    assert (
        drained_turn["clientMetrics"]["client_full_utterance_completion_ms"]
        != drained_turn["clientMetrics"]["client_first_audible_ms"]
    )

    # 2026-07-06 audio-stutter product repair: the browser exposes the one-time
    # v3 ws_super->REST migration, the default-query behavior, and reviewable
    # playback-underflow telemetry that reproduces the measured production stutter.
    pr = result["productRepair"]
    assert pr["available"] is True
    assert pr["clearedWsSuper"] is True and pr["afterMigratePin"] is None and pr["markerSet"] == "true"
    assert pr["queryUnpinnedHasEngine"] is False, "an unpinned session sends no engine param -> server REST default"
    assert pr["secondRunCleared"] is False, "the v3 migration is one-time (idempotent)"
    assert pr["optInCleared"] is False and pr["optInPin"] == "ws_super" and pr["queryOptInHasWsSuper"] is True
    # P1.2: a settings refresh must PRESERVE a deliberate post-migration ws_super
    # opt-in (the old generic-mismatch clear erased it whenever the default was rest).
    assert pr["optInSurvivesRefresh"] == "ws_super", "settings refresh must not erase a deliberate ws_super opt-in"
    assert pr["queryAfterRefreshHasWsSuper"] is True
    # P1.1: a measured >=40 ms playback underflow must GATE the terminal.
    assert pr["underflowTerminal"] == "degraded_underflow", "a measured underflow must block clean done"
    assert pr["cleanTerminal"] == "done"
    assert pr["stutterTelemetryRetained"] is True
    assert pr["underflow"]["count"] == 10
    assert abs(pr["underflow"]["totalMs"] - 4420.834) < 1.0
    assert pr["underflow"]["maxMs"] == 751

    # 2026-07-06 browser-audio REST-batch prebuffer fix: REST batch/parallel TTS
    # siblings can arrive out of order, so a lone REST first chunk must NOT flush
    # on the target duration or the 400ms escape hatch (it would play and END
    # before the sibling arrives -> underflow). It holds until chunk 2 (both
    # scheduled contiguous, zero gap) or a one-chunk 'done'. Post-terminal late
    # media is rejected, a barge clears the held queue, WS Super keeps its
    # low-latency start, and an abnormal REST end (SSE error) clears the held
    # chunk so it never plays after the turn terminal.
    rb = result["restBatch"]
    assert rb["available"] is True
    assert rb["restStartedAfterWait"] is False and rb["restStartedAfterChunk2"] is True
    assert rb["restScheduled"] == 2
    assert rb["restUnderflow"]["count"] == 0
    assert rb["restUnderflow"]["total"] == 0
    assert rb["restUnderflow"]["max"] == 0
    assert rb["shortRunwayHeld"] is True
    assert rb["longRunwayStarted"] is True
    assert rb["longRunwayDepth"] == 0
    assert rb["longRunwayScheduledAfterFirst"] == 1
    assert rb["longRunwayScheduledAfterSecond"] == 2
    assert rb["longRunwayUnderflow"] == 0
    assert rb["oneStartedAfterDone"] is True and rb["oneScheduled"] == 1
    assert rb["lateAccepted"] is True and rb["lateDepth"] == 0 and rb["lateScheduled"] == 0
    assert rb["abortDepthAfter"] == 0
    assert rb["wsStartedAfterWait"] is True
    assert rb["errClearedQueue"] is True
    assert rb["errTerminal"] == "error"
    assert rb["errScheduled"] == 0
    # 2026-07-06 follow-up (3 verifier defects): post-decode terminal race (Issue 1),
    # indexed REST ordering + dedup (Issue 2), and WS->REST fallback timer revocation
    # (Issue 3). Mirrors the .mjs restBatch asserts.
    assert rb["raceScheduled"] == 0
    assert rb["raceDepth"] == 0
    assert rb["dupDepth2"] == 1
    assert rb["dupStarted"] is False
    assert rb["dupScheduled"] == 0
    assert rb["oooDepth1"] == 0
    assert rb["oooScheduled"] == 2
    assert rb["oooUnderflow"] == 0
    assert rb["i3HasHelper"] is True
    assert rb["i3Engine"] == "rest"
    assert rb["i3TimerRevoked"] is True
    assert rb["i3StaleFired"] is False


def test_oracle_long_input_session_in_real_browser(tmp_path: Path) -> None:
    """The served Oracle path uses the accepted bounded session end to end."""
    node, chrome = _require_browser()
    result = _run_variant("canonical", evidence_dir=tmp_path, node=node, chrome=chrome)
    long_input = result["longInputSession"]
    identity = result["identity"]

    assert long_input["servedHtmlSha256"] == identity["servedHtmlSha256"]
    assert long_input["voiceInputSessionSha256"] == identity["voiceInputSessionSha256"]
    assert long_input["loaderPresent"] is True
    assert long_input["adapterPresent"] is True
    assert long_input["legacyRecordedChunksPresent"] is False

    pause = long_input["ordinaryPause"]
    assert pause["pauseMs"] == 1_500
    assert pause["requestsAfterOrdinaryPause"] == 0
    assert pause["requestsAfterFinalSilence"] == 1
    assert pause["turnIdDelta"] == 1
    assert pause["contentType"] == "application/json"
    assert pause["bodyKeys"] == [
        "schema",
        "source",
        "transcript",
        "transcription_model",
    ]

    rejected_tail = long_input["rejectedTailContinuation"]
    assert rejected_tail["continuationPauseMs"] == 1_500
    assert rejected_tail["finalSilenceMs"] == 3_000
    assert rejected_tail["acceptedPhraseArmed"] is True
    assert rejected_tail["trailingDecision"]["submit"] is False
    assert rejected_tail["trailingDecision"]["reason"] in {"too_short", "weak_peak"}
    assert rejected_tail["requestsAfterRejectedTail"] == 0
    assert rejected_tail["requestsAfterFinalSilence"] == 1
    assert rejected_tail["turnIdDelta"] == 1
    assert rejected_tail["transcript"] == "accepted phrase sentinel"
    assert rejected_tail["pcmSamplesAfterRejectedTail"] > 0
    rejected_inspect = rejected_tail["inspect"]
    assert rejected_inspect["state"] == "finalized"
    assert rejected_inspect["bargeCount"] == 1
    assert rejected_inspect["submissions"] == 1

    first_rejection = long_input["firstPhraseRejection"]
    assert first_rejection["decision"]["submit"] is False
    assert first_rejection["requests"] == 0
    assert first_rejection["turnIdDelta"] == 1
    assert first_rejection["activeSessionCleared"] is True
    assert first_rejection["inspect"]["state"] == "cancelled"
    assert first_rejection["inspect"]["submissions"] == 0

    virtual = long_input["virtualLongRun"]
    assert virtual["virtualDurationMs"] == 3 * 60_000
    assert virtual["maxAsrInFlight"] == 1
    assert virtual["asrCalls"] >= 30
    assert virtual["turnRequests"] == 1
    assert virtual["turnIdDelta"] == 1
    assert virtual["bodyKeys"] == [
        "schema",
        "source",
        "transcript",
        "transcription_model",
    ]
    assert virtual["envelope"]["schema"] == "Ms4VoiceTranscriptTurn.v1"
    assert virtual["envelope"]["transcription_model"] == "bounded-session"
    assert virtual["envelope"]["source"] == "voice_input_session"
    transcript = virtual["envelope"]["transcript"]
    assert transcript.index("alpha sentinel") < transcript.index("beta sentinel") < transcript.index("omega sentinel")
    assert transcript.count("repeated phrase") >= 2

    inspect = virtual["inspect"]
    assert inspect["state"] == "finalized"
    assert inspect["submissions"] == 1
    assert inspect["bargeCount"] == 1
    assert inspect["requestInFlight"] is False
    assert inspect["pendingWindow"] is False
    assert inspect["timerCount"] == 0
    assert inspect["maxObservedPcmSamples"] <= 48_000 * 30
    assert inspect["maxObservedTranscriptChars"] <= 65_536


@pytest.fixture(scope="module")
def _scheduler_runtime_probe(tmp_path_factory):
    """Run one canonical real-browser scheduler probe for all focused contracts."""
    node, chrome = _require_browser()
    evidence_dir = tmp_path_factory.mktemp("scheduler_runtime")
    result = _run_variant("canonical", evidence_dir=evidence_dir, node=node, chrome=chrome)
    return result["schedulerTiming"]


def test_rest_lone_first_3749125ms_has_bounded_cancel_owned_escape(
        _scheduler_runtime_probe) -> None:
    """A decoded, sub-4s REST c0 cannot wait forever for absent c1/done."""
    lone = _scheduler_runtime_probe["schedulerReview"]["restLoneFirstEscape"]
    assert lone["durationMs"] == pytest.approx(3749.125, abs=1e-9)

    before = lone["beforeDeadline"]
    assert before["started"] is False
    assert before["scheduled"] == 0
    assert before["queueDepth"] == 1
    assert before["holdReason"] == "awaiting_chunk_2"

    after = lone["afterDeadline"]
    assert after["scheduled"] == 1, (
        f"a lone decoded REST first chunk must escape after its bounded deadline: {lone!r}")
    assert before["timerArmed"] is True
    assert before["callbackCaptured"] is True
    assert before["requestedMs"] == 1000
    assert after == {
        "started": True,
        "scheduled": 1,
        "starts": 1,
        "queueDepth": 0,
        "timerCleared": True,
        "serverScheduledChunks": 1,
        "decodedChunks": 1,
    }

    cancel_before = lone["cancelBeforeBarge"]
    assert cancel_before == {
        "scheduled": 0,
        "queueDepth": 1,
        "timerArmed": True,
        "callbackCaptured": True,
        "requestedMs": 1000,
    }
    assert lone["cancelAfterBarge"] == {
        "ownerAdvanced": True,
        "scheduled": 0,
        "starts": 0,
        "queueDepth": 0,
        "timerCleared": True,
        "playbackRunStarted": False,
    }


def test_r9_outstanding_rest_sibling_defers_lone_timer_without_hiding_underflow(
        _scheduler_runtime_probe) -> None:
    replay = _scheduler_runtime_probe["schedulerReview"]["r9OutstandingSibling"]
    assert replay["input"] == {
        "firstDurationSec": pytest.approx(3.6586666666666665, abs=1e-12),
        "secondDurationSec": pytest.approx(6.677333333333333, abs=1e-12),
        "siblingDecodeMs": pytest.approx(4989.666666666667, abs=1e-9),
        "legacyTimerMs": 1000,
        "leadMs": 150,
    }
    assert replay["legacy"] == {
        "firstStartMs": 1150,
        "firstEndMs": pytest.approx(4808.666666666666, abs=1e-9),
        "gapMs": pytest.approx(181, abs=1e-9),
        "wouldCountAtThreshold": True,
        "thresholdMs": 40,
    }
    assert replay["beforeDeadline"] == {
        "scheduled": 0,
        "decoded": 1,
        "serverScheduled": 2,
        "queueDepth": 1,
        "timerArmed": True,
        "requestedMs": 1000,
    }
    assert replay["afterDeadline"] == {
        "scheduled": 0,
        "decoded": 1,
        "serverScheduled": 2,
        "queueDepth": 1,
        "timerArmed": False,
        "timerCleared": False,
        "playbackRunStarted": False,
        "holdReason": "awaiting_chunk_2",
        "starts": 0,
    }
    after = replay["afterSibling"]
    assert after["scheduled"] == 2
    assert after["decoded"] == after["serverScheduled"] == 2
    assert after["queueDepth"] == 0
    assert after["playbackRunStarted"] is True
    assert [row["label"] for row in after["starts"]] == ["r9-0", "r9-1"]
    assert after["rawGapMs"] == pytest.approx(0, abs=1e-9)
    assert after["underflow"] == {"count": 0, "totalMs": 0, "maxMs": 0}


def test_rest_startup_pair_runway_guard_in_real_browser(_scheduler_runtime_probe) -> None:
    """Exact physical REST/Vega startup-pair regression.

    The production browser scheduler receives the verified live arrival/decode
    timeline and decoded durations. The 3.3813334 s startup pair must gain only
    a small bounded release hold, then join chunk 3 with zero underflow.
    """
    pair = _scheduler_runtime_probe["startupPair"]

    assert pair["input"] == {
        "arrivalsMs": [15265, 15284, 19001],
        "decodeReadyMs": [15284, 15288, 19008],
        "durationsSec": [1.5786667, 1.8026667, 5.472],
    }
    observed = pair["observed"]
    assert len(observed["arrivalsMs"]) == len(observed["decodeMs"]) == 3
    assert observed["arrivalsMs"] == sorted(observed["arrivalsMs"])
    assert observed["decodeMs"] == sorted(observed["decodeMs"])
    assert observed["durationsSec"] == pytest.approx(pair["input"]["durationsSec"], abs=1e-6)
    assert observed["scheduledChunks"] == 3
    assert observed["audioErrors"] == 0 and observed["decodeErrors"] == 0
    assert pair["underflow"]["count"] == 0, \
        f"startup pair must bridge chunk 3 with zero underflow; measured replay={pair!r}"
    assert pair["underflow"]["totalMs"] == pytest.approx(0, abs=0.001)
    assert pair["underflow"]["maxMs"] == pytest.approx(0, abs=0.001)
    assert pair["terminalKind"] == "done"

    # Host-clock drift changes the absolute turn-relative timestamps. Measure the
    # one-time delay from the OBSERVED pair-ready decode and bound timer overhead
    # separately from the configured 200 ms production guard.
    assert pair["pairReadyDecodeMs"] == observed["decodeMs"][1]
    assert pair["startedImmediatelyAfterPair"] is False
    relative_delay = observed["firstScheduledMs"] - pair["pairReadyDecodeMs"]
    assert relative_delay == pair["addedFirstScheduleLatencyMs"]
    host_overhead = relative_delay - pair["guardCapMs"]
    assert -5 <= host_overhead <= 75, \
        f"guard timer overhead escaped its small host-jitter allowance: {pair!r}"

    # Raw continuity is stricter than the >=40 ms underflow counter: chunk 3 must
    # begin exactly at the startup pair's scheduled end (within float tolerance).
    assert abs(pair["pairEndToChunk3StartMs"]) <= 1.0, pair


def test_slow_cadence_short_pair_waits_for_sufficient_runway(
        _scheduler_runtime_probe) -> None:
    pair = _scheduler_runtime_probe["slowCadencePair"]
    assert pair["input"] == {
        "arrivalsMs": [16844, 20155, 25119, 29397, 29411, 29424],
        "decodeReadyMs": [16847, 20175, 25155, 29409, 29424, 29431],
        "durationsSec": [
            1.4826666666666666,
            2.2186666666666666,
            5.0986666666666665,
            8.309333333333333,
            10.538666666666666,
            5.290666666666667,
        ],
    }
    observed = pair["observed"]
    assert observed["arrivalsMs"] == pair["input"]["arrivalsMs"]
    assert observed["decodeMs"] == pair["input"]["decodeReadyMs"]
    assert observed["durationsSec"] == pytest.approx(
        pair["input"]["durationsSec"], abs=1e-9)
    assert observed["scheduledChunks"] == 6
    assert observed["audioErrors"] == 0 and observed["decodeErrors"] == 0

    # Fail-first acceptance: the old 149 ms timer produced exactly one normalized
    # 890 ms raw/countable gap and degraded_underflow for this cadence.
    assert pair["underflow"]["count"] == 0, \
        f"slow-cadence short pair must wait for useful runway; replay={pair!r}"
    assert pair["underflow"]["totalMs"] == pytest.approx(0, abs=0.001)
    assert pair["underflow"]["maxMs"] == pytest.approx(0, abs=0.001)
    assert abs(pair["pairEndToChunk2StartMs"]) <= 1.0, pair
    assert pair["terminalKind"] == "done"

    assert pair["pairRunwayMs"] == pytest.approx(3701.333333333333, abs=0.001)
    assert pair["firstSiblingArrivalGapMs"] == 3311
    assert pair["nextSiblingArrivalGapMs"] == 4964
    assert pair["timerArmedAfterPair"] is False
    assert pair["scheduledAfterPair"] == 0
    assert pair["firstScheduledBeforeChunk2"] is None
    assert observed["holdReasonAfterPair"] == "awaiting_chunk_3"
    assert observed["holdReason"] == "startup_three_runway_guard"
    assert observed["startupGuardRequestedAfterPairMs"] is None
    assert observed["startupGuardRequestedMs"] == 600
    # The controlled replay advances the turn/AudioContext clock without a
    # wall sleep. Chunk 4 therefore exercises the early-cancellation branch of
    # the bounded startup-three guard and releases the contiguous run.
    assert observed["firstScheduledMs"] == 29409
    assert observed["firstChunkHoldMs"] == 12562
    assert pair["addedFirstScheduleLatencyMs"] == 9234


def test_static_four_second_signal_does_not_override_slower_indexed_cadence(
        _scheduler_runtime_probe) -> None:
    latest = _scheduler_runtime_probe["cadenceBoundary"]["latestPhysical"]
    assert latest["acceptedArrivalsMs"] == [
        15151, 21314, 26935, 26941, 31392, 31400, 31406]
    assert latest["decodeMs"] == [
        15174, 21344, 26941, 26949, 31400, 31406, 31409]
    assert latest["durationsSec"] == pytest.approx([
        1.4293333333333333,
        2.592,
        4.405333333333333,
        5.290666666666667,
        6.912,
        5.802666666666667,
        2.8266666666666667,
    ], abs=1e-9)
    assert latest["pairRunwayMs"] == pytest.approx(4021.333333333333, abs=0.001)
    assert latest["availableRunwayMs"] == pytest.approx(
        4171.333333333333, abs=0.001)
    assert latest["indexedCadenceMs"] == 6163
    assert latest["cadenceTargetMs"] == pytest.approx(9244.5)
    assert latest["serverScheduledChunks"] == 7
    assert latest["duplicateChunks"] == 0
    assert latest["audioErrors"] == 0 and latest["decodeErrors"] == 0

    # Fail-first: current code lets the static >=4 s signal bypass stronger
    # indexed cadence evidence, producing one normalized 1359.333 ms underflow.
    assert latest["underflow"]["count"] == 0, \
        f"static runway must not override proven cadence insufficiency: {latest!r}"
    assert latest["underflow"]["totalMs"] == pytest.approx(0, abs=0.001)
    assert latest["underflow"]["maxMs"] == pytest.approx(0, abs=0.001)
    assert latest["sourceGapsMs"] == pytest.approx([0] * 6, abs=1e-9)
    assert latest["terminalKind"] == "done"

    pair = latest["pairCheckpoint"]
    assert pair["serverScheduledChunks"] > pair["decodedChunks"]
    assert pair["timerArmed"] is False
    assert pair["playbackRunStarted"] is False
    assert pair["scheduledChunks"] == 0
    assert pair["prebufferDepth"] == 2
    assert pair["holdReason"] == "awaiting_chunk_3"
    assert pair["requestedMs"] is None
    assert pair["firstScheduledMs"] is None
    assert pair["firstChunkHoldMs"] is None

    assert latest["scheduledChunks"] == 7
    assert latest["orderReleasedThrough"] == 7
    assert latest["sourceOrder"] == [
        f"latest-physical-{index}" for index in range(7)]
    assert latest["pairReadyDecodeMs"] == 21344
    # The fourth result followed chunk 3 by only 8 ms and cancelled the
    # startup-three guard immediately.
    assert latest["firstScheduledMs"] == 26949
    assert latest["firstChunkHoldMs"] == 11775
    assert latest["addedFirstScheduleLatencyMs"] == 5605
    assert latest["starts"][0]["scheduledContextNowMs"] \
        == pytest.approx(5538.666666666666)
    assert latest["starts"][0]["startAtMs"] \
        == pytest.approx(5688.666666666666)
    assert latest["starts"][0]["startAtMs"] \
        - latest["starts"][0]["scheduledContextNowMs"] == pytest.approx(150)


def test_q20_exact_nineteen_chunk_live_timing_holds_pair_until_chunk_three(
        _scheduler_runtime_probe) -> None:
    """Replay the rejected Q20 turn-2 browser receipt through production code.

    Q20 started a 6.496 s startup pair while 17 declared REST siblings were
    outstanding. The measured AudioContext advance to chunk 3 exceeded that
    runway by 1255.333 ms, which became the observed audible underflow. The
    repaired scheduler holds the exact pair through chunk 3; chunk 4 arrives
    10 ms later and cancels the bounded startup-three guard, releasing all 19
    chunks without changing their order or terminal accounting.
    """
    q20 = _scheduler_runtime_probe["cadenceBoundary"]["q20LiveTiming"]
    arrivals = [
        10561, 12063, 19961, 19976, 20637, 25395, 25405, 27135,
        30315, 30325, 35473, 35496, 39774, 39783, 41643, 42531,
        47386, 47454, 49096,
    ]
    decodes = [
        10565, 12070, 19975, 19985, 20644, 25405, 25413, 27142,
        30324, 30328, 35494, 35505, 39783, 39788, 41647, 42539,
        47414, 47461, 49099,
    ]
    durations = [
        1.856, 4.64, 9.141333333333334, 6.026666666666666,
        4.682666666666667, 7.8933333333333335, 6.261333333333333,
        4.224, 7.616, 3.2426666666666666, 8.682666666666666,
        7.381333333333333, 8.906666666666666, 3.7546666666666666,
        3.989333333333333, 6.730666666666667, 9.098666666666666,
        7.562666666666667, 1.664,
    ]
    assert q20["acceptedArrivalsMs"] == arrivals
    assert q20["decodeMs"] == decodes
    assert q20["durationsSec"] == pytest.approx(durations, abs=1e-12)
    assert q20["serverScheduledChunks"] == 19
    assert q20["decodedChunks"] == q20["scheduledChunks"] == 19
    assert q20["orderReleasedThrough"] == 19
    assert q20["sourceOrder"] == [f"q20-live-timing-{index}" for index in range(19)]

    pair = q20["pairCheckpoint"]
    assert pair == {
        "timerArmed": False,
        "playbackRunStarted": False,
        "scheduledChunks": 0,
        "prebufferDepth": 2,
        "serverScheduledChunks": 19,
        "decodedChunks": 2,
        "holdReason": "awaiting_chunk_3",
        "requestedMs": None,
        "firstScheduledMs": None,
        "firstChunkHoldMs": None,
    }
    pair_runway_ms = sum(durations[:2]) * 1000
    legacy_gap_ms = q20["contextAtDecodeMs"][2] - pair_runway_ms
    assert legacy_gap_ms == pytest.approx(1255.3333333333967, abs=1e-9)
    assert q20["firstScheduledMs"] == decodes[3]
    assert q20["firstChunkHoldMs"] == decodes[3] - decodes[0]
    assert q20["sourceGapsMs"] == pytest.approx([0] * 18, abs=1e-9)
    assert q20["underflow"] == {"count": 0, "totalMs": 0, "maxMs": 0}
    assert q20["audioErrors"] == q20["decodeErrors"] == q20["duplicateChunks"] == 0
    assert q20["terminalKind"] == "done"


def test_q26_exact_twenty_one_chunk_live_timing_has_no_playback_underflow(
        _scheduler_runtime_probe) -> None:
    """Replay the rejected Q26 turn-4 receipt through the real scheduler.

    The production turn decoded every chunk without an audio error, but its
    three-chunk startup runway expired 452.667 ms before chunk 5 could be
    scheduled.  The captured cadence is a permanent regression fixture: a
    clean turn must preserve order and finish with no mid-reply silence gap.
    """
    q26 = _scheduler_runtime_probe["cadenceBoundary"]["q26LiveTiming"]
    assert q26["serverScheduledChunks"] == 21
    assert q26["decodedChunks"] == q26["scheduledChunks"] == 21
    assert q26["orderReleasedThrough"] == 21
    assert q26["sourceOrder"] == [f"q26-live-timing-{index}" for index in range(21)]
    assert q26["pairCheckpoint"]["holdReason"] == "awaiting_chunk_3"
    assert q26["audioErrors"] == q26["decodeErrors"] == q26["duplicateChunks"] == 0
    assert q26["sourceGapsMs"] == pytest.approx([0] * 20, abs=1e-9)
    assert q26["underflow"] == {"count": 0, "totalMs": 0, "maxMs": 0}
    assert q26["terminalKind"] == "done"


def test_q29_capacity_two_response_start_preempts_wait_audio_under_four_seconds(
        _scheduler_runtime_probe) -> None:
    """Exact rejected Q29 c0/c1 timings take the response-start escape.

    Waiting audio remains available before decoded reply admission, but the real
    generated response owns the speech cursor once admitted.  The historical
    Q20/Q26 fixtures above remain on the conservative continuity path because
    their first decoded audio was already outside the response-start window.
    """
    cases = _scheduler_runtime_probe["schedulerReview"]["q29ResponseStart"]
    expected = {
        "turn5": {
            "firstDecodedMs": 2514,
            "secondDecodedMs": 4386,
            "durationsSec": [3.477333333333333, 4.224],
        },
        "turn6": {
            "firstDecodedMs": 2417,
            "secondDecodedMs": 4407,
            "durationsSec": [3.296, 5.056],
        },
    }
    for name, expected_case in expected.items():
        case = cases[name]
        assert case["input"]["firstDecodedMs"] == expected_case["firstDecodedMs"]
        assert case["input"]["secondDecodedMs"] == expected_case["secondDecodedMs"]
        assert case["input"]["durationsSec"] == pytest.approx(
            expected_case["durationsSec"], abs=1e-12)
        assert case["restTtsCapacity"] == 2
        assert case["restTtsCapacityProvenance"] == "measured_concurrency_probe"
        assert case["beforeDeadline"]["scheduled"] == 0
        assert case["beforeDeadline"]["prebufferDepth"] == 1
        assert case["beforeDeadline"]["timerArmed"] is True
        assert 0 <= case["beforeDeadline"]["requestedMs"] <= 1000
        after = case["afterDeadline"]
        assert after["scheduled"] == 1
        assert after["projectedOnsetMs"] < 4000
        assert after["holdReason"] == "response_start_slo_release"
        assert after["sloRelease"] is True
        assert after["placeholderStopped"] is True
        assert after["placeholderPreempted"] == 1
        assert after["placeholderKinds"] == ["reflex"]
        assert case["rawGapMs"] == pytest.approx(0, abs=1e-9)
        assert case["underflow"] == {"count": 0, "totalMs": 0, "maxMs": 0}


def test_response_start_deadline_boundary_releases_but_late_timer_stays_held(
        _scheduler_runtime_probe) -> None:
    """The reserved-margin boundary is inclusive; one millisecond late is not.

    The boundary still projects actual audible onset 100 ms inside the product
    SLO.  A delayed main-thread callback must not use the capacity-two escape
    after that reserve has already been consumed.
    """
    cases = _scheduler_runtime_probe["schedulerReview"]["q29ResponseStart"]
    boundary = cases["boundary"]
    assert boundary["beforeDeadline"]["requestedMs"] == 750
    assert boundary["afterDeadline"]["scheduled"] == 1
    assert boundary["afterDeadline"]["projectedOnsetMs"] == 3900
    assert boundary["afterDeadline"]["projectedOnsetMs"] < 4000
    assert boundary["afterDeadline"]["holdReason"] == "response_start_slo_release"
    assert boundary["afterDeadline"]["sloRelease"] is True

    late = cases["lateTimer"]
    assert late["beforeDeadline"]["requestedMs"] == 750
    assert late["afterDeadline"]["scheduled"] == 0
    assert late["afterDeadline"]["holdReason"] == "awaiting_chunk_2"
    assert late["afterDeadline"]["sloRelease"] is False
    assert late["afterDeadline"]["placeholderStopped"] is False


def test_response_start_escape_rejects_short_c0_and_unproven_capacity(
        _scheduler_runtime_probe) -> None:
    """Capacity and clock alone must never manufacture playback runway.

    The adversarial c0 is decoded at 1 s, carries only 100 ms of speech, and
    has an already-declared c1 that does not decode until 3 s. The former
    predicate would release c0 at 2 s, creating a deterministic 750 ms raw gap
    after the 150 ms WebAudio lead. The successor holds it and later releases
    the terminal pair contiguously. A Q29-shaped turn with only one observed
    replica also stays on the conservative path regardless of configured
    client concurrency.
    """

    cases = _scheduler_runtime_probe["schedulerReview"]["q29ResponseStart"]
    short = cases["shortFirstLateSibling"]
    assert short["restTtsCapacity"] == 2
    assert short["beforeDeadline"]["requestedMs"] == 1000
    assert short["afterDeadline"]["responseStartReleaseRunwayMs"] == 100
    assert short["afterDeadline"]["scheduled"] == 0
    assert short["afterDeadline"]["holdReason"] == "awaiting_chunk_2"
    assert short["afterDeadline"]["sloRelease"] is False
    assert short["afterDeadline"]["placeholderStopped"] is False
    hypothetical_gap_ms = (
        short["input"]["secondDecodedMs"]
        - (
            short["input"]["firstDecodedMs"]
            + short["beforeDeadline"]["requestedMs"]
            + 150
            + short["input"]["durationsSec"][0] * 1000
        )
    )
    assert hypothetical_gap_ms == pytest.approx(750, abs=1e-9)
    assert len(short["starts"]) == 2
    assert short["rawGapMs"] == pytest.approx(0, abs=1e-9)
    assert short["underflow"] == {"count": 0, "totalMs": 0, "maxMs": 0}

    conservative_cases = {
        "oneObservedReplica": (1, "scale_endpoint"),
        "unverifiedTwoReplicas": (2, "scale_response_unverified"),
    }
    for name, (capacity, provenance) in conservative_cases.items():
        case = cases[name]
        assert case["restTtsCapacity"] == capacity
        assert case["restTtsCapacityProvenance"] == provenance
        assert case["afterDeadline"]["responseStartReleaseRunwayMs"] \
            == pytest.approx(3477.333333333333, abs=1e-9)
        assert case["afterDeadline"]["scheduled"] == 0
        assert case["afterDeadline"]["holdReason"] == "awaiting_chunk_2"
        assert case["afterDeadline"]["sloRelease"] is False
        assert case["afterDeadline"]["placeholderStopped"] is False


def test_response_start_branch_is_present_in_durable_browser_diagnostics() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    for field in (
        "rest_tts_capacity_effective",
        "rest_tts_capacity_provenance",
        "response_start_slo_release",
        "response_start_release_requested_ms",
        "response_start_release_runway_ms",
        "reply_placeholder_audio_preempted",
        "reply_placeholder_kinds",
        "hold_reason",
    ):
        assert f"{field}:" in html
    assert "restTtsCapacity:" in html
    assert "restTtsCapacityProvenance:" in html


def test_cadence_sufficiency_boundary_uses_audible_hysteresis(
        _scheduler_runtime_probe) -> None:
    cases = _scheduler_runtime_probe["cadenceBoundary"]

    below = cases["meaningfulBelow"]
    assert below["serverScheduledChunks"] > below["decodedChunks"]
    assert below["cadenceTargetMs"] - below["availableRunwayMs"] == pytest.approx(60)
    assert below["timerArmed"] is False
    assert below["holdReason"] == "awaiting_chunk_3"

    exact_edges = [
        ("deficit39", 39, False, "awaiting_chunk_3"),
        ("deficit40", 40, False, "awaiting_chunk_3"),
        ("deficit41", 41, False, "awaiting_chunk_3"),
    ]
    for name, expected_deficit, timer_armed, reason in exact_edges:
        case = cases[name]
        assert case["sampleRate"] == 48000
        assert all(isinstance(frames, int) for frames in case["durationFrames"])
        computed_deficit = case["cadenceTargetMs"] - case["availableRunwayMs"]
        assert computed_deficit == pytest.approx(expected_deficit, abs=1e-9), case
        assert case["timerArmed"] is timer_armed, case
        assert case["holdReason"] == reason, case
        assert case["requestedMs"] is None

    tiny_below = cases["tinyBelow"]
    assert tiny_below["cadenceTargetMs"] - tiny_below["availableRunwayMs"] \
        == pytest.approx(1)
    assert tiny_below["timerArmed"] is False, tiny_below
    assert tiny_below["holdReason"] == "awaiting_chunk_3"

    equality = cases["equality"]
    assert equality["cadenceTargetMs"] == pytest.approx(equality["availableRunwayMs"])
    assert equality["timerArmed"] is False
    assert equality["holdReason"] == "awaiting_chunk_3"

    tiny_above = cases["tinyAbove"]
    assert tiny_above["availableRunwayMs"] - tiny_above["cadenceTargetMs"] \
        == pytest.approx(1)
    assert tiny_above["timerArmed"] is False
    assert tiny_above["holdReason"] == "awaiting_chunk_3"

    near = cases["nearSimultaneousOutstanding"]
    assert near["serverScheduledChunks"] > near["decodedChunks"]
    assert near["indexedCadenceMs"] == 19
    assert near["timerArmed"] is False
    assert near["requestedMs"] is None
    assert near["holdReason"] == "awaiting_chunk_3"

    adequate = cases["adequateOutstanding"]
    assert adequate["serverScheduledChunks"] > adequate["decodedChunks"]
    assert adequate["availableRunwayMs"] == pytest.approx(4000)
    assert adequate["playbackRunStarted"] is False
    assert adequate["scheduledChunks"] == 0
    assert adequate["prebufferDepth"] == 2
    assert adequate["holdReason"] == "awaiting_chunk_3"
    assert adequate["timerArmed"] is False


def test_cadence_uses_indexed_startup_pair_records_when_arrivals_interpose(
        _scheduler_runtime_probe) -> None:
    indexed = _scheduler_runtime_probe["cadenceBoundary"]["indexedInterposed"]
    assert indexed["acceptedArrivalsMs"] == [0, 10, 1000]
    assert indexed["acceptedCadenceMs"] == 10
    assert indexed["indexedCadenceMs"] == 1000
    assert indexed["orderReleasedThrough"] == 3
    assert indexed["serverScheduledChunks"] > indexed["decodedChunks"]
    assert indexed["cadenceTargetMs"] - indexed["availableRunwayMs"] \
        == pytest.approx(60)
    assert indexed["timerArmed"] is True, indexed
    assert indexed["playbackRunStarted"] is False
    assert indexed["scheduledChunks"] == 0
    assert indexed["prebufferDepth"] == 3
    assert indexed["holdReason"] == "startup_three_runway_guard"
    assert indexed["requestedMs"] == 600

    completed = _scheduler_runtime_probe["cadenceBoundary"][
        "indexedInterposedCompletion"]
    assert completed["acceptedArrivalsMs"] == [0, 10, 1000, 1600]
    assert completed["acceptedCadenceMs"] == 10
    assert completed["indexedCadenceMs"] == 1000
    assert completed["orderReleasedThrough"] == 4
    assert completed["sourceOrder"] == [
        "indexed-interposed-completion-0",
        "indexed-interposed-completion-1",
        "indexed-interposed-completion-2",
        "indexed-interposed-completion-3",
    ]
    assert completed["sourceGapsMs"] == pytest.approx([0, 0, 0], abs=1e-9)
    assert completed["scheduledChunks"] == 4
    assert completed["duplicateChunks"] == 0
    assert completed["audioErrors"] == 0 and completed["decodeErrors"] == 0
    assert completed["underflow"] == {"count": 0, "totalMs": 0, "maxMs": 0}
    assert completed["timerArmed"] is False
    assert completed["playbackRunStarted"] is True
    assert completed["terminalKind"] == "done"


def test_startup_guard_telemetry_is_truthful_and_additive(_scheduler_runtime_probe) -> None:
    pair = _scheduler_runtime_probe["startupPair"]
    observed = pair["observed"]
    assert observed["holdReasonAfterFirst"] == "awaiting_chunk_2"
    assert observed["holdReasonAfterPair"] == "startup_pair_runway_guard", pair
    assert observed["holdReason"] == "startup_pair_runway_guard", pair
    requested = observed["startupGuardRequestedMs"]
    assert isinstance(requested, (int, float)) and 0 < requested <= pair["guardCapMs"], pair
    assert requested == 200, "the exact 3.3813334 s pair must hit the evidenced 200 ms cap"
    assert observed["firstChunkHoldMs"] >= requested, pair


def test_real_done_active_flush_setup_is_explicit_and_isolated(
        _scheduler_runtime_probe) -> None:
    done = _scheduler_runtime_probe["schedulerReview"]["realDone"]
    assert done["flushStartMode"] == "explicit_production_flush", done
    assert done["preFlushState"] == {
        "timerArmed": True,
        "playbackRunStarted": False,
        "prebufferDepth": 2,
        "flushPromisePresent": False,
        "scheduledChunks": 0,
    }
    assert done["activeFlushState"] == {
        "gateCount": 1,
        "playbackRunStarted": True,
        "prebufferDepth": 0,
        "flushPromisePresent": True,
        "scheduledChunks": 0,
    }


def test_outstanding_rest_pair_terminal_done_flushes_without_hang(
        _scheduler_runtime_probe) -> None:
    done = _scheduler_runtime_probe["schedulerReview"]["outstandingPairDone"]
    assert done["heldBeforeDone"] == {
        "serverScheduledChunks": 3,
        "decodedChunks": 2,
        "scheduledChunks": 0,
        "prebufferDepth": 2,
        "playbackRunStarted": False,
        "timerArmed": False,
        "holdReason": "awaiting_chunk_3",
    }
    assert done["completed"] is True
    assert done["finalTerminal"] == "awaiting_audible"
    assert done["finalScheduled"] == 2
    assert done["finalPrebufferDepth"] == 0
    assert done["playbackRunStarted"] is True
    assert done["submitError"] is None
    assert done["order"] == ["review-0", "review-1"]


def test_inflight_flush_serializes_late_chunk_real_done_and_stale_timer(
        _scheduler_runtime_probe) -> None:
    review = _scheduler_runtime_probe["schedulerReview"]

    active = review["activeFlush"]
    assert active["concurrentGateBeforeFlushRelease"] is False, \
        f"chunk 3 entered sink admission before the startup pair drained: {active!r}"
    assert active["order"] == ["review-0", "review-1", "review-2"], active
    assert active["scheduled"] == 3

    done = review["realDone"]
    assert done["settledBeforeSinkRelease"] is False, \
        f"the real SSE done branch finalized while its pair flush was active: {done!r}"
    assert done["terminalBeforeSinkRelease"] is None
    assert done["scheduledBeforeSinkRelease"] == 0
    assert done["finalScheduled"] == 2
    assert done["finalTerminal"] == "awaiting_audible"
    assert done["order"] == ["review-0", "review-1"]
    assert done["error"] is None

    stale = review["staleTimer"]
    assert stale["callbackCaptured"] is True and stale["clearedByBarge"] is True
    assert stale["playbackRunStarted"] is False, \
        f"a stale guard callback revived playback after barge-in: {stale!r}"
    assert stale["scheduled"] == 0 and stale["starts"] == 0 and stale["queueDepth"] == 0


def test_done_awaiting_flush_barge_preserves_replacement_owner(
        _scheduler_runtime_probe) -> None:
    race = _scheduler_runtime_probe["schedulerReview"]["doneBargeRace"]
    assert race["awaitingDrainBeforeBarge"] is True, race
    assert race["terminalAfterBarge"] in {"aborted", "interrupted"}, race
    assert race["oldFinalTerminal"] == race["terminalAfterBarge"], \
        f"stale done continuation overwrote the canceled terminal: {race!r}"
    assert race["oldAcceptanceTerminal"] == race["terminalAfterBarge"], race
    assert race["oldScheduled"] == 0 and race["oldStarts"] == 0
    assert race["oldPlaybackRunStarted"] is False
    assert race["submitSettled"] is True and race["submitError"] is None

    assert race["activeReplacementPreserved"] is True
    assert race["replacementSessionPreserved"] is True
    assert race["replacementStoredSessionPreserved"] is True
    assert race["replacementPresence"] == "speaking"
    assert race["replacementSubtitle"] == "REPLACEMENT_UI_SENTINEL"
    assert race["replacementDetail"] == "REPLACEMENT_DETAIL_SENTINEL"
    assert race["replacementStatus"] == "REPLACEMENT_STATUS_SENTINEL"
    assert race["replacementAcceptancePreserved"] is True
    assert race["replacementAcceptanceTurnId"] == race["replacementTurnId"]


def test_startup_boundaries_early_third_and_active_flush_cancellation(
        _scheduler_runtime_probe) -> None:
    review = _scheduler_runtime_probe["schedulerReview"]

    boundary = review["loneBoundary"]
    assert boundary == {"started": True, "scheduled": 1, "timerArmed": False}

    adequate = review["adequatePair"]
    assert adequate["heldAfterFirst"] is True
    assert adequate["startedAfterPair"] is True
    assert adequate["scheduled"] == 2 and adequate["timerArmed"] is False
    assert adequate["order"] == ["review-0", "review-1"]

    early = review["earlyThird"]
    assert early["reasonAfterFirst"] == "awaiting_chunk_2"
    assert early["afterPair"]["started"] is False and early["afterPair"]["timerArmed"] is True
    assert early["startedAfterThird"] is True and early["timerCleared"] is True
    assert early["scheduled"] == 3 and early["underflow"] == 0
    assert early["order"] == ["review-0", "review-1", "review-2"]

    cancelled = review["activeFlushCancellation"]
    assert cancelled["ownerAdvanced"] is True
    assert cancelled["scheduled"] == 0 and cancelled["starts"] == 0
    assert cancelled["queueDepth"] == 0 and cancelled["timerCleared"] is True
    assert cancelled["playbackRunStarted"] is False
    assert cancelled["flushPromiseCleared"] is True


def test_oracle_undecodable_generated_audio_stays_blocked_in_real_browser(tmp_path: Path) -> None:
    node, chrome = _require_browser()
    result = _run_variant("undecodable", evidence_dir=tmp_path, node=node, chrome=chrome)
    blocked = result["retained"]["blockedState"]["lastVoiceTurn"]
    assert blocked["terminalKind"] == "blocked_zero_audio"
    assert blocked["telemetry"]["decodeErrors"] == 1
    assert result["audibleVerdict"]["silent"]["terminalKind"] == "degraded_silent"


def _has_run_dir(evidence_dir: Path, variant: str) -> bool:
    return any(Path(evidence_dir).glob(f"{variant}__*"))


def test_served_bytes_diverge_from_path_break_binding(tmp_path: Path) -> None:
    """Finding 6 ABA: the served-buffer digest reflects the ACTUAL served bytes,
    not the on-disk index; the run still cleans up fully."""
    node, chrome = _require_browser()
    fake_root = tmp_path / "fake_repo"
    (fake_root / "machine_spirit_4" / "web").mkdir(parents=True)
    mutated = INDEX_HTML.read_bytes() + b"\n<!-- ABA divergence -->\n"
    (fake_root / "machine_spirit_4" / "web" / "index.html").write_bytes(mutated)
    for static_name in (
        "ms4_voice_dsp.js",
        "voice_input_session.js",
        "oracle_remote_pwa.js",
        "service-worker.js",
        "manifest.webmanifest",
        "oracle-icon.svg",
    ):
        shutil.copyfile(
            ROOT / "machine_spirit_4" / "web" / static_name,
            fake_root / "machine_spirit_4" / "web" / static_name,
        )
    completed = _run_browser_transaction("canonical", evidence_dir=tmp_path, node=node, chrome=chrome, repo_root=fake_root)
    # The harness ran and reported the ACTUAL served bytes...
    assert completed.stdout_text.strip(), (
        "browser harness produced no result JSON before the expected served-bytes rejection: "
        f"returncode={completed.returncode}, primary={completed.primary_error}, "
        f"cleanup={completed.cleanup_error}, stderr={completed.stderr_text[-2000:]}"
    )
    served = json.loads(completed.stdout_text)["identity"]["servedHtmlSha256"]
    assert served == _sha256(mutated), "served digest must reflect the ACTUAL served bytes"
    assert served != _sha256(INDEX_HTML.read_bytes()), "an ABA-diverged buffer must not match the on-disk index"
    # ...and because served != on-disk index, the graph validation fails closed:
    # NO accepted evidence is committed for a divergent run.
    assert completed.committed is False and not _has_run_dir(tmp_path, "canonical")
    assert completed.cleanup_error and "served-bytes" in completed.cleanup_error
    assert completed.profile_residue.get("removed") is True and not Path(completed.profile_dir).exists()


def test_publish_is_non_replacing_and_unique(tmp_path: Path) -> None:
    """Finding 9: a pre-existing sentinel is untouched and two same-variant runs
    commit into distinct run directories (no overwrite, no mixed set)."""
    node, chrome = _require_browser()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("do-not-touch", encoding="utf-8")
    r1 = _run_variant("canonical", evidence_dir=tmp_path, node=node, chrome=chrome)
    r2 = _run_variant("canonical", evidence_dir=tmp_path, node=node, chrome=chrome)
    assert sentinel.read_text(encoding="utf-8") == "do-not-touch"
    assert r1["identity"]["runId"] != r2["identity"]["runId"]
    assert len(sorted(tmp_path.glob("canonical__*"))) == 2


# ---- R6 transaction / cleanup-ownership negatives --------------------------


def test_startup_failure_leaves_no_accepted_evidence(tmp_path: Path) -> None:
    """R6: a Chrome spawn/startup failure => harness nonzero => NO commit, profile
    removed, no surviving browser, no accepted evidence directory."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome,
        extra_env={"MS4_CHROME_PATH": str(HARNESS)}, timeout=60)
    assert completed.returncode != 0 and not completed.timed_out
    assert completed.committed is False and completed.publish_dir is None
    assert not _has_run_dir(tmp_path, "canonical")
    assert completed.profile_residue.get("removed") is True and not Path(completed.profile_dir).exists()
    assert _surviving_browser_pids(Path(completed.profile_dir).name) == []


def test_process_tree_terminated_on_hang(tmp_path: Path) -> None:
    """R6: a hung harness => the wrapper's Job Object proves active-count-zero tree
    death, removes the run-owned profile, and accepts NO evidence — even though the
    hung harness's own finally never ran (the exact R4 leak, now proven closed)."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome,
        extra_env={"MS4_BROWSER_TEST_HANG": "1"}, timeout=20)
    assert completed.timed_out is True
    marker = Path(completed.profile_dir).name
    time.sleep(2)
    assert _surviving_browser_pids(marker) == [], "process tree not cleaned up after hang"
    assert completed.tree_death and completed.tree_death.get("zero") is True, "Job active-count must reach zero"
    assert completed.profile_residue.get("removed") is True and not Path(completed.profile_dir).exists()
    assert completed.committed is False and not _has_run_dir(tmp_path, "canonical")


def test_harness_internal_cleanup_failure_rejects_run(tmp_path: Path) -> None:
    """R6: the harness's own Chrome-exit gate failure => nonzero => NO accepted
    evidence; the wrapper still removes the profile and leaves no staging."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome,
        extra_env={"MS4_BROWSER_TEST_CLEANUP_FAIL": "1"})
    assert completed.returncode != 0 and not completed.timed_out
    assert completed.committed is False and not _has_run_dir(tmp_path, "canonical")
    assert list(tmp_path.glob(".staging-*")) == []
    assert completed.profile_residue.get("removed") is True and not Path(completed.profile_dir).exists()


def test_successful_child_but_wrapper_profile_delete_failure_rejects_run(tmp_path: Path) -> None:
    """R6 (Reviewer A P1): a SUCCESSFUL child (exit 0, sealed staging) whose
    WRAPPER-owned profile deletion fails MUST reject the run — nonzero return, NO
    accepted publish directory, retained cleanup error. A green child is NOT proof
    of cleanup."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome, force_profile_delete_fail=True)
    assert completed.returncode != 0, "wrapper cleanup failure must fail the run even on a green child"
    assert completed.committed is False and completed.publish_dir is None
    assert not _has_run_dir(tmp_path, "canonical"), "no accepted evidence on wrapper cleanup failure"
    assert completed.cleanup_error and "profile" in completed.cleanup_error


def test_final_commit_failure_leaves_no_accepted_evidence(tmp_path: Path) -> None:
    """R6: a final-commit (publication) failure removes staging and creates NO
    accepted evidence directory; nonzero return."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome, force_commit_fail=True)
    assert completed.returncode != 0
    assert completed.committed is False and not _has_run_dir(tmp_path, "canonical")
    assert list(tmp_path.glob(".staging-*")) == []


def test_foreign_publish_target_is_never_deleted(tmp_path: Path) -> None:
    """R6 (Reviewer A P1): a pre-existing/foreign publish target is NEVER
    recursively deleted; the run refuses to commit and accepts no evidence."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome, force_publish_collision=True)
    assert completed.returncode != 0 and completed.committed is False
    pubs = list(tmp_path.glob("canonical__*"))
    assert len(pubs) == 1 and (pubs[0] / "FOREIGN.txt").read_text(encoding="utf-8") == "do-not-touch", \
        "a foreign publish target must survive untouched"
    assert list(tmp_path.glob(".staging-*")) == []


def test_direct_entry_cleanup_failure_returns_nonzero_and_no_evidence(tmp_path: Path) -> None:
    """R6 (Reviewer A P1): the DIRECT entrypoint enforces the SAME state machine —
    a wrapper cleanup failure returns nonzero, records the nonzero/uncommitted exit
    sidecar, and accepts no evidence."""
    node, chrome = _require_browser()
    rc = _run_direct_variant("canonical", tmp_path, _force_profile_delete_fail=True)
    assert rc != 0, "direct entry must return nonzero on cleanup failure"
    assert not _has_run_dir(tmp_path, "canonical")
    exit_txt = (tmp_path / "browser_canonical_direct.exit.txt").read_text(encoding="utf-8")
    assert "committed=False" in exit_txt and "returncode=0\n" not in exit_txt


def test_taskkill_failure_is_survived_by_job_object(tmp_path: Path) -> None:
    """R6 (Reviewer A P2): even if the harness's PID taskkill fails, the wrapper's
    handle-bound Job Object still proves tree death and removes the profile — the
    Job Object, not the PID taskkill, is the authoritative reaper (immune to PID
    reuse)."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome,
        extra_env={"MS4_BROWSER_TEST_TASKKILL_FAIL": "1"})
    assert completed.tree_death and completed.tree_death.get("zero") is True
    assert completed.profile_residue.get("removed") is True and not Path(completed.profile_dir).exists()
    assert _surviving_browser_pids(Path(completed.profile_dir).name) == []


def test_concurrent_same_variant_publication_is_isolated(tmp_path: Path) -> None:
    """R6: two same-variant runs into the SAME evidence dir each commit an
    isolated, sealed run directory with a distinct runId; no staging survives."""
    import concurrent.futures as _f

    node, chrome = _require_browser()
    with _f.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_run_variant, "canonical", evidence_dir=tmp_path, node=node, chrome=chrome)
            for _ in range(2)
        ]
        results = [fut.result() for fut in futures]
    assert len({r["identity"]["runId"] for r in results}) == 2
    assert len(sorted(tmp_path.glob("canonical__*"))) == 2
    assert list(tmp_path.glob(".staging-*")) == []


def test_no_unrelated_profile_is_deleted(tmp_path: Path) -> None:
    """R6: cleanup removes ONLY the exact run-owned profile; an UNRELATED
    ms4-oracle-browser-* directory is never touched."""
    node, chrome = _require_browser()
    unrelated = Path(tempfile.mkdtemp(prefix="ms4-oracle-browser-UNRELATED-"))
    (unrelated / "keep.txt").write_text("do-not-touch", encoding="utf-8")
    try:
        result = _run_variant("canonical", evidence_dir=tmp_path, node=node, chrome=chrome)
        assert result["ok"] is True
        assert unrelated.is_dir() and (unrelated / "keep.txt").read_text(encoding="utf-8") == "do-not-touch"
    finally:
        shutil.rmtree(unrelated, ignore_errors=True)


# ---- R6 process-control negatives (handle-count-stable) --------------------


def _process_handle_count():
    if not IS_WINDOWS:
        return None
    import gc
    gc.collect()  # release transient Popen process handles before measuring
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    count = ctypes.c_uint32(0)
    kernel32.GetProcessHandleCount(ctypes.c_void_p(kernel32.GetCurrentProcess()), ctypes.byref(count))
    return int(count.value)


def test_node_spawn_error_is_reaped_and_handle_stable(tmp_path: Path) -> None:
    """R6 (Reviewer A P2): a real Node spawn error closes the Job handle on every
    failed spawn (handle-count stable) and, through the transaction, reports a
    primary error, cleans the profile, and returns nonzero with no accepted run."""
    node, chrome = _require_browser()
    if IS_WINDOWS:
        bad = ["this-node-does-not-exist.exe", "-x"]
        for _ in range(2):  # warmup
            with pytest.raises(Exception):
                _run_node_in_tree(bad, cwd=str(ROOT), env=os.environ.copy(), timeout=15)
        before = _process_handle_count()
        for _ in range(8):
            with pytest.raises(Exception):
                _run_node_in_tree(bad, cwd=str(ROOT), env=os.environ.copy(), timeout=15)
        after = _process_handle_count()
        # A 1/spawn leak would be +8; the fixed primitive closes the Job handle.
        assert after - before <= 3, f"monotonic handle leak across 8 failed spawns: {before} -> {after}"
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node="this-node-does-not-exist.exe", chrome=chrome, timeout=15)
    assert completed.returncode != 0 and completed.primary_error
    assert completed.committed is False and not _has_run_dir(tmp_path, "canonical")
    # R8 P1-03/P1-4 (Reviewer A): a spawn/setup error is UNKNOWN tree death, so the
    # profile delete is WITHHELD and its residue stays surfaced — DEATH BEFORE
    # DELETE, never a fabricated/silent delete that contradicts the attestation.
    assert completed.profile_residue.get("removed") is not True
    assert (completed.tree_death or {}).get("zero") is not True


@pytest.mark.parametrize("hook", ["MS4_TEST_FORCE_JOB_SETUP_FAIL", "MS4_TEST_FORCE_ASSIGN_FAIL", "MS4_TEST_FORCE_RESUME_FAIL"])
def test_job_setup_assign_resume_failure_is_no_monotonic_leak(hook: str) -> None:
    """R6 (Reviewer A P2): Job configuration/assign/resume failure closes the Job
    handle and reaps any created process, so repeated failures do NOT grow the
    handle count MONOTONICALLY (the exact R5 defect was +1 handle per spawn — the
    unclosed Job handle). We warm up (absorb one-time DLL/pipe costs), then run 8
    forced failures and require growth far below one-per-spawn; a 1/spawn leak
    would be >= 8. Intermittent OS-reap timing on a suspended, never-resumed
    out-of-job child accounts for the small residual."""
    if not IS_WINDOWS:
        pytest.skip("Job Object handle discipline is Windows-specific")
    node, _chrome = _require_browser()
    env = os.environ.copy()
    env[hook] = "1"
    args = [node, "-e", "setTimeout(function(){}, 60000)"]
    for _ in range(2):  # warmup absorbs one-time costs
        with pytest.raises(Exception):
            _run_node_in_tree(args, cwd=str(ROOT), env=env, timeout=15)
    before = _process_handle_count()
    for _ in range(8):
        with pytest.raises(Exception):
            _run_node_in_tree(args, cwd=str(ROOT), env=env, timeout=15)
    after = _process_handle_count()
    # 8 forced failures: the fixed code must NOT leak ~1/spawn (that was +8).
    assert after - before <= 3, f"monotonic handle leak across 8 forced {hook}: {before} -> {after}"


# ---- R6 identity-bound ownership units -------------------------------------


def test_remove_owned_dir_refuses_outside_root(tmp_path: Path) -> None:
    """R6 (Reviewer A/B repro): a prefixed directory NOT component-contained under
    the expected parent is NEVER deleted, and the Windows sibling string-prefix
    trap (`C:\\Temp-sibling` vs `C:\\Temp`) is rejected."""
    outside = tmp_path / (PROFILE_PREFIX + "outside")
    outside.mkdir()
    (outside / "keep.txt").write_text("x", encoding="utf-8")
    other_parent = tmp_path / "some_other_parent"
    other_parent.mkdir()
    res = _remove_owned_dir(str(outside), expected_parent=str(other_parent),
                            prefix=PROFILE_PREFIX, captured_identity=None)
    assert res["removed"] is False and "refusing" in (res.get("error") or "")
    assert outside.is_dir(), "a directory outside the owned parent must not be removed"
    # Component containment must reject the sibling string-prefix trap.
    assert not _components_contained(str(tmp_path) + "-sibling", str(tmp_path))
    assert _components_contained(str(outside), str(tmp_path))


def test_remove_owned_dir_refuses_identity_swap() -> None:
    """R6 (Reviewer A P1 repro): capture identity, rename the owned dir aside, move
    a foreign dir to the SAME pathname; the identity-bound remover REFUSES (dev/ino
    mismatch) and never deletes the foreign replacement."""
    owned = Path(tempfile.mkdtemp(prefix=PROFILE_PREFIX))
    ident = _dir_identity(str(owned))
    aside = owned.with_name(owned.name + "-aside")
    owned.rename(aside)
    foreign = Path(tempfile.mkdtemp(prefix=PROFILE_PREFIX + "foreign-"))
    (foreign / "FOREIGN.txt").write_text("do-not-touch", encoding="utf-8")
    try:
        os.rename(str(foreign), str(owned))  # replacement at the SAME pathname
        res = _remove_owned_dir(str(owned), expected_parent=tempfile.gettempdir(),
                                prefix=PROFILE_PREFIX, captured_identity=ident)
        assert res["removed"] is False and "identity changed" in (res.get("error") or "")
        assert (owned / "FOREIGN.txt").read_text(encoding="utf-8") == "do-not-touch", \
            "a replacement/foreign object must never be deleted"
    finally:
        shutil.rmtree(owned, ignore_errors=True)
        shutil.rmtree(aside, ignore_errors=True)


def test_remove_owned_dir_refuses_reparse_point(tmp_path: Path) -> None:
    """R6: a symlink/reparse point at the owned pathname is refused (no-follow);
    the symlink target is never followed/deleted."""
    target = tmp_path / "real_target"
    target.mkdir()
    (target / "keep.txt").write_text("x", encoding="utf-8")
    link = Path(tempfile.gettempdir()) / (PROFILE_PREFIX + "link-" + uuid.uuid4().hex[:8])
    try:
        try:
            os.symlink(str(target), str(link), target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            pytest.skip("symlink creation not permitted in this environment")
        res = _remove_owned_dir(str(link), expected_parent=tempfile.gettempdir(),
                                prefix=PROFILE_PREFIX, captured_identity=None)
        err = (res.get("error") or "").lower()
        assert res["removed"] is False and ("reparse" in err or "symlink" in err)
        assert (target / "keep.txt").exists(), "the symlink target must not be followed/deleted"
    finally:
        try:
            os.remove(str(link))
        except OSError:
            pass
        shutil.rmtree(target, ignore_errors=True)


def test_remove_owned_dir_removes_exact_owned_dir() -> None:
    """R6 unit: an exact identity-matched owned dir with content is fully removed."""
    owned = Path(tempfile.mkdtemp(prefix=PROFILE_PREFIX))
    ident = _dir_identity(str(owned))
    (owned / "Default").mkdir()
    (owned / "Default" / "Cookies").write_bytes(b"\x00\x01\x02")
    res = _remove_owned_profile(str(owned), ident)
    assert res["removed"] is True and not owned.exists()


def test_remove_owned_dir_fails_closed_on_locked_file() -> None:
    """R6 unit (locked-file): a held-open file makes removal fail; the remover
    reports residue (fail closed), never silently claims success."""
    if not IS_WINDOWS:
        pytest.skip("open-handle delete-blocking is Windows-specific")
    owned = Path(tempfile.mkdtemp(prefix=PROFILE_PREFIX))
    ident = _dir_identity(str(owned))
    locked = owned / "locked.bin"
    handle = open(locked, "wb")
    handle.write(b"held open")
    handle.flush()
    try:
        res = _remove_owned_profile(str(owned), ident)
        assert res["removed"] is False and res["remaining"]
    finally:
        handle.close()
        shutil.rmtree(owned, ignore_errors=True)


# ---- R6 (advisory A4/A6) staging-graph validator units (fast, deterministic) --


def _build_valid_staging(root, variant="canonical", run_id="testrun0"):
    """Build a fully valid sealed staging graph (as the harness would), for the
    A4/A6 negative units to mutate."""
    sdir = Path(root) / f"{STAGING_PREFIX}{variant}__{run_id}"
    sdir.mkdir(parents=True)
    index_sha = "A" * 64
    harness_sha = "B" * 64
    voice_input_session_sha = "F" * 64
    proof_audio_sha = "C" * 64
    node_exec_sha = "D" * 64
    chrome_exec_sha = "E" * 64
    result = {"ok": True, "identity": {"variant": variant, "runId": run_id,
              "acceptanceTarget": True, "servedHtmlSha256": index_sha,
              "harnessSha256": harness_sha,
              "voiceInputSessionSha256": voice_input_session_sha}}
    result_bytes = (json.dumps(result, indent=2) + "\n").encode("utf-8")
    (sdir / f"oracle_browser_runtime__{variant}.json").write_bytes(result_bytes)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 48
    (sdir / f"oracle_final_state__{variant}.png").write_bytes(png)
    artifacts = {
        f"oracle_browser_runtime__{variant}.json": {"sha256": _sha256(result_bytes), "bytes": len(result_bytes)},
        f"oracle_final_state__{variant}.png": {"sha256": _sha256(png), "bytes": len(png)},
    }
    if variant == "canonical":  # R8 P1-08: canonical hive-cells is a mandatory member
        hive = b"\x89PNG\r\n\x1a\n" + b"\x11" * 40
        (sdir / "oracle_hive_cells__canonical.png").write_bytes(hive)
        artifacts["oracle_hive_cells__canonical.png"] = {"sha256": _sha256(hive), "bytes": len(hive)}
    pm = {"schema": "OracleR6PublishManifest.v1", "variant": variant, "runId": run_id,
          "harnessSha256": harness_sha,
          "voiceInputSessionSha256": voice_input_session_sha,
          "proofAudioSha256": proof_audio_sha,
          "nodeExecSha256": node_exec_sha,
          "chromeExecSha256": chrome_exec_sha,
          "artifacts": artifacts}
    pm_bytes = (json.dumps(pm, indent=2) + "\n").encode("utf-8")
    (sdir / "publish_manifest.json").write_bytes(pm_bytes)
    seal = {"schema": "OracleR6InnerSeal.v1", "sealed": True, "variant": variant, "runId": run_id,
            "fixtureNonce": "nonce", "servedHtmlSha256": index_sha, "harnessSha256": harness_sha,
            "voiceInputSessionSha256": voice_input_session_sha,
            "proofAudioSha256": proof_audio_sha, "nodeExecSha256": node_exec_sha,
            "chromeExecSha256": chrome_exec_sha,
            "artifacts": artifacts, "publishManifestSha256": _sha256(pm_bytes)}
    (sdir / "SEAL.json").write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    return sdir, result_bytes, index_sha


def test_graph_valid_staging_passes(tmp_path: Path) -> None:
    sdir, rb, isha = _build_valid_staging(tmp_path)
    ok, errs, parsed = _validate_staging_graph(str(sdir), variant="canonical", run_id="testrun0", stdout_bytes=rb, index_sha=isha)
    assert ok, errs
    assert parsed and parsed["ok"] is True


def test_graph_forged_seal_schema_rejected(tmp_path: Path) -> None:
    """A6: a forged inner seal (sealed:true but wrong schema) is rejected."""
    sdir, rb, isha = _build_valid_staging(tmp_path)
    seal = json.loads((sdir / "SEAL.json").read_text("utf-8"))
    seal["schema"] = "ForgedSeal.v9"
    (sdir / "SEAL.json").write_text(json.dumps(seal), encoding="utf-8")
    ok, errs, _ = _validate_staging_graph(str(sdir), variant="canonical", run_id="testrun0", stdout_bytes=rb, index_sha=isha)
    assert not ok and any("schema/sealed" in e for e in errs)


def test_graph_missing_required_artifact_rejected(tmp_path: Path) -> None:
    sdir, rb, isha = _build_valid_staging(tmp_path)
    (sdir / "oracle_final_state__canonical.png").unlink()
    ok, errs, _ = _validate_staging_graph(str(sdir), variant="canonical", run_id="testrun0", stdout_bytes=rb, index_sha=isha)
    assert not ok and any("missing" in e for e in errs)


def test_graph_artifact_hash_mismatch_rejected(tmp_path: Path) -> None:
    sdir, rb, isha = _build_valid_staging(tmp_path)
    (sdir / "oracle_final_state__canonical.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 99)
    ok, errs, _ = _validate_staging_graph(str(sdir), variant="canonical", run_id="testrun0", stdout_bytes=rb, index_sha=isha)
    assert not ok and any("hash/size mismatch" in e for e in errs)


def test_graph_text_masquerading_as_png_rejected(tmp_path: Path) -> None:
    """A6: a text file with a .png extension (updated in the seal so hashes match)
    is rejected by the PNG signature check."""
    sdir, rb, isha = _build_valid_staging(tmp_path)
    fake = b"this is not a PNG, it is plain text pretending to be one\n"
    (sdir / "oracle_final_state__canonical.png").write_bytes(fake)
    seal = json.loads((sdir / "SEAL.json").read_text("utf-8"))
    seal["artifacts"]["oracle_final_state__canonical.png"] = {"sha256": _sha256(fake), "bytes": len(fake)}
    pm = json.loads((sdir / "publish_manifest.json").read_text("utf-8"))
    pm["artifacts"] = seal["artifacts"]
    pm_bytes = (json.dumps(pm, indent=2) + "\n").encode("utf-8")
    (sdir / "publish_manifest.json").write_bytes(pm_bytes)
    seal["publishManifestSha256"] = _sha256(pm_bytes)
    (sdir / "SEAL.json").write_text(json.dumps(seal), encoding="utf-8")
    ok, errs, _ = _validate_staging_graph(str(sdir), variant="canonical", run_id="testrun0", stdout_bytes=rb, index_sha=isha)
    assert not ok and any("invalid PNG signature" in e for e in errs)


def test_graph_publish_manifest_not_bound_rejected(tmp_path: Path) -> None:
    """A6: mutating publish_manifest without updating the seal binding is rejected."""
    sdir, rb, isha = _build_valid_staging(tmp_path)
    (sdir / "publish_manifest.json").write_bytes(b'{"schema":"OracleR6PublishManifest.v1","tampered":true}\n')
    ok, errs, _ = _validate_staging_graph(str(sdir), variant="canonical", run_id="testrun0", stdout_bytes=rb, index_sha=isha)
    assert not ok and any("publish_manifest" in e for e in errs)


def test_graph_result_not_equal_stdout_rejected(tmp_path: Path) -> None:
    """A6: the staged result JSON must be byte-identical to captured stdout."""
    sdir, rb, isha = _build_valid_staging(tmp_path)
    ok, errs, _ = _validate_staging_graph(str(sdir), variant="canonical", run_id="testrun0", stdout_bytes=rb + b" ", index_sha=isha)
    assert not ok and any("byte-for-byte" in e for e in errs)


def test_graph_served_bytes_divergence_rejected(tmp_path: Path) -> None:
    """A6: a served-html digest != the on-disk index (ABA) is rejected."""
    sdir, rb, _isha = _build_valid_staging(tmp_path)
    ok, errs, _ = _validate_staging_graph(str(sdir), variant="canonical", run_id="testrun0", stdout_bytes=rb, index_sha="Z" * 64)
    assert not ok and any("served-bytes" in e for e in errs)


# ---- R7 fail-closed / handle-relative production-path regressions -------------


def test_unproven_tree_death_does_not_commit(tmp_path: Path) -> None:
    """R7 P1-1/P1-4 (A+B): an UNKNOWN tree death (Job query failed => active None,
    zero False) fails closed — profile delete withheld, NO commit, no accepted
    directory, nonzero effective exit."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome,
        extra_env={"MS4_TEST_FORCE_TREE_DEATH_UNKNOWN": "1"})
    assert completed.committed is False and not _has_run_dir(tmp_path, "canonical")
    assert completed.returncode != 0
    assert completed.cleanup_error and "tree death NOT proven" in completed.cleanup_error


def test_external_cancellation_propagates_with_no_accepted_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7 P1-4 (B): a KeyboardInterrupt in the process primitive PROPAGATES (is not
    swallowed into a normal result) and leaves NO accepted evidence, no owned
    profile/staging residue, and no surviving browser tree."""
    node, chrome = _require_browser()
    run_id = uuid.uuid4()
    monkeypatch.setattr(uuid, "uuid4", lambda: run_id)
    with pytest.raises(KeyboardInterrupt):
        _run_browser_transaction(
            "canonical", evidence_dir=tmp_path, node=node, chrome=chrome,
            extra_env={"MS4_TEST_FORCE_CANCEL": "1"})
    assert not _has_run_dir(tmp_path, "canonical")
    assert not list(Path(tmp_path).glob(".staging-canonical__*"))
    assert _surviving_browser_pids(f"{PROFILE_PREFIX}{run_id}") == []


def test_graph_mutation_before_rename_fails_closed(tmp_path: Path) -> None:
    """R7 P1-3 (A+B): a sealed artifact mutated AFTER validation and immediately
    BEFORE the commit is caught by the re-validation under the pinned handle just
    before the handle-relative rename; the run fails closed, no accepted dir."""
    node, chrome = _require_browser()
    completed = _run_browser_transaction(
        "canonical", evidence_dir=tmp_path, node=node, chrome=chrome,
        force_test_mutate_before_rename=True)
    assert completed.committed is False and not _has_run_dir(tmp_path, "canonical")
    assert completed.cleanup_error and "between validation and commit" in completed.cleanup_error


def test_hardlink_artifact_is_rejected(tmp_path: Path) -> None:
    """R7 P1-03/P2-2 (A+B): a hardlinked artifact (link_count>1) is rejected by the
    pre-commit artifact-safety check."""
    external = tmp_path / "external_payload.bin"
    external.write_bytes(b"external data pretending to be a screenshot")
    art = tmp_path / "oracle_final_state__canonical.png"
    try:
        os.link(str(external), str(art))
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hardlink creation not permitted in this environment")
    ok, reason = _artifact_is_safe(str(art))
    assert ok is False and "hardlink" in reason


def test_ads_artifact_is_rejected(tmp_path: Path) -> None:
    """R7 P2-2 (B): an artifact carrying an NTFS alternate data stream is rejected."""
    if not IS_WINDOWS:
        pytest.skip("NTFS ADS is Windows-specific")
    art = tmp_path / "oracle_final_state__canonical.png"
    art.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    with open(str(art) + ":reviewer_payload", "w", encoding="utf-8") as fh:
        fh.write("hidden stream payload")
    ok, reason = _artifact_is_safe(str(art))
    assert ok is False and "ADS" in reason


def _run_direct_variant(variant: str, evidence_dir: Path, _force_profile_delete_fail: bool = False) -> int:
    """R6 DIRECT entrypoint: the SAME transaction state machine as the pytest path
    (wrapper-owned tree-death proof + identity-bound profile deletion + atomic
    commit + attestation). Retains RAW stdout/stderr/exit + committed/error state,
    and returns the effective returncode (NONZERO on ANY cleanup failure)."""
    node = shutil.which("node")
    chrome = _chrome_path()
    if not node or not chrome:
        raise SystemExit("Node and Chrome/Edge are required for the direct Oracle browser replay")
    completed = _run_browser_transaction(
        variant, evidence_dir=Path(evidence_dir), node=node, chrome=chrome,
        force_profile_delete_fail=_force_profile_delete_fail)
    ev = Path(evidence_dir)
    ev.mkdir(parents=True, exist_ok=True)
    (ev / f"browser_{variant}_direct.stdout.bin").write_bytes(completed.stdout or b"")
    (ev / f"browser_{variant}_direct.stderr.bin").write_bytes(completed.stderr or b"")
    (ev / f"browser_{variant}_direct.exit.txt").write_text(
        f"returncode={completed.returncode}\ntimed_out={completed.timed_out}\n"
        f"committed={completed.committed}\ncleanup_error={completed.cleanup_error}\n"
        f"primary_error={completed.primary_error}\n", encoding="utf-8")
    return completed.returncode


# ---------------------------------------------------------------------------
# 2026-07-07 R4 Oracle latency observability (ADDITIVE) -- browser turn telemetry
#
# Lane: ms4_r4_observability_red (test-only). Fail-first contracts for ADDITIVE
# per-chunk attribution on the accepted no-gap REST cadence, plus a GREEN control
# proving the no-gap behavior itself. The .mjs harness reports observed telemetry
# under result["restObservability"] WITHOUT asserting (so the shared acceptance
# harness stays green and still commits); the RED/GREEN assertions live here.
#
# GROUNDED against frozen production index.html (READ-ONLY, verified this session):
#   * the per-turn telemetry object (index.html L6166-6183) records ONLY aggregate
#     counters (firstDecodedMs, decodedChunks, scheduledChunks, underflowGap*);
#     there are NO per-chunk arrival/decode timestamp arrays, no per-chunk decoded
#     duration, no first-chunk hold duration, and no hold-reason field.
#   * enqueueAudioChunk/flushPrebuffer (L5895-6058) implement the REST hold-until-
#     chunk-2 behavior but write none of the above attribution.
#   * the browser turn id is a client-side monotonic counter (currentVoiceTurnId);
#     no server-minted correlation id is present.
#
# Setup vs RED (R2 P2 finding 4): the ONLY sanctioned skip is an explicit PREFLIGHT
# absence of Node/Chrome BEFORE the run begins. Once Node+Chrome are detected
# present and the run is launched, a nonzero exit, timeout, empty stdout, malformed
# JSON, or an unavailable probe is a genuine FAILURE (fail-closed) -- never a skip --
# so a broken harness can never masquerade as "not a RED". One canonical real-Chrome
# execution is preserved.
#
# Correlation injection (R2 P1 finding 1): the correlation sub-probe drives a REAL
# SSE turn whose audio_chunk frames carry KNOWN server turn_id/chunk_id beside the
# real chunk index, then reads them back off the browser telemetry; the RED asserts
# equality with the injected ids (a correct propagation impl can satisfy it).
#
# Future production scope (R2 P2 finding 5): the corrective production work is scoped
# to voice.py + index.html ONLY. server.py was inspected READ-ONLY this session and
# forwards item.payload UNCHANGED to _sse_event (server.py L1288-1292; _sse_event
# L197-204 = json.dumps with no key filtering), so it neither strips nor rewrites
# provenance -- it is out of scope and NO server.py stripping RED is added.
#
# Evidence (R2 P2 finding 6): on a clean run the fixture preserves the BYTE-ORIGINAL
# canonical Chrome receipt (opt-in via MS4_R4_EVIDENCE_DIR) before any cleaned
# derivative, hashes it, and records browser version + served index.html hash.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# R2 P1 finding 2 repair -- PRODUCER -> RELAY -> BROWSER correlation fixture.
#
# The turn/chunk correlation ids ORIGINATE HERE (the producer contract), are
# serialized through the REAL server.py _sse_event relay (proving server.py
# preserves them unchanged), and the exact relayed SSE frame sequence is fed into
# the real browser SSE parser via the .mjs harness. The browser harness NEVER mints
# the expected ids (R2's tautology is removed): it only transports the fixture and
# reports the browser-observed telemetry; this Python runner owns the expected ids
# and asserts telemetry == producer fixture.
# ---------------------------------------------------------------------------


class _FakeSseHandler:
    """Minimal stand-in for the handler surface ``server._sse_event`` writes to: a
    byte sink ``wfile`` and no live ``connection`` (so the write-timeout branch is
    skipped). Captures the EXACT wire bytes the real relay emits."""

    def __init__(self) -> None:
        self.wfile = io.BytesIO()
        self.connection = None


def _silent_wav_bytes(sample_rate: int = 24000, ms: int = 120) -> bytes:
    """Deterministic, decodable 16-bit PCM mono WAV (silence). Fallback audio for
    the correlation fixture so the real browser decode path succeeds without
    depending on a repo audio asset."""
    n_samples = int(sample_rate * ms / 1000)
    data = b"\x00\x00" * n_samples
    byte_rate = sample_rate * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    return header + data


def _correlation_audio_base64() -> str:
    """Prefer the real proof audio (parity with the accepted R2 probe); fall back to
    a deterministic silent WAV. Either decodes in Chrome; the correlation contract is
    about ids, not audio content."""
    proof = ROOT / "machine_spirit_4" / "canned_audio" / "alloy" / "ack_listening.wav"
    try:
        raw = proof.read_bytes()
        if raw:
            return base64.b64encode(raw).decode("ascii")
    except OSError:
        pass
    return base64.b64encode(_silent_wav_bytes()).decode("ascii")


def _canonical_producer_audio_chunk_payloads(turn_id: str, chunk_ids, audio_b64: str):
    """The producer's audio_chunk payload shape (voice.py ``_try_emit_ready``:
    index/text/audio_base64/audio_mime/tts_ms/held_for_inorder_ms) PLUS the
    producer-minted correlation ids the GREEN contract requires: a turn_id stable
    across the turn and a chunk_id unique per chunk."""
    payloads = []
    for i, cid in enumerate(chunk_ids):
        payloads.append({
            "index": i,
            "text": f"Producer correlation sentence {i}.",
            "audio_base64": audio_b64,
            "audio_mime": "audio/wav",
            "tts_ms": 40 + i,
            "held_for_inorder_ms": 0,
            "turn_id": turn_id,
            "chunk_id": cid,
        })
    return payloads


def _relay_frame_via_server_sse(event: str, payload: dict) -> bytes:
    """Serialize ONE SSE frame through the REAL server.py relay (``_sse_event`` =
    json.dumps, no key filtering) and return the exact wire bytes. server.py is
    exercised READ-ONLY (byte-frozen this session)."""
    from machine_spirit_4.gateway import server as ms4_server

    handler = _FakeSseHandler()
    ok = ms4_server._sse_event(handler, event, payload)
    assert ok is True, "server._sse_event must accept the write to the byte sink"
    return handler.wfile.getvalue()


def _parse_sse_frame_bytes(frame_bytes: bytes):
    """Parse a single-event SSE frame the way the browser reader does (split
    'event:'/'data:' lines, JSON-decode the data). Returns (event, payload)."""
    text = frame_bytes.decode("utf-8")
    event = None
    data = ""
    for line in text.split("\n"):
        if line.startswith("event: "):
            event = line[len("event: "):].strip()
        elif line.startswith("data: "):
            data += line[len("data: "):]
    return event, (json.loads(data) if data else {})


def _build_producer_relay_correlation_fixture(dest_dir) -> dict:
    """Build the PRODUCER/RELAY correlation fixture (R2 finding 2 repair): the ids
    originate here (producer contract), the audio_chunk payloads are serialized
    through the REAL server.py ``_sse_event`` relay, and the exact relayed SSE frame
    sequence is written to a file the .mjs harness feeds into the real browser. The
    browser harness never mints these ids -- it consumes this file."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    turn_id = "ms4-prod-turn-" + uuid.uuid4().hex[:12]
    chunk_ids = [
        "ms4-prod-chunk-" + uuid.uuid4().hex[:10],
        "ms4-prod-chunk-" + uuid.uuid4().hex[:10],
    ]
    audio_b64 = _correlation_audio_base64()
    audio_payloads = _canonical_producer_audio_chunk_payloads(turn_id, chunk_ids, audio_b64)
    frames = [_relay_frame_via_server_sse("transcript", {"text": "correlation probe", "asr_ms": 1})]
    for payload in audio_payloads:
        frames.append(_relay_frame_via_server_sse("audio_chunk", payload))
    frames.append(_relay_frame_via_server_sse("done", {
        "session_id": "corr",
        "reply_text": "Correlation probe reply.",
        "metrics": {
            "audio_chunks": len(audio_payloads),
            "audio_client_written": len(audio_payloads),
            "audio_errors": 0,
            "total_ms": 5,
        },
    }))
    sse_frames = b"".join(frames).decode("utf-8")
    fixture = {
        "schema": "Ms4R3ProducerRelayCorrFixture.v1",
        "turn_id": turn_id,
        "chunk_ids": chunk_ids,
        "audio_chunk_count": len(audio_payloads),
        "relayed_by": "machine_spirit_4.gateway.server._sse_event",
        "sse_frames": sse_frames,
    }
    path = dest_dir / "producer_relay_corr_fixture.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    return {"path": path, "turn_id": turn_id, "chunk_ids": chunk_ids, "fixture": fixture}


def test_server_sse_relay_preserves_producer_correlation_ids() -> None:
    """GREEN (relay-preservation, R2 finding 2 middle leg): the REAL server.py SSE
    serializer (``_sse_event``) must relay producer audio_chunk payloads -- including
    the producer-minted turn_id/chunk_id -- byte-faithfully (json.dumps round-trip,
    no key filtering or rewrite). This proves server.py is NOT the correlation defect
    and keeps it OUT of the future GREEN scope.

    Exercised READ-ONLY against frozen server.py."""
    turn_id = "ms4-prod-turn-relaytest"
    chunk_ids = ["ms4-prod-chunk-r0", "ms4-prod-chunk-r1", "ms4-prod-chunk-r2"]
    audio_b64 = "QUJDMTIzX1JFTEFZ"  # opaque; the relay must not touch it
    payloads = _canonical_producer_audio_chunk_payloads(turn_id, chunk_ids, audio_b64)
    seen_turn = set()
    seen_chunk = []
    for original in payloads:
        frame = _relay_frame_via_server_sse("audio_chunk", original)
        # Wire shape the browser parses.
        assert frame.startswith(b"event: audio_chunk\n"), f"unexpected relay wire prefix: {frame[:40]!r}"
        assert frame.endswith(b"\n\n"), "SSE frame must terminate with a blank line"
        event, relayed = _parse_sse_frame_bytes(frame)
        assert event == "audio_chunk"
        # Every producer field survives the relay unchanged (turn_id/chunk_id/index/
        # audio/text): preservation, not rewrite.
        assert relayed == original, (
            f"server.py relay altered the producer payload: in={original} out={relayed}"
        )
        assert relayed["turn_id"] == turn_id
        seen_turn.add(relayed["turn_id"])
        seen_chunk.append(relayed["chunk_id"])
    assert seen_turn == {turn_id}, "relay must preserve one stable turn_id across chunks"
    assert seen_chunk == chunk_ids, "relay must preserve per-chunk chunk_id order/values"
    assert len(set(seen_chunk)) == len(seen_chunk), "relay must preserve chunk_id uniqueness"


def _preserve_r4_receipt(evidence_dir, completed, result) -> None:
    """R2 P2 finding 6: preserve the BYTE-ORIGINAL canonical Chrome receipt in the
    named R2 evidence directory BEFORE writing any cleaned human-readable
    derivative; hash it and record the browser version plus the served index.html
    hash. No-op unless ``MS4_R4_EVIDENCE_DIR`` is set (a normal pytest run keeps the
    receipt only in pytest temp storage)."""
    dest_root = os.environ.get("MS4_R4_EVIDENCE_DIR")
    if not dest_root:
        return
    dest = Path(dest_root) / "raw_receipts"
    dest.mkdir(parents=True, exist_ok=True)
    identity = (result or {}).get("identity") or {}
    manifest = {
        "preserved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harness_returncode": completed.returncode,
        "harness_committed": bool(completed.committed),
        "harness_timed_out": bool(completed.timed_out),
        "chrome": identity.get("chrome"),
        "chromeExecSha256": identity.get("chromeExecSha256"),
        "nodeVersion": identity.get("nodeVersion"),
        "servedHtmlSha256": identity.get("servedHtmlSha256"),
        "harnessSha256": identity.get("harnessSha256"),
        "runId": identity.get("runId"),
        "files": {},
    }

    def _record(name: str, data: bytes) -> Path:
        # Exclusive-create: never overwrite a prior byte-original receipt.
        final = dest / name
        stem = final.stem
        suffix = 0
        while final.exists():
            suffix += 1
            final = dest / f"{stem}.{suffix}{Path(name).suffix}"
        final.write_bytes(data)
        manifest["files"][final.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        return final

    # (1) BYTE-ORIGINAL raw harness stdout receipt (R2 P2 finding 3 repair): persist
    #     the EXACT CompletedProcess.stdout BYTES directly, BEFORE any decode/parse/
    #     clean. NEVER decode+re-encode -- errors='replace' silently corrupts any
    #     non-UTF-8 byte, so the old stdout_text.encode() path was not byte-original.
    raw_stdout = completed.stdout or b""
    raw_path = _record("canonical_harness_stdout.raw.json", raw_stdout)
    # Hash-and-compare the PRESERVED file against the exact source bytes.
    preserved = raw_path.read_bytes()
    source_sha = hashlib.sha256(raw_stdout).hexdigest()
    preserved_sha = hashlib.sha256(preserved).hexdigest()
    byte_original = preserved == raw_stdout and preserved_sha == source_sha
    manifest["raw_stdout_byte_original"] = {
        "verified": byte_original,
        "source_sha256": source_sha,
        "preserved_sha256": preserved_sha,
        "preserved_file": raw_path.name,
        "bytes": len(raw_stdout),
    }
    assert byte_original, (
        "raw stdout receipt is not byte-identical to CompletedProcess.stdout "
        f"(source_sha256={source_sha}, preserved_sha256={preserved_sha})"
    )
    # (2) BYTE-ORIGINAL committed canonical result artifact, copied verbatim from
    #     pytest temp storage.
    if completed.committed and completed.publish_dir:
        src = Path(completed.publish_dir) / "oracle_browser_runtime__canonical.json"
        if src.is_file():
            _record("oracle_browser_runtime__canonical.receipt.json", src.read_bytes())
    # Cleaned human-readable derivative LAST, only after the byte-originals.
    (dest / "receipt_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture(scope="module")
def _observability_probe(tmp_path_factory):
    """Run the canonical harness ONCE (smallest footprint) and return
    ``result["restObservability"]``. Skips cleanly (setup, not RED) on any
    harness/setup unavailability so a genuine assertion RED is never confused
    with a missing browser or a flaky harness run."""
    node = shutil.which("node")
    chrome = _chrome_path()
    if not node or not chrome:
        # PREFLIGHT ONLY (the sole sanctioned skip): absence before the run begins.
        pytest.skip("preflight: Node and Chrome/Edge absent before the observability run begins")
    evidence_dir = tmp_path_factory.mktemp("r4_observability")
    # R2 finding 2 repair: build the PRODUCER/RELAY correlation fixture (ids minted
    # here, frames serialized by the REAL server.py relay) and hand its path to the
    # harness. The harness consumes it -- it does not mint the ids.
    corr_fixture = _build_producer_relay_correlation_fixture(evidence_dir / "corr_fixture")
    completed = _run_browser_transaction(
        "canonical", evidence_dir=evidence_dir, node=node, chrome=chrome,
        extra_env={"MS4_BROWSER_CORR_FIXTURE": str(corr_fixture["path"])},
    )
    # R2 P2 finding 4: once Node+Chrome are present, every non-clean outcome is a
    # genuine fail-closed FAILURE, never a skip.
    if completed.timed_out:
        pytest.fail(
            "real-Chrome observability harness TIMED OUT after Node+Chrome were detected present "
            f"(primary={completed.primary_error}); fail-closed, not a skip"
        )
    if completed.returncode != 0:
        pytest.fail(
            f"real-Chrome observability harness exited nonzero rc={completed.returncode} after "
            f"Node+Chrome present (primary={completed.primary_error}); fail-closed, not a skip\n"
            f"stderr tail:\n{(completed.stderr_text or '')[-1200:]}"
        )
    if not (completed.stdout_text or "").strip():
        pytest.fail(
            "real-Chrome observability harness produced EMPTY stdout after Node+Chrome present; "
            "fail-closed, not a skip"
        )
    try:
        result = json.loads(completed.stdout_text)
    except ValueError as exc:
        pytest.fail(
            f"real-Chrome observability harness emitted MALFORMED JSON after Node+Chrome present "
            f"({exc}); fail-closed, not a skip"
        )
    obs = result.get("restObservability")
    if not isinstance(obs, dict) or obs.get("available") is not True:
        pytest.fail(
            f"restObservability probe UNAVAILABLE once Node+Chrome present ({obs}); "
            "fail-closed, not a skip"
        )
    # Stash the PRODUCER/RELAY fixture's expected ids (Python's source of truth) so
    # the correlation RED compares browser telemetry to the producer fixture -- never
    # to an id the harness minted (R2 finding 2 tautology removed).
    obs["_r3FixtureExpected"] = {
        "turn_id": corr_fixture["turn_id"],
        "chunk_ids": corr_fixture["chunk_ids"],
    }
    # R2 P2 finding 6: preserve the byte-original canonical receipt (opt-in) BEFORE
    # any cleaned derivative.
    _preserve_r4_receipt(evidence_dir, completed, result)
    return obs


def test_rest_no_gap_prebuffer_is_green_control(_observability_probe) -> None:
    """GREEN control (must PASS -- NOT a RED): the accepted no-gap REST behavior on
    the exact delayed-REST cadence. The lone first chunk stays HELD (not started)
    through the ~600ms inter-chunk gap, chunk 2 flushes both contiguous, and the
    playback underflow stays ZERO. This proves the cadence the RED attribution
    probe runs against is the healthy no-gap path (the control is genuinely
    passing, not an intentionally-passing assertion masquerading as RED)."""
    ng = _observability_probe["noGap"]
    assert ng["startedAfterChunk1"] is False, "lone REST chunk 1 must not start playback"
    assert ng["depthAfterChunk1"] == 1, "REST chunk 1 must be held in the prebuffer"
    assert ng["startedAfterWait"] is False, "REST chunk 1 must stay held across the ~600ms gap"
    assert ng["startedAfterChunk2"] is True, "REST chunk 2 must flush the held pair"
    assert ng["scheduled"] == 2, "both REST chunks must schedule"
    assert ng["underflow"]["count"] == 0, "no-gap REST schedule must have zero underflow gaps"
    assert ng["underflow"]["total"] == 0, "no-gap REST schedule must have zero total underflow ms"
    assert ng["underflow"]["max"] == 0, "no-gap REST schedule must have zero max underflow ms"


def test_browser_turn_telemetry_exposes_per_chunk_attribution(_observability_probe) -> None:
    """On the exact REST cadence the browser turn telemetry must
    expose per-chunk arrival timestamps, per-chunk decode timestamps, per-chunk
    decoded audio duration, the total first-chunk hold, and truthful startup-guard
    reason/requested-duration attribution."""
    attr = _observability_probe["attribution"]
    assert attr["arrivalCount"] == 2, \
        f"expected 2 per-chunk arrival timestamps; telemetry exposed {attr['arrivalCount']}"
    assert attr["decodeCount"] == 2, \
        f"expected 2 per-chunk decode timestamps; telemetry exposed {attr['decodeCount']}"
    assert attr["decodedDurationCount"] == 2, \
        f"expected 2 per-chunk decoded-duration samples; telemetry exposed {attr['decodedDurationCount']}"
    assert attr["firstChunkHoldMs"] is not None and attr["firstChunkHoldMs"] > 0, \
        f"expected a positive first-chunk hold duration; telemetry exposed {attr['firstChunkHoldMs']}"
    assert attr["holdReason"] == "startup_pair_runway_guard", \
        f"expected truthful startup guard reason; telemetry exposed {attr['holdReason']!r}"
    assert 0 < attr["startupGuardRequestedMs"] <= 200, \
        f"expected bounded requested startup guard; telemetry exposed {attr['startupGuardRequestedMs']!r}"


def test_browser_telemetry_carries_server_minted_correlation_id(_observability_probe) -> None:
    """RED (additive correlation contract, R2 P1 finding 2 REPAIR -- producer->relay->
    browser end to end): the correlation ids ORIGINATE in the Python producer fixture
    and are relayed through the REAL server.py ``_sse_event`` serializer; the harness
    only transports those exact frames into the real audio_chunk SSE path. The browser
    turn telemetry must PRESERVE the producer-originated turn_id/chunk_id so a browser
    turn joins 1:1 to the REST chunk metric (voice.py side). We assert telemetry equals
    the PRODUCER fixture (never a harness-minted id) plus per-chunk order + uniqueness;
    a correct index.html propagation impl (reading payload.turn_id/chunk_id in the
    audio_chunk handler) can satisfy this.

    Fails against current production: the audio_chunk handler ignores payload
    turn_id/chunk_id, so the telemetry surfaces no server ids."""
    corr = _observability_probe["correlation"]
    expected = _observability_probe["_r3FixtureExpected"]
    expected_turn = expected["turn_id"]
    expected_chunks = expected["chunk_ids"]
    # Vacuity guard: the injection path must actually have run (2 audio_chunk frames
    # reached the real SSE handler) so this is a genuine RED, not a no-op.
    assert corr.get("driven") is True, \
        f"correlation SSE turn did not drive (setup): {corr.get('error')!r}"
    assert corr.get("sseAudioChunks") == 2, \
        f"expected 2 producer/relay audio_chunk frames through the real SSE path; saw {corr.get('sseAudioChunks')}"
    # The harness must have CONSUMED the producer/relay fixture (server.py-serialized
    # frames), not minted the ids: the fixture ids the browser saw must equal the
    # Python producer ids exactly.
    assert corr.get("fixtureProvided") is True, \
        "correlation probe must consume the producer/relay fixture (MS4_BROWSER_CORR_FIXTURE)"
    assert corr.get("fixtureTurnId") == expected_turn, (
        "harness-consumed fixture turn id != producer fixture "
        f"(harness={corr.get('fixtureTurnId')!r}, producer={expected_turn!r})"
    )
    assert corr.get("fixtureChunkIds") == expected_chunks, (
        "harness-consumed fixture chunk ids != producer fixture "
        f"(harness={corr.get('fixtureChunkIds')!r}, producer={expected_chunks!r})"
    )
    assert expected_turn and len(expected_chunks) == 2, \
        "producer fixture must carry a stable turn id + 2 unique chunk ids"
    # Preservation + equality: telemetry turn id equals the PRODUCER turn id.
    assert corr["turnServerId"] == expected_turn, (
        "browser telemetry did not preserve the producer-originated turn id "
        f"(producer={expected_turn!r}, telemetry={corr['turnServerId']!r})"
    )
    # Ordered equality between producer per-chunk ids and telemetry ids.
    assert corr["chunkServerIds"] == expected_chunks, (
        "browser telemetry per-chunk server ids != producer ids "
        f"(producer={expected_chunks!r}, telemetry={corr['chunkServerIds']!r})"
    )
    # Per-chunk uniqueness.
    assert len(set(corr["chunkServerIds"])) == len(corr["chunkServerIds"]) == 2, \
        f"per-chunk server ids must be unique per chunk; got {corr['chunkServerIds']!r}"


# ---------------------------------------------------------------------------
# 2026-07-07 R5 adversarial per-chunk attribution (fail-first). The R4 telemetry
# arrays (chunkArrivalsMs/chunkDecodeMs/chunkDecodedDurationsSec/chunkServerIds)
# cannot claim 1:1 attribution unless they track the same ACCEPTED, UNIQUE chunk
# lifecycle. These RED contracts drive index.html's REAL enqueueAudioChunk in
# Chrome via the .mjs adversarial sub-probes (result.restObservability.adversarial)
# and assert a bounded per-turn per-chunk record keyed by the producer chunk_id
# (index fallback), where:
#   * a duplicate chunk_id/index does NOT inflate decoded/duration counts,
#   * a decode failure records 'decode_error' and inflates neither,
#   * a cancellation during decode records 'cancelled' and inflates neither,
#   * a late post-terminal chunk records 'late' and is not an accepted arrival,
#   * out-of-order unique chunks each record exactly one 'decoded' lifecycle,
#   * a second turn's records are isolated (no cross-turn bleed) and a stale chunk
#     for the old turn mutates neither turn's accepted accounting.
#
# Fails RED against frozen index.html (63CF8879): it has no per-chunk record, and a
# duplicate that decodes increments decodedChunks/chunkDecodeMs/chunkDecodedDurations
# BEFORE the playback dedup (index.html L5937-5948 vs L6001).
# ---------------------------------------------------------------------------


def _r5_adversarial(observability_probe: dict) -> dict:
    adv = observability_probe.get("adversarial")
    assert isinstance(adv, dict), f"restObservability.adversarial block missing: {adv!r}"
    return adv


def test_r5_duplicate_chunk_does_not_inflate_decode_attribution(_observability_probe) -> None:
    """RED: a duplicate chunk (same producer chunk_id, and separately same index) must
    be dropped BEFORE accepted-arrival accounting -- one arrival, one decode, one
    duration, one 'decoded' record, and one counted duplicate. Frozen index.html
    inflates decodedChunks/chunkDecodeMs/chunkDecodedDurationsSec to 2 on a duplicate
    and keeps no per-chunk record."""
    dup = _r5_adversarial(_observability_probe)["duplicate"]
    assert isinstance(dup, dict) and "error" not in dup, f"duplicate sub-probe error: {dup!r}"
    c = dup["byChunkId"]
    assert c["decodedChunks"] == 1, f"duplicate chunk_id inflated decodedChunks: {c!r}"
    assert c["arrivalCount"] == 1, f"duplicate chunk_id inflated arrival count: {c!r}"
    assert c["decodeCount"] == 1, f"duplicate chunk_id inflated chunkDecodeMs: {c!r}"
    assert c["durationCount"] == 1, f"duplicate chunk_id inflated decoded durations: {c!r}"
    assert c["duplicateChunks"] == 1, f"duplicate chunk_id not counted as a duplicate: {c!r}"
    assert c["recordCount"] == 1, f"expected exactly one per-chunk record: {c!r}"
    assert c["statuses"] == ["decoded"], f"expected one 'decoded' record status: {c!r}"
    i = dup["byIndex"]
    assert i["decodedChunks"] == 1, f"duplicate index inflated decodedChunks: {i!r}"
    assert i["decodeCount"] == 1, f"duplicate index inflated chunkDecodeMs: {i!r}"
    assert i["durationCount"] == 1, f"duplicate index inflated decoded durations: {i!r}"
    assert i["duplicateChunks"] == 1, f"duplicate index not counted as a duplicate: {i!r}"
    assert i["recordCount"] == 1, f"duplicate index must keep exactly one record: {i!r}"


def test_r5_decode_failure_records_terminal_status_no_decode_inflation(_observability_probe) -> None:
    """RED: a genuine decode failure records terminal status 'decode_error', counts
    the arrival + a decodeError, and inflates neither decodedChunks nor the decode/
    duration arrays. Frozen index.html keeps no per-chunk record/status."""
    df = _r5_adversarial(_observability_probe)["decodeFailure"]
    assert isinstance(df, dict) and "error" not in df, f"decodeFailure sub-probe error: {df!r}"
    assert df["returned"] is False, f"a genuine decode failure must return false: {df!r}"
    assert df["decodeErrors"] == 1, f"decode failure must count one decodeError: {df!r}"
    assert df["decodedChunks"] == 0, f"decode failure must not count a decoded chunk: {df!r}"
    assert df["arrivalCount"] == 1, f"the failed decode's arrival must still be counted once: {df!r}"
    assert df["decodeCount"] == 0, f"decode failure must not push chunkDecodeMs: {df!r}"
    assert df["durationCount"] == 0, f"decode failure must not push a decoded duration: {df!r}"
    assert df["recordCount"] == 1, f"expected exactly one per-chunk record: {df!r}"
    assert df["statuses"] == ["decode_error"], f"expected 'decode_error' record status: {df!r}"


def test_r5_cancellation_during_decode_records_terminal_status(_observability_probe) -> None:
    """RED: a cancellation (barge) DURING decode records terminal status 'cancelled',
    counts the arrival, and inflates neither decodedChunks nor the decode/duration
    arrays. Frozen index.html keeps no per-chunk record/status."""
    cd = _r5_adversarial(_observability_probe)["cancelMidDecode"]
    assert isinstance(cd, dict) and "error" not in cd, f"cancelMidDecode sub-probe error: {cd!r}"
    assert cd["returned"] is False, f"a barge during decode must return false: {cd!r}"
    assert cd["decodedChunks"] == 0, f"a cancelled chunk must not count as decoded: {cd!r}"
    assert cd["arrivalCount"] == 1, f"the cancelled chunk's arrival must be counted once: {cd!r}"
    assert cd["decodeCount"] == 0, f"a cancelled chunk must not push chunkDecodeMs: {cd!r}"
    assert cd["durationCount"] == 0, f"a cancelled chunk must not push a decoded duration: {cd!r}"
    assert cd["recordCount"] == 1, f"expected exactly one per-chunk record: {cd!r}"
    assert cd["statuses"] == ["cancelled"], f"expected 'cancelled' record status: {cd!r}"


def test_r5_late_post_terminal_chunk_records_terminal_status(_observability_probe) -> None:
    """RED: a late chunk arriving after the turn finalized ('done') is dropped
    (returns true, intentional), is NOT counted as an accepted arrival, and is
    recorded terminal 'late'. Frozen index.html keeps no per-chunk record/status."""
    lt = _r5_adversarial(_observability_probe)["latePostTerminal"]
    assert isinstance(lt, dict) and "error" not in lt, f"latePostTerminal sub-probe error: {lt!r}"
    assert lt["returned"] is True, f"post-terminal late drop must return true: {lt!r}"
    assert lt["arrivalCount"] == 0, f"a late post-terminal chunk must not be an accepted arrival: {lt!r}"
    assert lt["decodedChunks"] == 0, f"a late post-terminal chunk must not decode: {lt!r}"
    assert lt["decodeCount"] == 0, f"a late post-terminal chunk must not push chunkDecodeMs: {lt!r}"
    assert lt["recordCount"] == 1, f"expected exactly one per-chunk record: {lt!r}"
    assert lt["statuses"] == ["late"], f"expected 'late' record status: {lt!r}"


def test_r5_out_of_order_chunks_have_faithful_per_chunk_records(_observability_probe) -> None:
    """RED: two unique out-of-order chunks (index 1 before 0) each record exactly one
    'decoded' lifecycle, schedule contiguously (2), zero duplicates, zero underflow.
    Frozen index.html schedules correctly but keeps no per-chunk record."""
    oo = _r5_adversarial(_observability_probe)["outOfOrder"]
    assert isinstance(oo, dict) and "error" not in oo, f"outOfOrder sub-probe error: {oo!r}"
    assert oo["scheduled"] == 2, f"contiguous release must schedule both in order: {oo!r}"
    assert oo["underflow"] == 0, f"contiguous release must have zero underflow: {oo!r}"
    assert oo["arrivalCount"] == 2, f"two unique arrivals expected: {oo!r}"
    assert oo["decodeCount"] == 2, f"two unique decodes expected: {oo!r}"
    assert oo["durationCount"] == 2, f"two unique decoded durations expected: {oo!r}"
    assert oo["decodedChunks"] == 2, f"two unique decoded chunks expected: {oo!r}"
    assert oo["duplicateChunks"] == 0, f"out-of-order unique chunks are not duplicates: {oo!r}"
    assert oo["recordCount"] == 2, f"expected exactly two per-chunk records: {oo!r}"
    assert oo["statuses"] == ["decoded", "decoded"], f"both records must be 'decoded': {oo!r}"


def test_r5_second_turn_records_are_isolated_no_cross_turn_bleed(_observability_probe) -> None:
    """RED: a second turn's per-chunk records are isolated from the first (no
    cross-turn bleed), and a stale chunk for the OLD turn arriving after the new turn
    started mutates neither turn's accepted accounting. Frozen index.html keeps no
    per-chunk record at all."""
    st = _r5_adversarial(_observability_probe)["secondTurn"]
    assert isinstance(st, dict) and "error" not in st, f"secondTurn sub-probe error: {st!r}"
    assert st["aScheduled"] == 2, f"turn A must schedule its 2 chunks: {st!r}"
    assert st["bScheduled"] == 2, f"turn B must schedule its 2 chunks: {st!r}"
    assert st["aRecordCount"] == 2, f"turn A must keep exactly its 2 records (stale did not inflate): {st!r}"
    assert st["bRecordCount"] == 2, f"turn B must keep exactly its 2 records (no cross-turn bleed): {st!r}"
    assert st["aDecoded"] == 2 and st["bDecoded"] == 2, f"each turn decodes its own 2 chunks: {st!r}"
    assert st["aDecodeCount"] == 2 and st["bDecodeCount"] == 2, f"each turn pushes its own 2 decodes: {st!r}"
    assert st["aStatuses"] == ["decoded", "decoded"], f"turn A records both 'decoded': {st!r}"
    assert st["bStatuses"] == ["decoded", "decoded"], f"turn B records both 'decoded': {st!r}"
    assert st["staleReturn"] is False, f"a stale chunk for the old turn must be dropped (false): {st!r}"
    assert st["aArrivalCount"] == 2, f"the stale (barged) chunk must add no arrival to turn A: {st!r}"


# 2026-07-07 R6 pre-promotion hardening (fail-first). The fresh independent R5
# review (MS4_R5_INDEPENDENT_REVIEW.md) accepted the R5 telemetry as a source
# candidate but flagged two MANDATORY browser findings:
#   * F1 (LOW): only chunkRecords is bounded by VOICE_CHUNK_RECORDS_MAX; the compat
#     arrays (chunkArrivalsMs/chunkDecodeMs/chunkDecodedDurationsSec), decodedChunks
#     and chunkServerIds are NOT bounded (proven: 513 unique -> records=512 dropped=1
#     but arrays=513).
#   * F2 (LOW): a beyond-cap chunk (dropped at the cap, no record) is re-accepted and
#     decoded again on redelivery (arrivalsDelta=1 decodeDelta=1 decodedChunksDelta=1
#     dupDelta=0), i.e. dedup degrades past the cap.
# R6 makes the 512 cap govern the ENTIRE accepted per-turn telemetry lifecycle: at
# or beyond the cap the event is an intentional bounded drop BEFORE any arrival/
# decode/duration/playback accounting, producer-id accounting moves behind accepted-
# record admission, and a repeated beyond-cap key is never decoded or counted again.
# Driven through the REAL enqueueAudioChunk in real Chrome (capSaturation sub-probe).


def _r6_cap(observability_probe: dict) -> dict:
    adv = observability_probe.get("adversarial")
    assert isinstance(adv, dict), f"restObservability.adversarial block missing: {adv!r}"
    cap = adv.get("capSaturation")
    assert isinstance(cap, dict) and "error" not in cap, f"capSaturation sub-probe error: {cap!r}"
    return cap


def test_r6_cap_saturation_bounds_entire_per_turn_telemetry_surface(_observability_probe) -> None:
    """RED (F1): driving VOICE_CHUNK_RECORDS_MAX + 1 UNIQUE chunks into ONE turn must
    leave the ENTIRE per-turn telemetry surface bounded by the cap, not only
    chunkRecords. Frozen index.html caps chunkRecords (512, dropped>=1) but the
    compat arrays, decodedChunks and chunkServerIds run to 513."""
    cap = _r6_cap(_observability_probe)["afterSaturate"]
    n = cap["cap"]
    assert cap["unique_sent"] == n + 1, f"probe must drive cap+1 unique chunks: {cap!r}"
    assert cap["recordCount"] == n, f"chunkRecords must hard-cap at {n}: {cap!r}"
    assert cap["chunkRecordsDropped"] >= 1, f"the cap+1 chunk must be a counted bounded drop: {cap!r}"
    assert cap["arrivals"] == n, f"chunkArrivalsMs must be bounded at {n} (got {cap['arrivals']}): {cap!r}"
    assert cap["decodeMs"] == n, f"chunkDecodeMs must be bounded at {n} (got {cap['decodeMs']}): {cap!r}"
    assert cap["durations"] == n, f"chunkDecodedDurationsSec must be bounded at {n} (got {cap['durations']}): {cap!r}"
    assert cap["decodedChunks"] == n, f"decodedChunks must be bounded at {n} (got {cap['decodedChunks']}): {cap!r}"
    assert cap["serverIds"] == n, \
        f"chunkServerIds must be bounded at {n} (moved behind accepted-record admission): {cap!r}"


def test_r6_within_cap_redelivery_still_dedups_after_saturation(_observability_probe) -> None:
    """GREEN control: a WITHIN-cap chunk redelivered after saturation must still be
    deduplicated (no re-count) -- proves the cap fix did not break normal dedup."""
    wc = _r6_cap(_observability_probe)["withinCapRedeliver"]
    assert wc["arrivalsDelta"] == 0, f"within-cap redelivery must not add an arrival: {wc!r}"
    assert wc["decodeDelta"] == 0, f"within-cap redelivery must not add a decode: {wc!r}"
    assert wc["dupDelta"] == 1, f"within-cap redelivery must count exactly one duplicate: {wc!r}"


def test_r6_beyond_cap_chunk_stays_dropped_on_redelivery(_observability_probe) -> None:
    """RED (F2): a chunk first seen BEYOND the cap (dropped at the cap, no record)
    must REMAIN dropped on redelivery -- never decoded, arrival-counted, or decoded-
    counted again. It stays an intentional bounded drop. Frozen index.html re-accepts
    it (arrivalsDelta=1 decodeDelta=1 decodedChunksDelta=1 dupDelta=0)."""
    cap = _r6_cap(_observability_probe)
    bc = cap["beyondCapRedeliver"]
    n = cap["afterSaturate"]["cap"]
    assert bc["arrivalsDelta"] == 0, f"beyond-cap redelivery must add no arrival: {bc!r}"
    assert bc["decodeDelta"] == 0, f"beyond-cap redelivery must add no decode: {bc!r}"
    assert bc["decodedChunksDelta"] == 0, f"beyond-cap redelivery must not decode again: {bc!r}"
    assert bc["droppedDelta"] >= 1, f"beyond-cap redelivery must be a counted bounded drop: {bc!r}"
    assert bc["recordCountAfter"] == n, f"chunkRecords must stay hard-capped at {n} after redelivery: {bc!r}"


def test_raw_receipt_preserves_non_utf8_stdout_bytes(tmp_path: Path, monkeypatch) -> None:
    """R2 P2 finding 3 repair (deterministic, no Chrome): the canonical raw stdout
    receipt must be the EXACT ``CompletedProcess.stdout`` bytes, written directly
    before any decode/parse. Feeds stdout bytes that are NOT valid UTF-8 and proves
    the preserved file is byte-identical -- which the OLD ``stdout_text.encode()``
    (decode with errors='replace' then re-encode) path could NOT achieve. The receipt
    manifest records a byte-original verification (hash-and-compare)."""
    import types

    # stdout with invalid UTF-8 (lone 0xFF/0xFE) + a truncated multibyte lead
    # (0xE2 0x28) so a decode(errors='replace')+re-encode WOULD corrupt it.
    raw_stdout = b'{"ok":true,"note":"' + b"\xff\xfe\xe2\x28" + b'"}\n'
    assert raw_stdout.decode("utf-8", "replace").encode("utf-8") != raw_stdout, (
        "test premise: these bytes must not survive a decode(replace)+re-encode round-trip"
    )
    completed = types.SimpleNamespace(
        returncode=0,
        committed=False,
        timed_out=False,
        stdout=raw_stdout,
        stdout_text=raw_stdout.decode("utf-8", "replace"),
        publish_dir=None,
    )
    result = {"identity": {
        "runId": "rawtest", "chrome": {"Browser": "test"},
        "chromeExecSha256": "z", "nodeVersion": "vX",
        "servedHtmlSha256": "x", "harnessSha256": "y",
    }}
    monkeypatch.setenv("MS4_R4_EVIDENCE_DIR", str(tmp_path))
    _preserve_r4_receipt(tmp_path / "unused_pytest_dir", completed, result)

    raw_file = tmp_path / "raw_receipts" / "canonical_harness_stdout.raw.json"
    assert raw_file.is_file(), "byte-original raw stdout receipt must be written"
    preserved = raw_file.read_bytes()
    assert preserved == raw_stdout, "raw receipt must be byte-identical to CompletedProcess.stdout"
    assert hashlib.sha256(preserved).hexdigest() == hashlib.sha256(raw_stdout).hexdigest()

    manifest = json.loads((tmp_path / "raw_receipts" / "receipt_manifest.json").read_text("utf-8"))
    bo = manifest["raw_stdout_byte_original"]
    assert bo["verified"] is True
    assert bo["source_sha256"] == hashlib.sha256(raw_stdout).hexdigest()
    assert bo["preserved_sha256"] == bo["source_sha256"]
    assert bo["bytes"] == len(raw_stdout)


def test_voice_deadlines_allow_complete_long_answers_but_remain_finite() -> None:
    source = INDEX_HTML.read_text(encoding="utf-8")
    assert "scheduled > received ? 90000 : 30000" in source
    assert "return Number.isFinite(override) && override > 0 ? override : 660000;" in source
    assert "turnState.telemetry.runtimeGatedChunks += 1;" in source
    assert "Number(turnState.telemetry.runtimeGatedChunks || 0)" in source
    assert "turnState.telemetry.serverScheduledChunks += 1;" in source
    assert "turnState.telemetry.sseAudioChunks += 1;" in source
    # Both synthesis-declaration and audio-arrival branches must re-arm after
    # their counters change, so phase selection cannot lag one SSE frame.
    scheduled_branch = source.split(
        "turnState.telemetry.serverScheduledChunks += 1;", 1
    )[1].split("} else if (eventName === 'audio_chunk')", 1)[0]
    audio_branch = source.split(
        "turnState.telemetry.sseAudioChunks += 1;", 1
    )[1].split("if (turnState.telemetry.firstSseReceivedMs", 1)[0]
    assert "armStreamStall();" in scheduled_branch
    assert "armStreamStall();" in audio_branch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Direct tree-owning Oracle browser replay")
    parser.add_argument("--variant", required=True, choices=["canonical", "undecodable"])
    parser.add_argument("--evidence-dir", required=True)
    ns = parser.parse_args()
    rc = _run_direct_variant(ns.variant, Path(ns.evidence_dir).resolve())
    sys.exit(rc)
