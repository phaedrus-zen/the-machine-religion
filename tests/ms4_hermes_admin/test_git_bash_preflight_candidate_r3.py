from __future__ import annotations

import hashlib
import io
import os
import subprocess

from machine_spirit_4.hermes_admin import installer, state
from machine_spirit_4.scripts import runtime_common

import pytest


pytestmark = pytest.mark.real_git_bash_preflight


CANONICAL_GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"
WRONG_EXISTING_GIT_BASH = r"C:\Windows\System32\bash.exe"
NORMALIZED_CANONICAL_GIT_BASH = r"c:\PROGRAM FILES\Git\bin\.\BASH.EXE"
EXPECTED_GIT_BASH_SHA256 = "c79fa1cd106bd3e2321f87e580930142ba9309d17b968fa5e3e71dedbbdc36aa"
GIT_BASH_SMOKE_ARGV = [
    CANONICAL_GIT_BASH,
    "--noprofile",
    "--norc",
    "-c",
    "printf GIT_BASH_OK",
]
GIT_BASH_TEST_BYTES = b"sealed Git Bash test bytes"
_REAL_SHA256 = hashlib.sha256
_LOCAL_PYPI_MODE = {"mode": "pypi", "version": "0.18.1", "directory": None}
_DANGEROUS_BASH_ENV = {
    "BaSh_EnV": "unsafe-bash-env",
    "eNv": "unsafe-env",
    "bAsHoPtS": "unsafe-bashopts",
    "ShElLoPtS": "unsafe-shellopts",
    "pS4": "unsafe-ps4",
    "BaSh_XtRaCeFd": "9",
    "BaSh_FuNc_EvIl%%": "() { printf INJECTED; }",
}
_SAFE_ENV_NAME = "HERMES_GIT_BASH_SAFE_SENTINEL"


def _assert_exact_smoke_policy(argv, kwargs):
    assert argv == GIT_BASH_SMOKE_ARGV
    assert set(kwargs) == {
        "stdin",
        "capture_output",
        "text",
        "encoding",
        "errors",
        "timeout",
        "check",
        "shell",
        "env",
    }
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "strict"
    assert kwargs["timeout"] == installer._GIT_BASH_SMOKE_TIMEOUT_SECONDS
    assert kwargs["check"] is False
    assert kwargs["shell"] is False


def _record_local_install_mode(reached):
    """Local metadata read: allowed before Git-Bash on both trigger and job."""

    def install_mode():
        reached.append("install_mode")
        return dict(_LOCAL_PYPI_MODE)

    return install_mode


def _block_external_effect(reached, name, *, as_upgrade_error=False):
    def fail(*_args, **_kwargs):
        reached.append(name)
        if as_upgrade_error:
            raise installer.HermesUpgradeError(f"blocked next operation: {name}")
        raise AssertionError(f"{name} ran before Git Bash preflight completed")

    return fail


@pytest.mark.parametrize(
    (
        "case",
        "platform_name",
        "direct_candidate",
        "canonical_exists",
        "preflight_passes",
        "error_fragment",
    ),
    [
        ("windows-pinned", "win32", "startup", True, True, None),
        (
            "windows-normalized-equivalent",
            "win32",
            NORMALIZED_CANONICAL_GIT_BASH,
            True,
            True,
            None,
        ),
        (
            "windows-wrong-existing",
            "win32",
            WRONG_EXISTING_GIT_BASH,
            True,
            False,
            WRONG_EXISTING_GIT_BASH,
        ),
        ("windows-absent", "win32", None, True, False, "is required"),
        (
            "windows-canonical-missing",
            "win32",
            CANONICAL_GIT_BASH,
            False,
            False,
            "does not name an existing file",
        ),
        ("non-windows-untouched", "linux", WRONG_EXISTING_GIT_BASH, False, True, None),
    ],
    ids=lambda value: value if isinstance(value, str) and not value.startswith("C:") else None,
)
def test_trigger_update_preflight_precedes_validation_release_lookup_and_job_start(
    tmp_path,
    monkeypatch,
    case,
    platform_name,
    direct_candidate,
    canonical_exists,
    preflight_passes,
    error_fragment,
):
    """Git-Bash seal/hash/smoke precedes release lookup, job start, and fetch.

    Local ``install_mode()`` is a metadata read and may run first so a
    current/newer no-op can skip Git-Bash. After a passing preflight the next
    effect is either explicit-target job acceptance or release discovery.
    """
    monkeypatch.setattr(installer.sys, "platform", platform_name)
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", WRONG_EXISTING_GIT_BASH)

    startup_env = runtime_common.ms4_env()
    if platform_name == "win32":
        assert startup_env["HERMES_GIT_BASH_PATH"] == CANONICAL_GIT_BASH
    else:
        assert startup_env["HERMES_GIT_BASH_PATH"] == WRONG_EXISTING_GIT_BASH

    reached: list[str] = []
    file_checks: list[str] = []
    sealed_paths: list[str] = []
    sealed_handles: list[io.BytesIO] = []
    seal_events: list[object] = []
    digest_chunks: list[bytes] = []
    smoke_calls: list[tuple[list[str], dict[str, object]]] = []

    class FakePath:
        def __init__(self, value):
            self.value = str(value)

        def is_file(self):
            file_checks.append(self.value)
            if self.value.casefold() == WRONG_EXISTING_GIT_BASH.casefold():
                return True
            return canonical_exists

    class SealedBytes(io.BytesIO):
        def __enter__(self):
            seal_events.append("entered")
            return super().__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            seal_events.append("closed")
            return super().__exit__(exc_type, exc_value, traceback)

    def open_seal(path):
        sealed_paths.append(path)
        seal_events.append("opened")
        handle = SealedBytes(GIT_BASH_TEST_BYTES)
        sealed_handles.append(handle)
        return handle

    class ExpectedDigest:
        def update(self, chunk):
            digest_chunks.append(bytes(chunk))

        def hexdigest(self):
            return EXPECTED_GIT_BASH_SHA256

    def fake_sha256(initial=b""):
        digest = ExpectedDigest()
        if initial:
            digest.update(initial)
        return digest

    def smoke_run(argv, **kwargs):
        assert sealed_handles
        assert not sealed_handles[-1].closed
        seal_events.append("smoke")
        smoke_calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="GIT_BASH_OK",
            stderr="",
        )

    monkeypatch.setattr(installer, "Path", FakePath)
    monkeypatch.setattr(installer, "_open_windows_read_seal", open_seal, raising=False)
    monkeypatch.setattr(hashlib, "sha256", fake_sha256)
    monkeypatch.setattr(installer.subprocess, "run", smoke_run)
    monkeypatch.setattr(installer.versioning, "install_mode", _record_local_install_mode(reached))
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        _block_external_effect(reached, "latest_version", as_upgrade_error=True),
    )
    monkeypatch.setattr(
        installer.versioning,
        "recent_releases",
        _block_external_effect(reached, "recent_releases"),
    )
    monkeypatch.setattr(
        installer.state,
        "start_job",
        _block_external_effect(reached, "start_job", as_upgrade_error=True),
    )
    monkeypatch.setattr(
        installer, "_git_fetch_resolve", _block_external_effect(reached, "fetch")
    )
    monkeypatch.setattr(
        installer,
        "_preflight_disk_space",
        _block_external_effect(reached, "disk preflight"),
    )
    monkeypatch.setattr(
        installer,
        "_assert_editable_git_dir",
        _block_external_effect(reached, "git dir"),
    )
    for name, value in _DANGEROUS_BASH_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(_SAFE_ENV_NAME, "keep-me")
    expected_path = os.environ.get("PATH")

    candidate = startup_env["HERMES_GIT_BASH_PATH"] if direct_candidate == "startup" else direct_candidate
    paths = (
        ("trigger", lambda: installer.trigger_update(target_version="0.18.2")),
        ("job", lambda: installer.run_update_job(target_version="0.18.2")),
    )
    for update_path, invoke in paths:
        state._reset_for_tests(tmp_path / f"{case}-{update_path}.json")
        reached.clear()
        file_checks.clear()
        sealed_paths.clear()
        sealed_handles.clear()
        seal_events.clear()
        digest_chunks.clear()
        smoke_calls.clear()
        if candidate is None:
            monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        else:
            monkeypatch.setenv("HERMES_GIT_BASH_PATH", candidate)

        with pytest.raises(installer.HermesUpgradeError) as exc_info:
            invoke()

        message = str(exc_info.value)
        if preflight_passes:
            if update_path == "trigger":
                assert message == "blocked next operation: start_job"
                assert reached == ["install_mode", "start_job"]
            else:
                assert message == "blocked next operation: latest_version"
                assert reached == ["install_mode", "latest_version"]
        else:
            assert "HERMES_GIT_BASH_PATH" in message
            assert error_fragment in message
            assert reached == ["install_mode"]

        should_check_file = (
            platform_name == "win32"
            and candidate is not None
            and candidate.casefold().replace("/", "\\").replace("\\.\\", "\\")
            == CANONICAL_GIT_BASH.casefold()
        )
        assert file_checks == ([CANONICAL_GIT_BASH] if should_check_file else [])
        if platform_name == "win32" and preflight_passes:
            assert installer._WINDOWS_GIT_BASH_SHA256 == EXPECTED_GIT_BASH_SHA256
            assert sealed_paths == [CANONICAL_GIT_BASH]
            assert b"".join(digest_chunks) == GIT_BASH_TEST_BYTES
            assert seal_events == ["opened", "entered", "smoke", "closed"]
            assert len(sealed_handles) == 1
            assert sealed_handles[0].closed
            assert len(smoke_calls) == 1
            argv, kwargs = smoke_calls[0]
            _assert_exact_smoke_policy(argv, kwargs)
            smoke_env = kwargs["env"]
            assert isinstance(smoke_env, dict)
            folded_keys = {key.casefold() for key in smoke_env}
            assert folded_keys.isdisjoint(
                {"bash_env", "env", "bashopts", "shellopts", "ps4", "bash_xtracefd"}
            )
            assert not any(key.startswith("bash_func_") for key in folded_keys)
            assert smoke_env[_SAFE_ENV_NAME] == "keep-me"
            assert smoke_env.get("PATH") == expected_path
        else:
            assert sealed_paths == []
            assert sealed_handles == []
            assert seal_events == []
            assert digest_chunks == []
            assert smoke_calls == []


@pytest.mark.parametrize("update_path", ["trigger", "job"])
def test_git_bash_digest_mismatch_refuses_before_smoke_or_side_effects(
    tmp_path,
    monkeypatch,
    update_path,
):
    tampered_bytes = b"tampered Git Bash"
    actual_digest = _REAL_SHA256(tampered_bytes).hexdigest()
    assert actual_digest != EXPECTED_GIT_BASH_SHA256

    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", CANONICAL_GIT_BASH)
    reached: list[str] = []
    smoke_calls: list[tuple[object, object]] = []
    sealed_handles: list[io.BytesIO] = []

    class FakePath:
        def __init__(self, value):
            self.value = str(value)

        def is_file(self):
            return True

    def open_seal(path):
        assert path == CANONICAL_GIT_BASH
        handle = io.BytesIO(tampered_bytes)
        sealed_handles.append(handle)
        return handle

    def forbidden_smoke(*args, **kwargs):
        smoke_calls.append((args, kwargs))
        raise AssertionError("Git Bash smoke ran after a digest mismatch")

    monkeypatch.setattr(installer, "Path", FakePath)
    monkeypatch.setattr(installer, "_open_windows_read_seal", open_seal, raising=False)
    monkeypatch.setattr(installer.subprocess, "run", forbidden_smoke)
    monkeypatch.setattr(
        installer.versioning, "install_mode", _record_local_install_mode(reached)
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        _block_external_effect(reached, "latest_version"),
    )
    monkeypatch.setattr(
        installer.versioning,
        "recent_releases",
        _block_external_effect(reached, "recent_releases"),
    )
    monkeypatch.setattr(
        installer.state, "start_job", _block_external_effect(reached, "start_job")
    )
    monkeypatch.setattr(
        installer, "_git_fetch_resolve", _block_external_effect(reached, "fetch")
    )
    state._reset_for_tests(tmp_path / f"digest-mismatch-{update_path}.json")

    invoke = (
        (lambda: installer.trigger_update(target_version="0.18.2"))
        if update_path == "trigger"
        else (lambda: installer.run_update_job(target_version="0.18.2"))
    )
    with pytest.raises(installer.HermesUpgradeError) as exc_info:
        invoke()

    message = str(exc_info.value)
    assert "SHA-256" in message
    assert EXPECTED_GIT_BASH_SHA256 in message
    assert actual_digest in message
    assert smoke_calls == []
    assert len(sealed_handles) == 1
    assert sealed_handles[0].closed
    assert reached == ["install_mode"]


@pytest.mark.parametrize("update_path", ["trigger", "job"])
@pytest.mark.parametrize(
    ("case", "error_fragment"),
    [
        ("seal-open-error", "seal open failed"),
        ("sealed-read-error", "sealed read failed"),
        ("hash-error", "hash unavailable"),
    ],
)
def test_git_bash_seal_or_hash_error_refuses_before_side_effects(
    tmp_path,
    monkeypatch,
    update_path,
    case,
    error_fragment,
):
    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", CANONICAL_GIT_BASH)
    reached: list[str] = []
    smoke_calls: list[tuple[object, object]] = []
    sealed_handles: list[io.BytesIO] = []

    class FakePath:
        def __init__(self, value):
            self.value = str(value)

        def is_file(self):
            return True

    class ReadFailure(io.BytesIO):
        def read(self, _size=-1):
            raise OSError("sealed read failed")

    def open_seal(path):
        assert path == CANONICAL_GIT_BASH
        if case == "seal-open-error":
            raise OSError("seal open failed")
        handle = (
            ReadFailure(GIT_BASH_TEST_BYTES)
            if case == "sealed-read-error"
            else io.BytesIO(GIT_BASH_TEST_BYTES)
        )
        sealed_handles.append(handle)
        return handle

    def hash_factory():
        if case == "hash-error":
            raise ValueError("hash unavailable")
        return _REAL_SHA256()

    def forbidden_smoke(*args, **kwargs):
        smoke_calls.append((args, kwargs))
        raise AssertionError("Git Bash smoke ran after a seal/hash error")

    monkeypatch.setattr(installer, "Path", FakePath)
    monkeypatch.setattr(installer, "_open_windows_read_seal", open_seal, raising=False)
    monkeypatch.setattr(hashlib, "sha256", hash_factory)
    monkeypatch.setattr(installer.subprocess, "run", forbidden_smoke)
    monkeypatch.setattr(
        installer.versioning, "install_mode", _record_local_install_mode(reached)
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        _block_external_effect(reached, "latest_version"),
    )
    monkeypatch.setattr(
        installer.versioning,
        "recent_releases",
        _block_external_effect(reached, "recent_releases"),
    )
    monkeypatch.setattr(
        installer.state, "start_job", _block_external_effect(reached, "start_job")
    )
    monkeypatch.setattr(
        installer, "_git_fetch_resolve", _block_external_effect(reached, "fetch")
    )
    state._reset_for_tests(tmp_path / f"{case}-{update_path}.json")

    invoke = (
        (lambda: installer.trigger_update(target_version="0.18.2"))
        if update_path == "trigger"
        else (lambda: installer.run_update_job(target_version="0.18.2"))
    )
    with pytest.raises(installer.HermesUpgradeError) as exc_info:
        invoke()

    assert error_fragment in str(exc_info.value)
    assert smoke_calls == []
    assert reached == ["install_mode"]
    assert all(handle.closed for handle in sealed_handles)


@pytest.mark.parametrize("update_path", ["trigger", "job"])
@pytest.mark.parametrize(
    ("case", "smoke_outcome", "error_fragment"),
    [
        ("spawn-error", OSError("spawn refused"), "spawn refused"),
        (
            "timeout",
            subprocess.TimeoutExpired(GIT_BASH_SMOKE_ARGV, 5),
            "timed out",
        ),
        (
            "decode-error",
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
            "decode",
        ),
        (
            "nonzero",
            subprocess.CompletedProcess(
                GIT_BASH_SMOKE_ARGV,
                7,
                stdout="",
                stderr="smoke failed",
            ),
            "exit=7",
        ),
        (
            "stdout-extra-newline",
            subprocess.CompletedProcess(
                GIT_BASH_SMOKE_ARGV,
                0,
                stdout="GIT_BASH_OK\n",
                stderr="",
            ),
            "stdout mismatch",
        ),
    ],
)
def test_git_bash_smoke_failure_refuses_before_release_job_or_fetch_side_effects(
    tmp_path,
    monkeypatch,
    update_path,
    case,
    smoke_outcome,
    error_fragment,
):
    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", CANONICAL_GIT_BASH)
    reached: list[str] = []
    smoke_calls: list[tuple[list[str], dict[str, object]]] = []
    sealed_handles: list[io.BytesIO] = []

    class FakePath:
        def __init__(self, value):
            self.value = str(value)

        def is_file(self):
            return True

    class SealedBytes(io.BytesIO):
        pass

    def open_seal(path):
        assert path == CANONICAL_GIT_BASH
        handle = SealedBytes(GIT_BASH_TEST_BYTES)
        sealed_handles.append(handle)
        return handle

    class ExpectedDigest:
        def update(self, _chunk):
            return None

        def hexdigest(self):
            return EXPECTED_GIT_BASH_SHA256

    def fake_sha256(initial=b""):
        digest = ExpectedDigest()
        digest.update(initial)
        return digest

    def smoke_run(argv, **kwargs):
        assert sealed_handles
        assert not sealed_handles[-1].closed
        smoke_calls.append((list(argv), dict(kwargs)))
        if isinstance(smoke_outcome, BaseException):
            raise smoke_outcome
        return smoke_outcome

    monkeypatch.setattr(installer, "Path", FakePath)
    monkeypatch.setattr(installer, "_open_windows_read_seal", open_seal, raising=False)
    monkeypatch.setattr(hashlib, "sha256", fake_sha256)
    monkeypatch.setattr(installer.subprocess, "run", smoke_run)
    monkeypatch.setattr(
        installer.versioning, "install_mode", _record_local_install_mode(reached)
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        _block_external_effect(reached, "latest_version"),
    )
    monkeypatch.setattr(
        installer.versioning,
        "recent_releases",
        _block_external_effect(reached, "recent_releases"),
    )
    monkeypatch.setattr(
        installer.state, "start_job", _block_external_effect(reached, "start_job")
    )
    monkeypatch.setattr(
        installer, "_git_fetch_resolve", _block_external_effect(reached, "fetch")
    )
    state._reset_for_tests(tmp_path / f"{case}-{update_path}.json")

    invoke = (
        (lambda: installer.trigger_update(target_version="0.18.2"))
        if update_path == "trigger"
        else (lambda: installer.run_update_job(target_version="0.18.2"))
    )
    with pytest.raises(installer.HermesUpgradeError) as exc_info:
        invoke()

    assert error_fragment in str(exc_info.value)
    assert reached == ["install_mode"]
    assert len(smoke_calls) == 1
    assert len(sealed_handles) == 1
    assert sealed_handles[0].closed
    argv, kwargs = smoke_calls[0]
    _assert_exact_smoke_policy(argv, kwargs)


@pytest.mark.skipif(os.name != "nt", reason="Win32 share modes are Windows-only")
def test_windows_read_seal_blocks_write_and_replace_until_close_and_allows_loader(
    tmp_path,
):
    target = tmp_path / "sealed-target.exe"
    replacement = tmp_path / "sealed-replacement.exe"
    target.write_bytes(b"original")
    replacement.write_bytes(b"replacement")

    with installer._open_windows_read_seal(str(target)) as sealed:
        assert sealed.read() == b"original"
        with pytest.raises(OSError):
            target.open("wb")
        with pytest.raises(OSError):
            os.replace(replacement, target)

    with target.open("wb") as writable:
        writable.write(b"after-close")
    assert target.read_bytes() == b"after-close"
    replacement.write_bytes(b"replacement")
    os.replace(replacement, target)
    assert target.read_bytes() == b"replacement"

    with installer._open_windows_read_seal(CANONICAL_GIT_BASH) as sealed:
        assert _REAL_SHA256(sealed.read()).hexdigest() == EXPECTED_GIT_BASH_SHA256
        sealed.seek(0)
        smoke = subprocess.run(
            GIT_BASH_SMOKE_ARGV,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=installer._GIT_BASH_SMOKE_TIMEOUT_SECONDS,
            check=False,
            shell=False,
            env=installer._sanitized_git_bash_env(),
        )
        assert not sealed.closed

    assert smoke.returncode == 0
    assert smoke.stdout == "GIT_BASH_OK"
