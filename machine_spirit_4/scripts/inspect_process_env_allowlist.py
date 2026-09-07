"""Read allowlisted MS4 voice-reconcile env keys from a same-user process.

Never prints secrets. Unknown or secret-shaped keys are dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from typing import Any

ALLOWLIST = (
    "MS4_VOICE_CANDIDATE_RECONCILE_ENABLED",
    "MS4_VOICE_CANDIDATE_RECONCILE_SECS",
    "MS4_VOICE_CANDIDATE_RECONCILE_TIMEOUT_SECS",
    "MS4_HIVEMIND_URL",
    "MS4_HIVEMIND_HLI_URL",
    "MS4_OPEN_UI",
)

SECRET_MARKERS = (
    "key",
    "token",
    "secret",
    "password",
    "authorization",
    "credential",
    "cookie",
)

ReadEnvironFn = Callable[[int], dict[str, str]]


def filter_environ(raw: dict[str, str], allowlist: tuple[str, ...] = ALLOWLIST) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    allowed = {name.upper() for name in allowlist}
    for key, value in raw.items():
        name = str(key)
        if name.upper() not in allowed:
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in SECRET_MARKERS):
            continue
        cleaned[name] = str(value)
    return cleaned


def read_process_environ(pid: int) -> dict[str, str]:
    if os.name != "nt":
        raise OSError("windows_only")
    if pid <= 0:
        raise OSError("invalid_pid")
    if pid == os.getpid():
        return {str(key): str(value) for key, value in os.environ.items()}
    return _read_windows_process_environ(pid)


def _read_windows_process_environ(pid: int) -> dict[str, str]:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")

    class PROCESS_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("Reserved1", ctypes.c_void_p),
            ("PebBaseAddress", ctypes.c_void_p),
            ("Reserved2", ctypes.c_void_p * 2),
            ("UniqueProcessId", ctypes.c_void_p),
            ("Reserved3", ctypes.c_void_p),
        ]

    OpenProcess = kernel32.OpenProcess
    OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    OpenProcess.restype = wintypes.HANDLE
    ReadProcessMemory = kernel32.ReadProcessMemory
    ReadProcessMemory.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.LPVOID,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    ReadProcessMemory.restype = wintypes.BOOL
    CloseHandle = kernel32.CloseHandle
    CloseHandle.argtypes = [wintypes.HANDLE]
    CloseHandle.restype = wintypes.BOOL
    NtQueryInformationProcess = ntdll.NtQueryInformationProcess

    handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        raise OSError("open_process_failed")

    def read_bytes(address: int, size: int) -> bytes:
        buf = (ctypes.c_ubyte * size)()
        read = ctypes.c_size_t()
        ok = ReadProcessMemory(handle, ctypes.c_void_p(address), buf, size, ctypes.byref(read))
        if not ok or read.value != size:
            raise OSError("read_process_memory_failed")
        return bytes(buf)

    def read_ptr(address: int) -> int:
        raw = read_bytes(address, ctypes.sizeof(ctypes.c_void_p))
        return int.from_bytes(raw, "little")

    try:
        info = PROCESS_BASIC_INFORMATION()
        status = NtQueryInformationProcess(
            handle,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        )
        if status != 0 or not info.PebBaseAddress:
            raise OSError("query_peb_failed")
        peb = int(info.PebBaseAddress)
        params = read_ptr(peb + 0x20)
        if not params:
            raise OSError("missing_process_parameters")
        env_ptr = read_ptr(params + 0x80)
        env_size = int.from_bytes(read_bytes(params + 0x3F0, 8), "little")
        if not env_ptr or env_size <= 0 or env_size > 2_000_000:
            raise OSError("invalid_environment_block")
        block = read_bytes(env_ptr, env_size)
        text = block.decode("utf-16-le", "replace")
        pairs: dict[str, str] = {}
        for item in text.split("\x00"):
            if not item or item.startswith("=") or "=" not in item:
                continue
            key, value = item.split("=", 1)
            pairs[key] = value
        if not pairs:
            raise OSError("empty_environment_block")
        return pairs
    finally:
        CloseHandle(handle)


def inspect_pid(pid: int, *, read_environ: ReadEnvironFn | None = None) -> dict[str, Any]:
    reader = read_environ or read_process_environ
    try:
        raw = reader(int(pid))
        filtered = filter_environ(raw)
    except Exception:
        return {"ok": False, "pid": int(pid), "env": {}, "error": "reader_unavailable"}
    env = {key: filtered.get(key) for key in ALLOWLIST}
    return {"ok": True, "pid": int(pid), "env": env}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect allowlisted MS4 voice-reconcile env keys.")
    parser.add_argument("pid", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(inspect_pid(args.pid), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
