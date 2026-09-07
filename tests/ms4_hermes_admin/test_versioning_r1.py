"""F6 cross-platform Hermes-directory resolution.

``versioning._default_hermes_dir`` takes the platform name, environment, and
home directory as explicit arguments so every OS/arch decision is unit-testable
on any host without touching ``sys.platform`` or the real environment. These
tests pin the exact per-platform behavior and prove POSIX never silently reuses
the Windows ``~/Documents`` shape.

Note on ARM64 / Jetson / Thor: ``sys.platform`` is ``"linux"`` on both x86_64
and aarch64 Linux (the CPU architecture is reported by ``platform.machine()``,
which does NOT affect this resolution). Jetson/Thor therefore share the POSIX
branch with desktop Linux; the ``linux``/``linux-arm64`` cases below are
identical by construction, which is the intended, documented behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from machine_spirit_4.hermes_admin import versioning


# Constructed once; comparisons build the expected path from the SAME object so
# the assertions are OS-agnostic (a "/home/op" Path is a WindowsPath on Windows
# and a PosixPath on POSIX, but equals itself either way).
_HOME = Path("/home/op")
_WIN_HOME = Path("C:/Users/op")


@pytest.mark.parametrize(
    ("platform_name", "home", "expected_tail"),
    [
        ("win32", _WIN_HOME, ("Documents", "hermes-agent")),
        ("darwin", _HOME, ("Library", "Application Support", "hermes-agent")),
        ("linux", _HOME, (".local", "share", "hermes-agent")),
        # aarch64 Linux (Jetson/Thor) reports sys.platform == "linux": same branch.
        ("linux", _HOME, (".local", "share", "hermes-agent")),
        ("freebsd13", _HOME, (".local", "share", "hermes-agent")),
    ],
    ids=["windows", "macos", "linux-x86_64", "linux-arm64-jetson", "other-posix"],
)
def test_default_hermes_dir_per_platform_without_env(platform_name, home, expected_tail):
    result = versioning._default_hermes_dir(platform_name, {}, home)
    expected = home
    for part in expected_tail:
        expected = expected / part
    assert result == expected


def test_default_hermes_dir_posix_honors_xdg_data_home():
    result = versioning._default_hermes_dir("linux", {"XDG_DATA_HOME": "/xdg/data"}, _HOME)
    assert result == Path("/xdg/data") / "hermes-agent"


def test_default_hermes_dir_posix_never_reuses_windows_shape():
    linux = versioning._default_hermes_dir("linux", {}, _HOME)
    darwin = versioning._default_hermes_dir("darwin", {}, _HOME)
    windows_shape = _HOME / "Documents" / "hermes-agent"
    assert linux != windows_shape
    assert darwin != windows_shape


@pytest.mark.parametrize("platform_name", ["win32", "darwin", "linux", "freebsd13"])
def test_explicit_override_wins_on_every_platform(platform_name):
    override = "/explicit/hermes-checkout"
    result = versioning._default_hermes_dir(
        platform_name, {"MS4_HERMES_DIR": override}, _HOME
    )
    assert result == Path(override).expanduser()


def test_explicit_override_takes_priority_over_xdg():
    result = versioning._default_hermes_dir(
        "linux",
        {"MS4_HERMES_DIR": "/explicit/hermes", "XDG_DATA_HOME": "/xdg/data"},
        _HOME,
    )
    assert result == Path("/explicit/hermes").expanduser()


def test_hermes_dir_wrapper_honors_env_override(tmp_path, monkeypatch):
    override = tmp_path / "operator-chosen-hermes"
    monkeypatch.setenv("MS4_HERMES_DIR", str(override))
    assert versioning.hermes_dir() == Path(str(override)).expanduser()


def test_hermes_dir_wrapper_uses_platform_default_when_unset(monkeypatch):
    monkeypatch.delenv("MS4_HERMES_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    # Compare against the same helper bound to the live platform so the
    # assertion matches the branch this host actually takes, without asserting
    # a foreign OS's shape.
    result = versioning.hermes_dir()
    expected = versioning._default_hermes_dir(sys.platform, {}, Path.home())
    assert result == expected
