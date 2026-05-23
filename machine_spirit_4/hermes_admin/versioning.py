"""Hermes version detection + GitHub release awareness.

Mirrors HiveMind ``ollama_admin`` semver compare, cached latest-release
lookup, and ``is_safe_target_version`` shell-injection guard. The shape
is intentionally identical so the MS4 UI and HiveMind UI behave the
same way for the operator.

There is no vendored copy of Hermes anywhere in TMR. "Current version"
is read straight out of the active install (``importlib.metadata`` or
``pip show``), "latest version" is fetched from the public
``api.github.com/repos/NousResearch/hermes-agent/releases`` API, and
upgrades go through the operator's normal package manager (pip).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


GITHUB_REPO = "NousResearch/hermes-agent"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RECENT_RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases?per_page=20"
PYPI_LATEST_URL = "https://pypi.org/pypi/hermes-agent/json"

USER_AGENT = "MS4-HermesAdmin/1.0"
HTTP_TIMEOUT = 8
LATEST_TTL_SECS = 3600  # GitHub rate limits; Hermes releases ship every ~1-2 weeks.
RECENT_TTL_SECS = 3600

_SEMVER_RE = re.compile(
    r"^\s*v?(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?"
    r"(?:[-+][\w.\-]+)?\s*$"
)
_SAFE_VERSION_RE = re.compile(r"^[0-9A-Za-z._\-]{1,32}$")

# Hermes ships TWO version strings per release. pyproject (and `pip show`)
# carries the wheel version like `0.14.0`; the GitHub tag and CHANGELOG use
# a date-style alias like `v2026.5.16`. Wheel version is the authoritative
# value for semver compare (so `0.14.0 == 0.14.0` doesn't get tricked into
# claiming an update against a calendar tag). Release `name` fields look
# like `Hermes Agent v0.14.0 (v2026.5.16)`; extract the leading wheel
# version from there.
_RELEASE_NAME_VERSION_RE = re.compile(r"v?(\d+\.\d+\.\d+(?:-[\w.\-]+)?)")


@dataclass(frozen=True)
class CachedLatest:
    version: str
    published_at: str
    tag_name: str
    html_url: str
    fetched_at_unix: float


@dataclass(frozen=True)
class RecentRelease:
    version: str
    published_at: str
    prerelease: bool
    tag_name: str
    html_url: str


@dataclass
class _Cache:
    latest: CachedLatest | None = None
    recent: list[RecentRelease] = field(default_factory=list)
    recent_fetched_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_CACHE = _Cache()


def parse_semver(value: str | None) -> tuple[int, int, int] | None:
    """Parse a semver-ish string into a comparable ``(major, minor, patch)``.

    Strips a leading ``v`` and any pre-release/build suffix. Returns
    ``None`` if the string is not parseable so the caller treats it as
    "no opinion" rather than asserting an update is needed.
    """
    if not value:
        return None
    match = _SEMVER_RE.match(value)
    if not match:
        return None
    major = int(match.group("major"))
    minor = int(match.group("minor") or "0")
    patch = int(match.group("patch") or "0")
    return major, minor, patch


def update_available(current: str | None, latest: str | None) -> bool:
    """Strict ``current < latest``. Anything ambiguous returns False."""
    c = parse_semver(current)
    l = parse_semver(latest)
    if c is None or l is None:
        return False
    return c < l


def is_safe_target_version(value: str) -> bool:
    """Whitelist guard for ``target_version`` strings used in upgrade
    commands. Refuses anything that could be interpreted by the shell
    or by ``pip install`` as a marker expression.
    """
    if not value:
        return False
    if len(value) > 32:
        return False
    if not _SAFE_VERSION_RE.match(value):
        return False
    if any(c in value for c in (";", "&", "|", "`", "$", " ", "\n", "\t", "<", ">", "(", ")", "'", '"')):
        return False
    return True


def hermes_dir() -> Path:
    """Return the active Hermes checkout dir, honoring ``MS4_HERMES_DIR``."""
    raw = os.environ.get("MS4_HERMES_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Documents" / "hermes-agent"


def install_mode() -> dict[str, Any]:
    """Detect how Hermes is installed in the active interpreter.

    Returns ``{"mode": "editable", "directory": Path, ...}`` if Hermes
    is installed via ``pip install -e`` against a local checkout, or
    ``{"mode": "pypi", ...}`` when installed as a regular wheel.
    Returns ``{"mode": "missing", ...}`` if Hermes cannot be found at
    all. ``mode == "unknown"`` covers the rare case where the metadata
    record exists but neither shape matches (e.g. zip install).
    """
    try:
        import importlib.metadata as md

        dists = list(md.distributions(name="hermes-agent"))
        if not dists:
            dist = md.distribution("hermes-agent")
            dists = [dist]
    except Exception as exc:
        return {"mode": "missing", "error": str(exc), "directory": None}

    # A venv can legitimately carry both a wheel dist-info and an editable
    # direct_url record after upgrade/reinstall churn. `md.distribution()`
    # returns the first match on sys.path, which may be the wheel. Prefer an
    # editable record if any matching distribution has one.
    dist = _prefer_editable_distribution(dists)
    version = dist.version
    location: Path | None = None
    direct_url: dict[str, Any] | None = None
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        raw = None
    if raw:
        try:
            direct_url = json.loads(raw)
        except json.JSONDecodeError:
            direct_url = None

    editable = False
    if direct_url and isinstance(direct_url, dict):
        dir_info = direct_url.get("dir_info") if isinstance(direct_url.get("dir_info"), dict) else None
        editable = bool(dir_info and dir_info.get("editable"))
        url = direct_url.get("url")
        if isinstance(url, str) and url.startswith("file://"):
            location = _file_url_to_path(url)

    if editable and location is None:
        location = hermes_dir()

    if editable:
        return {
            "mode": "editable",
            "version": version,
            "directory": location,
            "direct_url": direct_url,
        }

    return {
        "mode": "pypi",
        "version": version,
        "directory": None,
        "direct_url": direct_url,
    }


def _prefer_editable_distribution(dists: list[Any]) -> Any:
    for dist in dists:
        try:
            raw = dist.read_text("direct_url.json")
        except Exception:
            raw = None
        if not raw:
            continue
        try:
            direct_url = json.loads(raw)
        except json.JSONDecodeError:
            continue
        dir_info = direct_url.get("dir_info") if isinstance(direct_url.get("dir_info"), dict) else None
        if dir_info and dir_info.get("editable"):
            return dist
    return dists[0]


def _file_url_to_path(url: str) -> Path | None:
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if os.name == "nt" and path.startswith("/") and len(path) >= 3 and path[2] == ":":
        path = path[1:]
    return Path(path)


def current_version() -> str | None:
    """Active Hermes version in the contained venv. ``None`` if missing."""
    info = install_mode()
    return info.get("version") if info.get("mode") != "missing" else None


def _http_get_json(url: str, timeout: int = HTTP_TIMEOUT) -> Any | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            data = response.read().decode("utf-8", "replace")
            return json.loads(data)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        return None
    except Exception:
        return None


def _wheel_version_from_release(payload: dict[str, Any]) -> str | None:
    """Pull the pip-style wheel version out of a GitHub release payload.

    Prefers the ``name`` field (``Hermes Agent v0.14.0 (v2026.5.16)``) over
    ``tag_name`` (``v2026.5.16``) because the wheel version is what ``pip
    show hermes-agent`` returns and therefore what ``current`` will be.
    Falls back to ``tag_name`` if no semver triple is found in the name.
    """
    name = payload.get("name")
    if isinstance(name, str):
        match = _RELEASE_NAME_VERSION_RE.search(name)
        if match:
            return match.group(1)
    tag = payload.get("tag_name")
    if isinstance(tag, str):
        cleaned = tag.lstrip("v")
        return cleaned or None
    return None


def latest_version(force_refresh: bool = False) -> CachedLatest | None:
    """Fetch the latest Hermes release. Cached for ``LATEST_TTL_SECS``.

    Uses GitHub Releases as the primary source for metadata
    (``published_at``, ``html_url``, ``tag_name``) and extracts the
    pip-style wheel version from the release ``name`` so the
    ``current`` (from ``pip show``) vs ``latest`` compare stays inside
    the same versioning space. Falls back to PyPI JSON when GitHub is
    unreachable.
    """
    now = time.time()
    with _CACHE.lock:
        cached = _CACHE.latest
        if cached and not force_refresh and (now - cached.fetched_at_unix) < LATEST_TTL_SECS:
            return cached

    payload = _http_get_json(LATEST_RELEASE_URL)
    if isinstance(payload, dict):
        version = _wheel_version_from_release(payload)
        tag = str(payload.get("tag_name") or (f"v{version}" if version else ""))
        if version:
            cached = CachedLatest(
                version=version,
                published_at=str(payload.get("published_at") or ""),
                tag_name=tag,
                html_url=str(payload.get("html_url") or ""),
                fetched_at_unix=now,
            )
            with _CACHE.lock:
                _CACHE.latest = cached
            return cached

    pypi_payload = _http_get_json(PYPI_LATEST_URL)
    if isinstance(pypi_payload, dict):
        info = pypi_payload.get("info") if isinstance(pypi_payload.get("info"), dict) else {}
        version = (info.get("version") if isinstance(info, dict) else None) or None
        if version:
            cached = CachedLatest(
                version=str(version),
                published_at="",
                tag_name=f"v{version}",
                html_url=f"https://pypi.org/project/hermes-agent/{version}/",
                fetched_at_unix=now,
            )
            with _CACHE.lock:
                _CACHE.latest = cached
            return cached

    with _CACHE.lock:
        return _CACHE.latest


def recent_releases(force_refresh: bool = False) -> list[RecentRelease]:
    """Recent Hermes releases for the "pin a version" dropdown.

    Returns wheel-version strings (``0.14.0``) the operator can pick from,
    each carrying the corresponding git ``tag_name`` (``v2026.5.16``) so
    the installer can run ``git checkout`` in editable mode without
    re-fetching the release list.
    """
    now = time.time()
    with _CACHE.lock:
        cached = list(_CACHE.recent)
        if cached and not force_refresh and (now - _CACHE.recent_fetched_at) < RECENT_TTL_SECS:
            return cached

    payload = _http_get_json(RECENT_RELEASES_URL)
    if not isinstance(payload, list):
        with _CACHE.lock:
            return list(_CACHE.recent)

    out: list[RecentRelease] = []
    for entry in payload[:20]:
        if not isinstance(entry, dict):
            continue
        tag = str(entry.get("tag_name") or "")
        version = _wheel_version_from_release(entry) or (tag.lstrip("v") if tag else "")
        if not version or not is_safe_target_version(version):
            continue
        if tag and not is_safe_target_version(tag.lstrip("v")):
            continue
        out.append(
            RecentRelease(
                version=version,
                published_at=str(entry.get("published_at") or ""),
                prerelease=bool(entry.get("prerelease")),
                tag_name=tag or f"v{version}",
                html_url=str(entry.get("html_url") or ""),
            )
        )
    with _CACHE.lock:
        _CACHE.recent = out
        _CACHE.recent_fetched_at = now
    return list(out)


def resolve_git_tag(target_version: str) -> str | None:
    """Look up the git tag that pairs with a wheel ``target_version``.

    Hermes ships pip-style wheel versions and date-style git tags from
    the same release; the version the operator sees is the wheel value,
    but ``git checkout`` requires the tag. This consults the cached
    release list, falling back to ``v<target_version>`` when the cache
    doesn't yet know about the requested wheel version.
    """
    if not target_version:
        return None
    cleaned = target_version.lstrip("v")
    if cleaned in {r.version for r in _CACHE.recent}:
        match = next(r for r in _CACHE.recent if r.version == cleaned)
        return match.tag_name or f"v{cleaned}"
    latest = _CACHE.latest
    if latest and latest.version == cleaned:
        return latest.tag_name or f"v{cleaned}"
    return f"v{cleaned}"


def git_describe(directory: Path) -> str | None:
    """Return ``git describe --tags`` for the editable Hermes checkout."""
    if not directory or not Path(directory).exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def version_info(force_refresh_latest: bool = False) -> dict[str, Any]:
    """One-call snapshot for the gateway/MCP/UI: current, latest, mode, etc."""
    from .state import last_update, update_in_progress

    mode = install_mode()
    current = mode.get("version")
    latest = latest_version(force_refresh=force_refresh_latest)
    last = last_update()

    latest_str = latest.version if latest else None
    return {
        "schema": "Ms4HermesVersion.v1",
        "current": current,
        "latest": latest_str,
        "update_available": update_available(current, latest_str),
        "update_in_progress": update_in_progress(),
        "latest_published_at": latest.published_at if latest else None,
        "latest_release_url": latest.html_url if latest else None,
        "install_mode": mode.get("mode"),
        "install_directory": str(mode.get("directory")) if mode.get("directory") else None,
        "git_describe": git_describe(mode["directory"]) if mode.get("directory") else None,
        "last_update": last.to_dict() if last else None,
        "source_repo": GITHUB_REPO,
    }


def _clear_caches_for_test() -> None:
    """Test-only helper to drop cached latest/recent."""
    with _CACHE.lock:
        _CACHE.latest = None
        _CACHE.recent = []
        _CACHE.recent_fetched_at = 0.0
