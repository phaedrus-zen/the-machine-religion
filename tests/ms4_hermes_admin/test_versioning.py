"""Versioning math + safety guard tests.

Mirrors the test coverage in HiveMind's ``ollama_admin.rs`` so MS4
behaves identically to the operator: strict semver compare, shell-safe
target_version guard, GitHub release fetch that survives rate limits.
"""

from __future__ import annotations

from machine_spirit_4.hermes_admin import versioning


def test_parse_semver_basic_and_optional_parts():
    assert versioning.parse_semver("0.13.0") == (0, 13, 0)
    assert versioning.parse_semver("v0.14.0") == (0, 14, 0)
    assert versioning.parse_semver("0.14") == (0, 14, 0)
    assert versioning.parse_semver("1") == (1, 0, 0)
    assert versioning.parse_semver(" 1.2.3 ") == (1, 2, 3)


def test_parse_semver_strips_prerelease_suffix():
    assert versioning.parse_semver("0.14.0-rc1") == (0, 14, 0)
    assert versioning.parse_semver("0.14.0+build.42") == (0, 14, 0)
    assert versioning.parse_semver("v1.0.0-alpha") == (1, 0, 0)


def test_parse_semver_rejects_nonsense():
    assert versioning.parse_semver("") is None
    assert versioning.parse_semver(None) is None  # type: ignore[arg-type]
    assert versioning.parse_semver("not-a-version") is None
    assert versioning.parse_semver("0.x.y") is None


def test_update_available_strict_compare():
    assert versioning.update_available("0.13.0", "0.14.0") is True
    assert versioning.update_available("0.14.0-rc1", "0.14.0") is False
    assert versioning.update_available("0.14.0", "0.14.0") is False
    assert versioning.update_available("0.15.0", "0.14.0") is False
    assert versioning.update_available(None, "0.14.0") is False
    assert versioning.update_available("0.13.0", None) is False
    assert versioning.update_available(None, None) is False


def test_is_safe_target_version_accepts_real_releases():
    for value in ["0.14.0", "0.14.0-rc1", "2026.5.16", "v2026.5.16", "1.0.0-alpha"]:
        assert versioning.is_safe_target_version(value), value


def test_is_safe_target_version_rejects_shell_injection_attempts():
    bad = [
        "",
        " ",
        "0.14.0; rm -rf /",
        "0.14.0 && touch /tmp/pwn",
        "0.14.0\nls",
        "$(whoami)",
        "`whoami`",
        "0.14.0 | nc evil 1234",
        "0.14.0 > /dev/null",
        "0.14.0,extras",
        "a" * 33,
    ]
    for value in bad:
        assert not versioning.is_safe_target_version(value), value


def test_install_mode_returns_known_shape():
    info = versioning.install_mode()
    assert info["mode"] in {"editable", "pypi", "missing"}
    if info["mode"] != "missing":
        assert isinstance(info["version"], str) and info["version"]


def test_recent_releases_filters_unsafe_versions(monkeypatch):
    versioning._clear_caches_for_test()
    payload = [
        {"tag_name": "v0.14.0", "published_at": "2026-05-16T00:00:00Z", "prerelease": False, "html_url": "https://example/0.14.0"},
        {"tag_name": "v0.13.0", "published_at": "2026-05-07T00:00:00Z", "prerelease": False, "html_url": "https://example/0.13.0"},
        {"tag_name": "v0.14.0; rm -rf /", "published_at": "2026-05-18T00:00:00Z", "prerelease": False, "html_url": "https://example/evil"},
    ]
    monkeypatch.setattr(versioning, "_http_get_json", lambda url, timeout=8: payload)
    releases = versioning.recent_releases(force_refresh=True)
    versions = [r.version for r in releases]
    assert "0.14.0" in versions
    assert "0.13.0" in versions
    assert all(versioning.is_safe_target_version(r.version) for r in releases)


def test_latest_version_falls_back_to_pypi_when_github_unreachable(monkeypatch):
    versioning._clear_caches_for_test()
    calls: list[str] = []

    def fake_get(url, timeout=8):
        calls.append(url)
        if url == versioning.LATEST_RELEASE_URL:
            return None
        if url == versioning.PYPI_LATEST_URL:
            return {"info": {"version": "0.14.0"}}
        raise AssertionError(url)

    monkeypatch.setattr(versioning, "_http_get_json", fake_get)
    latest = versioning.latest_version(force_refresh=True)
    assert latest is not None
    assert latest.version == "0.14.0"
    assert versioning.PYPI_LATEST_URL in calls


def test_version_info_self_consistent(monkeypatch):
    versioning._clear_caches_for_test()
    fake_payload = {
        "tag_name": "v0.14.0",
        "name": "Hermes Agent v0.14.0",
        "published_at": "2026-05-16T00:00:00Z",
        "html_url": "https://example/release",
    }
    monkeypatch.setattr(
        versioning,
        "_http_get_json",
        lambda url, timeout=8: fake_payload if url == versioning.LATEST_RELEASE_URL else None,
    )
    info = versioning.version_info(force_refresh_latest=True)
    assert info["schema"] == "Ms4HermesVersion.v1"
    assert info["latest"] == "0.14.0"
    assert info["source_repo"] == versioning.GITHUB_REPO
    assert "install_mode" in info
    assert "update_available" in info
    assert info["update_in_progress"] is False or info["update_in_progress"] is True


def test_latest_version_prefers_wheel_version_from_release_name(monkeypatch):
    """Hermes ships pip version (`0.14.0`) and date-style git tag
    (`v2026.5.16`) for the same release. ``latest_version().version``
    must return the pip-style value so comparing against ``current``
    (which comes from ``pip show``) doesn't lie."""
    versioning._clear_caches_for_test()
    payload = {
        "tag_name": "v2026.5.16",
        "name": "Hermes Agent v0.14.0 (v2026.5.16)",
        "published_at": "2026-05-16T09:59:15Z",
        "html_url": "https://github.com/NousResearch/hermes-agent/releases/tag/v2026.5.16",
    }
    monkeypatch.setattr(
        versioning,
        "_http_get_json",
        lambda url, timeout=8: payload if url == versioning.LATEST_RELEASE_URL else None,
    )
    latest = versioning.latest_version(force_refresh=True)
    assert latest is not None
    assert latest.version == "0.14.0"
    assert latest.tag_name == "v2026.5.16"
    assert versioning.update_available("0.14.0", latest.version) is False
    assert versioning.update_available("0.13.0", latest.version) is True


def test_version_info_does_not_advertise_unsigned_newer_release(monkeypatch):
    """GitHub can publish a newer wheel while the official tag stays unsigned.

    Oracle Retry/Pin must not treat that as an authorized update. Presence of
    signature bytes on the official annotated tag object is required before
    ``update_available`` becomes true; F4 still verifies the signer later.
    """
    versioning._clear_caches_for_test()
    monkeypatch.setattr(
        versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": None,
            "direct_url": None,
        },
    )
    monkeypatch.setattr(
        "machine_spirit_4.hermes_admin.state.last_update",
        lambda refresh=True: None,
    )

    def fake_get(url, timeout=8):
        if url == versioning.LATEST_RELEASE_URL:
            return {
                "tag_name": "v2026.8.18",
                "name": "Hermes Agent v0.20.4 (2026.8.18)",
                "published_at": "2026-08-18T07:26:46Z",
                "html_url": "https://github.com/NousResearch/hermes-agent/releases/tag/v2026.8.18",
            }
        if url.endswith("/git/refs/tags/v2026.8.18"):
            return {
                "object": {
                    "sha": "9f13bbbf8423427e159c78066356ca0e27ca6b74",
                    "type": "tag",
                }
            }
        if url.endswith("/git/tags/9f13bbbf8423427e159c78066356ca0e27ca6b74"):
            return {
                "tag": "v2026.8.18",
                "message": "Hermes Agent v0.20.4 (2026.8.18)\n\nRollup patch.\n",
                "object": {
                    "sha": "e624e9fde561e1add9388384012b295fde669ade",
                    "type": "commit",
                },
                "verification": {
                    "verified": False,
                    "reason": "unsigned",
                    "signature": None,
                },
            }
        return None

    monkeypatch.setattr(versioning, "_http_get_json", fake_get)
    info = versioning.version_info(force_refresh_latest=True)
    assert info["latest"] == "0.20.4"
    assert info["latest_tag"] == "v2026.8.18"
    assert info["latest_signature_state"] == "unsigned"
    assert info["update_available"] is False
    assert info["update_blocked_reason"] == "official_tag_unsigned"


def test_version_info_advertises_newer_release_only_when_tag_has_signature_bytes(
    monkeypatch,
):
    versioning._clear_caches_for_test()
    monkeypatch.setattr(
        versioning,
        "install_mode",
        lambda: {
            "mode": "editable",
            "version": "0.20.0",
            "directory": None,
            "direct_url": None,
        },
    )
    monkeypatch.setattr(
        "machine_spirit_4.hermes_admin.state.last_update",
        lambda refresh=True: None,
    )

    def fake_get(url, timeout=8):
        if url == versioning.LATEST_RELEASE_URL:
            return {
                "tag_name": "v2026.8.3",
                "name": "Hermes Agent v0.20.1 (2026.8.3)",
                "published_at": "2026-08-03T16:57:52Z",
                "html_url": "https://github.com/NousResearch/hermes-agent/releases/tag/v2026.8.3",
            }
        if url.endswith("/git/refs/tags/v2026.8.3"):
            return {
                "object": {
                    "sha": "7de39e700d2c329e15d32eb0b96e2f7cdd9fbdb2",
                    "type": "tag",
                }
            }
        if url.endswith("/git/tags/7de39e700d2c329e15d32eb0b96e2f7cdd9fbdb2"):
            return {
                "tag": "v2026.8.3",
                "message": (
                    "Hermes Agent v0.20.1 (2026.8.3)\n"
                    "-----BEGIN SSH SIGNATURE-----\n"
                    "U1NIU0lH\n"
                    "-----END SSH SIGNATURE-----\n"
                ),
                "object": {
                    "sha": "3c27eb6234bf91b8ceee9e9071591b31e9b148cb",
                    "type": "commit",
                },
                "verification": {
                    "verified": True,
                    "reason": "valid",
                    "signature": "-----BEGIN SSH SIGNATURE-----\nU1NIU0lH\n-----END SSH SIGNATURE-----\n",
                },
            }
        return None

    monkeypatch.setattr(versioning, "_http_get_json", fake_get)
    info = versioning.version_info(force_refresh_latest=True)
    assert info["latest"] == "0.20.1"
    assert info["latest_signature_state"] == "signed"
    assert info["update_available"] is True
    assert info["update_blocked_reason"] is None


def test_recent_releases_marks_unsigned_official_tags(monkeypatch):
    versioning._clear_caches_for_test()

    def fake_get(url, timeout=8):
        if url == versioning.RECENT_RELEASES_URL:
            return [
                {
                    "tag_name": "v2026.8.18",
                    "name": "Hermes Agent v0.20.4 (2026.8.18)",
                    "published_at": "2026-08-18T07:26:46Z",
                    "prerelease": False,
                    "html_url": "https://example/0.20.4",
                },
                {
                    "tag_name": "v2026.8.3",
                    "name": "Hermes Agent v0.20.0 (2026.8.3)",
                    "published_at": "2026-08-03T16:57:52Z",
                    "prerelease": False,
                    "html_url": "https://example/0.20.0",
                },
            ]
        if url.endswith("/git/refs/tags/v2026.8.18"):
            return {
                "object": {
                    "sha": "9f13bbbf8423427e159c78066356ca0e27ca6b74",
                    "type": "tag",
                }
            }
        if url.endswith("/git/tags/9f13bbbf8423427e159c78066356ca0e27ca6b74"):
            return {
                "tag": "v2026.8.18",
                "message": "Hermes Agent v0.20.4 (2026.8.18)\n",
                "object": {"sha": "e624e9fde561e1add9388384012b295fde669ade", "type": "commit"},
            }
        if url.endswith("/git/refs/tags/v2026.8.3"):
            return {
                "object": {
                    "sha": "7de39e700d2c329e15d32eb0b96e2f7cdd9fbdb2",
                    "type": "tag",
                }
            }
        if url.endswith("/git/tags/7de39e700d2c329e15d32eb0b96e2f7cdd9fbdb2"):
            return {
                "tag": "v2026.8.3",
                "message": "Hermes Agent v0.20.0 (2026.8.3)\n-----BEGIN SSH SIGNATURE-----\nU1NI\n-----END SSH SIGNATURE-----\n",
                "object": {"sha": "3c27eb6234bf91b8ceee9e9071591b31e9b148cb", "type": "commit"},
            }
        return None

    monkeypatch.setattr(versioning, "_http_get_json", fake_get)
    releases = versioning.recent_releases(force_refresh=True)
    by_version = {item.version: item for item in releases}
    assert by_version["0.20.4"].signature_state == "unsigned"
    assert by_version["0.20.0"].signature_state == "signed"


def test_resolve_git_tag_pairs_wheel_version_to_real_tag(monkeypatch):
    """`pip install` wants `0.14.0`; `git checkout` wants `v2026.5.16`.
    `resolve_git_tag` must look up the real tag from the recent release
    list when the operator picks a wheel version."""
    versioning._clear_caches_for_test()
    payload = [
        {
            "tag_name": "v2026.5.16",
            "name": "Hermes Agent v0.14.0 (v2026.5.16)",
            "published_at": "2026-05-16T00:00:00Z",
            "prerelease": False,
            "html_url": "https://example/0.14.0",
        },
        {
            "tag_name": "v2026.5.7",
            "name": "Hermes Agent v0.13.0 (v2026.5.7)",
            "published_at": "2026-05-07T00:00:00Z",
            "prerelease": False,
            "html_url": "https://example/0.13.0",
        },
    ]
    monkeypatch.setattr(versioning, "_http_get_json", lambda url, timeout=8: payload)
    versioning.recent_releases(force_refresh=True)
    assert versioning.resolve_git_tag("0.14.0") == "v2026.5.16"
    assert versioning.resolve_git_tag("0.13.0") == "v2026.5.7"
    assert versioning.resolve_git_tag("0.15.0") == "v0.15.0"  # fallback when unknown
