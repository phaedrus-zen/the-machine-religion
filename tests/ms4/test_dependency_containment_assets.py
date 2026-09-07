from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MS4 = ROOT / "machine_spirit_4"


def test_dependency_manifests_define_core_and_optional_groups():
    core = (MS4 / "requirements.txt").read_text(encoding="utf-8")
    full = (MS4 / "requirements-full-hermes.txt").read_text(encoding="utf-8")
    lock = json.loads((MS4 / "deps.lock.json").read_text(encoding="utf-8"))

    assert "C:/Users/" not in full
    assert "websockets" in full
    assert "playwright" in full
    assert "mss" in full
    assert "pyautogui" in full
    assert "pygetwindow" in full
    assert lock["schema"] == "Ms4DependencyLock.v1"
    assert "browser" in lock["capability_groups"]
    assert "desktop" in lock["capability_groups"]
    assert "hermes" in lock["capability_groups"]
    assert lock["capability_groups"]["hermes"]["editable_env"] == "MS4_HERMES_DIR"
    assert "jsonschema" in core


def test_setup_and_check_scripts_exist_with_expected_contracts():
    setup_py = (MS4 / "scripts" / "setup_ms4_runtime.py").read_text(encoding="utf-8")
    runtime_common = (MS4 / "scripts" / "runtime_common.py").read_text(encoding="utf-8")
    check_py = (MS4 / "scripts" / "check_ms4_deps.py").read_text(encoding="utf-8")

    assert ".venv" in runtime_common
    assert "venv_python" in setup_py
    assert "requirements-full-hermes.txt" in setup_py
    assert "runtime_manifest.json" in setup_py
    assert "playwright" in check_py
    assert "pyautogui" in check_py
    assert "mss" in check_py
    assert "desktop_control" in check_py


def test_runtime_env_pins_hermes_home():
    """The June 8 2026 Hermes upgrade moved the Windows platform-native
    data home to %LOCALAPPDATA%/hermes, orphaning the real config.yaml
    (with plugins.enabled: [ms4_consciousness]) in ~/.hermes and
    breaking every Depth Lobe job with HermesUnavailable. Hermes issue
    #18594: subprocess spawners must propagate HERMES_HOME explicitly.
    ms4_env() must therefore pin HERMES_HOME (default ~/.hermes,
    env-overridable) so the gateway, MCP, workers, and validators all
    resolve the same Hermes data home."""
    runtime_common = (MS4 / "scripts" / "runtime_common.py").read_text(encoding="utf-8")

    assert "def hermes_home" in runtime_common
    assert 'env.setdefault("HERMES_HOME", str(hermes_home()))' in runtime_common
    assert '".hermes"' in runtime_common


def _direct_gateway_bootstrap_snapshot(
    tmp_path: Path,
    *,
    hermes_home_override: Path | None = None,
) -> dict[str, object]:
    user_profile = tmp_path / "profile"
    hermes_home = hermes_home_override or user_profile / ".hermes"
    local_hermes_home = tmp_path / "local-app-data" / "hermes"
    hermes_home.mkdir(parents=True)
    local_hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - ms4_consciousness\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.pop("HERMES_HOME", None)
    env.pop("MS4_HERMES_DIR", None)
    if hermes_home_override is not None:
        env["HERMES_HOME"] = str(hermes_home_override)
    env["USERPROFILE"] = str(user_profile)
    env["HOME"] = str(user_profile)
    env["LOCALAPPDATA"] = str(local_hermes_home.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH")) if part
    )
    probe = (
        "import json, os\n"
        "from machine_spirit_4.gateway.server import build_runner\n"
        "snapshot = {\n"
        "    'hermes_home': os.environ.get('HERMES_HOME'),\n"
        "    'plugin': build_runner().health()['plugin'],\n"
        "}\n"
        "print('MS4_PLUGIN_BOOTSTRAP=' + json.dumps(snapshot, sort_keys=True))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    prefix = "MS4_PLUGIN_BOOTSTRAP="
    snapshots = [
        line.removeprefix(prefix)
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert snapshots, completed.stderr
    return json.loads(snapshots[-1])


def test_direct_gateway_bootstrap_pins_user_hermes_home_and_loads_plugin(tmp_path: Path):
    hermes_home = tmp_path / "profile" / ".hermes"
    snapshot = _direct_gateway_bootstrap_snapshot(tmp_path)

    assert (
        snapshot["hermes_home"] == str(hermes_home)
        and snapshot["plugin"]["found"] is True
        and snapshot["plugin"]["enabled"] is True
        and len(snapshot["plugin"]["hooks"]) == 5
    ), snapshot


def test_direct_gateway_bootstrap_preserves_explicit_hermes_home(tmp_path: Path):
    hermes_home = tmp_path / "explicit-hermes-home"
    snapshot = _direct_gateway_bootstrap_snapshot(
        tmp_path,
        hermes_home_override=hermes_home,
    )

    assert (
        snapshot["hermes_home"] == str(hermes_home)
        and snapshot["plugin"]["found"] is True
        and snapshot["plugin"]["enabled"] is True
        and len(snapshot["plugin"]["hooks"]) == 5
    ), snapshot


def test_runtime_env_uses_canonical_hivemind_url_with_legacy_alias():
    runtime_common = (MS4 / "scripts" / "runtime_common.py").read_text(encoding="utf-8")
    warden_service = json.loads((MS4 / "warden_service.json").read_text(encoding="utf-8"))

    assert '"MS4_HIVEMIND_URL"' in runtime_common
    assert '"MS4_HIVEMIND_HLI_URL"' in runtime_common
    assert 'env.setdefault("MS4_DEFAULT_MODEL", "nemotron-3-nano:4b")' in runtime_common
    assert 'env.setdefault("MS4_DEPTH_FALLBACK_MODEL", "nemotron-3-nano:30b")' in runtime_common
    assert warden_service["environment"]["MS4_HIVEMIND_URL"] == "http://127.0.0.1:6089"
    assert warden_service["environment"]["MS4_HIVEMIND_MCP_URL"] == "http://127.0.0.1:6105/mcp"


def test_launchers_require_contained_python():
    for script_name in ("run_ms4_gateway.py", "run_ms4_mcp.py", "start_ms4.py"):
        script = (MS4 / "scripts" / script_name).read_text(encoding="utf-8")
        assert "require_venv_python" in script

    start = (MS4 / "scripts" / "start_ms4.py").read_text(encoding="utf-8")
    assert '"machine_spirit_4.gateway.server"' in start
    assert '"machine_spirit_4.mcp.server"' in start
    assert 'ROOT / "machine_spirit_4" / "scripts" / "run_ms4_gateway.py"' not in start
    assert 'ROOT / "machine_spirit_4" / "scripts" / "run_ms4_mcp.py"' not in start


def test_supervisor_is_contained_python_watchdog():
    supervise = (MS4 / "scripts" / "supervise_ms4.py").read_text(encoding="utf-8")
    runtime_common = (MS4 / "scripts" / "runtime_common.py").read_text(encoding="utf-8")

    assert "venv_python" in supervise
    assert "is_port_listening" in supervise
    assert "(9080, 9180, 9181)" in supervise
    assert "--skip-validation" in supervise
    assert "supervisor" in supervise
    assert 'parser.add_argument("--watch"' in supervise
    assert "HiveMindMS4UserWatchdog" in supervise
    assert "MS4_WATCH_INTERVAL_SECS" in supervise
    assert 'getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)' in supervise
    assert "close_fds=True" in runtime_common


def test_ms4_autostart_story_is_scheduled_supervisor_not_warden():
    """Services and the visible Oracle window have explicit owners."""
    readme = (MS4 / "README.md").read_text(encoding="utf-8")
    warden_service = json.loads((MS4 / "warden_service.json").read_text(encoding="utf-8"))

    assert "MS4-Spirit-AutoStart" in readme
    assert "scripts/supervise_ms4.py" in readme
    assert "--watch --interval-seconds 300" in readme
    assert "user context" in readme
    assert "HiveMind Oracle.lnk" in readme
    assert "reference-only" in readme
    decision = warden_service["_autostart_decision"]
    assert "REFERENCE ONLY" in decision
    assert "MS4-Spirit-AutoStart" in decision
    assert "supervise_ms4.py" in decision
    assert "--watch" in decision
    assert "HiveMind Oracle.lnk" in decision


def test_runtime_scripts_are_cross_platform_python_entrypoints():
    forbidden_suffixes = {".ps1", ".sh", ".bat", ".cmd"}
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in forbidden_suffixes
        and ".git" not in path.parts
        and ".venv" not in path.parts
    ]

    assert offenders == []


def test_gateway_and_mcp_expose_dependency_status_assets():
    gateway = (MS4 / "gateway" / "server.py").read_text(encoding="utf-8")
    mcp_tools = (MS4 / "mcp" / "tools.py").read_text(encoding="utf-8")

    assert '"/deps/status"' in gateway
    assert "ms4.runtime.deps.status@v1" in mcp_tools


def test_fusion_smoke_defaults_to_resident_face_model():
    validator = (MS4 / "scripts" / "validate_ms4_fusion.py").read_text(encoding="utf-8")
    mcp_validator = (MS4 / "scripts" / "validate_ms4_mcp.py").read_text(encoding="utf-8")

    assert 'MS4_FUSION_TEST_MODEL", "llama3.1:8b"' in validator
    assert 'MS4_FUSION_TEST_MODEL", "qwen3-coder-next:latest"' not in validator
    assert '"depth_model_id": TEST_MODEL' in validator
    assert "ms4_chat_deep_completion" in validator
    assert "MS4_FUSION_CONTEXT_OK" in validator
    assert 'MS4_MCP_TEST_MODEL", "llama3.1:8b"' in mcp_validator
    assert '"model": "qwen3-coder-next:latest"' not in mcp_validator
    assert 'chat_payload.get("completed") is True' in mcp_validator
