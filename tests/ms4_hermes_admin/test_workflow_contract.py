"""Static contract checks for the scoped MS4 Hermes CI workflow."""

from __future__ import annotations

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ms4-hermes-admin.yml"

EXPECTED_RUNNERS = {
    "ubuntu-24.04": ("linux", "x64", "Linux", "X64", "x64"),
    "ubuntu-24.04-arm": ("linux", "arm64", "Linux", "ARM64", "arm64"),
    "windows-2025": ("windows", "x64", "Windows", "X64", "x64"),
    "windows-11-arm": ("windows", "arm64", "Windows", "ARM64", "arm64"),
    "macos-15-intel": ("macos", "x64", "macOS", "X64", "x64"),
    "macos-15": ("macos", "arm64", "macOS", "ARM64", "arm64"),
}


def _load_workflow():
    # BaseLoader preserves GitHub's `on` key instead of applying YAML 1.1 bools.
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_hosted_matrix_covers_supported_python_os_and_arch_cross_product():
    workflow = _load_workflow()
    job = workflow["jobs"]["hosted-contract"]
    matrix = job["strategy"]["matrix"]

    assert matrix["python_version"] == ["3.11", "3.12", "3.13"]
    assert set(matrix["runner"]) == set(EXPECTED_RUNNERS)
    assert "exclude" not in matrix

    includes = {item["runner"]: item for item in matrix["include"]}
    assert set(includes) == set(EXPECTED_RUNNERS)
    for runner, expected in EXPECTED_RUNNERS.items():
        item = includes[runner]
        actual = tuple(
            item[key]
            for key in ("os_family", "arch", "runner_os", "runner_arch", "python_arch")
        )
        assert actual == expected


def test_workflow_fails_closed_and_runs_owned_test_surfaces():
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = _load_workflow()
    hosted = workflow["jobs"]["hosted-contract"]

    assert "continue-on-error" not in workflow_text
    assert "pytest.skip" not in workflow_text
    assert "Require provenance executables" in workflow_text
    assert "Verify native matrix cell" in workflow_text
    assert "--ignore E731" in workflow_text
    for test_file in (
        "test_installer.py",
        "test_installer_r1.py",
        "test_isolation_regression_r2.py",
        "test_provenance_r1.py",
        "test_state.py",
        "test_versioning.py",
        "test_versioning_r1.py",
        "test_workflow_contract.py",
    ):
        assert f"tests/ms4_hermes_admin/{test_file}" in workflow_text
    assert "test_git_bash_preflight_candidate_r3.py" in workflow_text
    assert "--strict-config" in workflow_text
    assert "--strict-markers" in workflow_text
    assert hosted["strategy"]["fail-fast"] == "false"


def test_ci_tooling_matches_current_hermes_build_and_dev_requirements():
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    for requirement in (
        "setuptools==81.0.0",
        "pytest==9.0.2",
        "ruff==0.15.10",
        "PyYAML==6.0.3",
    ):
        assert requirement in workflow_text


def test_third_party_actions_are_commit_pinned():
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    uses = re.findall(r"uses:\s*([^\s#]+)", workflow_text)

    assert uses
    for action in uses:
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", action), action


def test_non_hosted_hardware_claim_is_an_explicit_manual_hold():
    workflow = _load_workflow()
    job = workflow["jobs"]["jetson-thor-hardware-gate"]

    assert "workflow_dispatch" in job["if"]
    assert job["runs-on"] == ["self-hosted", "linux", "ARM64", "hermes-jetson-thor"]
    assert job["strategy"]["matrix"]["python_version"] == ["3.11", "3.12", "3.13"]
    step_names = {step["name"] for step in job["steps"]}
    assert "Verify NVIDIA Tegra hardware identity" in step_names
