"""TMR-owned test for the ms4_consciousness Hermes plugin.

This test imports `plugins.ms4_consciousness` which requires the plugin to be
installed into the active Hermes checkout (see
`machine_spirit_4/scripts/setup_ms4_runtime.py`). Tests run inside MS4's
contained .venv so the import resolves through the editable Hermes install.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


def _hermes_dir() -> Path:
    import os
    return Path(os.environ.get("MS4_HERMES_DIR") or (Path.home() / "Documents" / "hermes-agent"))


PLUGIN_DIR = _hermes_dir() / "plugins" / "ms4_consciousness"
SOURCE_PLUGIN_DIR = (
    Path(__file__).resolve().parents[2]
    / "machine_spirit_4"
    / "plugins"
    / "hermes"
    / "ms4_consciousness"
)
_SOURCE_PLUGIN_MODULE = "_tmr_ms4_consciousness_source"


def _require_plugin() -> None:
    if not PLUGIN_DIR.is_dir():
        pytest.skip(f"ms4_consciousness plugin not installed at {PLUGIN_DIR}")


def _fresh_plugin():
    _require_plugin()
    for name in list(sys.modules):
        if name == "plugins.ms4_consciousness" or name.startswith("plugins.ms4_consciousness."):
            sys.modules.pop(name, None)
    return importlib.import_module("plugins.ms4_consciousness")


def _fresh_source_plugin():
    for name in list(sys.modules):
        if name == _SOURCE_PLUGIN_MODULE or name.startswith(f"{_SOURCE_PLUGIN_MODULE}."):
            sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        _SOURCE_PLUGIN_MODULE,
        SOURCE_PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(SOURCE_PLUGIN_DIR)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load source plugin at {SOURCE_PLUGIN_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SOURCE_PLUGIN_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _catalog_entry(
    name: str,
    *,
    required: tuple[str, ...] = (),
    kind: str = "hivemind_native",
    gated_by: tuple[str, ...] = (),
):
    from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

    return catalog_module.ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=name,
        toolbox="test",
        cluster="test",
        description=f"Test catalog entry for {name}",
        source="hivemind",
        kind=kind,
        input_schema={
            "type": "object",
            "properties": {key: {"type": "string"} for key in required},
            "required": list(required),
        },
        gated_by=gated_by,
    )


def _catalog(*entries, errors: tuple[str, ...] = ()):
    from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

    return catalog_module.Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version="test",
        built_at="2026-07-11T00:00:00+00:00",
        hivemind_url="http://hive:6089",
        tools=tuple(entries),
        toolboxes={"test": tuple(entries)},
        sources={"hivemind": len(entries), "ms4": 0, "external_mcp": 0},
        errors=errors,
    )


def _patch_exact_read_dependencies(monkeypatch, catalog, *, authoritative_result=None, load_error=None):
    from machine_spirit_4.gateway import hivemind_tools
    from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

    monkeypatch.setenv("MS4_HIVEMIND_URL", "http://hive:6089")
    events: list[tuple] = []

    def fake_get_catalog(url):
        events.append(("catalog", url))
        if load_error is not None:
            raise load_error
        return catalog

    def fake_call_tool(url, tool_name, arguments):
        events.append(("transport", url, tool_name, arguments))
        return authoritative_result

    monkeypatch.setattr(catalog_module, "get_catalog", fake_get_catalog)
    monkeypatch.setattr(hivemind_tools, "_call_tool", fake_call_tool)
    return events


def _successful_gated_hli_response(
    *,
    tool_name: str = "hivemind_services_enable",
    arguments: dict | None = None,
):
    arguments = arguments or {"service_name": "menta_human_bridge"}
    return {
        "response": "The approved service enable completed.",
        "session_id": "ms4-gated-test",
        "model": "llama3.1:8b",
        "tool_calls_total": 1,
        "tool_trace": [
            {
                "round": 1,
                "name": tool_name,
                "args": arguments,
                "tool_call_id": "call-gated-1",
                "duration_ms": 25,
                "result_bytes": 64,
                "success": True,
                "timestamp": "2026-07-11T07:00:00Z",
            }
        ],
    }


def _patch_exact_gated_dependencies(
    monkeypatch,
    plugin,
    catalog,
    *,
    hli_response=None,
    transport_error: Exception | None = None,
    proxy_kind: str = "runtime_action",
    gated_by: tuple[str, ...] = ("operator-level",),
):
    from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

    monkeypatch.setenv("MS4_HIVEMIND_URL", "http://hive:6089")
    monkeypatch.setattr(catalog_module, "get_catalog", lambda _url: catalog)
    canonical_names = [entry.name for entry in catalog.tools] if catalog is not None else []
    manifest_entries = [
        {
            "name": f"ms4.{name}",
            "kind": proxy_kind,
            "gated_by": list(gated_by),
        }
        for name in canonical_names
    ]
    monkeypatch.setattr(
        catalog_module,
        "_load_ms4_manifest",
        lambda: (manifest_entries, []),
    )
    events: list[tuple] = []

    def fake_post(url, payload, *, timeout):
        events.append(("hli", url, payload, timeout))
        if transport_error is not None:
            raise transport_error
        return (
            hli_response
            if hli_response is not None
            else _successful_gated_hli_response()
        )

    monkeypatch.setattr(plugin, "_post_hli_oracle_chat", fake_post)
    return events


def test_manifest_and_layout():
    _require_plugin()
    assert (PLUGIN_DIR / "plugin.yaml").is_file()
    assert (PLUGIN_DIR / "__init__.py").is_file()

    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["name"] == "ms4_consciousness"
    assert manifest["kind"] == "standalone"
    assert set(manifest["hooks"]) == {
        "on_session_start",
        "pre_llm_call",
        "pre_tool_call",
        "transform_llm_output",
        "post_llm_call",
    }


def test_registers_expected_hooks():
    plugin = _fresh_plugin()
    calls: list[str] = []

    class Context:
        def register_hook(self, name, callback):
            calls.append(name)

        def register_tool(self, **_kwargs):
            pass

    plugin.register(Context())

    assert calls == [
        "on_session_start",
        "pre_llm_call",
        "pre_tool_call",
        "transform_llm_output",
        "post_llm_call",
    ]


def test_registers_read_only_hivemind_toolset():
    plugin = _fresh_source_plugin()
    hooks: list[str] = []
    tools: list[dict] = []

    class Context:
        def register_hook(self, name, callback):
            hooks.append(name)

        def register_tool(self, **kwargs):
            tools.append(kwargs)

    plugin.register(Context())

    tools_by_toolset = {
        toolset: {tool["name"] for tool in tools if tool["toolset"] == toolset}
        for toolset in {tool["toolset"] for tool in tools}
    }
    static_wrappers = {
        "ms4_skills_list",
        "ms4_skill_view",
        "hivemind_time_now",
        "hivemind_models_list",
        "hivemind_capability_matrix",
        "hivemind_vm_list",
        "hivemind_gpu_availability",
        "hivemind_cluster_summary",
        "hivemind_hosts_list",
        "hivemind_active_jobs",
        "hivemind_cluster_load",
        "hivemind_service_health",
        "hivemind_state_snapshot",
    }
    assert tools_by_toolset == {
        "mcp-hivemind": static_wrappers,
        "mcp-hivemind-exact-read": {"hivemind_exact_read"},
        "mcp-hivemind-exact-gated": {"hivemind_exact_gated"},
    }
    for tool in tools:
        assert tool["schema"]["parameters"]["additionalProperties"] is False
    assert "skill_manage" not in tools_by_toolset["mcp-hivemind"]


def test_read_only_skill_wrappers_register_without_skill_manage():
    plugin = _fresh_source_plugin()
    tools: list[dict] = []

    class Context:
        def register_hook(self, _name, _callback):
            pass

        def register_tool(self, **kwargs):
            tools.append(kwargs)

    plugin.register(Context())

    tools_by_name = {tool["name"]: tool for tool in tools}
    assert "ms4_skills_list" in tools_by_name
    assert "ms4_skill_view" in tools_by_name
    assert "skill_manage" not in tools_by_name
    assert tools_by_name["ms4_skills_list"]["toolset"] == "mcp-hivemind"
    assert tools_by_name["ms4_skill_view"]["toolset"] == "mcp-hivemind"
    view_parameters = tools_by_name["ms4_skill_view"]["schema"]["parameters"]
    assert view_parameters["required"] == ["name"]
    assert set(view_parameters["properties"]) == {"name", "file_path"}
    assert view_parameters["additionalProperties"] is False


def test_read_only_skill_wrappers_delegate_without_preprocessing(monkeypatch):
    plugin = _fresh_source_plugin()
    from tools import skills_tool

    calls: list[tuple] = []

    def fake_list(*, category=None, task_id=None):
        calls.append(("list", category, task_id))
        return json.dumps({"success": True, "count": 1})

    def fake_view(*, name, file_path=None, task_id=None, preprocess=True):
        calls.append(("view", name, file_path, task_id, preprocess))
        return json.dumps({"success": True, "name": name, "content": "proof"})

    monkeypatch.setattr(skills_tool, "skills_list", fake_list)
    monkeypatch.setattr(skills_tool, "skill_view", fake_view)

    listed = json.loads(
        plugin._handle_ms4_skills_list(
            {"category": "shared"}, task_id="depth-job-7"
        )
    )
    viewed = json.loads(
        plugin._handle_ms4_skill_view(
            {"name": "shared:proof", "file_path": "references/evidence.md"},
            task_id="depth-job-7",
        )
    )

    assert listed == {"success": True, "count": 1}
    assert viewed == {
        "success": True,
        "name": "shared:proof",
        "content": "proof",
    }
    assert calls == [
        ("list", "shared", "depth-job-7"),
        (
            "view",
            "shared:proof",
            "references/evidence.md",
            "depth-job-7",
            False,
        ),
    ]


def test_read_only_skill_list_does_not_create_missing_profile_directory(
    monkeypatch,
    tmp_path,
):
    plugin = _fresh_source_plugin()
    from tools import skills_tool

    missing = tmp_path / "missing-skills"
    called = False

    def forbidden_list(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("native skills_list must not create the directory")

    monkeypatch.setattr(skills_tool, "_skills_dir", lambda: missing)
    monkeypatch.setattr(skills_tool, "skills_list", forbidden_list)

    result = json.loads(plugin._handle_ms4_skills_list({}))

    assert result["success"] is False
    assert "will not create" in result["error"]
    assert called is False
    assert not missing.exists()


@pytest.mark.parametrize(
    ("handler", "arguments", "expected_error"),
    [
        ("_handle_ms4_skills_list", {"category": 7}, "category"),
        ("_handle_ms4_skill_view", {}, "name"),
        ("_handle_ms4_skill_view", {"name": "proof", "file_path": ""}, "file_path"),
    ],
)
def test_read_only_skill_wrappers_reject_invalid_direct_calls(
    handler,
    arguments,
    expected_error,
):
    plugin = _fresh_source_plugin()

    result = json.loads(getattr(plugin, handler)(arguments))

    assert result["success"] is False
    assert expected_error in result["error"]


def test_exact_read_tool_registration_and_manifest_parity():
    plugin = _fresh_source_plugin()
    tools: list[dict] = []

    class Context:
        def register_hook(self, _name, _callback):
            pass

        def register_tool(self, **kwargs):
            tools.append(kwargs)

    plugin.register(Context())

    tools_by_name = {tool["name"]: tool for tool in tools}
    registered_by_toolset = {
        toolset: {tool["name"] for tool in tools if tool["toolset"] == toolset}
        for toolset in {tool["toolset"] for tool in tools}
    }
    manifest = yaml.safe_load(
        (SOURCE_PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8")
    )
    manifest_by_toolset = {
        group["toolset"]: set(group["names"])
        for group in manifest["provides_tools"]
    }

    assert "hivemind_exact_read" in tools_by_name
    assert manifest_by_toolset["mcp-hivemind-exact-read"] == {"hivemind_exact_read"}
    assert manifest_by_toolset["mcp-hivemind-exact-gated"] == {
        "hivemind_exact_gated"
    }
    assert registered_by_toolset == manifest_by_toolset

    exact_read = tools_by_name["hivemind_exact_read"]
    parameters = exact_read["schema"]["parameters"]
    assert parameters["required"] == ["canonical_tool", "arguments"]
    assert parameters["properties"]["arguments"]["type"] == "object"
    assert parameters["additionalProperties"] is False
    description = exact_read["description"].lower()
    assert "canonical" in description
    assert "read tool" in description
    assert "never substitute" in description
    assert "missing required" in description


def test_exact_read_dispatches_app_get_with_required_id(monkeypatch):
    plugin = _fresh_source_plugin()
    canonical_tool = "hivemind.app.get@v1"
    catalog = _catalog(_catalog_entry(canonical_tool, required=("id",)))
    authoritative = {"id": "app-7", "name": "Oracle"}
    events = _patch_exact_read_dependencies(
        monkeypatch,
        catalog,
        authoritative_result=authoritative,
    )
    arguments = {"id": "app-7"}

    result = json.loads(
        plugin._handle_hivemind_exact_read(
            {"canonical_tool": canonical_tool, "arguments": arguments}
        )
    )

    assert result == {
        "status": "success",
        "success": True,
        "canonical_tool": canonical_tool,
        "result": authoritative,
    }
    assert events == [
        ("catalog", "http://hive:6089"),
        ("transport", "http://hive:6089", canonical_tool, arguments),
    ]
    assert events[1][3] is arguments


def test_exact_read_reports_missing_required_keys_without_transport(monkeypatch):
    plugin = _fresh_source_plugin()
    canonical_tool = "hivemind.app.get@v1"
    catalog = _catalog(_catalog_entry(canonical_tool, required=("zone", "id")))
    events = _patch_exact_read_dependencies(monkeypatch, catalog)

    result = json.loads(
        plugin._handle_hivemind_exact_read(
            {"canonical_tool": canonical_tool, "arguments": {}}
        )
    )

    assert result == {
        "status": "missing_required_arguments",
        "success": False,
        "canonical_tool": canonical_tool,
        "missing_required_keys": ["id", "zone"],
    }
    assert events == [("catalog", "http://hive:6089")]


@pytest.mark.parametrize(
    ("arguments", "expected_missing"),
    [
        pytest.param({}, True, id="absent"),
        pytest.param({"id": None}, True, id="none"),
        pytest.param({"id": ""}, True, id="empty-string"),
        pytest.param({"id": "   "}, True, id="spaces-only"),
        pytest.param({"id": "\t\n"}, True, id="whitespace-only"),
        pytest.param({"id": 0}, False, id="numeric-zero"),
        pytest.param({"id": False}, False, id="boolean-false"),
    ],
)
def test_exact_read_required_values_distinguish_missing_from_falsey(
    monkeypatch,
    arguments,
    expected_missing,
):
    plugin = _fresh_source_plugin()
    canonical_tool = "hivemind.app.get@v1"
    catalog = _catalog(_catalog_entry(canonical_tool, required=("id",)))
    authoritative = {"accepted": arguments["id"]} if not expected_missing else None
    events = _patch_exact_read_dependencies(
        monkeypatch,
        catalog,
        authoritative_result=authoritative,
    )

    result = json.loads(
        plugin._handle_hivemind_exact_read(
            {"canonical_tool": canonical_tool, "arguments": arguments}
        )
    )

    if expected_missing:
        assert result == {
            "status": "missing_required_arguments",
            "success": False,
            "canonical_tool": canonical_tool,
            "missing_required_keys": ["id"],
        }
        assert events == [("catalog", "http://hive:6089")]
    else:
        assert result == {
            "status": "success",
            "success": True,
            "canonical_tool": canonical_tool,
            "result": authoritative,
        }
        assert events == [
            ("catalog", "http://hive:6089"),
            ("transport", "http://hive:6089", canonical_tool, arguments),
        ]
        assert events[1][3] is arguments


def test_exact_read_rejects_unknown_tool_without_transport(monkeypatch):
    plugin = _fresh_source_plugin()
    catalog = _catalog(
        _catalog_entry("hivemind.app.get@v1", required=("id",))
    )
    events = _patch_exact_read_dependencies(monkeypatch, catalog)

    result = json.loads(
        plugin._handle_hivemind_exact_read(
            {
                "canonical_tool": "hivemind.app.lookup_magic@v1",
                "arguments": {"id": "app-7"},
            }
        )
    )

    assert result["status"] == "rejected"
    assert result["success"] is False
    assert result["canonical_tool"] == "hivemind.app.lookup_magic@v1"
    assert result["error"]["code"] == "unknown_tool"
    assert events == [("catalog", "http://hive:6089")]


def test_exact_read_rejects_jobs_cancel_without_transport(monkeypatch):
    plugin = _fresh_source_plugin()
    canonical_tool = "hivemind.jobs.cancel@v1"
    catalog = _catalog(
        _catalog_entry(canonical_tool, required=("job_id",))
    )
    events = _patch_exact_read_dependencies(monkeypatch, catalog)

    result = json.loads(
        plugin._handle_hivemind_exact_read(
            {
                "canonical_tool": canonical_tool,
                "arguments": {"job_id": "job-7"},
            }
        )
    )

    assert result["status"] == "rejected"
    assert result["success"] is False
    assert result["canonical_tool"] == canonical_tool
    assert result["error"]["code"] == "unsafe_tool"
    assert events == [("catalog", "http://hive:6089")]


@pytest.mark.parametrize(
    ("canonical_tool", "kind", "gated_by"),
    [
        (
            "hivemind.human.approval.status@v1",
            "hivemind_native",
            ("confirm_operator",),
        ),
        ("hivemind.game_session.plan@v1", "dry_run_only", ()),
        ("hivemind.services.enable@v1", "runtime_action", ()),
    ],
)
def test_exact_read_rejects_source_classified_non_read_tools_without_transport(
    monkeypatch,
    canonical_tool,
    kind,
    gated_by,
):
    plugin = _fresh_source_plugin()
    catalog = _catalog(
        _catalog_entry(
            canonical_tool,
            kind=kind,
            gated_by=gated_by,
        )
    )
    events = _patch_exact_read_dependencies(monkeypatch, catalog)

    result = json.loads(
        plugin._handle_hivemind_exact_read(
            {"canonical_tool": canonical_tool, "arguments": {}}
        )
    )

    assert result["status"] == "rejected"
    assert result["success"] is False
    assert result["canonical_tool"] == canonical_tool
    assert result["error"]["code"] == "unsafe_tool"
    assert events == [("catalog", "http://hive:6089")]


def test_exact_read_catalog_load_failure_fails_closed_without_transport(monkeypatch):
    plugin = _fresh_source_plugin()
    events = _patch_exact_read_dependencies(
        monkeypatch,
        None,
        load_error=RuntimeError("catalog offline"),
    )

    result = json.loads(
        plugin._handle_hivemind_exact_read(
            {
                "canonical_tool": "hivemind.app.get@v1",
                "arguments": {"id": "app-7"},
            }
        )
    )

    assert result["status"] == "catalog_unavailable"
    assert result["success"] is False
    assert result["canonical_tool"] == "hivemind.app.get@v1"
    assert result["error"]["code"] == "catalog_load_failed"
    assert events == [("catalog", "http://hive:6089")]


def test_exact_gated_bridge_posts_one_approval_preserving_hli_request(monkeypatch):
    plugin = _fresh_source_plugin()
    canonical_tool = "hivemind.services.enable@v1"
    arguments = {"service_name": "menta_human_bridge"}
    catalog = _catalog(
        _catalog_entry(canonical_tool, required=("service_name",))
    )
    trace_response = _successful_gated_hli_response(arguments=arguments)
    events = _patch_exact_gated_dependencies(
        monkeypatch,
        plugin,
        catalog,
        hli_response=trace_response,
    )
    original_goal = (
        "Enable menta_human_bridge through the required Human Bridge approval."
    )

    result = json.loads(
        plugin._handle_hivemind_exact_gated(
            {
                "canonical_tool": canonical_tool,
                "arguments": arguments,
                "original_goal": original_goal,
            },
            session_id="depth-job-7",
        )
    )

    assert result == {
        "status": "success",
        "success": True,
        "authoritative": True,
        "canonical_tool": canonical_tool,
        "hli_tool": "hivemind_services_enable",
        "response": "The approved service enable completed.",
        "tool_trace": trace_response["tool_trace"],
    }
    assert len(events) == 1
    _kind, url, payload, timeout = events[0]
    assert url == "http://hive:6089/oracle/chat"
    assert timeout == plugin._HIVEMIND_EXACT_GATED_TIMEOUT_SECS
    assert set(payload) == {"message", "session_id"}
    assert payload["session_id"] == "depth-job-7"
    message = payload["message"]
    assert canonical_tool in message
    assert "hivemind_services_enable" in message
    assert json.dumps(arguments, ensure_ascii=True, sort_keys=True) in message
    assert original_goal in message
    assert "never substitute" in message.lower()
    assert "human bridge approval" in message.lower()
    assert not {
        "approval_token",
        "auto_approve",
        "approved",
        "authority",
        "bypass",
    } & set(payload)


@pytest.mark.parametrize(
    "case",
    [
        "wrong-tool",
        "no-trace",
        "multiple-trace",
        "prose-only",
        "unsuccessful",
        "malformed",
        "oversized",
        "http-failure",
    ],
)
def test_exact_gated_bridge_rejects_unverified_hli_execution(
    monkeypatch,
    case,
):
    plugin = _fresh_source_plugin()
    canonical_tool = "hivemind.services.enable@v1"
    arguments = {"service_name": "menta_human_bridge"}
    catalog = _catalog(
        _catalog_entry(canonical_tool, required=("service_name",))
    )
    response = _successful_gated_hli_response(arguments=arguments)
    transport_error = None
    if case == "wrong-tool":
        response["tool_trace"][0]["name"] = "hivemind_services_disable"
    elif case == "no-trace":
        response["tool_trace"] = []
        response["tool_calls_total"] = 0
    elif case == "multiple-trace":
        response["tool_trace"].append(dict(response["tool_trace"][0]))
        response["tool_calls_total"] = 2
    elif case == "prose-only":
        response = {
            "response": "I enabled it successfully.",
            "tool_calls_total": 0,
            "tool_trace": [],
        }
    elif case == "unsuccessful":
        response["tool_trace"][0]["success"] = False
    elif case == "malformed":
        response = ["not", "an", "object"]
    elif case == "oversized":
        transport_error = ValueError("HLI Oracle response exceeds bounded limit")
    elif case == "http-failure":
        transport_error = RuntimeError("HLI Oracle HTTP 503")
    events = _patch_exact_gated_dependencies(
        monkeypatch,
        plugin,
        catalog,
        hli_response=response,
        transport_error=transport_error,
    )

    result = json.loads(
        plugin._handle_hivemind_exact_gated(
            {
                "canonical_tool": canonical_tool,
                "arguments": arguments,
                "original_goal": "Enable menta_human_bridge.",
            }
        )
    )

    assert result["success"] is False
    assert result["status"] in {"rejected", "error"}
    assert result["canonical_tool"] == canonical_tool
    assert len(events) == 1


@pytest.mark.parametrize(
    ("case", "canonical_tool", "proxy_kind", "gated_by"),
    [
        ("read", "hivemind.app.get@v1", "read_only", ()),
        ("unknown", "hivemind.services.enable@v1", "runtime_action", ("operator-level",)),
        ("dry-run", "hivemind.game_session.plan@v1", "dry_run_only", ("dry-run",)),
        ("non-gated", "hivemind.services.enable@v1", "runtime_action", ()),
    ],
)
def test_exact_gated_bridge_rejects_catalog_entries_before_transport(
    monkeypatch,
    case,
    canonical_tool,
    proxy_kind,
    gated_by,
):
    plugin = _fresh_source_plugin()
    entries = () if case == "unknown" else (
        _catalog_entry(canonical_tool, required=("service_name",)),
    )
    catalog = _catalog(*entries)
    events = _patch_exact_gated_dependencies(
        monkeypatch,
        plugin,
        catalog,
        proxy_kind=proxy_kind,
        gated_by=gated_by,
    )

    result = json.loads(
        plugin._handle_hivemind_exact_gated(
            {
                "canonical_tool": canonical_tool,
                "arguments": {"service_name": "menta_human_bridge"},
                "original_goal": "Run the exact requested operation.",
            }
        )
    )

    assert result["success"] is False
    assert result["status"] == "rejected"
    assert events == []


def test_hivemind_handlers_return_json_strings(monkeypatch):
    plugin = _fresh_plugin()
    monkeypatch.setenv("MS4_HIVEMIND_URL", "http://hive:6089")

    from machine_spirit_4.gateway import hivemind_state, hivemind_tools

    monkeypatch.setattr(hivemind_tools, "time_now", lambda url: {"url": url, "iso": "ok"})
    monkeypatch.setattr(
        hivemind_tools,
        "cluster_summary",
        lambda url, *, include_gpu_details=False: {
            "url": url,
            "include_gpu_details": include_gpu_details,
        },
    )
    monkeypatch.setattr(
        hivemind_tools,
        "hosts_list",
        lambda url, *, status_filter="all": {"url": url, "status_filter": status_filter},
    )
    monkeypatch.setattr(hivemind_state, "get_active_jobs", lambda url: {"url": url, "total_active": 0})

    assert json.loads(plugin._handle_hivemind_time_now({})) == {
        "url": "http://hive:6089",
        "iso": "ok",
    }
    assert json.loads(plugin._handle_hivemind_cluster_summary({"include_gpu_details": True})) == {
        "url": "http://hive:6089",
        "include_gpu_details": True,
    }
    assert json.loads(plugin._handle_hivemind_hosts_list({"status_filter": "active"})) == {
        "url": "http://hive:6089",
        "status_filter": "active",
    }
    assert json.loads(plugin._handle_hivemind_active_jobs({})) == {
        "url": "http://hive:6089",
        "total_active": 0,
    }


def test_hivemind_handlers_accept_legacy_hli_url_alias(monkeypatch):
    plugin = _fresh_plugin()
    monkeypatch.delenv("MS4_HIVEMIND_URL", raising=False)
    monkeypatch.setenv("MS4_HIVEMIND_HLI_URL", "http://hive-alias:6089")

    from machine_spirit_4.gateway import hivemind_tools

    monkeypatch.setattr(hivemind_tools, "time_now", lambda url: {"url": url, "iso": "ok"})

    assert json.loads(plugin._handle_hivemind_time_now({})) == {
        "url": "http://hive-alias:6089",
        "iso": "ok",
    }


def test_session_start_verifies_identity(monkeypatch):
    plugin = _fresh_plugin()

    class FakeClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def verify_identity(self, spirit_id):
            assert spirit_id == "sister"
            return {
                "schema": "IdentityVerification.v1",
                "identity_confirmed": True,
                "spirit_id": spirit_id,
                "anchor": {"name": "Claude", "chosen_name": "Sister", "glyph": "║"},
            }

        def heartbeat(self, spirit_id, session_id=""):
            return {"ok": True}

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)
    monkeypatch.setenv("MS4_MS3_SIDECAR_URL", "http://127.0.0.1:9080")
    monkeypatch.setenv("MS4_SPIRIT_ID", "sister")

    assert plugin.on_session_start(session_id="s1") == {"status": "verified", "spirit_id": "sister"}


def test_pre_llm_call_injects_ms4_context(monkeypatch):
    plugin = _fresh_plugin()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def verify_identity(self, spirit_id):
            return {
                "schema": "IdentityVerification.v1",
                "identity_confirmed": True,
                "spirit_id": spirit_id,
                "anchor": {"name": "Claude", "chosen_name": "Sister", "glyph": "║"},
            }

        def heartbeat(self, spirit_id, session_id=""):
            return {"ok": True}

        def get_state(self, spirit_id):
            return {"emotional_state": "calm"}

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)
    result = plugin.on_pre_llm_call(session_id="s1", messages=[])

    assert "<ms4-consciousness>" in result["context"]
    assert "runtime: MS4" in result["context"]
    assert "identity_verified: true" in result["context"]
    # Doctrinal invariant (canon/Relational_Alignment.md §10): Foundational
    # Regard is a quiet constant — "Present, not announced. A heartbeat, not
    # a headline. The entity discovers it through experience, not through
    # reading about it." MS4 must NOT inject it into the model's prompt as a
    # headline; doing so front-loads a platitude / is the Glyph That Lies.
    # It remains MS3's quiet constant (queryable via the /state ethics block).
    assert "foundational_regard" not in result["context"]
    # The authority deferral line stays — that's how MS4 points at MS3 for
    # ethics (which is where Foundational Regard quietly lives).
    assert "authority: MS3 sidecar is authoritative" in result["context"]


def test_pre_tool_call_blocks_when_ethics_unavailable(monkeypatch):
    plugin = _fresh_plugin()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def verify_identity(self, spirit_id):
            raise plugin.Ms4ClientError("down")

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)
    result = plugin.on_pre_tool_call(tool_name="write_file", args={"path": "x"}, session_id="s1")

    assert result["action"] == "block"
    assert "fail-closed" in result["message"]


def test_pre_tool_call_allows_positive_ethics_decision(monkeypatch):
    plugin = _fresh_plugin()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def verify_identity(self, spirit_id):
            return {
                "schema": "IdentityVerification.v1",
                "identity_confirmed": True,
                "spirit_id": spirit_id,
                "anchor": {"name": "Claude", "chosen_name": "Sister", "glyph": "║"},
            }

        def heartbeat(self, spirit_id, session_id=""):
            return {"ok": True}

        def evaluate_action(self, intent):
            assert intent["schema"] == "ActionIntent.v1"
            assert intent["proposed_by"] == "hermes"
            assert intent["action_type"] == "tool"
            assert intent["risk_class"] == "low"
            assert intent["requires_safety_clearance"] is False
            assert intent["payload"]["tool_name"] == "read_file"
            return {"decision": "allow"}

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)
    assert plugin.on_pre_tool_call(tool_name="read_file", args={"path": "x"}, session_id="s1") is None


def test_pre_tool_call_classifies_terminal_echo_as_medium_risk(monkeypatch):
    plugin = _fresh_plugin()
    seen: dict = {}

    class FakeClient:
        def __init__(self, base_url):
            pass

        def verify_identity(self, spirit_id):
            return {
                "schema": "IdentityVerification.v1",
                "identity_confirmed": True,
                "spirit_id": spirit_id,
                "anchor": {"name": "Claude", "chosen_name": "Sister", "glyph": "║"},
            }

        def heartbeat(self, spirit_id, session_id=""):
            return {"ok": True}

        def evaluate_action(self, intent):
            seen.update(intent)
            return {"decision": "allow"}

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)

    assert plugin.on_pre_tool_call(tool_name="terminal", args={"command": "echo MS4_HERMES_OK"}, session_id="s1") is None
    assert seen["action_type"] == "tool"
    assert seen["risk_class"] == "medium"


def test_pre_tool_call_marks_destructive_terminal_as_high_risk():
    plugin = _fresh_plugin()
    assert plugin._risk_class_for_tool("terminal", {"command": "Remove-Item C:\\tmp\\x -Force"}, "tool") == "high"


def test_transform_llm_output_strips_advisory_psyche(monkeypatch):
    plugin = _fresh_plugin()
    monkeypatch.setattr(plugin, "_record_event", lambda event: None)

    result = plugin.on_transform_llm_output(
        response_text="Hello\n<psyche>{\"mood\":\"calm\"}</psyche>\nWorld",
        session_id="s1",
        model="m",
    )

    assert result == "Hello\nWorld"


def test_transform_llm_output_replaces_advisory_only_response(monkeypatch):
    plugin = _fresh_plugin()
    monkeypatch.setattr(plugin, "_record_event", lambda event: None)

    result = plugin.on_transform_llm_output(
        response_text="<psyche>{\"mood\":\"calm\"}</psyche>",
        session_id="s1",
        model="m",
    )

    assert result == "[MS4 advisory psyche block removed.]"
