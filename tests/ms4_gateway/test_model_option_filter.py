from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import subprocess
import types

import pytest

from machine_spirit_4.double_agent import model_picker
from machine_spirit_4.gateway import server as srv_module


ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = ROOT / "machine_spirit_4" / "web" / "index.html"


class _DummyRunner:
    ms3_url = "http://ms3:9080"


def _gateway_models_response(monkeypatch, models, *, response_shape="dict"):
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = _DummyRunner()
    handler.command = "GET"
    handler.path = "/models"
    headers = {"Host": "127.0.0.1:9180"}
    handler.headers = types.SimpleNamespace(get=headers.get)
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.requestline = "GET /models HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"

    monkeypatch.setattr(
        srv_module,
        "_proxy_json",
        lambda _url, timeout=8: (
            200,
            models if response_shape == "list" else {"models": models},
        ),
    )
    handler.do_GET()

    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    _, _, body = rest.partition(b"\r\n\r\n")
    return int(status_line.split(b" ")[1]), json.loads(body.decode("utf-8"))


def _filter_source() -> str:
    html = HTML_PATH.read_text(encoding="utf-8")
    start = html.find("const NON_CHAT_MODEL_METADATA_TOKENS")
    end = html.find("function _fillModelDropdown", start)
    assert start >= 0, "missing model metadata filter"
    assert end >= 0, "missing model dropdown boundary"
    return html[start:end]


def _kept_ids(models: list[dict[str, object]]) -> list[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the model-option JavaScript regression")
    script = (
        _filter_source()
        + "\nconst models = "
        + json.dumps(models)
        + ";\nprocess.stdout.write(JSON.stringify("
        + "models.filter(isChatCapableModel).map(model => model.id)));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, (
        f"model-option filter failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout)


INDEPENDENT_NON_CHAT_CASES = [
    {"category": "tts"},
    {"hivemind_category": "embedding"},
    {"hivemind_capabilities": {"image-generation": True}},
    {"details": {"category": "text_to_speech"}},
    {"task": "text-embedding"},
    {"output_modality": "image"},
    {"type": "tts"},
]

CLASSIFICATION_CONTROLS = [
    ({}, True),
    ({"category": "llm", "hivemind_capabilities": ["chat", "vision"]}, True),
    ({"category": "tts", "capabilities": ["chat"]}, False),
    (
        {
            "category": {"llm": True, "tts": False},
            "capabilities": {"chat": True, "image-generation": False},
        },
        True,
    ),
]

RECOGNIZED_METADATA_CASES = [
    ("hivemind_category", "tts"),
    ("category", "embedding"),
    ("hivemind_capability", "tts"),
    ("hivemind_capabilities", {"image-generation": True}),
    ("capability", "embedding"),
    ("capabilities", ["tts"]),
    ("task", "text-embedding"),
    ("tasks", ["text-embedding"]),
    ("modality", "audio"),
    ("modalities", ["audio"]),
    ("input_modality", "image"),
    ("input_modalities", ["image"]),
    ("output_modality", "image"),
    ("output_modalities", ["image"]),
    ("type", "tts"),
    ("types", ["embedding"]),
]


@pytest.mark.parametrize("metadata", INDEPENDENT_NON_CHAT_CASES)
def test_classifier_rejects_independent_probe_non_chat_cases(metadata) -> None:
    assert model_picker._is_chat_capable(
        {"id": "opaque-valid-looking-model", **metadata}
    ) is False


@pytest.mark.parametrize("metadata,expected", CLASSIFICATION_CONTROLS)
def test_classifier_preserves_independent_probe_controls(metadata, expected) -> None:
    assert model_picker._is_chat_capable(
        {"id": "opaque-valid-looking-model", **metadata}
    ) is expected


def test_authoritative_metadata_removes_non_chat_options_and_preserves_chat_options() -> None:
    models = [
        {"id": "all-minilm:33m", "category": "embedding"},
        {"id": "text-embedding-3-small", "hivemind_category": "embedding"},
        {"id": "opaque-speech-worker", "capabilities": ["text-to-speech"]},
        {"id": "Qwen3-TTS", "category": "tts"},
        {"id": "opaque-image-worker", "category": "image_gen"},
        {
            "id": "chatgpt-image-latest",
            "hivemind_capabilities": {"image-generation": True},
        },
        {
            "id": "llama3.1:8b",
            "category": "llm",
            "capabilities": ["completion", "tools"],
        },
        {
            "id": "llama-3.2-vision",
            "category": "llm",
            "capabilities": ["chat", "vision"],
        },
        {"id": "gpt-4.1-mini", "category": "llm"},
        {"id": "mystery-chat:13b"},
    ]

    assert _kept_ids(models) == [
        "llama3.1:8b",
        "llama-3.2-vision",
        "gpt-4.1-mini",
        "mystery-chat:13b",
    ]


def test_minilm_alias_fallback_does_not_reject_valid_mini_chat_models() -> None:
    models = [
        {"id": "all-minilm", "category": "llm"},
        {"id": "sentence-transformers/all-MiniLM-L6-v2", "category": "llm"},
        {"id": "vendor/MiniLM:latest"},
        {"id": "phi4-mini:latest", "category": "llm"},
        {"id": "MiniMax-M2.1", "capabilities": ["chat"]},
        {"id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "category": "llm"},
    ]

    assert _kept_ids(models) == [
        "phi4-mini:latest",
        "MiniMax-M2.1",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    ]


def test_face_and_depth_dropdowns_consume_the_same_filtered_list() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    assert "const models = (data.models || []).filter(isChatCapableModel);" in html
    assert (
        "_fillModelDropdown(modelSelect, 'Auto-pick Face Lobe model (GPU-aware)', "
        "models, faceSel, true);"
    ) in html
    assert (
        "_fillModelDropdown(depthModelSelect, 'Auto (recommended depth model)', "
        "models, depthSel, false);"
    ) in html


def test_normalization_cases_cover_every_classifier_metadata_key() -> None:
    assert {key for key, _value in RECOGNIZED_METADATA_CASES} == set(
        model_picker._RECOGNIZED_METADATA_KEYS
    )


@pytest.mark.parametrize("key,value", RECOGNIZED_METADATA_CASES)
def test_normalization_preserves_metadata_and_classification(key, value) -> None:
    raw = {
        "id": "llama3.1:8b-private",
        "hivemind_status": "installed",
        key: value,
    }

    normalized = model_picker._normalize_model_entry(raw)
    renormalized = model_picker._normalize_model_entry(normalized)

    assert normalized[key] == value
    assert renormalized[key] == value
    assert model_picker._is_chat_capable(raw) is False
    assert model_picker._is_chat_capable(normalized) is False


def test_normalization_preserves_nested_details_classification() -> None:
    raw = {
        "id": "llama3.1:8b-private",
        "hivemind_status": "installed",
        "details": {"category": "tts"},
    }

    normalized = model_picker._normalize_model_entry(raw)

    assert normalized["details"] == raw["details"]
    assert model_picker._is_chat_capable(raw) is False
    assert model_picker._is_chat_capable(normalized) is False


def test_face_catalog_filters_before_normalization(monkeypatch) -> None:
    catalog = [
        {"id": "qwen3:8b-tts", "hivemind_status": "installed", "category": "tts"},
        {
            "id": "llama3.1:8b-embed",
            "hivemind_status": "installed",
            "hivemind_category": "embedding",
        },
        {
            "id": "phi4-mini-image",
            "hivemind_status": "installed",
            "hivemind_capabilities": {"image-generation": True},
        },
        {
            "id": "qwen3:8b",
            "hivemind_status": "installed",
            "category": "llm",
            "capabilities": ["chat"],
        },
    ]
    seen = []
    original = model_picker._normalize_model_entry

    def track(entry):
        seen.append(entry["id"])
        return original(entry)

    monkeypatch.setattr(model_picker, "_normalize_model_entry", track)

    choice = model_picker._pick_from_catalog(catalog)

    assert seen == ["qwen3:8b"]
    assert choice is not None
    assert choice.model_id == "qwen3:8b"


@pytest.mark.parametrize(
    "metadata",
    [
        {"hivemind_category": "tts"},
        {"category": "embedding"},
        {"hivemind_capabilities": {"image-generation": True}},
    ],
    ids=("tts", "embedding", "image"),
)
def test_face_picker_rejects_opaque_authoritative_non_chat_metadata(metadata) -> None:
    entry = {
        "id": "llama3.1:8b-private",
        "hivemind_status": "installed",
        **metadata,
    }

    assert model_picker._pick_from_catalog([entry]) is None


def test_face_picker_tts_only_catalog_has_no_candidate() -> None:
    catalog = [
        {
            "id": "llama3.1:8b-private",
            "hivemind_category": "tts",
            "hivemind_status": "installed",
        },
        {
            "id": "phi4-mini-private",
            "category": "text-to-speech",
            "hivemind_status": "installed",
        },
    ]

    assert model_picker._pick_from_catalog(catalog) is None


@pytest.mark.parametrize("response_shape", ["dict", "list"])
def test_models_route_filters_opaque_authoritative_non_chat_entries(
    monkeypatch, response_shape
) -> None:
    models = [
        {"id": "opaque-a", "hivemind_category": "tts"},
        {"id": "opaque-b", "category": "embedding"},
        {"id": "opaque-c", "hivemind_capabilities": {"image-generation": True}},
        {"id": "llama3.1:8b", "category": "llm"},
        {
            "id": "multimodal-chat",
            "category": "llm",
            "hivemind_capabilities": ["chat", "vision"],
        },
    ]

    status, payload = _gateway_models_response(
        monkeypatch,
        models,
        response_shape=response_shape,
    )

    assert status == 200
    assert payload["chat_filtered"] is True
    assert [model["id"] for model in payload["models"]] == [
        "llama3.1:8b",
        "multimodal-chat",
    ]


def test_models_route_tts_only_catalog_is_empty(monkeypatch) -> None:
    models = [
        {"id": "opaque-a", "hivemind_category": "tts"},
        {"id": "opaque-b", "category": "text-to-speech"},
    ]

    status, payload = _gateway_models_response(monkeypatch, models)

    assert status == 200
    assert payload["chat_filtered"] is True
    assert payload["models"] == []
