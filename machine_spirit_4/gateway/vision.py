from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_VISION_MODEL = "qwen3-vl:4b-thinking"
FALLBACK_VISION_MODELS = ["llama3.2-vision:11b-instruct-q4_K_M"]


def _detect_image_mime(path: Path) -> str | None:
    header = path.read_bytes()[:64]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return mimetypes.guess_type(str(path))[0] if path.suffix.lower() in IMAGE_SUFFIXES else None


def _safe_image_path(raw_path: str) -> Path:
    if not raw_path or not isinstance(raw_path, str):
        raise ValueError("image_path is required")
    path = Path(os.path.expanduser(raw_path)).resolve()
    if not path.is_file():
        raise ValueError(f"image file not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image extension: {path.suffix}")
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"image is too large: {size} bytes > {MAX_IMAGE_BYTES}")
    if _detect_image_mime(path) is None:
        raise ValueError("file does not look like a supported image")
    return path


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    from .hivemind_state import hivemind_auth_headers

    headers = {"Content-Type": "application/json"}
    headers.update(hivemind_auth_headers())
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_hivemind_vlm_mcp(
    *,
    hivemind_url: str,
    image_base64: str,
    prompt: str,
    model: str,
    timeout: int,
) -> dict[str, Any] | None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "hivemind.vlm.describe_image@v1",
            "arguments": {
                "model": model,
                "prompt": prompt,
                "image_base64": image_base64,
                "temperature": 0,
                "max_tokens": 900,
            },
        },
    }
    response_payload = _post_json(f"{hivemind_url.rstrip('/')}/v1/mcp", payload, timeout)
    if response_payload.get("error"):
        return None
    result = response_payload.get("result", {})
    if result.get("isError") is True:
        return None
    content = result.get("content") or []
    if not content or not isinstance(content[0], dict):
        return None
    text = str(content[0].get("text") or "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    description = str(parsed.get("description") or parsed.get("message") or "")
    if not description.strip():
        return None
    return {
        "model": parsed.get("response_model") or parsed.get("model_requested") or model,
        "text": description,
        "reasoning_excerpt": None,
        "raw_response": parsed,
        "analysis_source": "hivemind_mcp:hivemind.vlm.describe_image@v1",
    }


def _call_hivemind_chat_completions(
    *,
    hivemind_url: str,
    data_url: str,
    prompt: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 900,
    }
    response_payload = _post_json(f"{hivemind_url.rstrip('/')}/v1/chat/completions", payload, timeout)
    message = response_payload.get("choices", [{}])[0].get("message", {})
    text = str(message.get("content") or "")
    if not text.strip():
        return {
            "model": response_payload.get("model") or model,
            "text": "",
            "reasoning_excerpt": str(message.get("reasoning_content"))[:1000] if message.get("reasoning_content") else None,
            "raw_response": response_payload,
            "analysis_source": "hivemind_openai_chat_completions",
            "empty_visible_text": True,
        }
    return {
        "model": response_payload.get("model") or model,
        "text": text,
        "reasoning_excerpt": str(message.get("reasoning_content"))[:1000] if message.get("reasoning_content") else None,
        "raw_response": response_payload,
        "analysis_source": "hivemind_openai_chat_completions",
    }


def _call_hivemind_vlm_chat(
    *,
    hivemind_url: str,
    image_base64: str,
    mime: str,
    prompt: str,
    model: str,
    prior_messages: list[dict[str, Any]] | None,
    timeout: int,
) -> dict[str, Any] | None:
    """Call the May-2026 ``hivemind.vlm.chat@v1`` tool. Returns the same
    shape as :func:`_call_hivemind_vlm_mcp` so the caller can swap it
    in without changing downstream code. Returns ``None`` on any
    failure so the caller falls back to the single-shot path.

    Useful when the operator is asking a follow-up about an image
    that was already part of the conversation (e.g. "what's that text
    in the upper left?"). The single-shot ``describe_image@v1``
    discards prior context; this path threads it through.
    """
    try:
        from . import hivemind_tools

        data_url = f"data:{mime};base64,{image_base64}"
        messages = list(prior_messages or [])
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )
        body = hivemind_tools.vlm_chat(
            hivemind_url,
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=900,
        )
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    text = ""
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        text = str(message.get("content") or "")
    if not text:
        text = str(body.get("text") or body.get("description") or "")
    if not text.strip():
        return None
    return {
        "model": body.get("model") or model,
        "text": text,
        "reasoning_excerpt": None,
        "raw_response": body,
        "analysis_source": "hivemind_mcp:hivemind.vlm.chat@v1",
    }


def _model_attempts(selected_model: str) -> list[str]:
    attempts = [selected_model]
    for fallback in FALLBACK_VISION_MODELS:
        if fallback not in attempts:
            attempts.append(fallback)
    return attempts


def analyze_local_image(
    *,
    hivemind_url: str,
    image_path: str,
    question: str | None = None,
    model: str | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    path = _safe_image_path(image_path)
    mime = _detect_image_mime(path) or "image/png"
    selected_model = model or os.environ.get("MS4_VISION_MODEL") or DEFAULT_VISION_MODEL
    prompt = question or (
        "Describe this image in detail. Include visible text, layout, colors, diagrams, "
        "and uncertainty. Do not invent text you cannot read."
    )
    image_base64 = base64.b64encode(path.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{image_base64}"
    analysis = None
    failed_attempts: list[dict[str, Any]] = []
    # Pass order:
    #   1. hivemind.vlm.chat@v1 (May-2026 multi-turn) — preferred
    #      because it accepts richer context if we ever pipe in a
    #      conversation, and HiveMind routes it the same way as the
    #      OpenAI chat-completions path.
    #   2. hivemind.vlm.describe_image@v1 (single-shot) — backwards
    #      compatible fallback.
    #   3. HiveMind /v1/chat/completions direct (legacy, used pre-
    #      vlm.describe_image rollout).
    # Each pass cycles through ``_model_attempts(selected_model)``
    # so a model that returns empty visible text on pass 1 can be
    # rescued by another model on pass 2 or 3.
    for attempt_model in _model_attempts(selected_model):
        analysis = _call_hivemind_vlm_chat(
            hivemind_url=hivemind_url,
            image_base64=image_base64,
            mime=mime,
            prompt=prompt,
            model=attempt_model,
            prior_messages=None,  # single-shot; multi-turn callers use analyze_local_image_with_history
            timeout=timeout,
        )
        if analysis is None:
            analysis = _call_hivemind_vlm_mcp(
                hivemind_url=hivemind_url,
                image_base64=image_base64,
                prompt=prompt,
                model=attempt_model,
                timeout=timeout,
            )
        if analysis is None:
            direct = _call_hivemind_chat_completions(
                hivemind_url=hivemind_url,
                data_url=data_url,
                prompt=prompt,
                model=attempt_model,
                timeout=timeout,
            )
            if direct.get("empty_visible_text"):
                failed_attempts.append({
                    "model": direct["model"],
                    "analysis_source": direct["analysis_source"],
                    "reason": "empty_visible_text",
                    "reasoning_excerpt": direct.get("reasoning_excerpt"),
                })
                analysis = None
            else:
                analysis = direct
        if analysis is not None and str(analysis.get("text") or "").strip():
            break
    if analysis is None or not str(analysis.get("text") or "").strip():
        raise RuntimeError(f"VLM returned no visible description text; failed_attempts={failed_attempts}")
    return {
        "schema": "Ms4VisionAnalysis.v1",
        "image_path": str(path),
        "mime_type": mime,
        "size_bytes": path.stat().st_size,
        "model": analysis["model"],
        "prompt": prompt,
        "text": analysis["text"],
        "reasoning_excerpt": analysis["reasoning_excerpt"],
        "analysis_source": analysis["analysis_source"],
        "failed_attempts": failed_attempts,
        "hivemind_url": hivemind_url.rstrip("/"),
    }
