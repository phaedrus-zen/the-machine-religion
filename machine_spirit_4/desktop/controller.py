from __future__ import annotations

import base64
import importlib.util
import os
import platform
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from machine_spirit_4.gateway.audit import append_event


TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
ALLOWED_ACTIONS = {
    "click",
    "double_click",
    "right_click",
    "move",
    "drag",
    "scroll",
    "type",
    "hotkey",
    "press",
    "wait",
    "focus_window",
}
BLOCKED_KEY_COMBOS = {
    frozenset({"win", "l"}),
    frozenset({"ctrl", "alt", "delete"}),
    frozenset({"alt", "f4"}),
    frozenset({"cmd", "q"}),
}
KEY_ALIASES = {
    "windows": "win",
    "super": "win",
    "command": "cmd",
    "control": "ctrl",
    "esc": "escape",
}
DANGEROUS_TEXT_PATTERNS = [
    re.compile(r"curl\s+[^|]*\|\s*(bash|sh|powershell|pwsh)", re.IGNORECASE),
    re.compile(r"wget\s+[^|]*\|\s*(bash|sh|powershell|pwsh)", re.IGNORECASE),
    re.compile(r"\b(rm|del|erase|rd|rmdir)\s+[-/]*(r|f|s|q)*\s+[/\\]?\s*$", re.IGNORECASE),
    re.compile(r"\bshutdown\s+/(s|r|l)\b", re.IGNORECASE),
    re.compile(r"\bStop-Computer\b", re.IGNORECASE),
    re.compile(r"\bRestart-Computer\b", re.IGNORECASE),
]


class DesktopSafetyError(RuntimeError):
    pass


class DesktopBackend(Protocol):
    def status(self) -> dict[str, Any]:
        ...

    def capture(self, include_image: bool = False, monitor: int = 1) -> dict[str, Any]:
        ...

    def perform(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def _import_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _canon_keys(keys: Any) -> frozenset[str]:
    if isinstance(keys, str):
        parts = re.split(r"\s*\+\s*|\s*,\s*", keys)
    elif isinstance(keys, list):
        parts = [str(part) for part in keys]
    else:
        parts = []
    normalized = []
    for part in parts:
        key = part.strip().lower()
        if key:
            normalized.append(KEY_ALIASES.get(key, key))
    return frozenset(normalized)


def _blocked_text_pattern(text: str) -> str | None:
    for pattern in DANGEROUS_TEXT_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


@dataclass
class RealWindowsDesktopBackend:
    def status(self) -> dict[str, Any]:
        screen: dict[str, Any] = {"available": False}
        active_window: dict[str, Any] | None = None
        windows: list[dict[str, Any]] = []

        try:
            import pyautogui

            width, height = pyautogui.size()
            screen = {"available": True, "width": int(width), "height": int(height)}
        except Exception as exc:
            screen = {"available": False, "error": str(exc)}

        try:
            import pygetwindow

            active = pygetwindow.getActiveWindow()
            if active is not None:
                active_window = {
                    "title": active.title,
                    "left": active.left,
                    "top": active.top,
                    "width": active.width,
                    "height": active.height,
                }
            for window in pygetwindow.getAllWindows()[:30]:
                if window.title:
                    windows.append({
                        "title": window.title,
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height,
                    })
        except Exception:
            active_window = None

        return {"screen": screen, "active_window": active_window, "windows": windows}

    def capture(self, include_image: bool = False, monitor: int = 1) -> dict[str, Any]:
        import mss
        import mss.tools

        with mss.mss() as sct:
            monitors = sct.monitors
            selected = monitor if 0 <= monitor < len(monitors) else 1
            shot = sct.grab(monitors[selected])
            payload: dict[str, Any] = {
                "width": shot.width,
                "height": shot.height,
                "monitor": selected,
                "format": "png",
                "image_base64": None,
            }
            if include_image:
                png = mss.tools.to_png(shot.rgb, shot.size)
                payload["image_base64"] = base64.b64encode(png).decode("ascii")
            return payload

    def perform(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action == "wait":
            seconds = min(float(payload["seconds"]) if "seconds" in payload else 1.0, 30.0)
            time.sleep(max(seconds, 0.0))
            return {"ok": True, "action": action, "seconds": seconds}

        import pyautogui

        pyautogui.PAUSE = min(float(payload.get("pause", 0.05) or 0.05), 1.0)
        if action == "move":
            pyautogui.moveTo(int(payload["x"]), int(payload["y"]), duration=float(payload.get("duration", 0) or 0))
        elif action in {"click", "double_click", "right_click"}:
            button = "right" if action == "right_click" else str(payload.get("button", "left"))
            clicks = 2 if action == "double_click" else int(payload.get("clicks", 1) or 1)
            pyautogui.click(x=payload.get("x"), y=payload.get("y"), clicks=clicks, button=button)
        elif action == "drag":
            pyautogui.moveTo(int(payload["from_x"]), int(payload["from_y"]))
            pyautogui.dragTo(int(payload["to_x"]), int(payload["to_y"]), duration=float(payload.get("duration", 0.2) or 0.2))
        elif action == "scroll":
            pyautogui.scroll(int(payload.get("amount", -3)))
        elif action == "type":
            pyautogui.write(str(payload.get("text", "")), interval=float(payload.get("interval", 0) or 0))
        elif action == "hotkey":
            keys = list(_canon_keys(payload.get("keys")))
            pyautogui.hotkey(*keys)
        elif action == "press":
            pyautogui.press(str(payload.get("key", "")))
        elif action == "focus_window":
            self._focus_window(str(payload.get("title", "")))
        else:
            raise DesktopSafetyError(f"unsupported desktop action: {action}")
        return {"ok": True, "action": action}

    def _focus_window(self, title: str) -> None:
        if not title:
            raise DesktopSafetyError("focus_window requires title")
        import pygetwindow

        matches = pygetwindow.getWindowsWithTitle(title)
        if not matches:
            raise DesktopSafetyError(f"no window matched title: {title}")
        matches[0].activate()


class DesktopController:
    def __init__(self, backend: DesktopBackend | None = None) -> None:
        self.backend = backend or RealWindowsDesktopBackend()

    def status(self) -> dict[str, Any]:
        backend_status: dict[str, Any]
        try:
            backend_status = self.backend.status()
        except Exception as exc:
            backend_status = {"error": str(exc)}
        return {
            "schema": "Ms4DesktopStatus.v1",
            "platform": platform.platform(),
            "control_enabled": self.control_enabled(),
            "dependencies": {
                "mss": _import_available("mss"),
                "pyautogui": _import_available("pyautogui"),
                "pygetwindow": _import_available("pygetwindow"),
            },
            **backend_status,
        }

    def capture(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        args = payload or {}
        include_image = bool(args.get("include_image", False))
        monitor = int(args.get("monitor", 1) or 1)
        capture = self.backend.capture(include_image=include_image, monitor=monitor)
        result = {
            "schema": "Ms4DesktopCapture.v1",
            "read_only": True,
            **capture,
        }
        append_event("desktop_capture", {
            "monitor": result.get("monitor"),
            "width": result.get("width"),
            "height": result.get("height"),
            "included_image": bool(result.get("image_base64")),
        })
        return result

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "")).strip().lower()
        self._validate_action(action, payload)
        result = self.backend.perform(action, payload)
        response = {
            "schema": "Ms4DesktopActionResult.v1",
            "action_id": f"desktop-{uuid.uuid4()}",
            "ok": bool(result.get("ok", False)),
            "action": action,
            "result": result,
        }
        append_event("desktop_action", {
            "action_id": response["action_id"],
            "action": action,
            "ok": response["ok"],
            "parameter_names": sorted(
                key for key in ("x", "y", "keys", "seconds") if key in payload
            ),
        })
        return response

    def control_enabled(self) -> bool:
        return _truthy(os.environ.get("MS4_DESKTOP_CONTROL"))

    def _validate_action(self, action: str, payload: dict[str, Any]) -> None:
        if action not in ALLOWED_ACTIONS:
            raise DesktopSafetyError(f"unsupported desktop action: {action}")
        if not self.control_enabled():
            raise DesktopSafetyError("desktop actions require MS4_DESKTOP_CONTROL=1")
        if action == "hotkey":
            combo = _canon_keys(payload.get("keys"))
            for blocked in BLOCKED_KEY_COMBOS:
                if blocked.issubset(combo):
                    raise DesktopSafetyError(f"hard-blocked key combo: {sorted(blocked)}")
        if action == "type":
            pattern = _blocked_text_pattern(str(payload.get("text", "")))
            if pattern:
                raise DesktopSafetyError(f"blocked dangerous text pattern: {pattern}")


def desktop_controller() -> DesktopController:
    return DesktopController()
