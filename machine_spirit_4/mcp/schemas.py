from __future__ import annotations

from typing import Any


class ToolInputError(ValueError):
    pass


def require_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"Missing required string argument: {key}")
    return value.strip()


def optional_string(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolInputError(f"Expected string argument: {key}")
    value = value.strip()
    return value or None


def require_object(arguments: dict[str, Any], key: str) -> dict[str, Any]:
    value = arguments.get(key)
    if not isinstance(value, dict):
        raise ToolInputError(f"Missing required object argument: {key}")
    return value
