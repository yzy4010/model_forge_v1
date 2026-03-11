"""Utility helpers for rule engine."""

from __future__ import annotations

from typing import Any, Mapping


def deep_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def now_monotonic() -> float:
    from time import monotonic

    return monotonic()
