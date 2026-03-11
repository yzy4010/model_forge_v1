"""Leaf conditions for rule evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from app.rule_engine.utils import now_monotonic


_COMPARATORS = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
}


def alias_condition(value: str | list[str]):
    expected = {value} if isinstance(value, str) else set(value)
    return lambda ctx: any(det["alias"] in expected for det in ctx["detections"])


def roi_condition(value: str | list[str]):
    expected = {value} if isinstance(value, str) else set(value)
    return lambda ctx: any(bool(set(det.get("roi_tags", ())) & expected) for det in ctx["detections"])


def count_condition(spec: Mapping[str, Any]):
    where = spec.get("where", {})
    cmp_name = next((k for k in _COMPARATORS if k in spec), "gte")
    target = int(spec.get(cmp_name, spec.get("value", 1)))

    def match(det: Dict[str, Any]) -> bool:
        if "alias" in where and det.get("alias") != where["alias"]:
            return False
        if "roi" in where and where["roi"] not in det.get("roi_tags", ()):  # type: ignore[arg-type]
            return False
        return True

    compare = _COMPARATORS[cmp_name]
    return lambda ctx: compare(sum(1 for det in ctx["detections"] if match(det)), target)


def duration_condition(spec: Mapping[str, Any], compile_expr, duration_state):
    seconds = float(spec.get("seconds", 0.0))
    ms = float(spec.get("ms", 0.0))
    threshold = seconds if seconds > 0 else (ms / 1000.0)
    child_expr = spec.get("condition")
    if not isinstance(child_expr, Mapping):
        raise ValueError("duration.condition must be an expression object")
    child = compile_expr(dict(child_expr))
    key = f"duration:{id(spec)}"

    def _pred(ctx: dict) -> bool:
        active = child(ctx)
        elapsed = duration_state.update(key, active, now_monotonic())
        return active and elapsed >= threshold

    return _pred
