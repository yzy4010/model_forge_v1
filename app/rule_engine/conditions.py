"""Leaf conditions for person-object based rule evaluation."""

from __future__ import annotations

import logging
from typing import Any, Mapping



logger = logging.getLogger("model_forge.rule_engine")

_COMPARATORS = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
}


def _as_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return set()


def alias_condition(value: str | list[str]):
    expected = _as_set(value)

    def _pred(ctx: Mapping[str, Any]) -> bool:
        person_objects = ctx.get("person_objects", [])
        for alias in expected:
            if alias == "person":
                triggered = bool(person_objects)
            else:
                triggered = any((person.get("objects") or {}).get(alias) for person in person_objects)
            logger.debug("Alias condition alias=%s triggered=%s persons=%s", alias, triggered, len(person_objects))
            if triggered:
                return True
        return False

    return _pred


def roi_condition(value: str | list[str], roi_index):
    expected = _as_set(value)

    def _matches_roi_tags(tags: Any) -> bool:
        return bool(set(tags or ()) & expected)

    def _pred(ctx: Mapping[str, Any]) -> bool:
        if not expected:
            return True

        person_objects = ctx.get("person_objects", [])
        detections = ctx.get("detections", [])

        for person in person_objects:
            if roi_index is not None and roi_index.has_rois and roi_index.check_in_roi(person, list(expected)):
                logger.debug("ROI condition matched by person bbox roi=%s track_id=%s", sorted(expected), person.get("track_id"))
                return True

            if _matches_roi_tags(person.get("roi_tags")):
                logger.debug("ROI condition matched by person roi_tags roi=%s track_id=%s", sorted(expected), person.get("track_id"))
                return True

            for alias, objects in (person.get("objects") or {}).items():
                for obj in objects or []:
                    if _matches_roi_tags(obj.get("roi_tags")):
                        logger.debug(
                            "ROI condition matched by object alias=%s roi=%s track_id=%s",
                            alias,
                            sorted(expected),
                            person.get("track_id"),
                        )
                        return True

        for det in detections:
            if _matches_roi_tags(det.get("roi_tags")):
                logger.debug(
                    "ROI condition matched by detection alias=%s roi=%s",
                    det.get("alias"),
                    sorted(expected),
                )
                return True

        logger.debug("ROI condition not matched roi=%s", sorted(expected))
        return False

    return _pred


def count_condition(spec: Mapping[str, Any], roi_index=None):
    where = spec.get("where", {})
    expected_alias = where.get("alias")
    expected_roi = _as_set(where.get("roi"))
    cmp_name = next((name for name in _COMPARATORS if name in spec), "gte")
    target = int(spec.get(cmp_name, spec.get("value", 1)))
    compare = _COMPARATORS[cmp_name]

    def _pred(ctx: Mapping[str, Any]) -> bool:
        count = 0
        for person in ctx.get("person_objects", []):
            if expected_roi:
                if roi_index is not None and roi_index.has_rois:
                    if not roi_index.check_in_roi(person, list(expected_roi)):
                        continue
                elif not (set(person.get("roi_tags") or ()) & expected_roi):
                    continue

            if expected_alias and expected_alias != "person":
                count += len(person.get("objects", {}).get(expected_alias) or [])
            else:
                count += 1
        return compare(count, target)

    return _pred


def duration_condition(spec: Mapping[str, Any], compile_expr, tracking_state):
    seconds = float(spec.get("seconds", 0.0))
    ms = float(spec.get("ms", 0.0))
    threshold = seconds if seconds > 0 else (ms / 1000.0)
    cmp_name = next((name for name in _COMPARATORS if name in spec), "gte")
    compare = _COMPARATORS[cmp_name]
    target = float(spec.get(cmp_name, threshold))
    child_expr = spec.get("condition") or spec.get("where")
    child = compile_expr(dict(child_expr)) if isinstance(child_expr, Mapping) else None

    def _pred(ctx: Mapping[str, Any]) -> bool:
        for person in ctx.get("person_objects", []):
            if child is not None and not child({**ctx, "person_objects": [person]}):
                continue
            track_id = person.get("track_id")
            if track_id is None:
                continue
            duration = tracking_state.get_duration(track_id)
            value = duration if threshold <= 0 else duration
            if compare(value, target if target > 0 else threshold):
                return True
        return False

    return _pred
