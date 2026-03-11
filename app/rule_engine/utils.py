"""Utility helpers for RuleEngine V3."""

from __future__ import annotations

from time import monotonic
from typing import Any, Mapping, Sequence, Tuple


BBox = Tuple[float, float, float, float]


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
    return monotonic()


def normalize_bbox(bbox: Sequence[float] | None) -> BBox | None:
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = map(float, bbox[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def compute_iou(box1: Sequence[float] | None, box2: Sequence[float] | None) -> float:
    b1 = normalize_bbox(box1)
    b2 = normalize_bbox(box2)
    if b1 is None or b2 is None:
        return 0.0

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = area1 + area2 - inter
    return inter / denom if denom > 0.0 else 0.0


def bbox_center(bbox: Sequence[float] | None) -> tuple[float, float] | None:
    b = normalize_bbox(bbox)
    if b is None:
        return None
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
