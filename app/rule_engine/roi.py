"""ROI helpers for RuleEngine V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from app.rule_engine.utils import bbox_center

Point = Tuple[float, float]


@dataclass(frozen=True)
class ROI:
    tag: str
    points: Tuple[Point, ...]


class ROIIndex:
    def __init__(self, roi_config: Mapping[str, Any] | None):
        self._rois: Tuple[ROI, ...] = tuple(self._parse_rois(roi_config or {}))

    def _parse_rois(self, cfg: Mapping[str, Any]) -> Iterable[ROI]:
        for item in cfg.get("rois", ()):
            pts = (item.get("geometry", {}) or {}).get("points") or item.get("points") or ()
            if len(pts) < 3:
                continue
            tag = item.get("semantic_tag") or item.get("name") or item.get("roi_id")
            if not tag:
                continue
            yield ROI(tag=str(tag), points=tuple((float(x), float(y)) for x, y in pts))

    @staticmethod
    def _inside_polygon(point: Point, polygon: Sequence[Point]) -> bool:
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            crosses = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
            if crosses:
                inside = not inside
            j = i
        return inside

    def tags_for_bbox(self, bbox: Sequence[float]) -> Tuple[str, ...]:
        center = bbox_center(bbox)
        if center is None:
            return ()
        tags: List[str] = []
        for roi in self._rois:
            if self._inside_polygon(center, roi.points):
                tags.append(roi.tag)
        return tuple(tags)

    def in_roi(self, bbox: Sequence[float], roi_name: str | Sequence[str]) -> bool:
        expected = {roi_name} if isinstance(roi_name, str) else set(roi_name)
        return bool(set(self.tags_for_bbox(bbox)) & {str(item) for item in expected})
