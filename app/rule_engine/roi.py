"""ROI helper and indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class ROI:
    tag: str
    points: Tuple[Point, ...]


class ROIIndex:
    """Preprocessed ROI index for fast point-in-polygon checks."""

    def __init__(self, roi_config: Mapping[str, Any] | None):
        self._rois: Tuple[ROI, ...] = tuple(self._parse_rois(roi_config or {}))

    def _parse_rois(self, cfg: Mapping[str, Any]) -> Iterable[ROI]:
        for item in cfg.get("rois", ()):
            geom = item.get("geometry", {})
            pts = geom.get("points") or item.get("points") or ()
            if len(pts) < 3:
                continue
            tag = item.get("semantic_tag") or item.get("name") or item.get("roi_id")
            if not tag:
                continue
            points = tuple((float(x), float(y)) for x, y in pts)
            yield ROI(tag=str(tag), points=points)

    @staticmethod
    def _inside_polygon(point: Point, polygon: Sequence[Point]) -> bool:
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def tags_for_bbox(self, bbox: Sequence[float]) -> Tuple[str, ...]:
        if len(bbox) < 4:
            return ()
        x1, y1, x2, y2 = map(float, bbox[:4])
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        tags: List[str] = []
        for roi in self._rois:
            if self._inside_polygon(center, roi.points):
                tags.append(roi.tag)
        return tuple(tags)
