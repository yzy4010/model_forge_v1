"""ROI helpers for RuleEngine V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from app.roi_engine import roi_engine

Point = Tuple[float, float]


@dataclass(frozen=True)
class ROI:
    tag: str
    points: Tuple[Point, ...]


class ROIIndex:
    def __init__(self, roi_config: Mapping[str, Any] | None):
        self._rois: Tuple[ROI, ...] = tuple(self._parse_rois(roi_config or {}))
        self._roi_polygons: dict[str, Tuple[Tuple[Point, ...], ...]] = {}
        for roi in self._rois:
            self._roi_polygons.setdefault(roi.tag, tuple())
            self._roi_polygons[roi.tag] = self._roi_polygons[roi.tag] + (roi.points,)

    @property
    def has_rois(self) -> bool:
        return bool(self._rois)

    def _parse_rois(self, cfg: Mapping[str, Any]) -> Iterable[ROI]:
        for item in cfg.get("rois", ()):
            pts = (item.get("geometry", {}) or {}).get("points") or item.get("points") or ()
            if len(pts) < 3:
                continue
            tag = item.get("semantic_tag") or item.get("name") or item.get("roi_id")
            if not tag:
                continue
            yield ROI(tag=str(tag), points=tuple((float(x), float(y)) for x, y in pts))

    def tags_for_bbox(self, bbox: Sequence[float]) -> Tuple[str, ...]:
        if not self._rois:
            return ()
        tags: List[str] = []
        for roi in self._rois:
            if roi_engine.center_in_roi(bbox, roi.points):
                tags.append(roi.tag)
        return tuple(tags)

    def check_in_roi(self, person: Mapping[str, Any], roi_name: str | Sequence[str]) -> bool:
        bbox = person.get("bbox") or ()
        return roi_engine.bbox_in_roi(bbox, roi_name, self._roi_polygons)

    def in_roi(self, bbox: Sequence[float], roi_name: str | Sequence[str]) -> bool:
        return roi_engine.bbox_in_roi(bbox, roi_name, self._roi_polygons)
