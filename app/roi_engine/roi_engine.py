#!/usr/bin/env python3
"""Independent geometry-focused ROI engine."""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, Mapping, Sequence


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    if len(point) < 2 or len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def center_in_roi(bbox: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox]
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return point_in_polygon(center, polygon)


def bbox_in_roi(
    bbox: Sequence[float],
    roi_name: str | Sequence[str],
    roi_polygons: Mapping[str, Iterable[Sequence[Sequence[float]]]],
) -> bool:
    if len(bbox) != 4 or not roi_polygons:
        return False
    expected = {str(roi_name)} if isinstance(roi_name, str) else {str(item) for item in roi_name}
    if not expected:
        return False
    for name in expected:
        for polygon in roi_polygons.get(name, ()):  # support one tag -> multi polygons
            if center_in_roi(bbox, polygon):
                return True
    return False


class ROIEngine:
    """Runtime ROI tagger for inference outputs."""

    def __init__(self, config: Dict[str, Any]):
        self._lock = threading.RLock()
        self._enabled_rois: list[dict[str, Any]] = []
        self.update(config)

    def update(self, config: Dict[str, Any]):
        with self._lock:
            self._enabled_rois = []
            for roi in config.get("rois", []):
                if not roi.get("enabled", False):
                    continue
                points = (roi.get("geometry") or {}).get("points") or roi.get("points") or []
                if len(points) < 3:
                    continue
                semantic_tag = roi.get("semantic_tag") or roi.get("name") or roi.get("roi_id")
                if not semantic_tag:
                    continue
                self._enabled_rois.append({"semantic_tag": str(semantic_tag), "polygon": points})

    def apply(self, detections):
        with self._lock:
            enabled_rois = tuple(self._enabled_rois)

        if not enabled_rois:
            return detections

        processed = []
        for detection in detections:
            if isinstance(detection, dict):
                bbox = detection.get("xyxy") or detection.get("bbox")
                if not bbox:
                    processed.append(detection)
                    continue
                detection.setdefault("roi_tags", [])
                for roi in enabled_rois:
                    if center_in_roi(bbox, roi["polygon"]) and roi["semantic_tag"] not in detection["roi_tags"]:
                        detection["roi_tags"].append(roi["semantic_tag"])
                processed.append(detection)
                continue

            bbox = getattr(detection, "xxyy", None) or getattr(detection, "bbox", None)
            if not bbox:
                processed.append(detection)
                continue
            for roi in enabled_rois:
                if center_in_roi(bbox, roi["polygon"]) and roi["semantic_tag"] not in detection.roi_tags:
                    detection.roi_tags.append(roi["semantic_tag"])
            processed.append(detection)
        return processed
