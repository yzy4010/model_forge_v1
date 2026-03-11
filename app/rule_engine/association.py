"""Stateless spatial association engine for per-frame object binding.

This module groups detections by alias and associates non-target objects to target
objects (default: ``person``) via IoU overlap.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


class AssociationEngine:
    """Build per-target associated objects for one inference frame.

    The engine is stateless and does not depend on tracking or rule definitions.
    It can run in "no-rule mode" where detections are simply grouped/associated.
    """

    def __init__(self, iou_threshold: float = 0.3, target_alias: str = "person"):
        """Initialize the association engine.

        Args:
            iou_threshold: Minimum IoU required for an object to belong to target.
            target_alias: Alias treated as association target. Defaults to person.
        """
        self.iou_threshold = float(iou_threshold)
        self.target_alias = str(target_alias)

    def group_by_alias(self, detections: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group detections by dynamic alias.

        Args:
            detections: Iterable of raw detection dictionaries.

        Returns:
            A dictionary of alias -> normalized detections list.
        """
        grouped: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
        for det in detections or []:
            alias = str(det.get("alias", "")).strip()
            if not alias:
                continue
            grouped[alias].append(
                {
                    "alias": alias,
                    "bbox": det.get("bbox") or det.get("xyxy") or [],
                    "score": float(det.get("score", 0.0)),
                    "track_id": det.get("track_id"),
                }
            )
        return dict(grouped)

    def compute_iou(self, box1: Sequence[float], box2: Sequence[float]) -> float:
        """Compute IoU for two ``[x1, y1, x2, y2]`` boxes."""
        if len(box1) != 4 or len(box2) != 4:
            return 0.0

        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
        area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
        denom = area1 + area2 - inter_area
        return inter_area / denom if denom > 0.0 else 0.0

    def match_objects(
        self,
        person_bbox: Sequence[float],
        objects: Mapping[str, List[Mapping[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Match dynamic alias objects to one target bbox using IoU threshold."""
        matched: Dict[str, List[Dict[str, Any]]] = {}
        for alias, dets in (objects or {}).items():
            if alias == self.target_alias:
                continue
            selected = [
                dict(det)
                for det in dets
                if self.compute_iou(person_bbox, det.get("bbox") or []) >= self.iou_threshold
            ]
            if selected:
                matched[alias] = selected
        return matched

    def build_associations(self, detections: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Build person-like association objects for one frame.

        Returns:
            List of person objects:
            {
                "track_id": optional[int],
                "bbox": [x1,y1,x2,y2],
                "score": float,
                "objects": {alias_name: [detections...]}
            }
        """
        grouped = self.group_by_alias(detections)
        targets = grouped.get(self.target_alias) or []
        if not targets:
            return []

        persons: List[Dict[str, Any]] = []
        for target in targets:
            bbox = target.get("bbox") or []
            persons.append(
                {
                    "track_id": target.get("track_id"),
                    "bbox": bbox,
                    "score": float(target.get("score", 0.0)),
                    "objects": self.match_objects(bbox, grouped),
                }
            )
        return persons
