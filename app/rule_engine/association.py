"""Spatial association for building per-person object groups."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

from app.rule_engine.utils import compute_iou, normalize_bbox


def _bbox_contains_point(bbox: List[float], x: float, y: float) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_center(bbox: List[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class AssociationEngine:
    def __init__(self, iou_threshold: float = 0.3, target_alias: str = "person"):
        self.iou_threshold = float(iou_threshold)
        self.target_alias = str(target_alias)

    def group_by_alias(self, detections: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
        for det in detections or []:
            alias = str(det.get("alias", "")).strip()
            bbox = normalize_bbox(det.get("bbox") or det.get("xyxy"))
            if not alias or bbox is None:
                continue
            grouped[alias].append(
                {
                    "alias": alias,
                    "bbox": list(bbox),
                    "score": float(det.get("score", 0.0)),
                    "track_id": det.get("track_id"),
                    "roi_tags": tuple(det.get("roi_tags") or ()),
                }
            )
        return dict(grouped)

    def match_objects(
        self,
        person_bbox: List[float],
        objects: Mapping[str, List[Mapping[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        matched: Dict[str, List[Dict[str, Any]]] = {}
        for alias, dets in (objects or {}).items():
            if alias == self.target_alias:
                continue
            selected = []
            for det in dets:
                det_bbox = list(det.get("bbox") or [])
                if len(det_bbox) != 4:
                    continue
                iou = compute_iou(person_bbox, det_bbox)
                cx, cy = _bbox_center(det_bbox)
                if iou >= self.iou_threshold or _bbox_contains_point(person_bbox, cx, cy):
                    selected.append(dict(det))
            if selected:
                matched[alias] = selected
        return matched

    def build_associations(self, detections: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        grouped = self.group_by_alias(detections)
        targets = grouped.get(self.target_alias, [])
        people: List[Dict[str, Any]] = []
        for person in targets:
            bbox = list(person.get("bbox") or [])
            people.append(
                {
                    "alias": self.target_alias,
                    "track_id": person.get("track_id"),
                    "bbox": bbox,
                    "score": float(person.get("score", 0.0)),
                    "roi_tags": tuple(person.get("roi_tags") or ()),
                    "objects": self.match_objects(bbox, grouped),
                }
            )
        return people
