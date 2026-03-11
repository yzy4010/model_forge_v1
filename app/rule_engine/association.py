"""Spatial association for building per-person object groups."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

from app.rule_engine.utils import compute_iou, normalize_bbox


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
            selected = [dict(det) for det in dets if compute_iou(person_bbox, det.get("bbox")) >= self.iou_threshold]
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
                    "objects": self.match_objects(bbox, grouped),
                }
            )
        return people
