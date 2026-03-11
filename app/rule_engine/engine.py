"""RuleEngine V3 entrypoint with association and tracking support."""

from __future__ import annotations

import threading
import logging
from typing import Any, Dict, List, Mapping

from app.rule_engine.association import AssociationEngine
from app.rule_engine.evaluator import ExpressionCompiler
from app.rule_engine.roi import ROIIndex
from app.rule_engine.schema import normalize_rules
from app.rule_engine.tracking_state import TrackStateManager
from app.rule_engine.utils import normalize_bbox


class RuleEngine:
    def __init__(self, rules: Any, roi_config: Mapping[str, Any] | None = None):
        self._logger = logging.getLogger("model_forge.rule_engine")
        self._lock = threading.RLock()
        self._association = AssociationEngine()
        self._tracking_state = TrackStateManager()
        self._roi_index = ROIIndex(roi_config)
        self._compiler = ExpressionCompiler(self._tracking_state, self._roi_index)

        self._compiled: List[tuple[str, Any]] = []
        for rule in normalize_rules(rules):
            self._compiled.append((rule.name, self._compiler.compile(rule.expr)))
        self._logger.info("RuleEngine initialized with %s compiled rules", len(self._compiled))

    def _normalize_detections(self, detections: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for det in detections or []:
            alias = str(det.get("alias", "")).strip()
            bbox = normalize_bbox(det.get("bbox") or det.get("xyxy"))
            if not alias or bbox is None:
                continue
            normalized.append(
                {
                    "alias": alias,
                    "bbox": list(bbox),
                    "score": float(det.get("score", 0.0)),
                    "track_id": det.get("track_id"),
                    "roi_tags": tuple(det.get("roi_tags") or (self._roi_index.tags_for_bbox(bbox) if self._roi_index.has_rois else ())),
                }
            )
        return normalized

    def process(self, detections: List[Mapping[str, Any]]) -> Dict[str, Any]:
        self._logger.debug("Start evaluating frame with %s raw detections", len(detections or []))
        normalized = self._normalize_detections(detections)
        self._logger.debug("Normalized detections count=%s", len(normalized))
        person_objects = self._association.build_associations(normalized)
        if self._roi_index.has_rois:
            for person in person_objects:
                person["roi_tags"] = tuple(self._roi_index.tags_for_bbox(person.get("bbox") or ()))
        self._tracking_state.update(person_objects)

        if not self._compiled:
            self._logger.debug("No compiled rules found, skip rule evaluation")
            return {"detections": normalized, "person_objects": person_objects, "results": {}}

        ctx = {
            "detections": normalized,
            "person_objects": person_objects,
            "tracking_state": self._tracking_state,
        }
        with self._lock:
            results = {name: bool(pred(ctx)) for name, pred in self._compiled}
        for rule_id, triggered in results.items():
            self._logger.debug("Rule evaluated rule_id=%s triggered=%s", rule_id, triggered)
        self._logger.info(
            "Rule evaluation finished: triggered=%s/%s",
            sum(1 for v in results.values() if v),
            len(results),
        )
        return {"detections": normalized, "person_objects": person_objects, "results": results}

    def evaluate(self, detections: List[Mapping[str, Any]]) -> Dict[str, bool]:
        return dict(self.process(detections).get("results") or {})
