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

        self._compiled: List[tuple[str, Mapping[str, Any], set[str], Any]] = []
        for rule in normalize_rules(rules):
            self._compiled.append(
                (
                    rule.name,
                    rule.expr,
                    self._extract_aliases(rule.expr),
                    self._compiler.compile(rule.expr),
                )
            )
        self._logger.info("RuleEngine initialized with %s compiled rules", len(self._compiled))

    def _extract_aliases(self, expr: Mapping[str, Any]) -> set[str]:
        aliases: set[str] = set()
        if not isinstance(expr, Mapping):
            return aliases

        alias_value = expr.get("alias")
        if isinstance(alias_value, str) and alias_value.strip():
            aliases.add(alias_value.strip())
        elif isinstance(alias_value, (list, tuple, set)):
            for alias in alias_value:
                alias_name = str(alias).strip()
                if alias_name:
                    aliases.add(alias_name)

        for key in ("and", "or"):
            for child in expr.get(key, []) or []:
                if isinstance(child, Mapping):
                    aliases.update(self._extract_aliases(child))

        child_not = expr.get("not")
        if isinstance(child_not, Mapping):
            aliases.update(self._extract_aliases(child_not))

        duration_cfg = expr.get("duration")
        if isinstance(duration_cfg, Mapping):
            child = duration_cfg.get("condition") or duration_cfg.get("where")
            if isinstance(child, Mapping):
                aliases.update(self._extract_aliases(child))
        return aliases

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
            results = {}
            for rule_id, rule_expr, alias_filters, pred in self._compiled:
                triggered = bool(pred(ctx))
                results[rule_id] = triggered

                available_aliases = {str(det.get("alias", "")).strip() for det in normalized if det.get("alias")}
                if alias_filters:
                    matched_models = sorted(alias for alias in alias_filters if alias in available_aliases)
                    unmatched_models = sorted(alias for alias in alias_filters if alias not in available_aliases)
                else:
                    matched_models = sorted(available_aliases)
                    unmatched_models = []

                self._logger.info(
                    "Rule evaluation rule_id=%s triggered=%s condition=%s matched_models=%s unmatched_models=%s",
                    rule_id,
                    triggered,
                    rule_expr,
                    matched_models,
                    unmatched_models,
                )
                if triggered:
                    self._logger.info(
                        "Rule triggered detail rule_id=%s condition=%s target_models=%s",
                        rule_id,
                        rule_expr,
                        matched_models,
                    )
        self._logger.info(
            "Rule evaluation finished: triggered=%s/%s",
            sum(1 for v in results.values() if v),
            len(results),
        )
        return {"detections": normalized, "person_objects": person_objects, "results": results}

    def evaluate(self, detections: List[Mapping[str, Any]]) -> Dict[str, bool]:
        return dict(self.process(detections).get("results") or {})
