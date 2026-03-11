"""Industrial-grade rule engine entrypoint."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping

from app.rule_engine.evaluator import ExpressionCompiler
from app.rule_engine.roi import ROIIndex
from app.rule_engine.schema import normalize_rules
from app.rule_engine.state import DurationState


class RuleEngine:
    """Thread-safe, compiled rule engine with ROI and temporal support."""

    def __init__(self, rules: Any, roi_config: Mapping[str, Any] | None = None):
        self._lock = threading.RLock()
        self._duration_state = DurationState()
        self._compiler = ExpressionCompiler(self._duration_state)
        self._roi_index = ROIIndex(roi_config)

        self._compiled: List[tuple[str, Any]] = []
        for rule in normalize_rules(rules):
            self._compiled.append((rule.name, self._compiler.compile(rule.expr)))

    def _normalize_detections(self, detections: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for det in detections or []:
            bbox = det.get("bbox") or det.get("xyxy") or ()
            roi_tags = tuple(det.get("roi_tags") or self._roi_index.tags_for_bbox(bbox))
            out.append(
                {
                    "alias": str(det.get("alias", "")),
                    "bbox": bbox,
                    "score": float(det.get("score", 0.0)),
                    "roi_tags": roi_tags,
                }
            )
        return out

    def evaluate(self, detections: List[Mapping[str, Any]]) -> Dict[str, bool]:
        normalized = self._normalize_detections(detections)
        ctx = {"detections": normalized}
        with self._lock:
            return {name: bool(pred(ctx)) for name, pred in self._compiled}
