from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, TypedDict

import cv2


class Detection(TypedDict):
    cls: int
    label: str
    conf: float
    xyxy: List[float]


class Summary(TypedDict):
    num_det: int
    top_conf: float


class Conclusion(TypedDict):
    detected: bool
    score: float
    text: str


class EventImage(TypedDict):
    data: str
    mime_type: str
    width: int
    height: int


class AliasResult(TypedDict, total=False):
    detections: List[Detection]
    summary: Summary
    conclusion: Conclusion
    image: EventImage


class InferFrameEvent(TypedDict):
    job_id: str
    scenario_id: str
    ts_ms: int
    frame_idx: int
    results: Dict[str, AliasResult]


@dataclass(frozen=True)
class AliasModel:
    yolo: Any
    conf: float
    iou: float
    imgsz: int
    max_det: int
    job_id: str
    scenario_id: str


def _encode_image(frame: Any) -> EventImage:
    height, width = frame.shape[:2]
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame image")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return EventImage(
        data=encoded,
        mime_type="image/jpeg",
        width=width,
        height=height,
    )


def _resolve_meta(models_by_alias: Mapping[str, AliasModel]) -> tuple[str, str]:
    job_ids = {model.job_id for model in models_by_alias.values()}
    scenario_ids = {model.scenario_id for model in models_by_alias.values()}
    if len(job_ids) != 1 or len(scenario_ids) != 1:
        raise ValueError("All alias models must share the same job_id and scenario_id")
    return job_ids.pop(), scenario_ids.pop()


def _build_detections(result: Any, names: Mapping[int, str]) -> List[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    clss = boxes.cls.tolist()

    detections: List[Detection] = []
    for cls_id, conf, coords in zip(clss, confs, xyxy):
        cls_int = int(cls_id)
        label = names.get(cls_int, str(cls_int))
        detections.append(
            Detection(
                cls=cls_int,
                label=label,
                conf=float(conf),
                xyxy=[float(value) for value in coords],
            )
        )
    return detections


def _build_conclusion(num_det: int, top_conf: float, detected: bool) -> Conclusion:
    if detected:
        text = f"detected {num_det} (top_conf={top_conf:.4f})"
    else:
        text = "no detections"
    score = float(top_conf) if detected else 0.0
    return Conclusion(detected=detected, score=score, text=text)


def run_frame(
    models_by_alias: Mapping[str, AliasModel],
    frame: Any,
    ts_ms: int,
    frame_idx: int,
) -> InferFrameEvent:
    job_id, scenario_id = _resolve_meta(models_by_alias)
    results: Dict[str, AliasResult] = {}

    for alias, model in models_by_alias.items():
        prediction = model.yolo.predict(
            frame,
            conf=model.conf,
            iou=model.iou,
            imgsz=model.imgsz,
            max_det=model.max_det,
            verbose=False,
        )
        result = prediction[0] if prediction else None
        names = getattr(model.yolo, "names", {}) or {}

        detections = _build_detections(result, names) if result is not None else []
        detections = [det for det in detections if det["conf"] >= model.conf]

        num_det = len(detections)
        top_conf = max((det["conf"] for det in detections), default=0.0)
        detected = num_det > 0 and top_conf >= model.conf

        if not detected:
            detections = []
            num_det = 0
            top_conf = 0.0

        summary = Summary(num_det=num_det, top_conf=float(top_conf))
        conclusion = _build_conclusion(num_det, top_conf, detected)

        alias_result: AliasResult = {
            "detections": detections,
            "summary": summary,
            "conclusion": conclusion,
        }

        if detected:
            alias_result["image"] = _encode_image(frame)

        results[alias] = alias_result

    return InferFrameEvent(
        job_id=job_id,
        scenario_id=scenario_id,
        ts_ms=ts_ms,
        frame_idx=frame_idx,
        results=results,
    )
