from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import cv2


def draw_detections(
    frame: Any,
    detections: Iterable[Mapping[str, Any]],
    title: str | None = None,
) -> Any:
    overlay = frame.copy()

    if title:
        cv2.putText(
            overlay,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for det in detections:
        xyxy = det.get("xyxy", [])
        if not isinstance(xyxy, Sequence) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in xyxy)
        label = str(det.get("label", ""))
        conf = det.get("conf")
        text = f"{label} {conf:.2f}" if isinstance(conf, (int, float)) else label

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if text:
            cv2.putText(
                overlay,
                text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return overlay


def _alias_color(alias: str) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    return palette[hash(alias) % len(palette)]


def _parse_bbox(det: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    xyxy = det.get("xyxy")
    if isinstance(xyxy, Sequence) and len(xyxy) == 4:
        x1, y1, x2, y2 = (int(value) for value in xyxy)
        return x1, y1, x2, y2

    bbox = det.get("bbox")
    if isinstance(bbox, Mapping):
        if all(key in bbox for key in ("x1", "y1", "x2", "y2")):
            return (
                int(bbox["x1"]),
                int(bbox["y1"]),
                int(bbox["x2"]),
                int(bbox["y2"]),
            )
        if all(key in bbox for key in ("x", "y", "w", "h")):
            x1 = int(bbox["x"])
            y1 = int(bbox["y"])
            x2 = x1 + int(bbox["w"])
            y2 = y1 + int(bbox["h"])
            return x1, y1, x2, y2

    if all(key in det for key in ("x", "y", "w", "h")):
        x1 = int(det["x"])
        y1 = int(det["y"])
        x2 = x1 + int(det["w"])
        y2 = y1 + int(det["h"])
        return x1, y1, x2, y2

    return None


def draw_alias_detections(
    frame: Any,
    results: Mapping[str, Mapping[str, Any]],
) -> Any:
    overlay = frame.copy()
    for alias, result in results.items():
        detections = result.get("detections", []) if isinstance(result, Mapping) else []
        color = _alias_color(alias)
        for det in detections:
            if not isinstance(det, Mapping):
                continue
            coords = _parse_bbox(det)
            if coords is None:
                continue
            x1, y1, x2, y2 = coords
            label = str(det.get("label", ""))
            conf = det.get("conf")
            conf_text = f"{conf:.2f}" if isinstance(conf, (int, float)) else ""
            text = f"{alias}:{label} {conf_text}".strip()

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            if text:
                cv2.putText(
                    overlay,
                    text,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    return overlay
