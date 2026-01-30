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
