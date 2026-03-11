from __future__ import annotations

import time

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.infer.job_registry import job_manager
from app.infer.visualize import draw_alias_detections
from app.roi_engine.roi_draw import draw_rois

router = APIRouter(tags=["preview"])


def _draw_rule_alerts(frame, alerts, results):
    if frame is None:
        return frame
    y = 30
    for alert in alerts or []:
        message = str(alert.get("message") or alert.get("rule_id") or "")
        if message:
            cv2.putText(
                frame,
                message,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28

        triggered_aliases = alert.get("triggered_aliases") or []
        if triggered_aliases:
            alias_line = f"Models: {', '.join(str(alias) for alias in triggered_aliases)}"
            cv2.putText(
                frame,
                alias_line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            y += 26

    return frame


@router.get("/preview/{job_id}")
def preview(job_id: str) -> StreamingResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    def gen():
        boundary = b"--frame\r\n"
        last_sent_after_stop = False
        while True:
            if job.stop_event.is_set() and last_sent_after_stop:
                break

            with job.raw_lock:
                frame = (
                    None
                    if job.latest_raw_frame_bgr is None
                    else job.latest_raw_frame_bgr.copy()
                )

            if frame is None:
                if job.stop_event.is_set():
                    break
                time.sleep(0.05)
                continue
            with job.res_lock:
                results = {} if job.latest_results is None else dict(job.latest_results)
                rule_results = (
                    {} if getattr(job, "latest_rule_results", None) is None
                    else dict(job.latest_rule_results)
                )
            frame = draw_alias_detections(frame, results)

            # 绘制 ROI 区域
            roi_config = getattr(job, 'roi_config', None)
            if roi_config:
                frame = draw_rois(frame, roi_config, results)

            frame = _draw_rule_alerts(frame, rule_results.get("alerts", []), results)
            height, width = frame.shape[:2]
            if width > 960:
                scale = 960 / width
                frame = cv2.resize(frame, (960, int(height * scale)))

            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
            )
            if not ok:
                time.sleep(0.05)
                continue

            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.1)

            if job.stop_event.is_set():
                last_sent_after_stop = True

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
