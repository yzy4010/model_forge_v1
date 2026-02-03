from __future__ import annotations

import time

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.infer.job_registry import job_manager
from app.infer.visualize import draw_alias_detections

router = APIRouter(tags=["preview"])


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
            frame = draw_alias_detections(frame, results)
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
