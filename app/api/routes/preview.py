from __future__ import annotations

import time

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.infer.job_registry import job_manager

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

            with job.frame_lock:
                frame = None if job.latest_frame_bgr is None else job.latest_frame_bgr.copy()

            if frame is None:
                if job.stop_event.is_set():
                    break
                time.sleep(0.05)
                continue

            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if not ok:
                time.sleep(0.05)
                continue

            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.05)

            if job.stop_event.is_set():
                last_sent_after_stop = True

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
