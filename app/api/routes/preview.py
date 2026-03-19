from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.infer.job_registry import job_manager
import logging

logger = logging.getLogger("model_forge.preview")

router = APIRouter(tags=["preview"])


@router.get("/preview/{job_id}")
def preview(job_id: str) -> StreamingResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    def gen():
        boundary = b"--frame\r\n"
        last_sent_after_stop = False
        last_ts_ms = -1

        while True:
            if job.stop_event.is_set() and last_sent_after_stop:
                break

            # 推理线程会预编码最新成品帧到 job.latest_encoded_jpg
            with job.preview_lock:
                ts_ms = int(getattr(job, "latest_encoded_ts_ms", 0) or 0)
                jpg_bytes = job.latest_encoded_jpg

            if not jpg_bytes:
                if job.stop_event.is_set():
                    break
                time.sleep(0.02)
                continue

            if ts_ms and ts_ms == last_ts_ms:
                if job.stop_event.is_set():
                    last_sent_after_stop = True
                time.sleep(0.01)
                continue

            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"

            if ts_ms:
                last_ts_ms = ts_ms
            if job.stop_event.is_set():
                last_sent_after_stop = True

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
