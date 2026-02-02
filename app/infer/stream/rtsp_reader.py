from __future__ import annotations

import time
from collections.abc import Generator
from threading import Event
from typing import Optional

import cv2


def iter_rtsp_frames(
    rtsp_url: str,
    sample_fps: float,
    stop_event: Optional[Event] = None,
) -> Generator[tuple[int, int, "cv2.Mat"], None, None]:
    """Yield frames from an RTSP stream at a target sample rate.

    Sampling is based on wall clock time. Frames are dropped if reading is slow
    to avoid lagging behind real time.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")

    interval_s = 1.0 / sample_fps
    next_emit_time = time.monotonic()
    frame_idx = 0

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from RTSP stream")

            now = time.monotonic()
            if now < next_emit_time:
                continue

            ts_ms = int(time.time() * 1000)
            yield frame_idx, ts_ms, frame
            frame_idx += 1
            next_emit_time = now + interval_s
    finally:
        cap.release()
