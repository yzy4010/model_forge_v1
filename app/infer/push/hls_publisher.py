"""HLS 推流发布器（带速率控制）。

独立线程以恒定帧率向 FFmpeg 写入帧，帧不足时自动重复上一帧。
与工序推理的 FFmpeg RTSP 推流逻辑完全一致——固定输出帧率，帧重复填充。

用法：
    publisher = RateControlledPublisher("outputs/hls/job_xxx", fps=6)
    publisher.set_frame(annotated_bgr_frame)   # 推理线程每帧调用
    publisher.close()
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger("model_forge.infer.hls")

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360
OUTPUT_FPS = 6


class RateControlledPublisher:
    """以恒定帧率向 FFmpeg 写帧，帧不足时重复最后一帧。"""

    def __init__(self, output_dir: str, width: int = OUTPUT_WIDTH,
                 height: int = OUTPUT_HEIGHT, fps: int = OUTPUT_FPS):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self._interval = 1.0 / fps
        self._latest_frame: bytes | None = None
        self._lock = threading.Lock()
        self._running = True
        self._process: subprocess.Popen | None = None
        self._start_ffmpeg()
        # 启动写入线程
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()
        logger.info("RateControlledPublisher start fps=%s output=%s", fps, output_dir)

    def _start_ffmpeg(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        manifest = str(Path(self.output_dir) / "index.m3u8")
        seg_pattern = str(Path(self.output_dir) / "seg_%03d.ts")
        gop = max(4, int(self.fps * 2))
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(gop),
            "-f", "hls",
            "-hls_time", "1",
            "-hls_list_size", "10",
            "-hls_flags", "delete_segments",
            "-hls_segment_filename", seg_pattern,
            manifest,
        ]
        try:
            self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError:
            logger.error("ffmpeg not found")
            self._process = None

    def set_frame(self, bgr_array) -> None:
        """推理线程调用：设置最新标注帧。"""
        import cv2
        h, w = bgr_array.shape[:2]
        if w != self.width or h != self.height:
            bgr_array = cv2.resize(bgr_array, (self.width, self.height),
                                    interpolation=cv2.INTER_AREA)
        with self._lock:
            self._latest_frame = bgr_array.tobytes()

    def _write_loop(self) -> None:
        """后台线程：以固定速率写入 FFmpeg。"""
        last_frame_bytes: bytes | None = None
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                data = self._latest_frame
            if data is None:
                data = last_frame_bytes
            else:
                last_frame_bytes = data
            if data is not None and self._process and self._process.stdin:
                try:
                    self._process.stdin.write(data)
                except BrokenPipeError:
                    logger.warning("RateControlledPublisher broken pipe")
                    break
                except Exception as e:
                    logger.warning("RateControlledPublisher write error: %s", e)
            # 控制写入速率
            elapsed = time.monotonic() - t0
            sleep_t = max(0, self._interval - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
            except Exception:
                pass
        if self._process:
            try:
                self._process.wait(timeout=3)
            except Exception:
                pass
