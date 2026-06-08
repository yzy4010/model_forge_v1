"""视频流发布器。

将标注后的帧推送到 RTSP 服务器（MediaMTX），由 RTSP 服务器自动生成 HLS，
前端通过 hls.js 播放。与工序推理使用完全相同的策略。

用法：
    publisher = StreamPublisher(job_id, fps=6)
    publisher.write_frame(annotated_bgr_frame)   # 每帧调用
    publisher.close()
"""

from __future__ import annotations

import logging
import subprocess
import os
import torch
from pathlib import Path

logger = logging.getLogger("model_forge.infer.stream")

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360
OUTPUT_FPS = 6

# MediaMTX 默认 HLS 输出路径：http://127.0.0.1:8888/{path}/index.m3u8
MTX_BASE_URL = os.getenv("MODEL_FORGE_MTX_URL", "http://127.0.0.1:8888")


def _ffmpeg_use_nvenc() -> bool:
    if os.getenv("MODEL_FORGE_FFMPEG_NVENC", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool(torch.cuda.is_available())


class StreamPublisher:
    """推流到 RTSP 服务器，由服务器生成 HLS。"""

    def __init__(self, output_dir: str, width: int = OUTPUT_WIDTH,
                 height: int = OUTPUT_HEIGHT, fps: int = OUTPUT_FPS):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.job_id = Path(output_dir).name
        self.process: subprocess.Popen | None = None
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._start()

    @property
    def hls_url(self) -> str:
        return f"{MTX_BASE_URL}/{self.job_id}"

    def _start(self) -> None:
        cmd = self._build_cmd()
        logger.info("StreamPublisher start job=%s rtsp=%s hls=%s/index.m3u8",
                    self.job_id, self._rtsp_url(), self.hls_url)
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError:
            logger.error("ffmpeg not found, stream push disabled")
            self.process = None

    def _rtsp_url(self) -> str:
        return f"rtsp://127.0.0.1:8554/{self.job_id}"

    def _build_cmd(self) -> list[str]:
        """与工序推理完全一致的 FFmpeg 命令。"""
        gop = max(4, int(self.fps * 2))
        use_nvenc = _ffmpeg_use_nvenc()
        if use_nvenc:
            return [
                "ffmpeg",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "h264_nvenc",
                "-preset", os.getenv("MODEL_FORGE_FFMPEG_NVENC_PRESET", "ll"),
                "-b:v", "2M",
                "-g", str(gop),
                "-profile:v", "main",
                "-rtsp_transport", "tcp",
                "-f", "rtsp",
                self._rtsp_url(),
            ]
        else:
            return [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "libx264",
                "-g", str(gop),
                "-rtsp_transport", "tcp",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-f", "rtsp",
                self._rtsp_url(),
            ]

    def write_frame(self, frame) -> None:
        """写入一帧 BGR 图像到 FFmpeg 管道。"""
        if self.process is None or self.process.stdin is None:
            return
        import cv2
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            logger.warning("StreamPublisher broken pipe, restarting...")
            self._start()
        except Exception as e:
            logger.warning("StreamPublisher write error: %s", e)

    def close(self) -> None:
        if self.process and self.process.stdin:
            try:
                self.process.stdin.close()
            except Exception:
                pass
        if self.process:
            try:
                self.process.wait(timeout=3)
            except Exception:
                pass
