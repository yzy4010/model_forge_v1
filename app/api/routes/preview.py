from __future__ import annotations

import time

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.infer.job_registry import job_manager
from app.infer.visualize import draw_alias_detections
from app.roi_engine.roi_draw import draw_rois
import logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["preview"])


@router.get("/preview/{job_id}")
def preview(job_id: str) -> StreamingResponse:
    # Step 1: 获取 Job 实例
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    logger.info("Previewing job_id=%s scenario_id=%s", job_id, job.scenario_id)

    # Step 2: 获取 RuleEngine 和 ROI 配置
    # 注意：这里我们允许 rule_engine 为 None，不再直接抛出 400 错误
    rule_engine = getattr(job, "rule_engine", None)
    roi_config = getattr(job, 'roi_config', None)

    logger.info("RuleEngine initialized: %s, ROI config: %s", rule_engine is not None, roi_config)

    # Step 3: 创建图像生成器
    def gen():
        boundary = b"--frame\r\n"
        last_sent_after_stop = False
        while True:
            if job.stop_event.is_set() and last_sent_after_stop:
                break

            with job.raw_lock:
                frame = None if job.latest_raw_frame_bgr is None else job.latest_raw_frame_bgr.copy()

            if frame is None:
                if job.stop_event.is_set():
                    break
                time.sleep(0.05)
                continue

            with job.res_lock:
                event_results = {} if job.latest_results is None else dict(job.latest_results)

            # 绘制模型原始检测框（始终执行，只要有检测结果）
            overlay = draw_alias_detections(frame, event_results)

            # --- 改进点：增加 rule_engine 存在性检查 ---
            current_roi_status = {}
            if rule_engine:
                # 只有 rule_engine 存在时才执行规则评估
                rule_output = rule_engine.evaluate_frame(event_results)
                current_roi_status = rule_output.get("roi_status", {})

            # 绘制 ROI 区域（如果 roi_config 存在，即使没走规则引擎，也可以绘制静态区域）
            if roi_config:
                overlay = draw_rois(overlay, roi_config, roi_status=current_roi_status)

            # 缩放处理
            display_frame = overlay
            height, width = display_frame.shape[:2]
            if width > 960:
                scale = 960 / width
                display_frame = cv2.resize(display_frame, (960, int(height * scale)))

            # 编码输出
            ok, jpg = cv2.imencode(
                ".jpg",
                display_frame,
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
