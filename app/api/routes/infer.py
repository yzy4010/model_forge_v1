from __future__ import annotations

import copy
import logging
import os
import time
from datetime import datetime
from threading import Thread
from typing import Dict
from urllib.parse import urlparse

import cv2
from fastapi import APIRouter, HTTPException

from app.api.schemas.infer import InferStartResponse, InferStreamRequest
from app.infer.inferencer import AliasModel, run_frame, validate_event
from app.infer.job import InferenceJob
from app.infer.job_registry import job_manager
from app.infer.model_loader import load_models
from app.infer.push import WebhookSender
from app.infer.visualize import draw_alias_detections
from app.roi_engine import ROIEngine
from typing import Optional
from app.roi_engine.roi_draw import draw_rois
from app.rule_engine import RuleParser, RuleEngine
from typing import Any

logger = logging.getLogger("model_forge.infer.routes")

router = APIRouter(prefix="/infer", tags=["infer"])

DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_WEBHOOK_URL = "http://127.0.0.1:18080/api/infer/events"


def _resolve_webhook_url() -> str:
    webhook_url = os.getenv("MODEL_FORGE_WEBHOOK_URL", DEFAULT_WEBHOOK_URL)
    if not webhook_url or not (
        webhook_url.startswith("http://") or webhook_url.startswith("https://")
    ):
        raise HTTPException(
            status_code=400,
            detail="webhook_url must start with http:// or https://",
        )
    parsed = urlparse(webhook_url)
    if parsed.hostname == "0.0.0.0":
        raise HTTPException(
            status_code=400,
            detail="webhook_url host cannot be 0.0.0.0",
        )
    if parsed.hostname == "localhost":
        userinfo = ""
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo = f"{userinfo}:{parsed.password}"
            userinfo = f"{userinfo}@"
        port = f":{parsed.port}" if parsed.port else ""
        normalized_netloc = f"{userinfo}127.0.0.1{port}"
        webhook_url = parsed._replace(netloc=normalized_netloc).geturl()
        logger.warning(
            "WEBHOOK_URL_HOST_LOCALHOST normalized to 127.0.0.1 for webhook_url=%s",
            webhook_url,
        )
    return webhook_url


def _build_models(req: InferStreamRequest, job_id: str) -> Dict[str, AliasModel]:
    loaded = load_models(req.scenario.models)
    params_by_alias = {model.alias: model.params for model in req.scenario.models}
    model_id_by_alias = {model.alias: model.model_id for model in req.scenario.models}
    models_by_alias: Dict[str, AliasModel] = {}
    for alias in [model.alias for model in req.scenario.models]:
        model = loaded.get(alias)
        if model is None:
            raise HTTPException(
                status_code=400,
                detail=f"Missing model config for alias '{alias}'",
            )
        params = params_by_alias.get(alias)
        if params is None:
            raise HTTPException(
                status_code=400,
                detail=f"Missing params for model alias '{alias}'",
            )
        model_id = model_id_by_alias.get(alias)
        if model_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Missing model_id for model alias '{alias}'",
            )
        models_by_alias[alias] = AliasModel(
            yolo=model.yolo,
            conf=params.conf,
            iou=params.iou,
            imgsz=params.imgsz,
            max_det=params.max_det,
            model_id=model_id,
            job_id=job_id,
            scenario_id=req.scenario.scenario_id,
        )
    return models_by_alias


def _mark_job_stopped(job: InferenceJob) -> None:
    job.status = "stopped"
    job.stopped_at = datetime.utcnow()


def _mark_job_failed(job: InferenceJob) -> None:
    job.status = "failed"
    job.failed_at = datetime.utcnow()


def _fmt_alias_summary(results: dict, aliases: list) -> str:
    parts = []
    for alias in aliases:
        result = results.get(alias, {}) or {}
        conclusion = result.get("conclusion", {}) or {}
        summary = result.get("summary", {}) or {}
        detected = bool(conclusion.get("detected", False))
        score = float(conclusion.get("score", 0.0) or 0.0)
        num_det = int(summary.get("num_det", 0) or 0)
        parts.append(f"{alias}:{'T' if detected else 'F'}({score:.2f},{num_det})")
    return " ".join(parts)


def _run_job(
        job_id: str,
        rtsp_url: str,
        sample_fps: float,
        models_by_alias: Dict[str, AliasModel],
        aliases: list[str],
        sender: WebhookSender,
        roi_config: Optional[dict] = None,
        rule_engine: Optional[RuleEngine] = None  # 接收规则引擎
) -> None:
    job = job_manager.get_job(job_id)
    if job is None:
        return

    t0 = time.time()
    logger.warning("Runner start job_id=%s scenario_id=%s", job_id, job.scenario_id)

    frames_done = 0
    failed = False

    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    interval_s = 1.0 / sample_fps
    next_emit_time = time.monotonic()
    frame_idx = 0

    # ================= ROI 初始化 =================
    roi_engine = None
    if roi_config:
        try:
            roi_engine = ROIEngine(roi_config)
            logger.info("ROI engine enabled for job %s", job_id)
            # 将 roi_config 保存到 job 对象中，以便 preview 接口使用
            job.roi_config = roi_config
        except Exception:
            logger.exception("Failed to initialize ROI engine, disabling ROI")
            roi_engine = None
    else:
        logger.info("ROI not configured for job %s", job_id)
    # =================================================
    try:
        while True:
            if job.stop_event.is_set():
                logger.warning(
                    "Runner stop_event set; breaking loop job_id=%s frame=%s",
                    job_id,
                    frame_idx,
                )
                break

            if job.status != "running":
                break

            now = time.monotonic()
            if now < next_emit_time:
                time.sleep(min(0.01, next_emit_time - now))
                continue

            with job.raw_lock:
                frame = None if job.latest_raw_frame_bgr is None else job.latest_raw_frame_bgr.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            ts_ms = int(time.time() * 1000)

            # ================= 模型推理 =================
            event = run_frame(models_by_alias, aliases, frame, ts_ms, frame_idx)
            event_results = event.get("results", {}) or {}
            # ===========================================

            # ================= ROI 应用 =================
            all_roi_tags = set()  # 记录本帧触发的所有 ROI 标签，用于规则引擎
            if roi_engine and event_results:
                try:
                    for alias_name, alias_result in event_results.items():
                        detections = alias_result.get("detections", [])
                        if detections:
                            processed = roi_engine.apply(detections)
                            alias_result["detections"] = processed
                            # 收集本帧所有被命中的标签
                            for det in processed:
                                tags = det.get("roi_tags", [])
                                if tags:
                                    all_roi_tags.update(tags)
                except Exception:
                    logger.exception("ROI apply failed")

            # ================= 规则引擎评估 (核心修复版) =================
            if rule_engine:
                # 明确声明类型，防止 IDE 报 'bool vs list' 警告
                engine_data: Dict[str, Any] = {
                    "roi": list(all_roi_tags)
                }

                # 映射 alias 状态
                for alias_name, res in event_results.items():
                    if not res:
                        engine_data[alias_name] = False
                        continue

                    # 关键修复：从 conclusion 字段提取 detected 状态
                    # 因为你的日志显示简化结果中使用了 result.get('conclusion', {}).get('detected')
                    conclusion = res.get("conclusion", {})
                    is_detected = conclusion.get("detected")

                    # 如果 conclusion 里拿不到，再尝试直接从第一层拿 (保底逻辑)
                    if is_detected is None:
                        is_detected = res.get("detected", False)

                    engine_data[alias_name] = is_detected

                # 执行评估
                try:
                    # 只有这里的 engine_data 显示 person: True，规则才会触发
                    logger.warning("RULE_ENGINE_INPUT -> %s", engine_data)
                    rule_engine.evaluate_frame(engine_data)
                except Exception:
                    logger.exception("Rule engine evaluation failed")
            # ==========================================================

            # 保持原有绘制逻辑
            overlay = draw_alias_detections(frame, event_results)

            # 画 ROI，并根据 roi_tags 判断是否命中
            if roi_config:
                try:
                    overlay = draw_rois(overlay, roi_config, event_results)
                except Exception:
                    logger.exception("ROI draw failed")

            with job.frame_lock:
                job.latest_frame_bgr = overlay
                job.latest_frame_ts_ms = ts_ms

            # 更新 overlay_path 指向的图片，添加 ROI 绘制
            for alias_name, alias_result in event_results.items():
                image_info = alias_result.get("image")
                if image_info and "overlay_path" in image_info:
                    overlay_path = image_info["overlay_path"]
                    try:
                        # 保存包含 ROI 的 overlay 到同一个路径
                        cv2.imwrite(overlay_path, overlay)
                        logger.info("Updated overlay image with ROI at: %s", overlay_path)
                    except Exception:
                        logger.exception("Failed to update overlay image with ROI")

            with job.res_lock:
                job.latest_results = copy.deepcopy(event_results)

            try:
                validate_event(event)
            except AssertionError as exc:
                logger.warning(
                    "Invalid inference event; dropped frame job_id=%s frame_idx=%s err=%s",
                    job_id,
                    frame_idx,
                    exc,
                )
                next_emit_time = now + interval_s
                continue

            elapsed = time.time() - t0
            qps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            alias_summary = _fmt_alias_summary(event_results, aliases)

            # 简化 event_results 信息，包含关键数据和ROI信息
            simplified_results = {}
            for alias, result in event_results.items():
                # 提取检测结果和ROI标签
                detections = []
                for det in result.get('detections', []):
                    detections.append({
                        'label': det.get('label', ''),
                        'conf': det.get('conf', 0.0),
                        'roi_tags': det.get('roi_tags', [])
                    })
                
                simplified_results[alias] = {
                    'detected': result.get('conclusion', {}).get('detected', False),
                    'score': result.get('conclusion', {}).get('score', 0.0),
                    'num_det': result.get('summary', {}).get('num_det', 0),
                    'detections': detections
                }

            logger.warning(
                "MF_FRAME_SUMMARY job=%s scenario_id=%s frame=%s qps=%.2f elapsed=%.3fs ts_ms=%d results={%s}",
                job_id,
                job.scenario_id,
                frame_idx,
                qps,
                elapsed,
                ts_ms,
                alias_summary
            )
            logger.debug("Event results: %s", simplified_results)

            sender.enqueue(event)

            job.frame_idx = frame_idx
            frames_done = frame_idx
            frame_idx += 1
            next_emit_time = max(next_emit_time + interval_s, time.monotonic())

    except Exception:
        failed = True
        logger.exception("Runner failed job_id=%s", job_id)
        _mark_job_failed(job)
        job.stop_event.set()

    finally:
        sender.stop(timeout_s=1.0)

        if job.stop_event.is_set() or job.status == "stopping":
            _mark_job_stopped(job)

        reason = "stopped" if job.stop_event.is_set() else "finished"
        elapsed = time.time() - t0

        logger.warning(
            "Runner exit job_id=%s scenario_id=%s reason=%s frames_done=%s elapsed_s=%.3f",
            job_id,
            job.scenario_id,
            reason,
            frames_done,
            elapsed,
        )


def _frame_grabber_loop(job_id: str, rtsp_url: str) -> None:
    job = job_manager.get_job(job_id)
    if job is None:
        return
    logger.warning("Grabber start job_id=%s scenario_id=%s", job_id, job.scenario_id)
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error("Grabber failed to open RTSP job_id=%s url=%s", job_id, rtsp_url)
        job.stop_event.set()
        return
    target_interval = 1.0 / 15.0
    next_tick = time.monotonic()
    try:
        while True:
            if job.stop_event.is_set():
                break
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with job.raw_lock:
                job.latest_raw_frame_bgr = frame
            now = time.monotonic()
            next_tick = max(next_tick + target_interval, now)
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()
        logger.warning("Grabber exit job_id=%s scenario_id=%s", job_id, job.scenario_id)


@router.post("/stream", response_model=InferStartResponse)
def start_infer_stream(req: InferStreamRequest) -> InferStartResponse:
    # 打印整个模型，查看 rule_config 是否有值
    logger.info(f"请求数据详情: {req.model_dump()}")
    if not req.rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url is required")

    job_id = job_manager.start_job(
        {
            "scenario_id": req.scenario.scenario_id,
            "rtsp_url": req.rtsp_url,
            "sample_fps": req.sample_fps,
        }
    )

    try:
        models_by_alias = _build_models(req, job_id)
        aliases = [model.alias for model in req.scenario.models]
        webhook_url = _resolve_webhook_url()
        logger.warning("WEBHOOK_URL_USED=%s", webhook_url)
        sender = WebhookSender(webhook_url)

        # ================= 规则引擎初始化 =================
        rule_engine = None
        # 尝试从 scenario 中获取 rule_config
        rule_config = getattr(req.scenario, "rule_config", None)
        logger.info("Rule 配置接收: Rule config received: %s", rule_config)
        if rule_config:
            try:
                # 转换 Pydantic 模型为 dict 列表
                raw_rules = [r.model_dump() if hasattr(r, "model_dump") else r for r in rule_config]
                rules = [RuleParser.parse_rule(r) for r in raw_rules if r.get("enabled", True)]
                if rules:
                    rule_engine = RuleEngine(rules)
                    logger.info("Rule engine initialized with %d rules for job %s", len(rules), job_id)
            except Exception:
                logger.exception("Failed to initialize Rule engine")
        # =================================================

    except Exception:
        job_manager.stop_job(job_id)
        raise

    # 传递 ROI 配置
    roi_config = getattr(req.scenario, "roi_config", None)  # 从请求中获取 ROI 配置
    logger.info("ROI 配置接收: ROI config received: %s", roi_config)
    if roi_config is not None:
        roi_config = roi_config.model_dump()
        # logger.info("ROI 配置转换: ROI config after model_dump: %s", roi_config)

    sample_fps = req.sample_fps if req.sample_fps else DEFAULT_SAMPLE_FPS

    # 将 rule_engine 传入 _run_job
    thread = Thread(
        target=_run_job,
        args=(job_id, req.rtsp_url, sample_fps, models_by_alias, aliases, sender, roi_config, rule_engine),
        daemon=True,
    )

    job = job_manager.get_job(job_id)
    if job is not None:
        job.thread = thread
        job.sender = sender
        grabber_thread = Thread(
            target=_frame_grabber_loop,
            args=(job_id, req.rtsp_url),
            daemon=True,
        )
        job.grabber_thread = grabber_thread
        grabber_thread.start()
    thread.start()

    return InferStartResponse(job_id=job_id, status="running")


@router.post("/{job_id}/stop")
def stop_infer_stream(job_id: str) -> dict:
    try:
        snapshot = job_manager.stop_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": snapshot["job_id"],
        "status": snapshot["status"],
        "stopped_at": snapshot["stopped_at"],
    }
