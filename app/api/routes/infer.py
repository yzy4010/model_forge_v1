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
from app.roi_engine.roi_draw import draw_rois_by_rule
from app.rule_engine import RuleParser, RuleEngine
from typing import Any
import asyncio
from app.infer.push.socket_manager import socket_manager
from app.api.services.SceneConfigServer import scene_config_db

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


def _redact_url(url: str) -> str:
    if not url:
        return url
    # simple redact for scheme://user:pass@
    try:
        return url.split("://", 1)[0] + "://***:***@" + url.split("@", 1)[1]
    except Exception:
        return url


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

            # ================= ROI 应用 (保持不变，用于提供基础数据) =================
            all_roi_tags = set()
            if roi_engine and event_results:
                try:
                    for alias_name, alias_result in event_results.items():
                        detections = alias_result.get("detections", [])
                        if detections:
                            processed = roi_engine.apply(detections)
                            alias_result["detections"] = processed
                            for det in processed:
                                tags = det.get("roi_tags", [])
                                if tags:
                                    all_roi_tags.update(tags)
                except Exception:
                    logger.exception("ROI apply failed")

            # ================= 规则引擎评估 (强绑定修复版) =================
            triggered_rois = set()
            triggered_aliases = set()  # 新增：记录是哪些别名触发了告警
            active_alerts = []  # 修改点 新增列表，用于存储告警详情

            if rule_engine:
                # 1. 构造“带空间信息”的 engine_data
                # 以前是 {'smoking': True}, 现在是 {'smoking': ['safe_zone']}
                engine_data: Dict[str, Any] = {
                    "roi": list(all_roi_tags)  # 全局命中的所有 ROI 标签
                }

                for alias_name, res in event_results.items():
                    if not res:
                        engine_data[alias_name] = []
                        continue

                    # 提取该别名下【所有检测框】命中的 ROI 标签集合
                    alias_hit_tags = set()
                    detections = res.get("detections", [])
                    for det in detections:
                        tags = det.get("roi_tags", [])
                        alias_hit_tags.update(tags)

                    # 存入列表，供 AtomicCondition 的 alias_in_roi 操作符使用
                    engine_data[alias_name] = list(alias_hit_tags)

                # 2. 执行规则评估
                try:
                    # 这里的日志会显示精确的绑定关系，例如：'smoking': ['safe_zone']
                    logger.warning("RULE_ENGINE_INPUT (Strong Bound) -> %s", engine_data)

                    for rule in rule_engine.rules:
                        # 确保 Rule 类有 enabled 属性
                        if getattr(rule, 'enabled', True):
                            # 评估时，AtomicCondition 会去匹配具体的 ROI 标签
                            if rule.root_condition.evaluate(engine_data):

                                # 记录触发了告警的别名 (例如: 'smoking')
                                # 假设你的 Rule 对象有获取涉及别名的方法，如果没有，我们可以从 condition 里拿
                                involved_aliases = rule.get_involved_aliases()
                                triggered_aliases.update(involved_aliases)

                                # 触发告警
                                rule.action(rule.rule_id, engine_data, rule.action_params)
                                # 提取该规则涉及到的 ROI 标签用于变色
                                involved = rule.get_involved_rois()
                                triggered_rois.update(involved)
                                rule_name = getattr(rule, "name", None) or rule.rule_id

                                #  修改点：收集准备推送到 Java 端的规则详情
                                active_alerts.append({
                                    "rule_id": rule.rule_id,
                                    "rule_name": rule_name,
                                    "message": rule.action_params.get("message", "实时告警"),
                                    "level": rule.action_params.get("level", "warning")
                                })

                except Exception:
                    logger.exception("Rule engine evaluation failed")

            # ================= 绘制逻辑 (生成用于展示的图片) =================
            # 先画检测框，再画 ROI
            overlay = draw_alias_detections(frame, event_results)

            if roi_config:
                try:
                    # 这里使用当前帧计算出的 triggered_rois 进行变色绘制
                    overlay = draw_rois_by_rule(overlay, roi_config, triggered_rois)
                except Exception:
                    logger.exception("ROI draw failed")

            # 1. 优先同步结果状态 (供预览接口判断变色)
            with job.res_lock:
                job.latest_results = copy.deepcopy(event_results)
                # 这里的变量名必须与 preview 接口 getattr 的名字完全一致
                job.latest_triggered_rois = triggered_rois.copy()
                logger.debug(
                    "SYNC_SAVE job_id=%s job_obj_id=%s triggered_rois=%s",
                    job_id,
                    id(job),
                    list(job.latest_triggered_rois),
                )

            # 2. 更新当前展示帧 (已经画好框和变色 ROI 的图片)
            with job.frame_lock:
                job.latest_frame_bgr = overlay
                job.latest_frame_ts_ms = ts_ms

            # 3. 更新物理文件 (overlay_path)
            # ================= 定向保存部分 =================
            if len(triggered_rois) > 0:
                for alias_name in triggered_aliases:
                    # 只从 event_results 里挑选真正触发了规则的别名进行保存
                    alias_result = event_results.get(alias_name)
                    if not alias_result:
                        continue

                    image_info = alias_result.get("image")
                    if image_info and "overlay_path" in image_info:
                        overlay_path = image_info["overlay_path"]
                        try:
                            cv2.imwrite(overlay_path, overlay)
                            logger.info("💾 ALARM_SAVE alias=%s overlay_path=%s", alias_name, overlay_path)
                        except Exception:
                            logger.exception("Failed to save alarm image alias=%s", alias_name)
            else:
                # 可选：如果没触发，可以考虑是否删除旧的本地图片，或者直接跳过（推荐直接跳过）
                # 无规则，或者保存所以模型推理图片
                pass
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

            # ================= 消息推送 =================
            # 修改点构造包含规则信息的 Payload
            push_payload = {
                "job_id": job_id,
                "scenario_id": job.scenario_id,
                "frame_idx": frame_idx,
                "ts_ms": ts_ms,
                "results": simplified_results,  # 使用你已经简化好的推理结果
                "triggered_rules": active_alerts,  # 触发的具体规则详情
                "triggered_rois": list(triggered_rois)  # 触发的 ROI 标签列表
            }

            # 修改点 推送
            if sender:
                # 注意：如果之前用了 validate_event(event)，
                sender.enqueue(push_payload)

            # === SocketIO 推送开始 ===
            try:
                # 尝试获取 socket_manager 绑定的主 loop
                # 如果你在 startup 没存，可以尝试用 asyncio.get_event_loop()
                # 但在多线程下，最好是通过单例传递过来的 loop
                main_loop = getattr(socket_manager, 'loop', None)

                if main_loop and main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        socket_manager.emit_event("inference_update", push_payload, room=job_id),
                        main_loop
                    )
                    # 修改后（测试用）：去掉 room 参数，直接广播给所有人
                    # asyncio.run_coroutine_threadsafe(
                    #     socket_manager.emit_event("inference_update", push_payload),
                    #     main_loop
                    # )
                else:
                    # 如果这里报错，说明 startup_event 没跑或者没存上 loop
                    logger.error("❌ 未找到主事件循环，无法推送 SocketIO 消息")
            except Exception as e:
                logger.error(f"SocketIO 跨线程推送失败: {e}")

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
            ts_ms = int(time.time() * 1000)
            with job.raw_lock:
                job.latest_raw_frame_bgr = frame
                job.latest_raw_frame_ts_ms = ts_ms
            now = time.monotonic()
            next_tick = max(next_tick + target_interval, now)
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()
        logger.warning("Grabber exit job_id=%s scenario_id=%s", job_id, job.scenario_id)


def _preview_encoder_loop(job_id: str) -> None:
    job = job_manager.get_job(job_id)
    if job is None:
        return

    preview_width = int(os.getenv("MODEL_FORGE_PREVIEW_WIDTH", "640") or "640")
    preview_width = max(160, preview_width)
    preview_fps = float(os.getenv("MODEL_FORGE_PREVIEW_FPS", "8") or "8")
    if preview_fps <= 0:
        preview_fps = 8.0
    min_interval_s = 1.0 / preview_fps
    next_at = time.monotonic()
    quality = int(os.getenv("MODEL_FORGE_PREVIEW_JPEG_QUALITY", "55") or "55")
    quality = max(30, min(95, quality))

    last_raw_ts_ms = -1

    while not job.stop_event.is_set():
        now = time.monotonic()
        if now < next_at:
            time.sleep(min(0.01, next_at - now))
            continue
        next_at = max(next_at + min_interval_s, time.monotonic())

        with job.raw_lock:
            raw_ts_ms = int(getattr(job, "latest_raw_frame_ts_ms", 0) or 0)
            raw = None if job.latest_raw_frame_bgr is None else job.latest_raw_frame_bgr.copy()

        if raw is None:
            continue
        if raw_ts_ms and raw_ts_ms == last_raw_ts_ms:
            continue
        last_raw_ts_ms = raw_ts_ms

        # snapshot inference overlays (cheap locks, avoid deep copy here)
        with job.res_lock:
            results = None if job.latest_results is None else dict(job.latest_results)
            triggered_rois = set(getattr(job, "latest_triggered_rois", set()))

        roi_config = getattr(job, "roi_config", None)

        frame = raw
        try:
            if results:
                frame = draw_alias_detections(frame, results)
            if roi_config:
                frame = draw_rois_by_rule(frame, roi_config, triggered_rois)
        except Exception:
            logger.exception("Preview overlay draw failed job_id=%s", job_id)

        try:
            h, w = frame.shape[:2]
            if w > preview_width:
                scale = preview_width / w
                frame = cv2.resize(
                    frame,
                    (preview_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, jpg_buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if ok:
                with job.preview_lock:
                    job.latest_encoded_jpg = jpg_buf.tobytes()
                    job.latest_encoded_ts_ms = raw_ts_ms
        except Exception:
            logger.exception("Preview encode failed job_id=%s", job_id)


@router.post("/stream", response_model=InferStartResponse)
def start_infer_stream(req: InferStreamRequest) -> InferStartResponse:
    logger.info(
        "INFER_STREAM_START scenario_id=%s models=%s sample_fps=%s rtsp_url=%s",
        req.scenario.scenario_id,
        len(req.scenario.models or []),
        req.sample_fps,
        _redact_url(req.rtsp_url or ""),
    )
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
        logger.info("WEBHOOK_URL_USED=%s", _redact_url(webhook_url))
        sender = WebhookSender(webhook_url)

        # ================= 规则引擎初始化 =================
        rule_engine = None
        # 尝试从 scenario 中获取 rule_config
        rule_config = getattr(req.scenario, "rule_config", None)
        logger.info("RULE_CONFIG_RECEIVED enabled=%s", bool(rule_config))
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
    logger.info("ROI_CONFIG_RECEIVED enabled=%s", roi_config is not None)
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
        preview_thread = Thread(
            target=_preview_encoder_loop,
            args=(job_id,),
            daemon=True,
        )
        job.preview_thread = preview_thread
        preview_thread.start()
    thread.start()

    return InferStartResponse(job_id=job_id, status="running")


@router.post("/{job_id}/stop")
def stop_infer_stream(job_id: str) -> dict:
    try:
        snapshot = job_manager.stop_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    try:
        rows = scene_config_db.mark_stopped_by_job_id(job_id)
        if rows:
            logger.info("scene_configs status=0 by job_id=%s rows=%s", job_id, rows)
    except Exception as e:
        logger.warning("scene_configs 按 job_id 更新 status=0 失败 job_id=%s: %s", job_id, e)
    return {
        "job_id": snapshot["job_id"],
        "status": snapshot["status"],
        "stopped_at": snapshot["stopped_at"],
    }
