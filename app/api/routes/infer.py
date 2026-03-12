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
from app.infer.storage import save_triggered_image
from app.infer.visualize import draw_alias_detections
from app.roi_engine import ROIEngine
from typing import Optional
from app.roi_engine.roi_draw import draw_rois
from app.rule_engine import RuleEngine

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



def _extract_triggered_aliases(expr: object) -> set[str]:
    aliases: set[str] = set()
    if isinstance(expr, dict):
        alias_value = expr.get("alias")
        if isinstance(alias_value, str) and alias_value:
            aliases.add(alias_value)
        elif isinstance(alias_value, list):
            aliases.update(str(item) for item in alias_value if str(item))
        for key in ("and", "or", "all", "any"):
            branch = expr.get(key)
            if isinstance(branch, list):
                for item in branch:
                    aliases.update(_extract_triggered_aliases(item))
        nested_not = expr.get("not")
        if isinstance(nested_not, dict):
            aliases.update(_extract_triggered_aliases(nested_not))
        nested_conditions = expr.get("conditions")
        if isinstance(nested_conditions, dict):
            aliases.update(_extract_triggered_aliases(nested_conditions))
    elif isinstance(expr, list):
        for item in expr:
            aliases.update(_extract_triggered_aliases(item))
    return aliases




def _normalize_rule_expr_to_or(expr: object) -> object:
    """Normalize incoming rule expression to OR logic for grouped conditions."""
    if isinstance(expr, dict):
        normalized = {}
        for key, value in expr.items():
            if key in ("and", "all") and isinstance(value, list):
                normalized["or"] = [_normalize_rule_expr_to_or(item) for item in value]
            elif key in ("or", "any") and isinstance(value, list):
                normalized["or"] = [_normalize_rule_expr_to_or(item) for item in value]
            elif key == "not" and isinstance(value, dict):
                normalized[key] = _normalize_rule_expr_to_or(value)
            elif key == "conditions" and isinstance(value, dict):
                normalized[key] = _normalize_rule_expr_to_or(value)
            else:
                normalized[key] = _normalize_rule_expr_to_or(value) if isinstance(value, (dict, list)) else value
        return normalized
    if isinstance(expr, list):
        return [_normalize_rule_expr_to_or(item) for item in expr]
    return expr
def _build_rule_overlay_text(alerts: list[dict], aliases: list[str], event_results: dict) -> list[str]:
    lines: list[str] = []
    if alerts:
        lines.extend(str(alert.get("message") or alert.get("rule_id") or "").strip() for alert in alerts)
    return [line for line in lines if line]


def _collect_rule_target_aliases(alerts: list[dict], event_results: Dict[str, dict]) -> set[str]:
    aliases: set[str] = set()
    for alert in alerts or []:
        for alias in alert.get("triggered_aliases") or []:
            if isinstance(alias, str) and alias:
                aliases.add(alias)

    if aliases:
        return aliases

    # 没有明确触发别名时，回退到当前帧检测到的模型别名
    return {
        alias
        for alias, result in (event_results or {}).items()
        if bool((result.get("conclusion") or {}).get("detected", False))
    }


def _draw_rule_info(frame, lines: list[str]):
    if frame is None:
        return frame
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            str(line),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return frame

def _run_job(
        job_id: str,
        rtsp_url: str,
        sample_fps: float,
        models_by_alias: Dict[str, AliasModel],
        aliases: list[str],
        sender: WebhookSender,
        roi_config: Optional[dict] = None,
        rule_engine: Optional[RuleEngine] = None,
        rule_meta: Optional[Dict[str, Dict[str, str]]] = None,
        rule_aliases: Optional[Dict[str, set[str]]] = None,
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

    def _build_rule_alerts(rule_eval: Dict[str, bool], event_results: Dict[str, dict]) -> list[dict]:
        alerts = []
        for rid, triggered in (rule_eval or {}).items():
            if not triggered:
                continue
            meta = (rule_meta or {}).get(rid, {})
            aliases_for_rule = sorted((rule_aliases or {}).get(rid, set()))
            detected_aliases = []
            for alias in aliases_for_rule:
                detected = bool(
                    (event_results.get(alias, {}) or {})
                    .get("conclusion", {})
                    .get("detected", False)
                )
                if detected:
                    detected_aliases.append(alias)
            alerts.append(
                {
                    "rule_id": rid,
                    "name": meta.get("name", rid),
                    "message": meta.get("message") or f"Rule triggered: {rid}",
                    "triggered_aliases": detected_aliases,
                    "condition": meta.get("condition"),
                }
            )
        return alerts

    def _log_rule_evaluation(rule_eval: Dict[str, bool], event_results: Dict[str, dict]) -> None:
        if not rule_eval:
            logger.debug("Rule evaluation skipped or empty rule set")
            return

        for rid, triggered in rule_eval.items():
            aliases_for_rule = sorted((rule_aliases or {}).get(rid, set()))
            satisfied_aliases = []
            unsatisfied_aliases = []
            for alias in aliases_for_rule:
                detected = bool((event_results.get(alias, {}) or {}).get("conclusion", {}).get("detected", False))
                if detected:
                    satisfied_aliases.append(alias)
                else:
                    unsatisfied_aliases.append(alias)

            meta = (rule_meta or {}).get(rid, {})
            logger.info(
                "Rule evaluation rule_id=%s triggered=%s condition=%s matched_models=%s unmatched_models=%s",
                rid,
                triggered,
                meta.get("condition"),
                satisfied_aliases,
                unsatisfied_aliases,
            )

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
            if roi_engine and event_results:
                try:
                    for alias_name, alias_result in event_results.items():
                        detections = alias_result.get("detections", [])
                        if detections:
                            processed = roi_engine.apply(detections)
                            alias_result["detections"] = processed
                except Exception:
                    logger.exception("ROI apply failed")
            # ===========================================

            # 保持原有绘制逻辑
            overlay = draw_alias_detections(frame, event_results)

            # 画 ROI，并根据 roi_tags 判断是否命中
            if roi_config:
                try:
                    overlay = draw_rois(overlay, roi_config, event_results)
                except Exception:
                    logger.exception("ROI draw failed")

            rule_results = {"alerts": [], "results": {}}
            if rule_engine and event_results:
                try:
                    detections_for_rule = []
                    for alias_name, alias_result in event_results.items():
                        for det in alias_result.get("detections", []):
                            det_obj = dict(det)
                            det_obj["alias"] = alias_name
                            det_obj["bbox"] = det.get("xyxy")
                            det_obj["score"] = det.get("conf", 0.0)
                            detections_for_rule.append(det_obj)
                    rule_eval = rule_engine.evaluate(detections_for_rule)
                    _log_rule_evaluation(rule_eval, event_results)
                    rule_results = {"results": rule_eval, "alerts": _build_rule_alerts(rule_eval, event_results)}
                except Exception:
                    logger.exception("Rule engine evaluate failed")

            event["rule_results"] = rule_results
            overlay = _draw_rule_info(
                overlay,
                _build_rule_overlay_text(rule_results.get("alerts", []), aliases, event_results),
            )

            with job.frame_lock:
                job.latest_frame_bgr = overlay
                job.latest_frame_ts_ms = ts_ms

            # 仅在有规则触发时保存 overlay 图
            if rule_results.get("alerts"):
                target_aliases = _collect_rule_target_aliases(rule_results.get("alerts", []), event_results)
                for alias_name in sorted(target_aliases):
                    alias_result = event_results.get(alias_name, {})
                    image_info = alias_result.get("image") if isinstance(alias_result, dict) else None
                    overlay_path = image_info.get("overlay_path") if isinstance(image_info, dict) else None
                    try:
                        overlay_path = save_triggered_image(
                            overlay,
                            job_id,
                            frame_idx,
                            alias_name,
                            output_path=overlay_path,
                        )
                        if isinstance(alias_result, dict):
                            alias_result["image"] = {
                                "type": "file",
                                "overlay_path": overlay_path,
                            }
                        logger.info("Saved rule-matched overlay image: %s", overlay_path)
                    except Exception:
                        logger.exception("Failed to save rule-matched overlay image")
            else:
                for alias_name, alias_result in event_results.items():
                    image_info = alias_result.get("image")
                    if image_info and "overlay_path" in image_info:
                        try:
                            if os.path.exists(image_info["overlay_path"]):
                                os.remove(image_info["overlay_path"])
                        except Exception:
                            logger.exception("Failed to remove non-rule overlay image")

            with job.res_lock:
                job.latest_results = copy.deepcopy(event_results)
                job.latest_rule_results = copy.deepcopy(rule_results)

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
    except Exception:
        job_manager.stop_job(job_id)
        raise

    # 传递 ROI 配置
    roi_config = getattr(req.scenario, "roi_config", None)  # 从请求中获取 ROI 配置
    logger.info("ROI 配置接收: ROI config received: %s", roi_config)
    if roi_config is not None:
        roi_config = roi_config.model_dump()
        logger.info("ROI 配置转换: ROI config after model_dump: %s", roi_config)

    raw_rules = getattr(req.scenario, "rule_config", None)
    rule_engine = None
    rule_meta: Dict[str, Dict[str, str]] = {}
    rule_aliases: Dict[str, set[str]] = {}
    logger.info("Rule 配置接收: Rule config received: %s",raw_rules)
    if raw_rules:
        try:
            compiled_rules = []
            for item in raw_rules:
                item_dict = item.model_dump()
                if not item_dict.get("enabled", True):
                    continue
                rule_id = str(item_dict.get("rule_id") or "")
                expr = _normalize_rule_expr_to_or(item_dict.get("expr") or item_dict.get("conditions"))
                if not rule_id or not expr:
                    continue
                compiled_rules.append({"name": rule_id, "expr": expr, "enabled": True})
                rule_meta[rule_id] = {
                    "name": item_dict.get("name") or rule_id,
                    "message": item_dict.get("message") or f"Rule triggered: {rule_id}",
                    "condition": expr,
                }
                rule_aliases[rule_id] = _extract_triggered_aliases(expr)
            if compiled_rules:
                rule_engine = RuleEngine(compiled_rules, roi_config=roi_config)
                logger.info("任务已启动规则引擎 %s 包含 %s 规则", job_id, len(compiled_rules))
                logger.info(f"规则详情: {compiled_rules}")
        except Exception:
            logger.exception("Failed to initialize RuleEngine, disabling rules")
            rule_engine = None
            rule_meta = {}
            rule_aliases = {}

    sample_fps = req.sample_fps if req.sample_fps else DEFAULT_SAMPLE_FPS
    thread = Thread(
        target=_run_job,
        args=(job_id, req.rtsp_url, sample_fps, models_by_alias, aliases, sender, roi_config, rule_engine, rule_meta, rule_aliases),
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
