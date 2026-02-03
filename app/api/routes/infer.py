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
            event = run_frame(models_by_alias, aliases, frame, ts_ms, frame_idx)
            event_results = event.get("results", {}) or {}
            overlay = draw_alias_detections(frame, event_results)
            with job.frame_lock:
                job.latest_frame_bgr = overlay
                job.latest_frame_ts_ms = ts_ms
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
            logger.warning(
                "MF_FRAME_SUMMARY job=%s frame=%s qps~%.2f results={%s}",
                job_id,
                frame_idx,
                qps,
                alias_summary,
            )
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

    sample_fps = req.sample_fps if req.sample_fps else DEFAULT_SAMPLE_FPS
    thread = Thread(
        target=_run_job,
        args=(job_id, req.rtsp_url, sample_fps, models_by_alias, aliases, sender),
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
