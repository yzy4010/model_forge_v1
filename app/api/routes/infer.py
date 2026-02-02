from __future__ import annotations

import logging
import os
from threading import Thread
from typing import Dict
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException

from app.api.schemas.infer import InferStartResponse, InferStreamRequest
from app.infer.inferencer import AliasModel, run_frame, validate_event
from app.infer.job import InferenceJob
from app.infer.job_manager import JobManager
from app.infer.model_loader import load_models
from app.infer.push import WebhookSender
from app.infer.stream.rtsp_reader import iter_rtsp_frames

logger = logging.getLogger("model_forge.infer.routes")

router = APIRouter(prefix="/infer", tags=["infer"])

job_manager = JobManager()

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
    for alias, model in loaded.items():
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


def _update_job_on_exit(job: InferenceJob | None) -> None:
    if job and job.is_running():
        job.stop()


def _run_job(
    job_id: str,
    rtsp_url: str,
    sample_fps: float,
    models_by_alias: Dict[str, AliasModel],
    sender: WebhookSender,
) -> None:
    try:
        for frame_idx, ts_ms, frame in iter_rtsp_frames(rtsp_url, sample_fps):
            job = job_manager.get_job(job_id)
            if job is None or not job.is_running():
                break
            event = run_frame(models_by_alias, frame, ts_ms, frame_idx)
            try:
                validate_event(event)
            except AssertionError as exc:
                logger.warning(
                    "Invalid inference event; dropped frame job_id=%s frame_idx=%s err=%s",
                    job_id,
                    frame_idx,
                    exc,
                )
                continue
            sender.enqueue(event)
            job.frame_idx = frame_idx
    except Exception:
        logger.exception("Inference job failed (job_id=%s)", job_id)
    finally:
        sender.stop(timeout_s=1.0)
        _update_job_on_exit(job_manager.get_job(job_id))


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
        webhook_url = _resolve_webhook_url()
        logger.warning("WEBHOOK_URL_USED=%s", webhook_url)
        sender = WebhookSender(webhook_url)
    except Exception:
        job_manager.stop_job(job_id)
        raise

    sample_fps = req.sample_fps if req.sample_fps else DEFAULT_SAMPLE_FPS
    thread = Thread(
        target=_run_job,
        args=(job_id, req.rtsp_url, sample_fps, models_by_alias, sender),
        daemon=True,
    )
    thread.start()

    return InferStartResponse(job_id=job_id, status="running")


@router.post("/{job_id}/stop")
def stop_infer_stream(job_id: str) -> dict:
    stopped = job_manager.stop_job(job_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "status": "stopped"}
