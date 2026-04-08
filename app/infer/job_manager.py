from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import HTTPException

from app.infer.job import InferenceJob

logger = logging.getLogger("model_forge.infer.job_manager")


@dataclass
class JobRequestPayload:
    scenario_id: Optional[str] = None
    rtsp_url: Optional[str] = None
    sample_fps: Optional[float] = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, InferenceJob] = {}
        self._lock = Lock()

    @staticmethod
    def _require_scenario_id(raw: Optional[str]) -> str:
        if raw is None:
            raise HTTPException(status_code=400, detail="scenario_id is required")
        s = str(raw).strip()
        if not s:
            raise HTTPException(status_code=400, detail="scenario_id is required")
        return s

    def _scenario_has_active_infer(self, scenario_id: str) -> bool:
        """同一 scenario 在 running 或 stopping 时视为占用，阻塞再启。"""
        for job in self._jobs.values():
            sid = job.scenario_id
            if sid is None:
                continue
            if str(sid).strip() != scenario_id:
                continue
            if job.status in ("running", "stopping"):
                return True
        return False

    def start_job(self, req: Any) -> str:
        with self._lock:
            payload = self._normalize_payload(req)
            scenario_id = self._require_scenario_id(payload.scenario_id)
            if self._scenario_has_active_infer(scenario_id):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"scenario '{scenario_id}' already has an active inference job "
                        "(running or stopping); wait for it to finish or stop it first"
                    ),
                )
            job_id = uuid4().hex
            self._jobs[job_id] = InferenceJob(
                job_id=job_id,
                scenario_id=scenario_id,
                rtsp_url=payload.rtsp_url,
                sample_fps=payload.sample_fps,
                status="running",
                started_at=datetime.utcnow(),
                frame_idx=0,
            )
            return job_id

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in {"stopped", "failed"}:
                return job.snapshot()
            logger.warning("Stop requested job_id=%s status=%s", job_id, job.status)
            job.stop_event.set()
            job.status = "stopping"
            thread = job.thread
            grabber_thread = job.grabber_thread
            preview_thread = job.preview_thread
            sender = job.sender
        if thread is not None:
            thread.join(timeout=3.0)
            alive = thread.is_alive()
            logger.warning("Stop join job_id=%s thread_alive=%s", job_id, alive)
        if grabber_thread is not None:
            grabber_thread.join(timeout=2.0)
            alive = grabber_thread.is_alive()
            logger.warning("Stop join job_id=%s grabber_alive=%s", job_id, alive)
        if preview_thread is not None:
            preview_thread.join(timeout=2.0)
            alive = preview_thread.is_alive()
            logger.warning("Stop join job_id=%s preview_alive=%s", job_id, alive)
        if sender is not None:
            sender.stop(timeout_s=1.0)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.status = "stopped"
            job.stopped_at = datetime.utcnow()
            return job.snapshot()

    def get_job(self, job_id: str) -> Optional[InferenceJob]:
        with self._lock:
            return self._jobs.get(job_id)

    @staticmethod
    def _normalize_payload(req: Any) -> JobRequestPayload:
        if isinstance(req, JobRequestPayload):
            return req
        if isinstance(req, dict):
            return JobRequestPayload(
                scenario_id=req.get("scenario_id"),
                rtsp_url=req.get("rtsp_url"),
                sample_fps=req.get("sample_fps"),
            )
        return JobRequestPayload(
            scenario_id=getattr(req, "scenario_id", None),
            rtsp_url=getattr(req, "rtsp_url", None),
            sample_fps=getattr(req, "sample_fps", None),
        )
