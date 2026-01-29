from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import HTTPException

from app.infer.job import InferenceJob


@dataclass
class JobRequestPayload:
    scenario_id: Optional[str] = None
    rtsp_url: Optional[str] = None
    sample_fps: Optional[float] = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, InferenceJob] = {}
        self._lock = Lock()

    def start_job(self, req: Any) -> str:
        with self._lock:
            running_job = self._get_running_job()
            if running_job is not None:
                raise HTTPException(status_code=409, detail="inference job already running")
            payload = self._normalize_payload(req)
            job_id = uuid4().hex
            self._jobs[job_id] = InferenceJob(
                job_id=job_id,
                scenario_id=payload.scenario_id,
                rtsp_url=payload.rtsp_url,
                sample_fps=payload.sample_fps,
                status="running",
                started_at=datetime.utcnow(),
                frame_idx=0,
            )
            return job_id

    def stop_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.stop()
            return True

    def get_job(self, job_id: str) -> Optional[InferenceJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def _get_running_job(self) -> Optional[InferenceJob]:
        for job in self._jobs.values():
            if job.is_running():
                return job
        return None

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
