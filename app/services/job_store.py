from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class JobRecord:
    job_id: str
    status: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobStore:
    def __init__(self, outputs_root: Path | None = None) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()
        self._outputs_root = outputs_root or Path("outputs")

    def create_job(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = JobRecord(job_id=job_id, status="queued")

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is not None:
            return record
        record = self._load_from_disk(job_id)
        if record is None:
            return None
        with self._lock:
            self._jobs[job_id] = record
        return record

    def update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                record = JobRecord(job_id=job_id, status=fields.get("status", "queued"))
                self._jobs[job_id] = record
            if "status" in fields and fields["status"] is not None:
                record.status = fields["status"]
            if "error" in fields:
                record.error = fields["error"]
            if "result" in fields:
                record.result = fields["result"]

    def _load_from_disk(self, job_id: str) -> Optional[JobRecord]:
        meta_path = self._outputs_root / job_id / "meta.json"
        if not meta_path.exists():
            return None
        try:
            payload = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return None
        result = dict(payload)
        result.pop("job_id", None)
        result.pop("status", None)
        result.pop("error", None)
        raw_run_dir = payload.get("raw_run_dir")
        artifacts_dir = payload.get("artifacts_dir")
        if raw_run_dir is not None:
            result["raw_run_dir"] = str(Path(raw_run_dir))
        if artifacts_dir is not None:
            result["artifacts_dir"] = str(Path(artifacts_dir))
        result["meta_path"] = str(meta_path.resolve())
        return JobRecord(
            job_id=payload.get("job_id", job_id),
            status=payload.get("status", "unknown"),
            error=payload.get("error"),
            result=result,
        )


job_store = JobStore()
