from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class JobRecord:
    status: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            self._jobs[job_id] = JobRecord(status=status)

    def get_status(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_result(
        self,
        job_id: str,
        status: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                record = JobRecord(status=status or "queued")
                self._jobs[job_id] = record
            if status is not None:
                record.status = status
            if error is not None:
                record.error = error
            if result is not None:
                record.result = result


job_store = JobStore()
