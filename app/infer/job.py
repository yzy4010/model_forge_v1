from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class InferenceJob:
    job_id: str
    scenario_id: Optional[str]
    rtsp_url: Optional[str]
    sample_fps: Optional[float]
    status: str
    started_at: Optional[datetime]
    frame_idx: int = 0

    def stop(self) -> None:
        self.status = "stopped"

    def is_running(self) -> bool:
        return self.status == "running"
