from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Optional, TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from app.infer.push import WebhookSender


@dataclass
class InferenceJob:
    job_id: str
    scenario_id: Optional[str]
    rtsp_url: Optional[str]
    sample_fps: Optional[float]
    status: str
    started_at: Optional[datetime]
    frame_idx: int = 0
    stop_event: Event = field(default_factory=Event)
    thread: Optional[Thread] = None
    sender: Optional["WebhookSender"] = None
    stopped_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    latest_frame_bgr: Optional[Any] = None
    latest_frame_ts_ms: int = 0
    frame_lock: Lock = field(default_factory=Lock)

    def stop(self) -> None:
        self.stop_event.set()
        self.status = "stopped"
        self.stopped_at = datetime.utcnow()

    def is_running(self) -> bool:
        return self.status == "running"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scenario_id": self.scenario_id,
            "rtsp_url": self.rtsp_url,
            "sample_fps": self.sample_fps,
            "status": self.status,
            "started_at": self.started_at.isoformat() + "Z"
            if self.started_at
            else None,
            "stopped_at": self.stopped_at.isoformat() + "Z"
            if self.stopped_at
            else None,
            "failed_at": self.failed_at.isoformat() + "Z"
            if self.failed_at
            else None,
            "frame_idx": self.frame_idx,
        }
