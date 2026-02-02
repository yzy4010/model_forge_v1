from typing import List, Optional

from pydantic import BaseModel, Field


class InferParams(BaseModel):
    conf: float = Field(0.25, description="Confidence threshold")
    iou: float = Field(0.45, description="IoU threshold")
    imgsz: int = Field(640, description="Inference image size")
    max_det: int = Field(50, description="Maximum detections per frame")


class ScenarioModel(BaseModel):
    alias: str = Field(..., description="Model alias")
    model_id: str = Field(..., description="Model identifier")
    weights_path: str = Field(..., description="Path to model weights")
    labels: Optional[List[str]] = Field(
        default=None, description="Labels for model outputs"
    )
    params: InferParams = Field(..., description="Inference parameters")


class ScenarioSnapshot(BaseModel):
    scenario_id: str = Field(..., description="Scenario identifier")
    models: List[ScenarioModel] = Field(..., description="Scenario model snapshots")


class InferStreamRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP stream URL")
    sample_fps: Optional[float] = Field(
        2.0, description="Sampling FPS for RTSP reader"
    )
    scenario: ScenarioSnapshot = Field(..., description="Scenario snapshot")


class InferStartResponse(BaseModel):
    job_id: str = Field(..., description="Inference job identifier")
    status: str = Field(..., description="Start status")
