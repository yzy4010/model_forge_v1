from typing import List, Optional, Dict, Any
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


# ================= 新增 ROI 结构 =================

class ROIResolution(BaseModel):
    width: int
    height: int


class ROIGeometry(BaseModel):
    points: List[List[int]]


class ROIItem(BaseModel):
    roi_id: str
    name: str
    semantic_tag: str
    enabled: bool = True
    geometry: ROIGeometry


class ROIConfig(BaseModel):
    camera_id: str
    config_version: int
    resolution: ROIResolution
    rois: List[ROIItem]


# ================= 修改 ScenarioSnapshot =================

class ScenarioSnapshot(BaseModel):
    scenario_id: str = Field(..., description="Scenario identifier")
    models: List[ScenarioModel] = Field(..., description="Scenario model snapshots")
    roi_config: Optional[ROIConfig] = None   # 👈 关键新增


class InferStreamRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP stream URL")
    sample_fps: Optional[float] = Field(
        2.0, description="Sampling FPS for RTSP reader"
    )
    scenario: ScenarioSnapshot = Field(..., description="Scenario snapshot")


class InferStartResponse(BaseModel):
    job_id: str = Field(..., description="Inference job identifier")
    status: str = Field(..., description="Start status")