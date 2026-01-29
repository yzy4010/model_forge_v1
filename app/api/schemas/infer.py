from typing import Dict, Optional

from pydantic import BaseModel, Field


class ScenarioModel(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model name")
    model_path: str = Field(..., description="Path to model weights")


class ScenarioSnapshot(BaseModel):
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Scenario name")
    model: ScenarioModel = Field(..., description="Scenario model snapshot")
    alias_map: Dict[str, str] = Field(
        default_factory=dict, description="Alias mapping for model labels"
    )


class InferParams(BaseModel):
    conf: float = Field(0.25, description="Confidence threshold")
    iou: float = Field(0.45, description="IoU threshold")
    imgsz: int = Field(640, description="Inference image size")
    max_det: int = Field(100, description="Maximum detections per frame")


class InferStreamRequest(BaseModel):
    stream_url: str = Field(..., description="Stream URL or source")
    scenario: ScenarioSnapshot = Field(..., description="Scenario snapshot")
    params: InferParams = Field(..., description="Inference parameters")


class InferStartResponse(BaseModel):
    stream_id: str = Field(..., description="Stream identifier")
    status: str = Field(..., description="Start status")
