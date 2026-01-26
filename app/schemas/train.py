from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class YoloTrainRequest(BaseModel):
    model_name_or_path: str = Field(..., description="YOLO model name or path")
    data_yaml: str = Field(..., description="Path to dataset YAML")
    strategy: str = Field(..., description="Training strategy label")
    device_policy: str = Field(..., description="Device policy, e.g. auto/cpu")
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    note: Optional[str] = Field(default=None, description="Optional note")


class TrainResponse(BaseModel):
    job_id: str
    status: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
