from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field


class YoloTrainRequest(BaseModel):
    model_name_or_path: str = Field(
        "yolov8n.pt", description="YOLO model name or path"
    )
    data_yaml: str = Field(..., description="Path to dataset YAML")
    strategy: str = Field(..., description="Training strategy label")
    device_policy: str = Field(..., description="Device policy, e.g. auto/cpu")
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    note: Optional[str] = Field(default=None, description="Optional note")
    train_mode: Literal["from_pretrained", "finetune", "resume"] = Field(
        "from_pretrained", description="Training mode"
    )
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    base_job_id: Optional[str] = Field(default=None, description="Base job id")
    base_weight: Optional[Literal["best.pt", "last.pt"]] = Field(
        default=None, description="Base weight name"
    )
    resume_from: Optional[str] = Field(
        default=None, description="Resume from run directory"
    )


class TrainResponse(BaseModel):
    job_id: str
    status: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
