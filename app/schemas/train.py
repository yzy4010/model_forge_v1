from typing import Any, Dict, Optional, Literal, Union

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


class YoloTrainNewParams(BaseModel):
    epochs: int = Field(50, description="Number of training epochs")
    imgsz: int = Field(640, description="Training image size")
    batch: Union[int, str] = Field("auto", description="Batch size or auto")
    patience: int = Field(10, description="Early stopping patience")
    augment_level: str = Field("default", description="Augmentation level label")
    lr_scale: float = Field(1.0, description="Learning rate scale factor")
    device_policy: str = Field("auto", description="Device policy, e.g. auto/cpu")
    seed: Optional[int] = Field(None, description="Random seed")
    val: bool = Field(True, description="Run validation")
    save: bool = Field(True, description="Save checkpoints")


class YoloTrainNewRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset identifier")
    model_spec: str = Field(..., description="YOLO model spec name")
    params: YoloTrainNewParams = Field(..., description="Training parameters")


class YoloTrainContinueParams(BaseModel):
    epochs: int = Field(20, description="Number of training epochs")
    imgsz: int = Field(640, description="Training image size")
    batch: Union[int, str] = Field("auto", description="Batch size or auto")
    patience: int = Field(10, description="Early stopping patience")
    augment_level: str = Field("default", description="Augmentation level label")
    lr_scale: float = Field(1.0, description="Learning rate scale factor")
    freeze: int = Field(0, description="Freeze layers count")
    device_policy: str = Field("auto", description="Device policy, e.g. auto/cpu")
    seed: Optional[int] = Field(None, description="Random seed")
    val: bool = Field(True, description="Run validation")
    save: bool = Field(True, description="Save checkpoints")


class YoloTrainContinueRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset identifier")
    base_job_id: str = Field(..., description="Base job identifier")
    continue_strategy: Literal["finetune_best", "resume_last"] = Field(
        "finetune_best", description="Continue strategy"
    )
    params: YoloTrainContinueParams = Field(..., description="Training parameters")
