from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EventImage(BaseModel):
    data: str = Field(..., description="Base64-encoded image payload")
    mime_type: str = Field("image/jpeg", description="Image MIME type")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")


class Detection(BaseModel):
    label: str = Field(..., description="Detected class label")
    conf: float = Field(..., description="Detection confidence")
    bbox: List[float] = Field(
        ..., description="Bounding box in xyxy format [x1, y1, x2, y2]"
    )


class Summary(BaseModel):
    detected: bool = Field(..., description="Whether any detection exists")
    count: int = Field(..., description="Total detections count")
    max_conf: float = Field(..., description="Maximum confidence")


class Conclusion(BaseModel):
    status: str = Field(..., description="Conclusion status")
    message: Optional[str] = Field(default=None, description="Conclusion message")


class AliasResult(BaseModel):
    alias: str = Field(..., description="Alias label")
    detected: bool = Field(..., description="Whether detected for this alias")
    detections: List[Detection] = Field(
        default_factory=list, description="Detections for this alias"
    )
    summary: Summary = Field(..., description="Alias summary")
    conclusion: Conclusion = Field(..., description="Alias conclusion")
    image: Optional[EventImage] = Field(
        default=None, description="Only present when detected=true"
    )


class InferFrameEvent(BaseModel):
    event: str = Field("infer_frame", description="Event type")
    stream_id: str = Field(..., description="Stream identifier")
    frame_id: str = Field(..., description="Frame identifier")
    timestamp: float = Field(..., description="Frame timestamp")
    results: Dict[str, AliasResult] = Field(
        default_factory=dict, description="Alias results keyed by alias"
    )
