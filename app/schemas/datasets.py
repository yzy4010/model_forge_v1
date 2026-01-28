from typing import Any, Dict, Optional

from pydantic import BaseModel


class DatasetRegisterLocalRequest(BaseModel):
    name: Optional[str] = None
    root_dir: str
    data_yaml: Optional[str] = None


class DatasetRegisterLocalResponse(BaseModel):
    dataset_id: str
    name: Optional[str]
    root_dir: str
    resolved_data_yaml_path: str
    stats: Dict[str, Any]
    created_at: Optional[str]


class DatasetInfoResponse(BaseModel):
    dataset_id: str
    name: Optional[str]
    root_dir: str
    resolved_data_yaml_path: str
    stats: Dict[str, Any]
    created_at: Optional[str]
