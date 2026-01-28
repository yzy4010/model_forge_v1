from typing import Any, Dict

from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    dataset_dir: str
    extracted_dir: str
    resolved_data_yaml_path: str
    stats: Dict[str, Any]
