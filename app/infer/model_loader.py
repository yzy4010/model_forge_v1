from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List

from fastapi import HTTPException
from ultralytics import YOLO

from app.api.schemas.infer import ScenarioModel

logger = logging.getLogger("model_forge.infer")


@dataclass(frozen=True)
class LoadedModel:
    alias: str
    model_id: str
    weights_path: Path
    yolo: YOLO


_MODEL_CACHE: Dict[str, YOLO] = {}
_CACHE_LOCK = Lock()


def _ensure_weights_available(weights_path: Path) -> None:
    if weights_path.exists():
        return
    raise HTTPException(
        status_code=404,
        detail=f"Model weights not found at {weights_path}",
    )


def _resolve_alias(model: ScenarioModel) -> str:
    return model.alias or model.model_id


def load_models(models: List[ScenarioModel]) -> Dict[str, LoadedModel]:
    loaded: Dict[str, LoadedModel] = {}
    for model in models:
        weights_path = Path(model.weights_path)
        _ensure_weights_available(weights_path)
        with _CACHE_LOCK:
            yolo = _MODEL_CACHE.get(model.model_id)
            if yolo is None:
                logger.info(
                    "Loading YOLO model for model_id=%s from %s",
                    model.model_id,
                    weights_path,
                )
                yolo = YOLO(str(weights_path))
                _MODEL_CACHE[model.model_id] = yolo
            else:
                logger.info("Model cache hit for model_id=%s", model.model_id)
        alias = _resolve_alias(model)
        loaded[alias] = LoadedModel(
            alias=alias,
            model_id=model.model_id,
            weights_path=weights_path,
            yolo=yolo,
        )
    return loaded
