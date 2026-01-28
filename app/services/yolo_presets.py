from __future__ import annotations

from typing import Dict


_AUGMENT_PRESETS: Dict[str, Dict[str, float]] = {
    # 平台约定映射：关闭所有增强
    "off": {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    # 轻量增强：轻微颜色+小幅平移/缩放
    "light": {
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    # 默认增强：YOLO 训练常用默认强度
    "default": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    # 强增强：增加 mixup/copy-paste
    "strong": {
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "degrees": 0.0,
        "translate": 0.2,
        "scale": 0.7,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
    },
}


def resolve_augment_params(level: str) -> Dict[str, float]:
    """Map platform augmentation level to Ultralytics train args."""
    key = level.strip().lower()
    if key not in _AUGMENT_PRESETS:
        raise ValueError(f"unsupported augment_level: {level}")
    return dict(_AUGMENT_PRESETS[key])


def supports_freeze_param() -> bool:
    try:
        from ultralytics.cfg import DEFAULT_CFG
    except Exception:
        return False
    return "freeze" in DEFAULT_CFG
