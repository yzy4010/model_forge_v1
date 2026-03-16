from __future__ import annotations

from pathlib import Path

import cv2


def _sanitize_alias(alias: str) -> str:
    return alias.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_overlay(frame, job_id: str, frame_idx: int, alias: str) -> str:
    output_dir = Path("outputs") / f"infer_{job_id}" / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_alias = _sanitize_alias(alias)
    filename = f"{frame_idx:08d}_{safe_alias}.jpg"
    path = output_dir / filename

    ok = cv2.imwrite(path.as_posix(), frame)
    if not ok:
        raise ValueError(f"Failed to write overlay image to {path}")

    return path.as_posix()


def get_overlay_path(job_id: str, frame_idx: int, alias: str) -> str:
    """仅生成图片路径，不执行物理写入"""
    output_dir = Path("outputs") / f"infer_{job_id}" / "overlays"
    # 这里依然保留创建目录的逻辑，确保路径是可用的
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_alias = _sanitize_alias(alias)
    filename = f"{frame_idx:08d}_{safe_alias}.jpg"
    path = output_dir / filename

    return path.as_posix()
