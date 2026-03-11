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


def save_triggered_image(frame, job_id: str, frame_idx: int, alias: str, output_path: str | None = None) -> str:
    """Save the frame for a triggered rule and return the persisted path."""
    path = output_path or save_overlay(frame, job_id, frame_idx, alias)
    ok = cv2.imwrite(path, frame)
    if not ok:
        raise ValueError(f"Failed to write triggered image to {path}")
    return path
