from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import shutil

from ultralytics import YOLO


_ARTIFACT_FILES = ["best.pt", "last.pt", "results.csv", "args.yaml"]


def _find_latest_run_dir(project_dir: Path) -> Path:
    run_dirs = [path for path in project_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {project_dir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def run_yolo_train(
    task,
    data_yaml: str,
    model_name_or_path: str,
    outputs_dir: Path,
    hyperparams: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    project_dir = outputs_dir / "runs"
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name_or_path)
    train_args = {
        "data": data_yaml,
        "project": str(project_dir),
        "name": "train",
        "device": device,
        **hyperparams,
    }

    results = model.train(**train_args)

    run_dir = None
    if hasattr(results, "save_dir"):
        run_dir = Path(results.save_dir)
    if run_dir is None and hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
        run_dir = Path(model.trainer.save_dir)
    if run_dir is None:
        run_dir = _find_latest_run_dir(project_dir)

    artifacts_dir = outputs_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    uploaded: List[str] = []

    for filename in _ARTIFACT_FILES:
        src_path = run_dir / filename
        if not src_path.exists():
            continue
        dest_path = artifacts_dir / filename
        shutil.copy2(src_path, dest_path)
        try:
            task.upload_artifact(name=filename, artifact_object=str(dest_path))
        except Exception:
            # Keep training result even if ClearML artifact upload fails in offline mode.
            pass
        uploaded.append(filename)

    return {
        "run_dir": str(run_dir),
        "artifacts_dir": str(artifacts_dir),
        "artifacts": uploaded,
    }
