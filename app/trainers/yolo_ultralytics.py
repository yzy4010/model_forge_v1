from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil

# 禁用 Ultralytics 自带的 ClearML 回调，避免与我们平台的 ClearML Task 冲突
try:
    from ultralytics.utils import SETTINGS
    SETTINGS["clearml"] = False
except Exception:
    pass


from ultralytics import YOLO


_ARTIFACT_FILES = ["best.pt", "last.pt", "results.csv", "args.yaml"]


def _find_latest_run_dir(project_dir: Path) -> Optional[Path]:
    run_dirs = [
        path
        for path in project_dir.glob("train*")
        if path.is_dir() and path.name.startswith("train")
    ]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _collect_artifacts(run_dir: Path, artifacts_dir: Path) -> Tuple[List[str], List[str]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    uploaded: List[str] = []
    missing: List[str] = []

    file_map = {
        "best.pt": run_dir / "weights" / "best.pt",
        "last.pt": run_dir / "weights" / "last.pt",
        "results.csv": run_dir / "results.csv",
        "args.yaml": run_dir / "args.yaml",
    }

    for filename in _ARTIFACT_FILES:
        src_path = file_map[filename]
        if not src_path.exists():
            missing.append(filename)
            continue
        dest_path = artifacts_dir / filename
        shutil.copy2(src_path, dest_path)
        uploaded.append(filename)

    return uploaded, missing


def run_yolo_train(
    task,
    data_yaml: str,
    model_name_or_path: str,
    outputs_dir: Path,
    hyperparams: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    job_root = outputs_dir.resolve()
    job_root.mkdir(parents=True, exist_ok=True)
    project_dir = (job_root / "raw").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name_or_path)
    name = "train"
    train_args = {
        "task": "detect",
        "data": data_yaml,
        "project": str(project_dir),
        "name": name,
        "exist_ok": False,
        "device": device,
        **hyperparams,
    }

    model.train(**train_args)

    run_dir = _find_latest_run_dir(project_dir)
    if run_dir is None:
        fallback_dir = Path("runs") / "detect" / "outputs" / job_root.name / "raw"
        run_dir = _find_latest_run_dir(fallback_dir)
    if run_dir is None:
        run_dir = project_dir

    artifacts_dir = job_root / "artifacts"
    uploaded, missing = _collect_artifacts(run_dir, artifacts_dir)

    for filename in uploaded:
        dest_path = artifacts_dir / filename
        try:
            task.upload_artifact(name=filename, artifact_object=str(dest_path))
        except Exception:
            # Keep training result even if ClearML artifact upload fails in offline mode.
            pass

    return {
        "raw_run_dir": str(project_dir),
        "run_dir": str(run_dir.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "artifacts": uploaded,
        "missing_artifacts": missing,
    }
