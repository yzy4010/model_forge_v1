from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import contextlib
import csv
import json
import logging
import shutil
import sys
import threading
import time

# 禁用 Ultralytics 自带的 ClearML 回调，避免与我们平台的 ClearML Task 冲突
try:
    from ultralytics.utils import SETTINGS
    SETTINGS["clearml"] = False
except Exception:
    pass


from ultralytics import YOLO


_ARTIFACT_FILES = ["best.pt", "last.pt", "results.csv", "args.yaml"]
_METRIC_FIELDS = {
    "epoch",
    "epochs",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95",
    "metrics/mAP50",
    "precision",
    "recall",
    "map50",
    "map50-95",
    "map50_95",
    "loss",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
}


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class MetricsWatcher(threading.Thread):
    def __init__(
        self,
        project_dir: Path,
        logs_dir: Path,
        poll_interval: float = 1.5,
        resume_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._project_dir = project_dir
        self._logs_dir = logs_dir
        self._poll_interval = poll_interval
        self._resume_dir = resume_dir
        self._stop_event = threading.Event()
        self._last_epoch: Optional[int] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        metrics_path = self._logs_dir / "metrics.jsonl"
        while not self._stop_event.is_set():
            run_dir = self._resolve_run_dir()
            results_path = run_dir / "results.csv"
            if results_path.exists():
                payload = self._read_latest_metrics(results_path)
                if payload:
                    epoch = payload.get("epoch")
                    if epoch is None or epoch != self._last_epoch:
                        with metrics_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        if epoch is not None:
                            self._last_epoch = epoch
            time.sleep(self._poll_interval)

    def _resolve_run_dir(self) -> Path:
        if self._resume_dir and self._resume_dir.exists():
            return self._resume_dir
        run_dir = _find_latest_run_dir(self._project_dir)
        return run_dir or self._project_dir

    def _read_latest_metrics(self, results_path: Path) -> Optional[Dict[str, Any]]:
        try:
            lines = results_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        rows = [row for row in csv.reader(lines) if any(row)]
        if len(rows) < 2:
            return None
        headers = rows[0]
        last_row = rows[-1]
        if len(last_row) < len(headers):
            return None
        payload = dict(zip(headers, last_row))
        filtered: Dict[str, Any] = {}
        for key, value in payload.items():
            if key not in _METRIC_FIELDS:
                continue
            filtered[key] = _coerce_value(value)
        epoch_value = payload.get("epoch")
        if epoch_value is not None:
            filtered["epoch"] = _coerce_value(epoch_value)
        return filtered or None


def _coerce_value(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


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
    train_mode: str,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    job_root = outputs_dir.resolve()
    job_root.mkdir(parents=True, exist_ok=True)
    project_dir = (job_root / "raw").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = (job_root / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = logs_dir / "train.log"
    resume_dir = Path(resume_from).resolve() if resume_from else None

    if train_mode == "resume":
        if resume_dir is None or not resume_dir.exists():
            raise ValueError("resume_from must be provided and exist for resume mode")
        model = YOLO(str(resume_dir))
    else:
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

    metrics_watcher = MetricsWatcher(project_dir, logs_dir, resume_dir=resume_dir)
    metrics_watcher.start()
    logger = logging.getLogger("model_forge.train")
    with train_log_path.open("a", encoding="utf-8") as log_handle:
        tee_stream = TeeStream(sys.stdout, log_handle)
        tee_err_stream = TeeStream(sys.stderr, log_handle)
        handler = logging.StreamHandler(tee_stream)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            with contextlib.redirect_stdout(tee_stream), contextlib.redirect_stderr(
                tee_err_stream
            ):
                if train_mode == "resume":
                    train_args["resume"] = True
                logger.info("Starting YOLO training with mode=%s", train_mode)
                model.train(**train_args)
        except Exception as exc:
            if train_mode == "resume":
                raise RuntimeError(f"Resume failed: {exc}") from exc
            raise
        finally:
            metrics_watcher.stop()
            metrics_watcher.join(timeout=5)
            logger.removeHandler(handler)

    run_dir = resume_dir or _find_latest_run_dir(project_dir)
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
