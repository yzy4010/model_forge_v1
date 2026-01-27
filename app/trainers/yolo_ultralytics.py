from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import contextlib
import csv
import json
import logging
import re
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

from app.services.meta import utc_now_iso


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
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PROGRESS_LINE_RE = re.compile(
    r"(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"(?P<gpu_mem>\S+)\s+"
    r"(?P<box_loss>[-+]?\d*\.?\d+)\s+"
    r"(?P<cls_loss>[-+]?\d*\.?\d+)\s+"
    r"(?P<dfl_loss>[-+]?\d*\.?\d+)\s+"
    r"(?P<instances>\d+)\s+"
    r"(?P<imgsz>\d+):"
    r"(?:\s+(?P<progress_pct>\d+)%\s+)?"
    r".*?(?P<iter>\d+)/(?P<iters>\d+)"
)


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
                        payload["type"] = "metrics"
                        payload["ts"] = utc_now_iso()
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


class ProgressWatcher(threading.Thread):
    def __init__(
        self,
        logs_dir: Path,
        poll_interval: float = 0.4,
        emit_interval: float = 1.0,
    ) -> None:
        super().__init__(daemon=True)
        self._logs_dir = logs_dir
        self._poll_interval = poll_interval
        self._emit_interval = emit_interval
        self._stop_event = threading.Event()
        self._last_key: Optional[Tuple[int, int]] = None
        self._last_emit = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        train_log_path = self._logs_dir / "train.log"
        metrics_path = self._logs_dir / "metrics.jsonl"
        buffer = ""
        while not self._stop_event.is_set():
            if not train_log_path.exists():
                time.sleep(self._poll_interval)
                continue
            with train_log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(0, 0)
                while not self._stop_event.is_set():
                    chunk = handle.read()
                    if not chunk:
                        time.sleep(self._poll_interval)
                        continue
                    chunk = chunk.replace("\r", "\n")
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop() if lines else ""
                    for raw_line in lines:
                        cleaned = _clean_log_line(raw_line)
                        if not cleaned:
                            continue
                        payload = _parse_progress_line(cleaned)
                        if not payload:
                            continue
                        epoch = payload.get("epoch")
                        iteration = payload.get("iter")
                        if epoch is None or iteration is None:
                            continue
                        key = (int(epoch), int(iteration))
                        now = time.monotonic()
                        if key == self._last_key:
                            continue
                        if now - self._last_emit < self._emit_interval:
                            continue
                        payload["type"] = "progress"
                        payload["ts"] = utc_now_iso()
                        with metrics_path.open("a", encoding="utf-8") as metrics_handle:
                            metrics_handle.write(
                                json.dumps(payload, ensure_ascii=False) + "\n"
                            )
                        self._last_key = key
                        self._last_emit = now


def _coerce_value(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def _clean_log_line(line: str) -> str:
    if not line:
        return ""
    cleaned = _ANSI_ESCAPE_RE.sub("", line)
    cleaned = cleaned.replace("\r", "")
    cleaned = cleaned.strip()
    return cleaned


def _parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    match = _PROGRESS_LINE_RE.search(line)
    if not match:
        return None
    payload: Dict[str, Any] = {}
    for key, value in match.groupdict().items():
        if value is None:
            continue
        if key == "gpu_mem":
            payload[key] = value
        else:
            payload[key] = _coerce_value(value)
    if "progress_pct" not in payload:
        percent_match = re.search(r"(\d+)%", line)
        if percent_match:
            payload["progress_pct"] = _coerce_value(percent_match.group(1))
    return payload


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
    progress_watcher = ProgressWatcher(logs_dir)
    progress_watcher.start()
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
            progress_watcher.stop()
            progress_watcher.join(timeout=5)
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
