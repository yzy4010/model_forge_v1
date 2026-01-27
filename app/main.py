from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from clearml import Task
from fastapi import FastAPI, HTTPException
import csv

from app.schemas.train import TrainResponse, TrainStatusResponse, YoloTrainRequest
from app.services.job_store import job_store
from app.services.meta import utc_now_iso, write_meta
from app.trainers.yolo_ultralytics import run_yolo_train

app = FastAPI(title="ModelForge v1")


@app.get("/")
def read_root() -> dict:
    return {"message": "ModelForge v1 is running"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "project": "model_forge_v1"}


def _resolve_device(device_policy: str) -> str:
    if device_policy.lower() in {"auto", "cpu"}:
        return "cpu"
    return device_policy


def _coerce_value(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def _filter_rows(rows: Iterable[list[str]]) -> list[list[str]]:
    return [row for row in rows if any(cell.strip() for cell in row)]


def _resolve_results_csv(job_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    artifacts_path = job_dir / "artifacts" / "results.csv"
    if artifacts_path.exists():
        return artifacts_path, "artifacts/results.csv"
    raw_results = list((job_dir / "raw").glob("train*/results.csv"))
    if not raw_results:
        return None, None
    latest = max(raw_results, key=lambda path: path.stat().st_mtime)
    return latest, "raw/train/results.csv"


def _get_value(payload: dict[str, str], options: Iterable[str]) -> Optional[Any]:
    for key in options:
        if key in payload and payload[key] != "":
            return _coerce_value(payload[key])
    return None


def _build_event(payload: dict[str, str]) -> Optional[dict[str, Any]]:
    epoch_value = _get_value(payload, ["epoch"])
    if epoch_value is None:
        return None
    metrics: dict[str, Any] = {}
    losses: dict[str, Any] = {}
    map50 = _get_value(payload, ["metrics/mAP50(B)", "metrics/mAP50", "map50"])
    if map50 is not None:
        metrics["map50"] = map50
    map5095 = _get_value(
        payload,
        ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map50-95", "map50_95"],
    )
    if map5095 is not None:
        metrics["map5095"] = map5095
    precision = _get_value(payload, ["metrics/precision(B)", "precision"])
    if precision is not None:
        metrics["precision"] = precision
    recall = _get_value(payload, ["metrics/recall(B)", "recall"])
    if recall is not None:
        metrics["recall"] = recall

    box_loss = _get_value(payload, ["train/box_loss", "box_loss", "val/box_loss"])
    if box_loss is not None:
        losses["box_loss"] = box_loss
    cls_loss = _get_value(payload, ["train/cls_loss", "cls_loss", "val/cls_loss"])
    if cls_loss is not None:
        losses["cls_loss"] = cls_loss
    dfl_loss = _get_value(payload, ["train/dfl_loss", "dfl_loss", "val/dfl_loss"])
    if dfl_loss is not None:
        losses["dfl_loss"] = dfl_loss

    event: dict[str, Any] = {"epoch": epoch_value, "ts": utc_now_iso()}
    if metrics:
        event["metrics"] = metrics
    if losses:
        event["losses"] = losses
    return event


def _run_yolo_job(job_id: str, request: YoloTrainRequest) -> None:
    created_at = utc_now_iso()
    job_store.update_job(job_id, status="running")
    outputs_dir = Path("outputs") / job_id
    task = None
    dataset_hash = None
    data_path = Path(request.data_yaml)
    if data_path.exists():
        dataset_hash = sha1(data_path.read_bytes()).hexdigest()
    dataset_meta = {
        "data_yaml": request.data_yaml,
        "dataset_id": request.dataset_id,
        "dataset_hash": dataset_hash,
    }
    name_or_path = (
        request.resume_from
        if request.train_mode == "resume" and request.resume_from
        else request.model_name_or_path
    )
    model_meta = {
        "name_or_path": name_or_path,
        "base": {
            "job_id": request.base_job_id,
            "weight": request.base_weight,
        },
    }
    try:
        Task.set_offline(True)
        task_name = f"yolo-train-{job_id}"
        task = Task.init(project_name="ModelForge", task_name=task_name)
        task.connect(
            {
                "model_name_or_path": request.model_name_or_path,
                "data_yaml": request.data_yaml,
                "device_policy": request.device_policy,
                "strategy": request.strategy,
                "note": request.note,
                "train_mode": request.train_mode,
                "dataset_id": request.dataset_id,
                "base_job_id": request.base_job_id,
                "base_weight": request.base_weight,
                "resume_from": request.resume_from,
            },
            name="train_config",
        )
        task.connect(request.hyperparams, name="hyperparams")

        hyperparams: Dict[str, Any] = dict(request.hyperparams)
        hyperparams.setdefault("workers", 0)
        device = _resolve_device(request.device_policy)

        result = run_yolo_train(
            task=task,
            data_yaml=request.data_yaml,
            model_name_or_path=request.model_name_or_path,
            outputs_dir=outputs_dir,
            hyperparams=hyperparams,
            device=device,
            train_mode=request.train_mode,
            resume_from=request.resume_from,
        )
        finished_at = utc_now_iso()
        meta_path = write_meta(
            outputs_dir=outputs_dir,
            job_id=job_id,
            status="completed",
            raw_run_dir=result.get("raw_run_dir"),
            run_dir=result.get("run_dir"),
            artifacts_dir=result.get("artifacts_dir"),
            artifacts=result.get("artifacts", []),
            missing_artifacts=result.get("missing_artifacts", []),
            created_at=created_at,
            finished_at=finished_at,
            error=None,
            clearml_offline_note="ClearML offline artifacts are stored under ~/.clearml/cache/offline",
            train_mode=request.train_mode,
            dataset=dataset_meta,
            model=model_meta,
        )
        result["meta_path"] = str(meta_path.resolve())
        result["created_at"] = created_at
        result["finished_at"] = finished_at
        job_store.update_job(job_id, status="completed", result=result)
    except Exception as exc:
        finished_at = utc_now_iso()
        meta_path = write_meta(
            outputs_dir=outputs_dir,
            job_id=job_id,
            status="failed",
            raw_run_dir=None,
            run_dir=None,
            artifacts_dir=str((outputs_dir / "artifacts").resolve()),
            artifacts=[],
            missing_artifacts=["best.pt", "last.pt", "results.csv", "args.yaml"],
            created_at=created_at,
            finished_at=finished_at,
            error=str(exc),
            clearml_offline_note="ClearML offline artifacts are stored under ~/.clearml/cache/offline",
            train_mode=request.train_mode,
            dataset=dataset_meta,
            model=model_meta,
        )
        job_store.update_job(
            job_id,
            status="failed",
            error=str(exc),
            result={
                "raw_run_dir": None,
                "artifacts_dir": str((outputs_dir / "artifacts").resolve()),
                "artifacts": [],
                "missing_artifacts": ["best.pt", "last.pt", "results.csv", "args.yaml"],
                "meta_path": str(meta_path.resolve()),
                "created_at": created_at,
                "finished_at": finished_at,
            },
        )
    finally:
        if task is not None:
            task.close()


@app.post("/train/yolo", response_model=TrainResponse)
def train_yolo(request: YoloTrainRequest) -> TrainResponse:
    job_id = uuid4().hex
    job_store.create_job(job_id)
    thread = Thread(target=_run_yolo_job, args=(job_id, request), daemon=True)
    thread.start()
    return TrainResponse(job_id=job_id, status="queued")


@app.get("/train/{job_id}", response_model=TrainStatusResponse)
def get_train_status(job_id: str) -> TrainStatusResponse:
    record = job_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    return TrainStatusResponse(
        job_id=job_id,
        status=record.status,
        error=record.error,
        result=record.result,
    )


@app.get("/train/{job_id}/logs")
def get_train_logs(job_id: str, offset: int = 0, limit: int = 20000) -> dict:
    job_dir = Path("outputs") / job_id
    log_path = job_dir / "logs" / "train.log"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job_id not found")
    if offset < 0 or limit <= 0:
        raise HTTPException(status_code=400, detail="Invalid offset/limit")
    if not log_path.exists():
        return {"offset": offset, "next_offset": offset, "text": ""}
    with log_path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(limit)
    text = data.decode("utf-8", errors="replace")
    next_offset = offset + len(data)
    return {"offset": offset, "next_offset": next_offset, "text": text}


@app.get("/train/{job_id}/stream")
def get_train_stream(job_id: str, cursor: int = 0) -> dict:
    record = job_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    if cursor < 0:
        raise HTTPException(status_code=400, detail="Invalid cursor")
    job_dir = Path("outputs") / job_id
    results_path, source = _resolve_results_csv(job_dir)
    if results_path is None:
        return {
            "cursor": cursor,
            "events": [],
            "status": record.status,
            "source": None,
            "waiting_for": "results.csv not created yet (emitted at epoch end)",
        }
    try:
        rows = _filter_rows(
            list(csv.reader(results_path.read_text(encoding="utf-8").splitlines()))
        )
    except OSError:
        rows = []
    if len(rows) < 2:
        return {
            "cursor": 0,
            "events": [],
            "status": record.status,
            "source": source,
            "waiting_for": None,
        }
    headers = rows[0]
    if headers:
        headers[0] = headers[0].lstrip("\ufeff")
    data_rows = rows[1:]
    start_index = min(cursor, len(data_rows))
    events = []
    for row in data_rows[start_index:]:
        if len(row) < len(headers):
            continue
        payload = dict(zip(headers, row))
        event = _build_event(payload)
        if event:
            events.append(event)
    return {
        "cursor": len(data_rows),
        "events": events,
        "status": record.status,
        "source": source,
        "waiting_for": None,
    }
