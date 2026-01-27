from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Any, Dict
from uuid import uuid4

from clearml import Task
from fastapi import FastAPI, HTTPException

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


def _run_yolo_job(job_id: str, request: YoloTrainRequest) -> None:
    created_at = utc_now_iso()
    job_store.update_job(job_id, status="running")
    outputs_dir = Path("outputs") / job_id
    task = None
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
