from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Iterable, Optional
import re
from uuid import uuid4

from clearml import Task
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import csv
import json
import os
import shutil
import zipfile
import yaml

from app.schemas.datasets import DatasetUploadResponse
from app.schemas.train import TrainResponse, TrainStatusResponse, YoloTrainRequest
from app.services.job_store import job_store
from app.services.meta import utc_now_iso, write_meta
from app.trainers.yolo_ultralytics import run_yolo_train

app = FastAPI(title="ModelForge v1")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def _get_value(payload: dict[str, str], options: Iterable[str]) -> Optional[Any]:
    for key in options:
        if key in payload and payload[key] != "":
            return _coerce_value(payload[key])
    return None


def _parse_results_csv(results_path: Path) -> list[dict[str, Any]]:
    try:
        lines = results_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = [row for row in csv.reader(lines) if any(row)]
    if len(rows) < 2:
        return []
    headers = rows[0]
    if headers:
        headers[0] = headers[0].lstrip("\ufeff")
    data_rows = rows[1:]
    results: list[dict[str, Any]] = []
    for row in data_rows:
        if len(row) < len(headers):
            continue
        payload = dict(zip(headers, row))
        epoch_value = _get_value(payload, ["epoch"])
        if epoch_value is None:
            continue
        entry = {
            "epoch": epoch_value,
            "box_loss": _get_value(payload, ["train/box_loss", "box_loss", "val/box_loss"]),
            "cls_loss": _get_value(payload, ["train/cls_loss", "cls_loss", "val/cls_loss"]),
            "dfl_loss": _get_value(payload, ["train/dfl_loss", "dfl_loss", "val/dfl_loss"]),
            "precision": _get_value(payload, ["metrics/precision(B)", "precision"]),
            "recall": _get_value(payload, ["metrics/recall(B)", "recall"]),
            "map50": _get_value(payload, ["metrics/mAP50(B)", "metrics/mAP50", "map50"]),
            "map50_95": _get_value(
                payload,
                ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map50-95", "map50_95"],
            ),
        }
        results.append(entry)
    return results


def _find_latest_run_dir(raw_dir: Path) -> Optional[Path]:
    run_dirs = [
        path
        for path in raw_dir.glob("train*")
        if path.is_dir() and path.name.startswith("train")
    ]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _find_results_csv(job_dir: Path) -> Optional[Path]:
    artifacts_results = job_dir / "artifacts" / "results.csv"
    if artifacts_results.exists():
        return artifacts_results
    raw_dir = job_dir / "raw"
    run_dir = _find_latest_run_dir(raw_dir) if raw_dir.exists() else None
    if run_dir:
        candidate = run_dir / "results.csv"
        if candidate.exists():
            return candidate
    return None


def _safe_extract_zip(zip_path: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    base_dir = extracted_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute():
                raise HTTPException(status_code=400, detail="zip entry contains absolute path")
            resolved_path = (base_dir / member_path).resolve()
            if not str(resolved_path).startswith(str(base_dir) + os.sep):
                raise HTTPException(status_code=400, detail="zip entry outside extracted directory")
            if member.is_dir():
                resolved_path.mkdir(parents=True, exist_ok=True)
                continue
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, resolved_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def _coerce_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        try:
            sorted_items = sorted(raw_names.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError):
            sorted_items = sorted(raw_names.items(), key=lambda item: str(item[0]))
        return [str(value) for _, value in sorted_items]
    raise HTTPException(status_code=400, detail="names must be a list or mapping")


def _normalize_dataset_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    parts = list(candidate.parts)
    if candidate.is_absolute() and candidate.anchor in parts:
        parts = [part for part in parts if part != candidate.anchor]
    normalized_parts = [part for part in parts if part not in (".", "..", "")]
    if not normalized_parts:
        return base_dir.resolve()
    return (base_dir / Path(*normalized_parts)).resolve()


def _count_files(directory: Path, extensions: set[str]) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _count_labels(directory: Path) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix == ".txt")


@app.post("/datasets/upload", response_model=DatasetUploadResponse)
def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
) -> DatasetUploadResponse:
    dataset_id = uuid4().hex
    datasets_root = Path("datasets")
    dataset_dir = datasets_root / dataset_id
    extracted_dir = dataset_dir / "extracted"
    raw_zip_path = dataset_dir / "raw.zip"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with raw_zip_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    _safe_extract_zip(raw_zip_path, extracted_dir)

    data_yaml_path = extracted_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise HTTPException(status_code=400, detail="data.yaml not found in extracted dataset")

    raw_data_text = data_yaml_path.read_text(encoding="utf-8")
    raw_data = yaml.safe_load(raw_data_text)
    if not isinstance(raw_data, dict):
        raise HTTPException(status_code=400, detail="data.yaml must contain a mapping")

    if "train" not in raw_data or "val" not in raw_data:
        raise HTTPException(status_code=400, detail="data.yaml must include train and val entries")

    if "nc" not in raw_data or "names" not in raw_data:
        raise HTTPException(status_code=400, detail="data.yaml must include nc and names entries")

    try:
        nc_value = int(raw_data["nc"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="nc must be an integer") from None

    names_list = _coerce_names(raw_data["names"])
    if len(names_list) != nc_value:
        raise HTTPException(status_code=400, detail="names length must match nc")

    train_dir = _normalize_dataset_path(extracted_dir, str(raw_data["train"]))
    val_dir = _normalize_dataset_path(extracted_dir, str(raw_data["val"]))
    test_dir = None
    if raw_data.get("test") is not None:
        test_dir = _normalize_dataset_path(extracted_dir, str(raw_data["test"]))

    if not train_dir.exists():
        raise HTTPException(status_code=400, detail="train images directory not found")
    if not val_dir.exists():
        raise HTTPException(status_code=400, detail="val images directory not found")

    train_labels_dir = train_dir.parent / "labels"
    val_labels_dir = val_dir.parent / "labels"
    if not train_labels_dir.exists():
        raise HTTPException(status_code=400, detail="train labels directory not found")
    if not val_labels_dir.exists():
        raise HTTPException(status_code=400, detail="val labels directory not found")

    train_images = _count_files(train_dir, IMAGE_EXTENSIONS)
    val_images = _count_files(val_dir, IMAGE_EXTENSIONS)
    test_images = _count_files(test_dir, IMAGE_EXTENSIONS) if test_dir and test_dir.exists() else 0

    train_labels = _count_labels(train_labels_dir)
    val_labels = _count_labels(val_labels_dir)
    test_labels_dir = test_dir.parent / "labels" if test_dir else None
    test_labels = _count_labels(test_labels_dir) if test_labels_dir else 0

    resolved_data = dict(raw_data)
    resolved_data["train"] = train_dir.resolve().as_posix()
    resolved_data["val"] = val_dir.resolve().as_posix()
    resolved_data["nc"] = nc_value
    resolved_data["names"] = names_list
    if test_dir and test_dir.exists():
        resolved_data["test"] = test_dir.resolve().as_posix()
    else:
        resolved_data.pop("test", None)

    resolved_yaml_path = dataset_dir / "resolved_data.yaml"
    resolved_yaml_path.write_text(
        yaml.safe_dump(resolved_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    stats = {
        "train": {"images": train_images, "labels": train_labels},
        "val": {"images": val_images, "labels": val_labels},
        "test": {"images": test_images, "labels": test_labels},
        "nc": nc_value,
        "names": names_list,
    }

    meta = {
        "dataset_id": dataset_id,
        "name": name,
        "dataset_dir": str(dataset_dir.resolve()),
        "raw_zip_path": str(raw_zip_path.resolve()),
        "extracted_dir": str(extracted_dir.resolve()),
        "resolved_data_yaml_path": str(resolved_yaml_path.resolve()),
        "stats": stats,
        "raw_data_yaml": raw_data,
        "raw_data_yaml_text": raw_data_text,
    }

    meta_path = dataset_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        dataset_dir=str(dataset_dir.resolve()),
        extracted_dir=str(extracted_dir.resolve()),
        resolved_data_yaml_path=str(resolved_yaml_path.resolve()),
        stats=stats,
    )


def _find_args_yaml(job_dir: Path) -> Optional[Path]:
    artifacts_args = job_dir / "artifacts" / "args.yaml"
    if artifacts_args.exists():
        return artifacts_args
    raw_dir = job_dir / "raw"
    run_dir = _find_latest_run_dir(raw_dir) if raw_dir.exists() else None
    if run_dir:
        candidate = run_dir / "args.yaml"
        if candidate.exists():
            return candidate
    return None


def _parse_epochs_from_args(args_path: Path) -> Optional[int]:
    try:
        lines = args_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    pattern = re.compile(r"^\s*epochs\s*:\s*(.+?)\s*$")
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip().strip("'\"")
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _resolve_epochs_total(job_dir: Path) -> Optional[int]:
    meta_path = job_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        hyperparams = meta.get("hyperparams") or {}
        epochs_value = hyperparams.get("epochs")
        if epochs_value is not None:
            try:
                return int(float(epochs_value))
            except (TypeError, ValueError):
                return None
    args_path = _find_args_yaml(job_dir)
    if args_path:
        return _parse_epochs_from_args(args_path)
    return None


def _build_logs_summary(job_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    job_dir = Path("outputs") / job_id
    results_path = _find_results_csv(job_dir)
    results = _parse_results_csv(results_path) if results_path else []
    best_entry = max(
        (entry for entry in results if entry.get("map50") is not None),
        key=lambda entry: entry["map50"],
        default=None,
    )
    metrics_summary = {
        "best_epoch": best_entry["epoch"] if best_entry else None,
        "best_map50": best_entry["map50"] if best_entry else None,
        "best_map50_95": best_entry["map50_95"] if best_entry else None,
    }
    dataset = meta.get("dataset") or {}
    model = meta.get("model") or {}
    return {
        "job_id": job_id,
        "train_mode": meta.get("train_mode"),
        "dataset_id": dataset.get("dataset_id"),
        "model_name_or_path": model.get("name_or_path"),
        "hyperparams": meta.get("hyperparams") or {},
        "artifacts": meta.get("artifacts") or [],
        "metrics_summary": metrics_summary,
        "created_at": meta.get("created_at"),
        "finished_at": meta.get("finished_at"),
    }


def _build_result_bundle(job_id: str, meta: dict[str, Any]) -> Path:
    job_dir = Path("outputs") / job_id
    bundle_dir = job_dir / "bundle"
    zip_path = bundle_dir / f"model_forge_{job_id}.zip"
    if zip_path.exists():
        return zip_path.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = job_dir / "artifacts"
    artifact_candidates = {
        "best.pt": artifacts_dir / "best.pt",
        "last.pt": artifacts_dir / "last.pt",
        "results.csv": artifacts_dir / "results.csv",
        "args.yaml": artifacts_dir / "args.yaml",
    }

    dataset = meta.get("dataset") or {}
    readme_lines = [
        f"job_id: {job_id}",
        f"train_mode: {meta.get('train_mode')}",
        f"dataset_id: {dataset.get('dataset_id')}",
        f"created_at: {meta.get('created_at')}",
        f"finished_at: {meta.get('finished_at')}",
        "",
    ]
    readme_content = "\n".join(readme_lines)

    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(meta_path, arcname="meta.json")
        bundle.writestr("README.txt", readme_content)
        for filename, source_path in artifact_candidates.items():
            if not source_path.exists():
                continue
            bundle.write(source_path, arcname=f"artifacts/{filename}")
    return zip_path.resolve()


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
    hyperparams: Dict[str, Any] = dict(request.hyperparams)
    hyperparams.setdefault("workers", 0)
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
            hyperparams=hyperparams,
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
            hyperparams=hyperparams,
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
def get_train_logs(job_id: str) -> dict:
    record = job_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    if record.status != "completed":
        raise HTTPException(
            status_code=409, detail="training logs available after completion"
        )
    job_dir = Path("outputs") / job_id
    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="meta.json invalid")
    return _build_logs_summary(job_id, meta)


@app.get("/train/{job_id}/progress")
def get_train_progress(job_id: str, history_tail_size: int = 5) -> dict:
    record = job_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    job_dir = Path("outputs") / job_id
    results_path = _find_results_csv(job_dir)
    results = _parse_results_csv(results_path) if results_path else []
    epochs_total = _resolve_epochs_total(job_dir)
    if not results:
        return {
            "job_id": job_id,
            "status": record.status,
            "epochs_total": epochs_total,
            "epochs_done": 0,
            "last": None,
            "best_so_far": None,
            "history_tail": [],
        }
    best_entry = max(
        (entry for entry in results if entry.get("map50") is not None),
        key=lambda entry: entry["map50"],
        default=None,
    )
    return {
        "job_id": job_id,
        "status": record.status,
        "epochs_total": epochs_total,
        "epochs_done": len(results),
        "last": results[-1],
        "best_so_far": best_entry,
        "history_tail": results[-max(history_tail_size, 0) :] if history_tail_size else [],
    }


@app.get("/train/{job_id}/result")
def get_train_result(job_id: str) -> dict:
    record = job_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    if record.status != "completed":
        raise HTTPException(status_code=409, detail="training not finished")
    job_dir = Path("outputs") / job_id
    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="meta.json invalid")
    results_path = _find_results_csv(job_dir)
    results = _parse_results_csv(results_path) if results_path else []
    best_entry = max(
        (entry for entry in results if entry.get("map50") is not None),
        key=lambda entry: entry["map50"],
        default=None,
    )
    best_metrics = (
        {
            "epoch": best_entry.get("epoch"),
            "map50": best_entry.get("map50"),
            "map50_95": best_entry.get("map50_95"),
            "precision": best_entry.get("precision"),
            "recall": best_entry.get("recall"),
        }
        if best_entry
        else None
    )
    epochs_total = _resolve_epochs_total(job_dir)
    epochs_done = (
        len(results)
        if results
        else (epochs_total if epochs_total is not None else 0)
    )
    dataset = meta.get("dataset") or {}
    model = meta.get("model") or {}
    bundle_path = _build_result_bundle(job_id, meta)
    return {
        "job_id": job_id,
        "status": record.status,
        "train_mode": meta.get("train_mode"),
        "dataset_id": dataset.get("dataset_id"),
        "model_name_or_path": model.get("name_or_path"),
        "hyperparams": meta.get("hyperparams") or {},
        "epochs_total": epochs_total,
        "epochs_done": epochs_done,
        "best": best_metrics,
        "artifacts": meta.get("artifacts") or [],
        "bundle": {"zip_path": str(bundle_path)},
        "created_at": meta.get("created_at"),
        "finished_at": meta.get("finished_at"),
    }


@app.get("/train/{job_id}/stream")
def get_train_stream(job_id: str) -> dict:
    return get_train_result(job_id)
