from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Iterable, Optional
import re
from uuid import uuid4

from clearml import Task
from fastapi import FastAPI, HTTPException
import csv
import json
import zipfile
import yaml

from app.schemas.datasets import (
    DatasetInfoResponse,
    DatasetRegisterLocalRequest,
    DatasetRegisterLocalResponse,
)
from app.schemas.train import (
    TrainResponse,
    YoloTrainContinueRequest,
    YoloTrainNewRequest,
    YoloTrainRequest,
)
from app.services.job_store import job_store
from app.services.meta import utc_now_iso, write_meta
from app.services.yolo_presets import resolve_augment_params, supports_freeze_param
from app.trainers.yolo_ultralytics import run_yolo_train

app = FastAPI(title="ModelForge v1")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_SPEC_WEIGHTS = {
    "yolov8s": "yolov8s.pt",
    "yolov8n": "yolov8n.pt",
    "yolov26": "yolov26.pt",
}


@app.get("/")
def read_root() -> dict:
    return {"message": "ModelForge v1 is running"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "project": "model_forge_v1"}


def _cuda_is_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


def _resolve_device_policy(device_policy: str) -> tuple[str, str]:
    normalized = device_policy.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="unsupported device_policy")
    cuda_available = _cuda_is_available()
    if normalized == "cuda":
        if not cuda_available:
            raise HTTPException(status_code=400, detail="cuda not available")
        return "cuda", "cuda"
    if normalized == "cpu":
        return "cpu", "cpu"
    if cuda_available:
        return "cuda", "cuda"
    return "cpu", "cpu"


def _resolve_device(device_policy: str) -> str:
    device, _ = _resolve_device_policy(device_policy)
    return device


def _resolve_batch_size(batch: int | str, device_kind: str) -> int:
    if isinstance(batch, str):
        if batch.lower() != "auto":
            raise HTTPException(status_code=400, detail="unsupported batch value")
        return 8 if device_kind == "cuda" else 2
    if batch <= 0:
        raise HTTPException(status_code=400, detail="batch must be positive")
    return batch


def _resolve_lr0(base_lr0: float, lr_scale: float) -> float:
    if lr_scale not in {0.5, 1.0, 2.0}:
        raise HTTPException(status_code=400, detail="unsupported lr_scale")
    return base_lr0 * lr_scale


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


def _resolve_model_spec(model_spec: str) -> str:
    normalized = model_spec.strip().lower()
    if normalized not in MODEL_SPEC_WEIGHTS:
        raise HTTPException(status_code=400, detail="unsupported model_spec")
    weight = MODEL_SPEC_WEIGHTS[normalized]
    if normalized == "yolov26" and not Path(weight).exists():
        raise HTTPException(
            status_code=400,
            detail="yolov26.pt not found; please provide weights",
        )
    return weight


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


def _load_job_meta(job_id: str) -> dict[str, Any]:
    meta_path = Path("outputs") / job_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="base_job_id not found")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="base job meta invalid") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="base job meta invalid")
    return payload


def _resolve_resume_dir(meta: dict[str, Any], base_job_id: str) -> Optional[Path]:
    run_dir = meta.get("run_dir")
    if run_dir:
        candidate = Path(str(run_dir))
        if candidate.exists():
            return candidate
    raw_run_dir = meta.get("raw_run_dir")
    if raw_run_dir:
        candidate = Path(str(raw_run_dir))
        if candidate.exists():
            latest = _find_latest_run_dir(candidate)
            if latest:
                return latest
    raw_dir = Path("outputs") / base_job_id / "raw"
    if raw_dir.exists():
        latest = _find_latest_run_dir(raw_dir)
        if latest:
            return latest
    return None


def _load_resolved_data_yaml(dataset_id: str) -> Path:
    dataset_dir = Path("datasets") / dataset_id
    resolved_path = dataset_dir / "resolved_data.yaml"
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="dataset_id not found")
    try:
        payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=500, detail="resolved_data.yaml invalid") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="resolved_data.yaml must contain a mapping")
    return resolved_path


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


def _resolve_data_yaml_source(root_dir: Path, data_yaml: Optional[str]) -> Path:
    if data_yaml is None or str(data_yaml).strip() == "":
        return (root_dir / "data.yaml").expanduser().resolve()
    candidate = Path(str(data_yaml)).expanduser()
    if not candidate.is_absolute():
        candidate = root_dir / candidate
    return candidate.resolve()


def _build_resolved_data_yaml(
    root_dir: Path,
    nc_value: int,
    names_list: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_images_dir = root_dir / "train" / "images"
    train_labels_dir = root_dir / "train" / "labels"
    if (
        not train_images_dir.exists()
        or not train_images_dir.is_dir()
        or not train_labels_dir.exists()
        or not train_labels_dir.is_dir()
    ):
        raise HTTPException(status_code=400, detail="train structure invalid")

    valid_images_dir = root_dir / "valid" / "images"
    valid_labels_dir = root_dir / "valid" / "labels"
    val_images_dir = root_dir / "val" / "images"
    val_labels_dir = root_dir / "val" / "labels"

    if (
        valid_images_dir.exists()
        and valid_images_dir.is_dir()
        and valid_labels_dir.exists()
        and valid_labels_dir.is_dir()
    ):
        val_images = valid_images_dir
        val_labels = valid_labels_dir
    elif (
        val_images_dir.exists()
        and val_images_dir.is_dir()
        and val_labels_dir.exists()
        and val_labels_dir.is_dir()
    ):
        val_images = val_images_dir
        val_labels = val_labels_dir
    else:
        raise HTTPException(status_code=400, detail="missing val/valid")

    test_images_dir = root_dir / "test" / "images"
    test_labels_dir = root_dir / "test" / "labels"

    resolved_data = {
        "train": train_images_dir.resolve().as_posix(),
        "val": val_images.resolve().as_posix(),
        "nc": nc_value,
        "names": names_list,
    }
    if test_images_dir.exists():
        resolved_data["test"] = test_images_dir.resolve().as_posix()

    stats = {
        "nc": nc_value,
        "names": names_list,
        "train_images": _count_files(train_images_dir, IMAGE_EXTENSIONS),
        "valid_images": _count_files(val_images, IMAGE_EXTENSIONS),
        "test_images": _count_files(test_images_dir, IMAGE_EXTENSIONS),
        "train_labels": _count_labels(train_labels_dir),
        "valid_labels": _count_labels(val_labels),
        "test_labels": _count_labels(test_labels_dir),
    }
    return resolved_data, stats


@app.post("/datasets/register_local", response_model=DatasetRegisterLocalResponse)
def register_local_dataset(
    request: DatasetRegisterLocalRequest,
) -> DatasetRegisterLocalResponse:
    root_dir = Path(request.root_dir).expanduser()
    if not root_dir.exists() or not root_dir.is_dir():
        raise HTTPException(status_code=400, detail="root_dir invalid")
    root_dir = root_dir.resolve()

    data_yaml_source = _resolve_data_yaml_source(root_dir, request.data_yaml)
    if not data_yaml_source.exists() or not data_yaml_source.is_file():
        raise HTTPException(status_code=400, detail="missing data.yaml")

    try:
        raw_data = yaml.safe_load(data_yaml_source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail="data.yaml invalid") from exc
    if not isinstance(raw_data, dict):
        raise HTTPException(status_code=400, detail="data.yaml invalid")

    if "nc" not in raw_data or "names" not in raw_data:
        raise HTTPException(status_code=400, detail="nc names missing")

    try:
        nc_value = int(raw_data["nc"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="nc invalid") from None

    names_list = _coerce_names(raw_data["names"])
    if len(names_list) != nc_value:
        raise HTTPException(status_code=400, detail="nc names mismatch")

    resolved_data, stats = _build_resolved_data_yaml(root_dir, nc_value, names_list)

    dataset_id = uuid4().hex
    created_at = utc_now_iso()
    dataset_dir = Path("datasets") / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml_path = dataset_dir / "resolved_data.yaml"
    resolved_yaml_path.write_text(
        yaml.safe_dump(resolved_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    meta = {
        "dataset_id": dataset_id,
        "name": request.name,
        "root_dir": str(root_dir),
        "data_yaml_source_path": str(data_yaml_source),
        "resolved_data_yaml_path": str(resolved_yaml_path.resolve()),
        "stats": stats,
        "created_at": created_at,
    }
    meta_path = dataset_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return DatasetRegisterLocalResponse(
        dataset_id=dataset_id,
        name=request.name,
        root_dir=str(root_dir),
        resolved_data_yaml_path=str(resolved_yaml_path.resolve()),
        stats=stats,
        created_at=created_at,
    )


@app.get("/datasets/{dataset_id}", response_model=DatasetInfoResponse)
def get_dataset_info(dataset_id: str) -> DatasetInfoResponse:
    dataset_dir = Path("datasets") / dataset_id
    meta_path = dataset_dir / "dataset_meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="dataset_id not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="dataset_meta.json is invalid") from exc
    root_dir = meta.get("root_dir") or meta.get("dataset_dir") or ""
    return DatasetInfoResponse(
        dataset_id=meta.get("dataset_id", dataset_id),
        name=meta.get("name"),
        root_dir=str(root_dir),
        resolved_data_yaml_path=meta.get("resolved_data_yaml_path"),
        stats=meta.get("stats", {}),
        created_at=meta.get("created_at"),
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


def _run_yolo_job(
    job_id: str,
    request: YoloTrainRequest,
    meta_hyperparams: Optional[Dict[str, Any]] = None,
    train_hyperparams: Optional[Dict[str, Any]] = None,
) -> None:
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
    hyperparams: Dict[str, Any] = dict(train_hyperparams or request.hyperparams)
    hyperparams.setdefault("workers", 0)
    meta_hparams: Dict[str, Any]
    if meta_hyperparams is None:
        meta_hparams = dict(hyperparams)
    else:
        meta_hparams = dict(meta_hyperparams)
        meta_hparams.setdefault("workers", hyperparams.get("workers", 0))
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
        task.connect(meta_hparams, name="hyperparams")

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
            hyperparams=meta_hparams,
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
            hyperparams=meta_hparams,
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


@app.post("/train/yolo/new", response_model=TrainResponse)
def train_yolo_new(request: YoloTrainNewRequest) -> TrainResponse:
    resolved_data_path = _load_resolved_data_yaml(request.dataset_id)
    model_name_or_path = _resolve_model_spec(request.model_spec)
    params_payload = request.params.dict()
    device_policy = params_payload.get("device_policy", "auto")
    augment_level = params_payload.get("augment_level", "default")
    lr_scale = params_payload.get("lr_scale", 1.0)
    meta_hyperparams = {key: value for key, value in params_payload.items() if value is not None}
    train_hyperparams = dict(meta_hyperparams)
    train_hyperparams.pop("augment_level", None)
    train_hyperparams.pop("lr_scale", None)
    train_hyperparams.pop("device_policy", None)
    _, device_kind = _resolve_device_policy(device_policy)
    batch_value = train_hyperparams.pop("batch", "auto")
    train_hyperparams["batch"] = _resolve_batch_size(batch_value, device_kind)
    try:
        augment_params = resolve_augment_params(augment_level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    train_hyperparams.update(augment_params)
    base_lr0 = float(train_hyperparams.get("lr0", 0.01))
    train_hyperparams["lr0"] = _resolve_lr0(base_lr0, lr_scale)
    meta_hyperparams["batch"] = train_hyperparams["batch"]
    meta_hyperparams["lr0"] = train_hyperparams["lr0"]
    yolo_request = YoloTrainRequest(
        model_name_or_path=model_name_or_path,
        data_yaml=str(resolved_data_path.resolve()),
        strategy="default",
        device_policy=device_policy,
        hyperparams=train_hyperparams,
        dataset_id=request.dataset_id,
        train_mode="from_pretrained",
    )
    job_id = uuid4().hex
    job_store.create_job(job_id)
    outputs_dir = Path("outputs") / job_id
    outputs_dir.mkdir(parents=True, exist_ok=True)
    thread = Thread(
        target=_run_yolo_job,
        args=(job_id, yolo_request, meta_hyperparams, train_hyperparams),
        daemon=True,
    )
    thread.start()
    return TrainResponse(job_id=job_id, status="queued")


@app.post("/train/yolo/continue", response_model=TrainResponse)
def train_yolo_continue(request: YoloTrainContinueRequest) -> TrainResponse:
    base_record = job_store.get_job(request.base_job_id)
    if base_record is None:
        raise HTTPException(status_code=404, detail="base_job_id not found")
    base_meta = _load_job_meta(request.base_job_id)
    base_artifacts_dir = Path("outputs") / request.base_job_id / "artifacts"
    best_weight = base_artifacts_dir / "best.pt"
    if not best_weight.exists():
        raise HTTPException(status_code=400, detail="best.pt not found for base_job_id")
    resolved_data_path = _load_resolved_data_yaml(request.dataset_id)
    params_payload = request.params.dict()
    device_policy = params_payload.get("device_policy", "auto")
    augment_level = params_payload.get("augment_level", "default")
    lr_scale = params_payload.get("lr_scale", 1.0)
    freeze_value = params_payload.get("freeze", 0)
    meta_hyperparams = {key: value for key, value in params_payload.items() if value is not None}
    train_hyperparams = dict(meta_hyperparams)
    train_hyperparams.pop("augment_level", None)
    train_hyperparams.pop("lr_scale", None)
    train_hyperparams.pop("device_policy", None)
    train_hyperparams.pop("freeze", None)
    _, device_kind = _resolve_device_policy(device_policy)
    batch_value = train_hyperparams.pop("batch", "auto")
    train_hyperparams["batch"] = _resolve_batch_size(batch_value, device_kind)
    try:
        augment_params = resolve_augment_params(augment_level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    train_hyperparams.update(augment_params)
    base_lr0 = float(train_hyperparams.get("lr0", 0.01))
    train_hyperparams["lr0"] = _resolve_lr0(base_lr0, lr_scale)
    warnings: list[str] = []

    train_mode = "finetune"
    resume_from = None
    model_name_or_path = str(best_weight.resolve())
    base_weight = "best.pt"
    if request.continue_strategy == "resume_last":
        train_mode = "resume"
        base_weight = "last.pt"
        last_weight = base_artifacts_dir / "last.pt"
        if last_weight.exists():
            resume_from = str(last_weight.resolve())
        else:
            resume_dir = _resolve_resume_dir(base_meta, request.base_job_id)
            if resume_dir:
                resume_from = str(resume_dir.resolve())
        if resume_from is None:
            raise HTTPException(
                status_code=400, detail="resume_last not supported for base_job_id"
            )
    if train_mode == "finetune" and freeze_value:
        if supports_freeze_param():
            train_hyperparams["freeze"] = freeze_value
        else:
            warnings.append("当前版本不支持 freeze")

    if warnings:
        meta_hyperparams["warnings"] = warnings
    meta_hyperparams["batch"] = train_hyperparams["batch"]
    meta_hyperparams["lr0"] = train_hyperparams["lr0"]

    yolo_request = YoloTrainRequest(
        model_name_or_path=model_name_or_path,
        data_yaml=str(resolved_data_path.resolve()),
        strategy=request.continue_strategy,
        device_policy=device_policy,
        hyperparams=train_hyperparams,
        dataset_id=request.dataset_id,
        train_mode=train_mode,
        base_job_id=request.base_job_id,
        base_weight=base_weight,
        resume_from=resume_from,
    )
    job_id = uuid4().hex
    job_store.create_job(job_id)
    outputs_dir = Path("outputs") / job_id
    outputs_dir.mkdir(parents=True, exist_ok=True)
    thread = Thread(
        target=_run_yolo_job,
        args=(job_id, yolo_request, meta_hyperparams, train_hyperparams),
        daemon=True,
    )
    thread.start()
    return TrainResponse(job_id=job_id, status="queued")


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

