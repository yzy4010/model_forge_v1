from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_meta(
    *,
    outputs_dir: Path,
    job_id: str,
    status: str,
    raw_run_dir: Optional[str],
    run_dir: Optional[str],
    artifacts_dir: Optional[str],
    artifacts: Iterable[str],
    missing_artifacts: Iterable[str],
    created_at: str,
    finished_at: str,
    error: Optional[str] = None,
    clearml_offline_note: Optional[str] = None,
    train_mode: Optional[str] = None,
    dataset: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    meta_path = outputs_dir / "meta.json"
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "raw_run_dir": raw_run_dir,
        "run_dir": run_dir,
        "artifacts_dir": artifacts_dir,
        "artifacts": list(artifacts),
        "missing_artifacts": list(missing_artifacts),
        "created_at": created_at,
        "finished_at": finished_at,
        "error": error,
    }
    if clearml_offline_note:
        payload["clearml_offline_note"] = clearml_offline_note
    if train_mode:
        payload["train_mode"] = train_mode
    if dataset:
        payload["dataset"] = dataset
    if model:
        payload["model"] = model
    if hyperparams is not None:
        payload["hyperparams"] = hyperparams

    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return meta_path
