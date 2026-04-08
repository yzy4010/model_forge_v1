# app/api/infer_runner_api.py
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.services.dbHelp import db
from app.services.video_stream_legacy.common.redisManager import RedisManager
from app.services.infer_runner import InferRunner

router = APIRouter(prefix="/infer", tags=["infer"])
logger = logging.getLogger("model_forge.fjy_infer")


class SceneProcessConfigSave(BaseModel):
    """工序推理场景主配置保存（对应表 scene_process_configs）"""

    scene_process_id: str = Field(..., max_length=64, description="工序场景唯一ID")
    rtsp_url: str = Field(..., max_length=512, description="RTSP 视频流地址")
    scene_process_name: Optional[str] = Field(None, max_length=128, description="工序推理场景名称")
    cameras_id: str = Field(..., max_length=64, description="关联摄像头ID")
    cameras_name: str = Field(..., max_length=64, description="关联摄像头名称")
    action_id: str = Field(..., max_length=64, description="关联检测动作ID")
    action_name: str = Field(..., max_length=64, description="关联检测动作名称")

# 简单做法：模块级单例，整个进程共用一个 RedisManager + InferRunner
_redis = RedisManager()
_runner = InferRunner(_redis)


def _get_cameras_id_by_scene_process_id(scene_process_id: str) -> Optional[str]:
    """根据 scene_process_id 查 scene_process_configs.cameras_id。"""
    rows = db.fetch_dict(
        "SELECT cameras_id FROM scene_process_configs WHERE scene_process_id=%s LIMIT 1",
        (scene_process_id,),
    )
    if not rows:
        return None
    cid = rows[0].get("cameras_id")
    if cid is None:
        return None
    s = str(cid).strip()
    return s if s else None


def _mark_scene_process_running_by_scene_process_id(scene_process_id: str) -> None:
    """推理已启动成功时，将对应 scene_process_id 行的 status 置为 1。"""
    try:
        db.execute(
            "UPDATE scene_process_configs SET status=1 WHERE scene_process_id=%s",
            (scene_process_id,),
        )
    except Exception as e:
        logger.warning(
            "scene_process_configs 更新 status=1 失败 scene_process_id=%s: %s",
            scene_process_id,
            e,
        )


def _mark_scene_process_stopped_by_scene_process_id(scene_process_id: str) -> None:
    """推理停止成功（或确认无任务）时，将对应 scene_process_id 行的 status 置为 0。"""
    try:
        db.execute(
            "UPDATE scene_process_configs SET status=0 WHERE scene_process_id=%s",
            (scene_process_id,),
        )
    except Exception as e:
        logger.warning(
            "scene_process_configs 更新 status=0 失败 scene_process_id=%s: %s",
            scene_process_id,
            e,
        )


@router.post("/start/{scene_process_id}")
def start_infer(scene_process_id: str):
    """
    根据工序场景 ID 启动推理：先查 scene_process_configs 得到 cameras_id，再调用底层推理启动。
    启动成功（ok=true）时，将该 scene_process_id 对应行的 status 置为 1。
    """
    cameras_id = _get_cameras_id_by_scene_process_id(scene_process_id)
    if not cameras_id:
        return {
            "ok": False,
            "msg": f"未找到工序配置或 cameras_id 为空：scene_process_id={scene_process_id}",
            "error": None,
            "taskid": None,
        }
    result = _runner.start(cameras_id)
    if result.get("ok"):
        _mark_scene_process_running_by_scene_process_id(scene_process_id)
    return result


@router.post("/stop/{scene_process_id}")
def stop_infer(scene_process_id: str):
    """
    根据工序场景 ID 停止推理：先查 scene_process_configs 得到 cameras_id，再调用底层停止。
    停止成功（ok=true，含已停止、无运行任务）时，将该 scene_process_id 对应行的 status 置为 0。
    """
    cameras_id = _get_cameras_id_by_scene_process_id(scene_process_id)
    if not cameras_id:
        return {
            "ok": False,
            "msg": f"未找到工序配置或 cameras_id 为空：scene_process_id={scene_process_id}",
        }
    result = _runner.stop(cameras_id)
    if result.get("ok"):
        _mark_scene_process_stopped_by_scene_process_id(scene_process_id)
    return result


@router.get("/status")
def list_infer_tasks():
    """
    查看当前所有推理任务状态。
    """
    return _runner.status()


@router.get("/camera-configs")
def list_all_camera_configs():
    """
    从 Java 拉取全部摄像头推理配置（getCameraRetrievalListFromPython），用于查全量。
    """
    return _runner.list_all_java_processor_configs()


@router.post("/scene-process-config")
def save_scene_process_config(body: SceneProcessConfigSave):
    """
    保存工序推理场景配置：写入 `scene_process_configs`。
    若 `scene_process_id` 已存在则更新（依赖该字段 UNIQUE）。
    """
    sql = """
        INSERT INTO scene_process_configs (
            scene_process_id, rtsp_url, scene_process_name,
            cameras_id, cameras_name, action_id, action_name
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            rtsp_url = VALUES(rtsp_url),
            scene_process_name = VALUES(scene_process_name),
            cameras_id = VALUES(cameras_id),
            cameras_name = VALUES(cameras_name),
            action_id = VALUES(action_id),
            action_name = VALUES(action_name),
            updated_at = CURRENT_TIMESTAMP
    """
    params = (
        body.scene_process_id,
        body.rtsp_url,
        body.scene_process_name,
        body.cameras_id,
        body.cameras_name,
        body.action_id,
        body.action_name,
    )
    conn, cursor = db.get_conn()
    try:
        cursor.execute(sql, params)
        conn.commit()
        return {
            "ok": True,
            "msg": "保存成功",
            "scene_process_id": body.scene_process_id,
            "affected_rows": cursor.rowcount,
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "msg": str(e),
                "scene_process_id": body.scene_process_id,
            },
        ) from e
    finally:
        cursor.close()
        conn.close()


_SCENE_PROCESS_LIST_SQL = """
    SELECT id, scene_process_id, rtsp_url, scene_process_name,
           cameras_id, cameras_name, action_id, action_name,
           status, created_at, updated_at
    FROM scene_process_configs
    ORDER BY updated_at DESC
"""


def _normalize_scene_process_row(row: dict) -> dict:
    """与表 scene_process_configs.status 对齐：0 未运行，1 运行中。"""
    out = dict(row)
    try:
        st = out.get("status")
        out["status"] = int(st) if st is not None else 0
    except (TypeError, ValueError):
        out["status"] = 0
    return out


@router.get("/scene-process-configs")
def list_scene_process_configs(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=200, description="每页条数"),
):
    """
    分页查询工序推理场景配置列表（表 `scene_process_configs`），按 `updated_at` 倒序。
    每条含 status：0 未运行，1 运行中。
    """
    result = db.fetch_dict_page(
        _SCENE_PROCESS_LIST_SQL.strip(),
        None,
        page=page,
        page_size=page_size,
    )
    if result.get("data") is None:
        raise HTTPException(
            status_code=500,
            detail={"msg": "工序推理场景列表查询失败"},
        )
    items = [_normalize_scene_process_row(r) for r in (result["data"] or [])]
    return {
        "ok": True,
        "items": items,
        "total": result["total"],
        "page": result["page"],
        "page_size": page_size,
        "total_pages": result["total_pages"],
    }
