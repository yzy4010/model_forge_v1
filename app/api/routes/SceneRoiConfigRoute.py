from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from app.api.services import SceneRoiConfigServer
import json

SceneRoiConfig_Router = APIRouter()
scene_roi_config_db = SceneRoiConfigServer.scene_roi_config_db

@SceneRoiConfig_Router.post("/scene/roi")
def create_roi(cfg: dict = Body(...)):
    """
    新增ROI区域配置
    """
    try:
        scene_id = cfg.get("scene_id")
        camera_id = cfg.get("camera_id")
        roi_id = cfg.get("roi_id")
        name = cfg.get("name")
        semantic_tag = cfg.get("semantic_tag")
        enabled = cfg.get("enabled", True)
        config_version = cfg.get("config_version", 1)
        resolution_width = cfg.get("resolution_width", 640)
        resolution_height = cfg.get("resolution_height", 360)
        geometry = cfg.get("geometry")
        # geometry 若非字符串则转为json字符串
        if geometry is not None and not isinstance(geometry, str):
            geometry = json.dumps(geometry, ensure_ascii=False)
        # 判空
        if not all([scene_id, camera_id, roi_id, name]):
            raise HTTPException(status_code=400, detail="scene_id, camera_id, roi_id, name 为必填项")
        _id = scene_roi_config_db.create_roi(
            scene_id, camera_id, roi_id, name, semantic_tag, enabled,
            config_version, resolution_width, resolution_height, geometry
        )
        return JSONResponse(content={"code": 200, "msg": "ROI配置创建成功", "id": _id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI配置创建失败: {e}"})


@SceneRoiConfig_Router.put("/scene/roi/{roi_id}")
def update_roi(roi_id: str, cfg: dict = Body(...)):
    """
    修改ROI区域配置
    """
    try:
        name = cfg.get("name")
        semantic_tag = cfg.get("semantic_tag")
        enabled = cfg.get("enabled")
        config_version = cfg.get("config_version")
        resolution_width = cfg.get("resolution_width")
        resolution_height = cfg.get("resolution_height")
        geometry = cfg.get("geometry")
        # geometry 若非字符串则转为json字符串
        if geometry is not None and not isinstance(geometry, str):
            geometry = json.dumps(geometry, ensure_ascii=False)
        # 构造更新字段
        fields = {}
        if name is not None:
            fields["name"] = name
        if semantic_tag is not None:
            fields["semantic_tag"] = semantic_tag
        if enabled is not None:
            fields["enabled"] = enabled
        if config_version is not None:
            fields["config_version"] = config_version
        if resolution_width is not None:
            fields["resolution_width"] = resolution_width
        if resolution_height is not None:
            fields["resolution_height"] = resolution_height
        if geometry is not None:
            fields["geometry"] = geometry
        if not fields:
            raise HTTPException(status_code=400, detail="没有需要更新的字段")
        # 构造sql
        sets = []
        values = []
        for k, v in fields.items():
            sets.append(f"{k}=%s")
            values.append(v)
        sql = f"UPDATE scene_roi_configs SET {', '.join(sets)} WHERE roi_id=%s"
        values.append(roi_id)
        updated = scene_roi_config_db.db.execute(sql, tuple(values))
        if updated == 0:
            raise HTTPException(status_code=404, detail="ROI未找到或无参数更新")
        return JSONResponse(content={"code": 200, "msg": "ROI配置更新成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI配置更新失败: {e}"})


@SceneRoiConfig_Router.delete("/scene/roi/{roi_id}")
def delete_roi(roi_id: str):
    """
    删除ROI区域配置
    """
    try:
        sql = "DELETE FROM scene_roi_configs WHERE roi_id=%s"
        deleted = scene_roi_config_db.db.execute(sql, (roi_id,))
        if deleted == 0:
            raise HTTPException(status_code=404, detail="ROI未找到")
        return JSONResponse(content={"code": 200, "msg": "ROI配置删除成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI配置删除失败: {e}"})


@SceneRoiConfig_Router.get("/scene/roi/{roi_id}")
def get_roi(roi_id: str):
    """
    查询单个ROI区域配置详情
    """
    try:
        roi = scene_roi_config_db.get_roi(roi_id)
        if not roi:
            raise HTTPException(status_code=404, detail="ROI未找到")
        return JSONResponse(content={"code": 200, "data": roi})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI查询失败: {e}"})


@SceneRoiConfig_Router.get("/scene/rois")
def list_rois(scene_id: str = Query(None), page: int = Query(1, gt=0), page_size: int = Query(10, gt=0, le=100)):
    """
    分页查询ROI区域配置列表
    """
    try:
        all_rois = scene_roi_config_db.list_rois(scene_id) or []
        total = len(all_rois)
        start = (page - 1) * page_size
        end = start + page_size
        rois = all_rois[start:end]
        return JSONResponse(content={
            "code": 200,
            "data": rois,
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI分页查询失败: {e}"})