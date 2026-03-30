from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from app.api.services.RoiConfigServer import roi_config_db
import json

RoiConfigServer_Router = APIRouter()

@RoiConfigServer_Router.post("/roi")
def create_roi(roi: dict = Body(...)):
    """
    新增ROI区域配置
    """
    try:
        camera_id = roi.get("camera_id")
        roi_id = roi.get("roi_id")
        name = roi.get("name")
        semantic_tag = roi.get("semantic_tag")
        enabled = roi.get("enabled", True)
        config_version = roi.get("config_version", 1)
        resolution_width = roi.get("resolution_width", 640)
        resolution_height = roi.get("resolution_height", 360)
        geometry = roi.get("geometry")

        # 判空
        if not all([camera_id, roi_id, name]):
            raise HTTPException(status_code=400, detail="camera_id, roi_id, name 为必填项")

        _id = roi_config_db.create_roi(camera_id, roi_id, name, semantic_tag, enabled, config_version, resolution_width, resolution_height, geometry)
        return JSONResponse(content={"code": 200, "msg": "ROI区域创建成功", "id": _id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI区域创建失败: {e}"})


@RoiConfigServer_Router.put("/roi/{roi_id}")
def update_roi(roi_id: str, roi: dict = Body(...)):
    """
    修改ROI区域配置
    """
    try:
        name = roi.get("name")
        semantic_tag = roi.get("semantic_tag")
        enabled = roi.get("enabled")
        geometry = roi.get("geometry")

        updated = roi_config_db.update_roi(roi_id, name, semantic_tag, enabled, geometry)
        if updated == 0:
            raise HTTPException(status_code=404, detail="ROI区域未找到或无参数更新")
        return JSONResponse(content={"code": 200, "msg": "ROI区域更新成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI区域更新失败: {e}"})


@RoiConfigServer_Router.delete("/roi/{roi_id}")
def delete_roi(roi_id: str):
    """
    删除ROI区域配置
    """
    try:
        deleted = roi_config_db.delete_roi(roi_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="ROI区域未找到")
        return JSONResponse(content={"code": 200, "msg": "ROI区域删除成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI区域删除失败: {e}"})


@RoiConfigServer_Router.get("/roi/{roi_id}")
def get_roi(roi_id: str):
    """
    查询单个ROI区域详情
    """
    try:
        roi = roi_config_db.get_roi(roi_id)
        if not roi:
            raise HTTPException(status_code=404, detail="ROI区域未找到")
        # 字符串化geometry为json
        if roi and "geometry" in roi and roi["geometry"]:
            try:
                roi["geometry"] = json.loads(roi["geometry"])
            except Exception:
                pass
        return JSONResponse(content={"code": 200, "data": roi})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI区域查询失败: {e}"})


@RoiConfigServer_Router.get("/rois")
def list_rois(page: int = Query(1, gt=0), page_size: int = Query(10, gt=0, le=100)):
    """
    分页查询ROI区域列表
    """
    try:
        all_rois = roi_config_db.list_rois() or []
        total = len(all_rois)
        start = (page - 1) * page_size
        end = start + page_size
        rois = all_rois[start:end]
        for roi in rois:
            if "geometry" in roi and roi["geometry"]:
                try:
                    roi["geometry"] = json.loads(roi["geometry"])
                except Exception:
                    pass
        return JSONResponse(content={"code": 200, "data": rois, "total": total, "page": page, "page_size": page_size})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"ROI区域分页查询失败: {e}"})