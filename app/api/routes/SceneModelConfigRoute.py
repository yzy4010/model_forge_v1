from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from app.api.services.SceneModelConfigServer import scene_model_config_db
import json

SceneModelConfig_Router = APIRouter()


@SceneModelConfig_Router.post("/scene/model")
def create_model(cfg: dict = Body(...)):
    """
    新增模型配置
    """
    try:
        scene_id = cfg.get("scene_id")
        alias = cfg.get("alias")
        model_id = cfg.get("model_id")
        weights_path = cfg.get("weights_path")
        labels = cfg.get("labels")
        params = cfg.get("params")
        if labels is not None and not isinstance(labels, str):
            labels = json.dumps(labels, ensure_ascii=False)
        if params is not None and not isinstance(params, str):
            params = json.dumps(params, ensure_ascii=False)
        if not all([scene_id, alias, model_id, weights_path]):
            raise HTTPException(status_code=400, detail="scene_id, alias, model_id, weights_path 为必填项")
        _id = scene_model_config_db.create_model(scene_id, alias, model_id, weights_path, labels, params)
        return JSONResponse(content={"code": 200, "msg": "模型配置创建成功", "id": _id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"模型配置创建失败: {e}"})


@SceneModelConfig_Router.put("/scene/model/{model_id}")
def update_model(model_id: str, cfg: dict = Body(...)):
    """
    修改模型配置
    """
    try:
        alias = cfg.get("alias")
        weights_path = cfg.get("weights_path")
        labels = cfg.get("labels")
        params = cfg.get("params")
        if labels is not None and not isinstance(labels, str):
            labels = json.dumps(labels, ensure_ascii=False)
        if params is not None and not isinstance(params, str):
            params = json.dumps(params, ensure_ascii=False)
        updated = scene_model_config_db.update_model(model_id, alias, weights_path, labels, params)
        if updated == 0:
            raise HTTPException(status_code=404, detail="模型未找到或无参数更新")
        return JSONResponse(content={"code": 200, "msg": "模型配置更新成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"模型配置更新失败: {e}"})


@SceneModelConfig_Router.delete("/scene/model/{model_id}")
def delete_model(model_id: str):
    """
    删除模型配置
    """
    try:
        deleted = scene_model_config_db.delete_model(model_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="模型未找到")
        return JSONResponse(content={"code": 200, "msg": "模型配置删除成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"模型配置删除失败: {e}"})


@SceneModelConfig_Router.get("/scene/model/{model_id}")
def get_model(model_id: str):
    """
    查询单个模型配置详情
    """
    try:
        model = scene_model_config_db.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="模型未找到")
        return JSONResponse(content={"code": 200, "data": model})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"模型查询失败: {e}"})


@SceneModelConfig_Router.get("/scene/models")
def list_models(scene_id: str = Query(None), page: int = Query(1, gt=0), page_size: int = Query(10, gt=0, le=100)):
    """
    分页查询模型配置列表
    """
    try:
        all_models = scene_model_config_db.list_models(scene_id) or []
        total = len(all_models)
        start = (page - 1) * page_size
        end = start + page_size
        models = all_models[start:end]
        return JSONResponse(content={
            "code": 200,
            "data": models,
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"模型分页查询失败: {e}"})