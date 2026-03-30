from fastapi import APIRouter, HTTPException, Query,Body
from fastapi.responses import JSONResponse
from app.api.services import ModelConfigServer
import json


model_config_db = ModelConfigServer.model_config_db
model_router = APIRouter()

@model_router.post("/models")
def create_model(model: dict = Body(...)):
    """
    创建新的模型配置
    """
    try:
        alias = model.get("alias")
        model_id = model.get("model_id")
        weights_path = model.get("weights_path")
        labels = json.dumps(model.get("labels", [])) if isinstance(model.get("labels"), list) else model.get("labels")
        params = json.dumps(model.get("params", {})) if isinstance(model.get("params"), dict) else model.get("params")

        if not all([alias, model_id, weights_path]):
            raise HTTPException(status_code=400, detail="alias, model_id, weights_path are required.")

        _id = model_config_db.create_model(alias, model_id, weights_path, labels, params)
        return JSONResponse(content={"code": 200, "msg": "模型创建成功", "id": _id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"模型创建失败: {e}"})

@model_router.get("/models")
def list_models():
    """
    获取所有模型列表
    """
    try:
        models = model_config_db.list_models() or []
        for m in models:
            try:
                m["labels"] = json.loads(m["labels"]) if m.get("labels") and isinstance(m["labels"], str) else m.get("labels", [])
            except Exception:
                m["labels"] = m.get("labels")
            try:
                m["params"] = json.loads(m["params"]) if m.get("params") and isinstance(m["params"], str) else m.get("params", {})
            except Exception:
                m["params"] = m.get("params")
        return JSONResponse(content={"code": 200, "data": models})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"获取模型列表失败: {e}"})

@model_router.get("/models/{model_id}")
def get_model(model_id: str):
    """
    获取指定模型详细配置
    """
    try:
        models = model_config_db.get_model(model_id)
        if not models or not isinstance(models, list):
            raise HTTPException(status_code=404, detail="模型不存在")
        m = models[0]
        try:
            m["labels"] = json.loads(m["labels"]) if m.get("labels") and isinstance(m["labels"], str) else m.get("labels", [])
        except Exception:
            m["labels"] = m.get("labels")
        try:
            m["params"] = json.loads(m["params"]) if m.get("params") and isinstance(m["params"], str) else m.get("params", {})
        except Exception:
            m["params"] = m.get("params")
        return JSONResponse(content={"code": 200, "data": m})
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"获取模型失败: {e}"})

@model_router.put("/models/{model_id}")
def update_model(model_id: str, model: dict = Body(...)):
    """
    更新指定模型信息
    """
    try:
        alias = model.get("alias")
        weights_path = model.get("weights_path")
        labels = json.dumps(model.get("labels", [])) if isinstance(model.get("labels"), list) else model.get("labels")
        params = json.dumps(model.get("params", {})) if isinstance(model.get("params"), dict) else model.get("params")

        rows = model_config_db.update_model(model_id, alias=alias, weights_path=weights_path, labels=labels, params=params)
        if rows == 0:
            raise HTTPException(status_code=404, detail="模型不存在或无更新")
        return JSONResponse(content={"code": 200, "msg": "模型更新成功"})
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"模型更新失败: {e}"})

@model_router.delete("/models/{model_id}")
def delete_model(model_id: str):
    """
    删除指定模型
    """
    try:
        rows = model_config_db.delete_model(model_id)
        if rows == 0:
            raise HTTPException(status_code=404, detail="模型不存在")
        return JSONResponse(content={"code": 200, "msg": "模型删除成功"})
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"模型删除失败: {e}"})


