from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from app.api.services import SceneRuleConfigServer
import json

SceneRuleConfig_Route = APIRouter()
scene_rule_config_db = SceneRuleConfigServer.scene_rule_config_db

@SceneRuleConfig_Route.post("/scene/rule")
def create_rule(cfg: dict = Body(...)):
    """
    新增规则配置
    """
    try:
        scene_id = cfg.get("scene_id")
        rule_id = cfg.get("rule_id")
        name = cfg.get("name")
        enabled = cfg.get("enabled", True)
        conditions = cfg.get("conditions")
        action_type = cfg.get("action_type")
        action_params = cfg.get("action_params")
        # 非字符串json类型转储
        if conditions is not None and not isinstance(conditions, str):
            conditions = json.dumps(conditions, ensure_ascii=False)
        if action_params is not None and not isinstance(action_params, str):
            action_params = json.dumps(action_params, ensure_ascii=False)
        # 判空
        if not all([scene_id, rule_id, name]):
            raise HTTPException(status_code=400, detail="scene_id, rule_id, name 为必填项")
        _id = scene_rule_config_db.create_rule(
            scene_id, rule_id, name, enabled, conditions, action_type, action_params
        )
        return JSONResponse(content={"code": 200, "msg": "规则配置创建成功", "id": _id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则配置创建失败: {e}"})


@SceneRuleConfig_Route.put("/scene/rule/{rule_id}")
def update_rule(rule_id: str, cfg: dict = Body(...)):
    """
    修改规则配置
    """
    try:
        name = cfg.get("name")
        enabled = cfg.get("enabled")
        conditions = cfg.get("conditions")
        action_type = cfg.get("action_type")
        action_params = cfg.get("action_params")
        # 非字符串json类型转储
        if conditions is not None and not isinstance(conditions, str):
            conditions = json.dumps(conditions, ensure_ascii=False)
        if action_params is not None and not isinstance(action_params, str):
            action_params = json.dumps(action_params, ensure_ascii=False)
        # 没有字段更新
        if (
            name is None and enabled is None and
            conditions is None and action_type is None and action_params is None
        ):
            raise HTTPException(status_code=400, detail="没有需要更新的字段")
        updated = scene_rule_config_db.update_rule(
            rule_id, name, enabled, conditions, action_type, action_params
        )
        if updated == 0:
            raise HTTPException(status_code=404, detail="规则未找到")
        return JSONResponse(content={"code": 200, "msg": "规则配置更新成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则配置更新失败: {e}"})


@SceneRuleConfig_Route.delete("/scene/rule/{rule_id}")
def delete_rule(rule_id: str):
    """
    删除规则配置
    """
    try:
        deleted = scene_rule_config_db.delete_rule(rule_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="规则未找到")
        return JSONResponse(content={"code": 200, "msg": "规则配置删除成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则配置删除失败: {e}"})


@SceneRuleConfig_Route.get("/scene/rule/{rule_id}")
def get_rule(rule_id: str):
    """
    查询单个规则配置详情
    """
    try:
        rule = scene_rule_config_db.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail="规则未找到")
        return JSONResponse(content={"code": 200, "data": rule})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则查询失败: {e}"})


@SceneRuleConfig_Route.get("/scene/rules")
def list_rules(
    scene_id: str = Query(None),
    page: int = Query(1, gt=0),
    page_size: int = Query(10, gt=0, le=100)
):
    """
    分页查询规则配置列表
    """
    try:
        all_rules = scene_rule_config_db.list_rules(scene_id) or []
        total = len(all_rules)
        start = (page - 1) * page_size
        end = start + page_size
        rules = all_rules[start:end]
        return JSONResponse(content={
            "code": 200,
            "data": rules,
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则分页查询失败: {e}"})
