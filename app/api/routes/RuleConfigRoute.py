from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from app.api.services import RuleConfigServer
import json

RuleConfig_Router = APIRouter()
rule_config_db = RuleConfigServer.rule_config_db  # 假定有对应的db对象和方法实现

@RuleConfig_Router.post("/rule")
def create_rule(rule: dict = Body(...)):
    """
    新增规则配置
    """
    try:
        rule_id = rule.get("rule_id")
        name = rule.get("name")
        enabled = rule.get("enabled", True)
        conditions = json.dumps(rule.get("conditions", {})) if isinstance(rule.get("conditions"), dict) else rule.get("conditions")
        action_type = rule.get("action_type")
        action_params = json.dumps(rule.get("action_params", {})) if isinstance(rule.get("action_params"), dict) else rule.get("action_params")
        # 判空
        if not all([rule_id, name]):
            raise HTTPException(status_code=400, detail="rule_id, name 为必填项")
        _id = rule_config_db.create_rule(rule_id, name, enabled, conditions, action_type, action_params)
        return JSONResponse(content={"code": 200, "msg": "规则创建成功", "id": _id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则创建失败: {e}"})

@RuleConfig_Router.put("/rule/{rule_id}")
def update_rule(rule_id: str, rule: dict = Body(...)):
    """
    修改规则配置
    """
    try:
        name = rule.get("name")
        enabled = rule.get("enabled")
        conditions = json.dumps(rule.get("conditions")) if isinstance(rule.get("conditions"), dict) else rule.get("conditions")
        action_type = rule.get("action_type")
        action_params = json.dumps(rule.get("action_params")) if isinstance(rule.get("action_params"), dict) else rule.get("action_params")

        updated = rule_config_db.update_rule(rule_id, name, enabled, conditions, action_type, action_params)
        if updated == 0:
            raise HTTPException(status_code=404, detail="规则未找到或无参数更新")
        return JSONResponse(content={"code": 200, "msg": "规则更新成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则更新失败: {e}"})

@RuleConfig_Router.delete("/rule/{rule_id}")
def delete_rule(rule_id: str):
    """
    删除规则配置
    """
    try:
        deleted = rule_config_db.delete_rule(rule_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="规则未找到")
        return JSONResponse(content={"code": 200, "msg": "规则删除成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则删除失败: {e}"})

@RuleConfig_Router.get("/rule/{rule_id}")
def get_rule(rule_id: str):
    """
    查询单个规则详情
    """
    try:
        rule = rule_config_db.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail="规则未找到")
        # 格式化JSON字段为dict
        if rule and "conditions" in rule and rule["conditions"]:
            try:
                rule["conditions"] = json.loads(rule["conditions"])
            except Exception:
                pass
        if rule and "action_params" in rule and rule["action_params"]:
            try:
                rule["action_params"] = json.loads(rule["action_params"])
            except Exception:
                pass
        return JSONResponse(content={"code": 200, "data": rule})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则查询失败: {e}"})

@RuleConfig_Router.get("/rules")
def list_rules(page: int = Query(1, gt=0), page_size: int = Query(10, gt=0, le=100)):
    """
    分页查询规则配置列表
    """
    try:
        all_rules = rule_config_db.list_rules() or []
        total = len(all_rules)
        start = (page - 1) * page_size
        end = start + page_size
        rules = all_rules[start:end]
        for rule in rules:
            if "conditions" in rule and rule["conditions"]:
                try:
                    rule["conditions"] = json.loads(rule["conditions"])
                except Exception:
                    pass
            if "action_params" in rule and rule["action_params"]:
                try:
                    rule["action_params"] = json.loads(rule["action_params"])
                except Exception:
                    pass
        return JSONResponse(content={"code": 200, "data": rules, "total": total, "page": page, "page_size": page_size})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"规则分页查询失败: {e}"})