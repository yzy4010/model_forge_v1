import logging

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.services import SceneConfigServer
from app.api.services import SceneModelConfigServer
from app.api.services import SceneRoiConfigServer
from app.api.services import SceneRuleConfigServer
from app.api.schemas.infer import InferParams, InferStreamRequest
from app.api.routes.infer import start_infer_stream

import json

logger = logging.getLogger("model_forge.scene_config")

SceneConfig_Router = APIRouter()


scene_config_db = SceneConfigServer.SceneConfigDB()
scene_model_config_db = SceneModelConfigServer.SceneModelConfigDB()
scene_roi_config_db = SceneRoiConfigServer.SceneRoiConfigDB()
scene_rule_config_db = SceneRuleConfigServer.SceneRuleConfigDB()


def _sanitize_rois_for_infer(roi_config: dict) -> dict:
    """保证 ROI 结构满足 InferStreamRequest / ROIConfig 校验。"""
    out = dict(roi_config)
    rois = []
    for roi in out.get("rois") or []:
        if not isinstance(roi, dict):
            continue
        r = dict(roi)
        geom = r.get("geometry")
        if not isinstance(geom, dict):
            geom = {}
        if "points" not in geom:
            geom = {**geom, "points": []}
        r["geometry"] = geom
        if r.get("semantic_tag") is None:
            r["semantic_tag"] = ""
        rois.append(r)
    out["rois"] = rois
    return out


def _build_infer_stream_request_from_full_config(data: dict) -> InferStreamRequest:
    """将 full_config 的 data 转为启动推理所需的 InferStreamRequest。"""
    scenario = data.get("scenario") or {}
    default_params = InferParams().model_dump()
    models_payload = []
    for m in scenario.get("models") or []:
        if not isinstance(m, dict):
            continue
        alias = m.get("alias")
        model_id = m.get("model_id")
        weights_path = m.get("weights_path")
        if not all([alias, model_id, weights_path]):
            continue
        merged_params = {**default_params, **(m.get("params") or {})}
        models_payload.append(
            {
                "alias": alias,
                "model_id": model_id,
                "weights_path": weights_path,
                "labels": m.get("labels") or [],
                "params": merged_params,
            }
        )
    if not models_payload:
        raise ValueError("场景下没有可用的模型配置（需包含 alias、model_id、weights_path）")

    roi_raw = scenario.get("roi_config") or {}
    roi_config = None
    if isinstance(roi_raw, dict) and roi_raw.get("rois"):
        roi_config = _sanitize_rois_for_infer(roi_raw)

    rules = scenario.get("rule_config") or []
    rule_config = rules if rules else None

    payload = {
        "rtsp_url": data["rtsp_url"],
        "sample_fps": float(data.get("sample_fps") or 2.0),
        "scenario": {
            "scenario_id": scenario["scenario_id"],
            "models": models_payload,
            "roi_config": roi_config,
            "rule_config": rule_config,
        },
    }
    return InferStreamRequest.model_validate(payload)


def _start_infer_after_full_config(data: dict) -> dict:
    """
    根据完整配置启动推理。成功返回 job_id；失败不抛 HTTP，由 infer 字段说明原因。
    始终包含 job_id 键（成功为 str，未启动/失败为 None），便于前端统一解析。
    """
    if not data.get("rtsp_url"):
        return {"started": False, "job_id": None, "error": "缺少 rtsp_url，无法启动推理"}
    try:
        req = _build_infer_stream_request_from_full_config(data)
    except ValueError as e:
        return {"started": False, "job_id": None, "error": str(e)}
    except ValidationError as e:
        return {"started": False, "job_id": None, "error": f"推理请求参数校验失败: {e}"}

    try:
        resp = start_infer_stream(req)
        return {
            "started": True,
            "job_id": resp.job_id,
            "status": resp.status,
        }
    except HTTPException as e:
        detail = e.detail
        if not isinstance(detail, str):
            detail = str(detail)
        return {
            "started": False,
            "job_id": None,
            "error": detail,
            "http_status": e.status_code,
        }
    except Exception as e:
        logger.exception("启动推理失败 scene_id=%s", data.get("scenario", {}).get("scenario_id"))
        return {"started": False, "job_id": None, "error": str(e)}


def _build_scene_full_config_data(scene_id: str) -> dict:
    """
    从数据库组装与保存接口一致的 full_config 结构（不含推理启动）。
    """
    # 1. 主配置
    main = scene_config_db.get_scene_config(scene_id)
    rtsp_url = main[0].get("rtsp_url") if main and isinstance(main, list) else None
    sample_fps = main[0].get("sample_fps") if main and isinstance(main, list) else 4

    # 2. 模型配置
    models_raw = scene_model_config_db.list_models(scene_id) or []
    models = []
    for m in models_raw:
        params = None
        if m.get("params"):
            try:
                params = json.loads(m["params"]) if isinstance(m["params"], str) else m["params"]
            except Exception:
                params = m["params"]
        labels = None
        if m.get("labels"):
            try:
                labels = json.loads(m["labels"]) if isinstance(m["labels"], str) else m["labels"]
            except Exception:
                labels = m["labels"]
        models.append({
            "alias": m.get("alias"),
            "model_id": m.get("model_id"),
            "weights_path": m.get("weights_path"),
            "labels": labels or [],
            "params": params or {}
        })

    # 3. ROI配置
    roi_raw = scene_roi_config_db.list_rois(scene_id) or []
    roi_config = {}
    if roi_raw:
        first = roi_raw[0]
        camera_id = first.get("camera_id", "")
        config_version = first.get("config_version", 1)
        resolution = {
            "width": first.get("resolution_width", 640),
            "height": first.get("resolution_height", 360),
        }
        rois = []
        for roi in roi_raw:
            geometry = None
            if roi.get("geometry"):
                try:
                    geometry = json.loads(roi["geometry"]) if isinstance(roi["geometry"], str) else roi["geometry"]
                except Exception:
                    geometry = roi["geometry"]
            else:
                geometry = {}
            rois.append({
                "roi_id": roi.get("roi_id"),
                "name": roi.get("name"),
                "semantic_tag": roi.get("semantic_tag"),
                "enabled": roi.get("enabled", True),
                "geometry": geometry
            })
        roi_config = {
            "camera_id": camera_id,
            "config_version": config_version,
            "resolution": resolution,
            "rois": rois
        }
    else:
        roi_config = {}

    # 4. 规则配置
    rules_raw = scene_rule_config_db.list_rules(scene_id) or []
    rule_config = []
    for rule in rules_raw:
        conditions = {}
        if rule.get("conditions"):
            try:
                conditions = json.loads(rule["conditions"]) if isinstance(rule["conditions"], str) else rule["conditions"]
            except Exception:
                conditions = rule["conditions"]
        action_type = rule.get("action_type")
        action_params = {}
        if rule.get("action_params"):
            try:
                action_params = json.loads(rule["action_params"]) if isinstance(rule["action_params"], str) else rule["action_params"]
            except Exception:
                action_params = rule["action_params"]

        rule_config.append({
            "rule_id": rule.get("rule_id"),
            "name": rule.get("name"),
            "enabled": rule.get("enabled", True),
            "conditions": conditions,
            "action": {
                "type": action_type,
                "params": action_params
            }
        })

    scenario_id = main[0].get("scene_id") if main and isinstance(main, list) and "scene_id" in main[0] else scene_id

    return {
        "rtsp_url": rtsp_url,
        "sample_fps": sample_fps,
        "scenario": {
            "scenario_id": scenario_id,
            "models": models,
            "roi_config": roi_config,
            "rule_config": rule_config
        }
    }


# 静态路径需写在 /scenes/{scene_id} 之前，避免 scene_id 误匹配为 full_config
@SceneConfig_Router.get("/scenes/full_config/{scene_id}")
async def get_scene_full_config_api(scene_id: str):
    """
    仅查询：根据场景 id 返回完整配置（GET 无副作用，符合 REST）。
    """
    try:
        data = _build_scene_full_config_data(scene_id)
        return JSONResponse(content={"code": 200, "msg": "success", "data": data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"failed: {e}"})


@SceneConfig_Router.post("/scenes/full_config/{scene_id}")
async def post_scene_full_config_and_start_infer(scene_id: str):
    """
    查询完整配置后，用同一份数据启动推理流（POST 表示有副作用）。
    前端请使用 POST 调用本接口；若用 GET 会 405。
    """
    try:
        data = _build_scene_full_config_data(scene_id)
        infer_info = _start_infer_after_full_config(data)
        # 根级 job_id：与 POST /infer/stream 对齐，避免前端只读根对象时拿不到
        job_id = infer_info.get("job_id")
        if infer_info.get("started") and job_id:
            try:
                scene_config_db.update_scene_infer_state(scene_id, str(job_id), 1)
            except Exception as ex:
                logger.warning(
                    "scene_configs 写入 job_id/status 失败 scene_id=%s job_id=%s: %s",
                    scene_id,
                    job_id,
                    ex,
                )
        return JSONResponse(
            content={
                "code": 200,
                "msg": "success",
                "data": data,
                "infer": infer_info,
                "job_id": job_id,
                "status": infer_info.get("status") if infer_info.get("started") else None,
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"failed: {e}"})


@SceneConfig_Router.post("/scenes")
def create_scene_config(cfg: dict = Body(...)):
    """
    新增推理场景主配置
    """
    try:
        scene_id = cfg.get("scene_id")
        rtsp_url = cfg.get("rtsp_url")
        sample_fps = cfg.get("sample_fps", 4)
        scenario_name = cfg.get("scenario_name")
        # 判空
        if not all([scene_id, rtsp_url]):
            raise HTTPException(status_code=400, detail="scene_id, rtsp_url 为必填项")
        _id = scene_config_db.create_scene_config(scene_id, rtsp_url, sample_fps, scenario_name)
        return JSONResponse(content={"code": 200, "msg": "场景配置创建成功", "id": _id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"场景配置创建失败: {e}"})


@SceneConfig_Router.put("/scenes/{scene_id}")
def update_scene_config(scene_id: str, cfg: dict = Body(...)):
    """
    修改推理场景主配置
    """
    try:
        rtsp_url = cfg.get("rtsp_url")
        sample_fps = cfg.get("sample_fps")
        scenario_name = cfg.get("scenario_name")
        updated = scene_config_db.update_scene_config(scene_id, rtsp_url, sample_fps, scenario_name)
        if updated == 0:
            raise HTTPException(status_code=404, detail="场景未找到或无参数更新")
        return JSONResponse(content={"code": 200, "msg": "场景配置更新成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"场景配置更新失败: {e}"})


@SceneConfig_Router.delete("/scenes/{scene_id}")
def delete_scene_config(scene_id: str):
    """
    删除推理场景主配置
    """
    try:
        deleted = scene_config_db.delete_scene_config(scene_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="场景未找到")
        return JSONResponse(content={"code": 200, "msg": "场景配置删除成功"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"场景配置删除失败: {e}"})


@SceneConfig_Router.get("/scenes/{scene_id}")
def get_scene_config(scene_id: str):
    """
    查询单个推理场景配置详情
    """
    try:
        scene = scene_config_db.get_scene_config(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="场景未找到")
        return JSONResponse(content={"code": 200, "data": scene})
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"场景查询失败: {e}"})


def _normalize_scene_list_item(row: dict) -> dict:
    """列表项中与 scene_configs 表对齐：保证含 status（0 未运行，1 运行中）、job_id。"""
    out = dict(row)
    try:
        st = out.get("status")
        out["status"] = int(st) if st is not None else 0
    except (TypeError, ValueError):
        out["status"] = 0
    if "job_id" not in out:
        out["job_id"] = None
    return out


@SceneConfig_Router.get("/scenes")
def list_scene_configs(page: int = Query(1, gt=0), page_size: int = Query(10, gt=0, le=100)):
    """
    分页查询推理场景主配置列表。
    每条记录含表字段：含 status（0 未运行，1 运行中）、job_id 等（见 scene_configs）。
    """
    try:
        all_scenes = scene_config_db.list_scene_configs() or []
        total = len(all_scenes)
        start = (page - 1) * page_size
        end = start + page_size
        scenes = [_normalize_scene_list_item(s) for s in all_scenes[start:end]]
        return JSONResponse(content={
            "code": 200,
            "data": scenes,
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"场景分页查询失败: {e}"})

@SceneConfig_Router.post("/scenes/save")
def save_scene_config(scene_data: dict = Body(...)):
    """
    保存推理场景配置（主配置、模型、ROI、规则等）
    :param scene_data: 前端传入的json结构，参考示例
    :return: 保存结果
    """
    try:
        print("保存主配置")


        # 1. 解析主配置和场景ID、RTSP等
        scenario = scene_data.get("scenario", {})
        scene_id = scenario.get("scenario_id")
        rtsp_url = scene_data.get("rtsp_url")
        sample_fps = scene_data.get("sample_fps", 4)
        scenario_name = scenario.get("scenario_name")

        # 2. 保存主配置（id为数据库自增，无需传递，自动生成）
        _id = scene_config_db.create_scene_config(
            scene_id=scene_id,
            rtsp_url=rtsp_url,
            sample_fps=sample_fps,
            scenario_name=scenario_name
        )

        print("保存主配置")

        # 3. 保存模型配置（id自动生成）
        for model in scenario.get("models", []):
            alias = model.get("alias")
            model_id = model.get("model_id")
            weights_path = model.get("weights_path")
            labels = json.dumps(model.get("labels", [])) if isinstance(model.get("labels"), list) else model.get("labels")
            params = json.dumps(model.get("params", {})) if isinstance(model.get("params"), dict) else model.get("params")
            # 不传id，由create_model自动生成
            scene_model_config_db.create_model(
                scene_id=scene_id,
                alias=alias,
                model_id=model_id,
                weights_path=weights_path,
                labels=labels,
                params=params
            )

            print("保存model")


        # 4. 保存ROI配置（id自动生成）
        roi_conf = scenario.get("roi_config", {})
        camera_id = roi_conf.get("camera_id")
        config_version = roi_conf.get("config_version", 1)
        width = roi_conf.get("resolution", {}).get("width", 640)
        height = roi_conf.get("resolution", {}).get("height", 360)
        for roi in roi_conf.get("rois", []):
            roi_id = roi.get("roi_id")
            name = roi.get("name")
            semantic_tag = roi.get("semantic_tag")
            enabled = roi.get("enabled", True)
            geometry = json.dumps(roi.get("geometry", {})) if isinstance(roi.get("geometry"), dict) else roi.get("geometry")
            # 不传id，由create_roi_config自动生成
            scene_roi_config_db.create_roi(
                scene_id=scene_id,
                camera_id=camera_id,
                config_version=config_version,
                resolution_width=width,
                resolution_height=height,
                roi_id=roi_id,
                name=name,
                semantic_tag=semantic_tag,
                enabled=enabled,
                geometry=geometry
            )

            print("保存ROI配置")
        
        # 5. 保存规则配置（id自动生成）
        for rule in scenario.get("rule_config", []):
            rule_id = rule.get("rule_id")
            name = rule.get("name")
            enabled = rule.get("enabled", True)
            conditions = json.dumps(rule.get("conditions", {})) if isinstance(rule.get("conditions"), dict) else rule.get("conditions")
            action = rule.get("action", {})
            action_type = action.get("type")
            action_params = json.dumps(action.get("params", {})) if isinstance(action.get("params"), dict) else action.get("params")
            # 不传id，由create_rule自动生成
            scene_rule_config_db.create_rule(
                scene_id=scene_id,
                rule_id=rule_id,
                name=name,
                enabled=enabled,
                conditions=conditions,
                action_type=action_type,
                action_params=action_params,
            )
            print("保存规则配置")

        return JSONResponse(content={"code": 200, "msg": "保存成功", "scene_id": scene_id})

    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"保存失败: {e}"})
