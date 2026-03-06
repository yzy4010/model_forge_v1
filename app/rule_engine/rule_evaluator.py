from typing import Union, Dict, List
import logging
from app.rule_engine.rule_protocol import AtomicCondition, LogicalGroup

logger = logging.getLogger("model_forge.rule_evaluator")


def evaluate_condition(condition: Union[AtomicCondition, LogicalGroup], event_results: dict) -> bool:
    """
    递归评估单帧是否匹配规则。
    """
    if isinstance(condition, AtomicCondition):
        # 场景 A: 指定了 alias
        if condition.alias:
            model_data = event_results.get(condition.alias, {})
            detections = model_data.get("detections", []) if isinstance(model_data, dict) else []

            if not detections:
                return False

            for det in detections:
                # Label 模糊匹配
                det_label = str(det.get("label", ""))
                label_ok = (condition.label is None) or (condition.label.lower() in det_label.lower())
                # ROI 匹配
                roi_tags = det.get("roi_tags", [])
                roi_ok = (condition.roi is None) or (condition.roi in roi_tags)

                if label_ok and roi_ok:
                    return True
            return False

        # 场景 B: 未指定 alias，只指定了 ROI (全局 ROI 检查)
        elif condition.roi:
            # 检查所有的检测框是否包含这个 roi 标签
            for alias, data in event_results.items():
                detections = data.get("detections", []) if isinstance(data, dict) else []
                for det in detections:
                    if condition.roi in det.get("roi_tags", []):
                        return True
            return False

        return False

    elif isinstance(condition, LogicalGroup):
        # 评估 AND 条件（all）
        if condition.all is not None:
            if not condition.all:
                return False
            return all(evaluate_condition(c, event_results) for c in condition.all)

        # 评估 OR 条件（any）
        if condition.any is not None:
            if not condition.any:
                return False
            return any(evaluate_condition(c, event_results) for c in condition.any)

    return False


def collect_rois(condition: Union[AtomicCondition, LogicalGroup], event_results: dict, roi_status: Dict[str, str]):
    """
    收集真正导致规则触发的 ROI 标识。
    """
    if isinstance(condition, AtomicCondition):
        # 重新运行原子评估，确保它是真的触发者
        if evaluate_condition(condition, event_results):
            if condition.roi:
                roi_status[condition.roi] = "alert"
            # 特殊情况：如果只定义了 alias 没定 roi，但规则整体触发了，
            # 我们可以把该 alias 命中的所有 ROI 都标红（根据业务需求决定）
            else:
                model_data = event_results.get(condition.alias, {})
                detections = model_data.get("detections", []) if isinstance(model_data, dict) else []
                for det in detections:
                    for tag in det.get("roi_tags", []):
                        roi_status[tag] = "alert"

    elif isinstance(condition, LogicalGroup):
        # 递归检查所有可能的子路径
        sub_items = []
        if condition.all:
            sub_items.extend(condition.all)
        if condition.any:
            sub_items.extend(condition.any)

        for c in sub_items:
            collect_rois(c, event_results, roi_status)