"""
rule_engine.py

ModelForge V2 RuleEngine 核心接口。
- 调用 rule_evaluator.py 进行条件评估和 ROI 收集
- 输出 alerts + roi_status
"""

import logging
from typing import List, Dict
from app.rule_engine.rule_protocol import RuleConfig
from app.rule_engine.rule_evaluator import evaluate_condition, collect_rois

logger = logging.getLogger("model_forge.rule_engine")


class RuleEngine:
    """
    RuleEngine 核心类，用于评估每帧的规则触发情况。
    Stateless，适合实时视频流处理。
    """

    def __init__(self, rules):

        # 允许传入 dict 或 RuleConfig
        self.rules = []

        for r in rules:
            if isinstance(r, RuleConfig):
                self.rules.append(r)
            else:
                self.rules.append(RuleConfig(**r))

    def evaluate_frame(self, event_results: dict) -> dict:
        alerts = []
        roi_status: Dict[str, str] = {}

        for rule in self.rules:
            if not rule.enabled:
                continue

            # 1. 评估整条规则是否触发
            if evaluate_condition(rule.conditions, event_results):
                # 2. 命中后，添加报警信息
                alerts.append({
                    "rule_id": rule.rule_id,
                    "level": rule.action.level,
                    "message": rule.action.message
                })
                # 3. 传入 event_results 进行上下文相关的 ROI 收集
                collect_rois(rule.conditions, event_results, roi_status)

        return {
            "alerts": alerts,
            "roi_status": roi_status
        }