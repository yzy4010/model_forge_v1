"""
rule_protocol.py

定义 ModelForge V2 规则引擎的数据协议。
使用 Pydantic v2，支持：
- AtomicCondition
- LogicalGroup（递归支持 all/any）
- RuleAction
- RuleConfig
"""

from __future__ import annotations
from typing import List, Optional, Union
from pydantic import BaseModel, Field

# ----------------------------
# AtomicCondition
# ----------------------------
class AtomicCondition(BaseModel):
    """
    原子条件，用于规则引擎中的最小判断单元。

    Attributes:
        alias (str): 模型别名，例如 "smoking"。
        roi (Optional[str]): 可选，ROI 区域标识。
        label (Optional[str]): 可选，检测标签，例如 "smoke"。
    """
    alias: str = Field(..., description="模型别名")
    roi: Optional[str] = Field(None, description="可选 ROI 区域标识")
    label: Optional[str] = Field(None, description="可选标签")

# ----------------------------
# Forward reference for recursive type
# ----------------------------
ConditionType = Union['AtomicCondition', 'LogicalGroup']

# ----------------------------
# LogicalGroup
# ----------------------------
class LogicalGroup(BaseModel):
    """
    逻辑组合条件，支持 AND(all) / OR(any) 嵌套逻辑。

    Attributes:
        all (Optional[List[ConditionType]]): 所有条件均满足时触发（AND）。
        any (Optional[List[ConditionType]]): 任意条件满足时触发（OR）。
    """
    all: Optional[List[ConditionType]] = Field(None, description="AND 组合")
    any: Optional[List[ConditionType]] = Field(None, description="OR 组合")

# ----------------------------
# RuleAction
# ----------------------------
class RuleAction(BaseModel):
    """
    规则触发后的动作信息。

    Attributes:
        type (str): 动作类型，例如 "alert"。
        level (str): 严重等级，例如 "high", "medium", "low"。
        message (str): 报警或提示信息。
    """
    type: str = Field(..., description="动作类型，例如 'alert'")
    level: str = Field(..., description="严重等级，例如 'high'")
    message: str = Field(..., description="报警或提示信息")

# ----------------------------
# RuleConfig
# ----------------------------
class RuleConfig(BaseModel):
    """
    单条规则配置。

    Attributes:
        rule_id (str): 规则唯一标识。
        name (str): 规则名称。
        enabled (bool): 是否启用。
        conditions (ConditionType): 条件逻辑。
        action (RuleAction): 触发动作。
    """
    rule_id: str = Field(..., description="规则唯一标识")
    name: str = Field(..., description="规则名称")
    enabled: bool = Field(True, description="是否启用")
    conditions: ConditionType = Field(..., description="条件逻辑")
    action: RuleAction = Field(..., description="触发动作")

    # ------------------------
    # 辅助方法：JSON 序列化
    # ------------------------
    def to_json(self, **kwargs) -> str:
        """
        将规则配置序列化为 JSON 字符串。
        """
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> 'RuleConfig':
        """
        从 JSON 字符串解析规则配置。
        """
        return cls.model_validate_json(json_str)

# ----------------------------
# 递归重建，支持 LogicalGroup 的嵌套解析
# ----------------------------
LogicalGroup.model_rebuild()
RuleConfig.model_rebuild()