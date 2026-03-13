# rule_engine/__init__.py

# 从各个模块导入核心类
from .condition import BaseCondition, AtomicCondition, LogicGroup
from .rule import Rule
from .rule_engine import RuleEngine
from .rule_parser import RuleParser
from .actions import alert_action, save_image_action

# 定义 __all__，这决定了外部使用 "from rule_engine import *" 时能拿到的内容
__all__ = [
    "BaseCondition",
    "AtomicCondition",
    "LogicGroup",
    "Rule",
    "RuleEngine",
    "RuleParser",
    "alert_action",
    "save_image_action"
]