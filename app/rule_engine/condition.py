from typing import Any, List, Dict


class BaseCondition:
    def evaluate(self, event_data: dict) -> bool:
        raise NotImplementedError


class AtomicCondition(BaseCondition):
    """原子条件：支持 alias 和 roi 的灵活判断"""

    def __init__(self, key: str, operator: str, value: Any):
        self.key = key  # 对应 data 中的键
        self.operator = operator  # 操作符: 'alias_check', 'contains'
        self.value = value  # 目标值

    def evaluate(self, event_data: dict) -> bool:
        # 1. 从数据中提取对应的值
        actual_value = event_data.get(self.key)

        # 2. 根据操作符进行判定
        if self.operator == "eq":
            # 用于判断 person: True
            return actual_value == self.value

        elif self.operator == "contains":
            # 用于判断 "danger_zone" 是否在 ['danger_zone'] 列表中
            if isinstance(actual_value, list):
                return self.value in actual_value
            return False

        elif self.operator == "alias_check":  # 兼容你之前的旧逻辑名
            return actual_value == True

        return False


class LogicGroup(BaseCondition):
    """逻辑组：实现嵌套的 AND, OR, NOT"""

    def __init__(self, mode: str, conditions: List[BaseCondition]):
        self.mode = mode.upper()
        self.conditions = conditions

    def evaluate(self, event_data: dict) -> bool:
        if self.mode == "AND":
            # 短路评估：遇到 False 立即停止
            return all(c.evaluate(event_data) for c in self.conditions)
        elif self.mode == "OR":
            # 短路评估：遇到 True 立即停止
            return any(c.evaluate(event_data) for c in self.conditions)
        elif self.mode == "NOT":
            if not self.conditions: return True
            return not self.conditions[0].evaluate(event_data)
        return False