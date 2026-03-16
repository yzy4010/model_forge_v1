from typing import Any, List, Dict


class BaseCondition:
    def evaluate(self, event_data: dict) -> bool:
        raise NotImplementedError

    def get_roi_values(self) -> list:
        """
        定义一个空接口，确保子类可以被统一调用。
        默认返回空列表，表示该条件不涉及具体的 ROI 标签。
        """
        return []


class AtomicCondition(BaseCondition):
    """原子条件：支持 alias 和 roi 的灵活判断"""

    def __init__(self, key: str, operator: str, value: Any):
        self.key = key  # 对应 data 中的键
        self.operator = operator  # 操作符: 'alias_check', 'contains'
        self.value = value  # 目标值

    def evaluate(self, event_data: dict) -> bool:
        actual_val = event_data.get(self.key)

        # 核心：处理 属性+空间 强绑定
        if self.operator == "alias_in_roi":
            # 此时 self.key 是 "smoking", self.value 是 "danger_zone"
            # actual_val 是我们在 _run_job 里存入的列表 ["safe_zone"]
            if isinstance(actual_val, list):
                return self.value in actual_val
            return False

        # 兼容原有的简单布尔判断
        if self.operator == "exists":
            return len(actual_val) > 0 if isinstance(actual_val, list) else bool(actual_val)

        # 兼容原有的全局 ROI 判断
        if self.operator == "contains":
            roi_list = event_data.get("roi", [])
            return self.value in roi_list if isinstance(roi_list, list) else False

        return False

    # 在 AtomicCondition 类中
    def get_roi_values(self) -> list:
        # 关键：必须包含我们新定义的 alias_in_roi 操作符
        if self.operator in ["contains", "alias_in_roi"]:
            return [self.value]
        return []


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

    # 在 LogicGroup (any/all) 类中
    def get_roi_values(self) -> list:
        values = []
        for c in self.conditions:
            values.extend(c.get_roi_values())
        return values