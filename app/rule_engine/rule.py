import time


class Rule:
    def __init__(self, rule_id: str, root_condition, action, action_params=None, enabled=True):
        self.rule_id = rule_id
        self.root_condition = root_condition
        self.action = action
        self.action_params = action_params or {}
        self.enabled = enabled

        # --- 新增状态控制 ---
        self.last_triggered_time = 0
        self.cooldown_seconds = self.action_params.get("cooldown", 5)  # 默认 5 秒内不重复告警

    def evaluate(self, event_data: dict) -> bool:
        return self.root_condition.evaluate(event_data)

    def trigger(self, event_data: dict):
        current_time = time.time()

        # 冷却时间检查
        if current_time - self.last_triggered_time < self.cooldown_seconds:
            return  # 还在冷却中，跳过执行

        self.last_triggered_time = current_time
        self.action(self.rule_id, event_data, self.action_params)

    def get_involved_rois(self) -> set:
        involved = set()
        # 调用根条件的递归获取方法
        if self.root_condition:
            involved.update(self.root_condition.get_roi_values())
        return involved