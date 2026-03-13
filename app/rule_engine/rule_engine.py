import logging
logger = logging.getLogger(__name__)

class RuleEngine:
    def __init__(self, rules: list):
        self.rules = rules
        self.logger = logging.getLogger("RuleEngine")

    def evaluate_frame(self, event_data):
        for rule in self.rules:
            # 核心调试：打印判定结果
            res = rule.root_condition.evaluate(event_data)

            # 这里会告诉你为什么没触发
            logger.debug(f"Rule {rule.rule_id} evaluation: {res}")

            if res:
                # 只有这里为 True，action 才会执行
                rule.action(rule.rule_id, event_data, rule.action_params)