from .condition import AtomicCondition, LogicGroup
from .rule import Rule
from .actions import alert_action, save_image_action


class RuleParser:
    ACTION_MAP = {
        "alert": alert_action,
        "save_image": save_image_action
    }

    @classmethod
    def parse_condition(cls, config: dict):
        if "any" in config:
            return LogicGroup("OR", [cls.parse_condition(c) for c in config["any"]])
        if "all" in config:
            return LogicGroup("AND", [cls.parse_condition(c) for c in config["all"]])
        if "not" in config:
            return LogicGroup("NOT", [cls.parse_condition(config["not"])])

        # ================= 核心修复部分 =================
        # 匹配原子条件：
        # 1. 如果是 alias，直接用别名（如 "person"）作为 key，去 engine_data 找 bool 值
        if "alias" in config:
            alias_value = config["alias"]
            return AtomicCondition(key=alias_value, operator="eq", value=True)

        # 2. 如果是 roi，去 engine_data 的 "roi" 键里找对应的标签
        if "roi" in config:
            return AtomicCondition(key="roi", operator="contains", value=config["roi"])
        # ===============================================

        raise ValueError(f"Unknown condition config: {config}")

    @classmethod
    def parse_rule(cls, rule_dict: dict) -> Rule:
        rule_id = rule_dict.get("rule_id", "R_UNKNOWN")
        # 注意：这里要获取 action 里的 params，而不是整个 action 配置
        action_cfg = rule_dict.get("action", {})
        action_params = action_cfg.get("params", {})

        # 解析逻辑树
        root_cond = cls.parse_condition(rule_dict.get("conditions", {}))

        # 获取动作函数
        action_func = cls.ACTION_MAP.get(action_cfg.get("type"), alert_action)

        return Rule(
            rule_id=rule_id,
            root_condition=root_cond,
            action=action_func,
            action_params=action_params  # 修正：只传 params 字典
        )