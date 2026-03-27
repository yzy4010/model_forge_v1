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
        # 1. 处理逻辑组合递归 (any/all/not) 保持不变
        if "any" in config:
            return LogicGroup("OR", [cls.parse_condition(c) for c in config["any"]])
        if "all" in config:
            return LogicGroup("AND", [cls.parse_condition(c) for c in config["all"]])
        if "not" in config:
            return LogicGroup("NOT", [cls.parse_condition(config["not"])])

        # ================= 核心修复部分：强绑定解析 =================

        # 1. 匹配【空间+属性】强绑定条件
        # 场景：{"alias": "smoking", "roi": "danger_zone"}
        if "alias" in config and "roi" in config:
            return AtomicCondition(
                key=config["alias"],  # 以别名为 key，如 "smoking"
                operator="alias_in_roi",  # 使用新定义的强绑定操作符
                value=config["roi"]  # 以 ROI 标签为判定值，如 "danger_zone"
            )

        # 2. 匹配【纯属性】存在性条件
        # 场景：{"alias": "person"} -> 只要画面任何地方有人就触发
        if "alias" in config:
            return AtomicCondition(
                key=config["alias"],
                operator="exists",  # 建议改为 exists，兼容列表判定
                value=True
            )

        # 3. 匹配【纯空间】触发条件
        # 场景：{"roi": "danger_zone"} -> 只要该区域有任何东西命中就触发
        if "roi" in config:
            return AtomicCondition(
                key="roi",
                operator="contains",
                value=config["roi"]
            )
        # =========================================================

        raise ValueError(f"Unknown condition config: {config}")

    @classmethod
    def parse_rule(cls, rule_dict: dict) -> Rule:
        rule_id = rule_dict.get("rule_id", "R_UNKNOWN")
        rule_name = rule_dict.get("name")

        enabled = rule_dict.get("enabled", True)  # 获取 JSON 中的 enabled 状态

        # 注意：这里要获取 action 里的 params，而不是整个 action 配置
        action_cfg = rule_dict.get("action", {})
        action_params = action_cfg.get("params", {})

        # 解析逻辑树
        root_cond = cls.parse_condition(rule_dict.get("conditions", {}))

        # 获取动作函数
        action_func = cls.ACTION_MAP.get(action_cfg.get("type"), alert_action)

        return Rule(
            rule_id=rule_id,
            name=rule_name,
            root_condition=root_cond,
            action=action_func,
            action_params=action_params,  # 修正：只传 params 字典
            enabled = enabled # <--- 传进去
        )