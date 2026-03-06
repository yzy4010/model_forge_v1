"""
example_rules.py

ModelForge V2 示例规则集，用于开发、调试和测试 RuleEngine。
包含单条件规则和嵌套 all/any 条件规则。
"""

from rule_protocol import AtomicCondition, LogicalGroup, RuleAction, RuleConfig

# ----------------------------
# 示例规则列表
# ----------------------------
example_rules = [

    # 规则1: 危险区域吸烟
    RuleConfig(
        rule_id="R1",
        name="危险区域吸烟",
        enabled=True,
        conditions=AtomicCondition(
            alias="smoking",
            roi="danger_zone"
        ),
        action=RuleAction(
            type="alert",
            level="high",
            message="危险区域检测到吸烟"
        )
    ),

    # 规则2: 仓库区未穿防护背心
    RuleConfig(
        rule_id="R2",
        name="仓库区未穿防护背心",
        enabled=True,
        conditions=AtomicCondition(
            alias="vest",
            roi="storage_zone",
            label="vest"
        ),
        action=RuleAction(
            type="alert",
            level="medium",
            message="仓库区检测到未穿防护背心"
        )
    ),

    # 规则3: 安全帽佩戴异常，any 条件组合示例
    RuleConfig(
        rule_id="R3",
        name="安全帽佩戴异常",
        enabled=True,
        conditions=LogicalGroup(
            any=[
                AtomicCondition(alias="helmet", roi="danger_zone", label="no_helmet"),
                AtomicCondition(alias="helmet", roi="construction_zone", label="no_helmet")
            ]
        ),
        action=RuleAction(
            type="alert",
            level="high",
            message="检测到安全帽佩戴异常"
        )
    ),

    # 规则4: 多模型组合示例（all 条件）
    RuleConfig(
        rule_id="R4",
        name="危险区域吸烟且未戴安全帽",
        enabled=True,
        conditions=LogicalGroup(
            all=[
                AtomicCondition(alias="smoking", roi="danger_zone"),
                AtomicCondition(alias="helmet", roi="danger_zone", label="no_helmet")
            ]
        ),
        action=RuleAction(
            type="alert",
            level="high",
            message="危险区域吸烟且未戴安全帽"
        )
    ),

]

# ----------------------------
# 可以直接导入 example_rules 使用
# from example_rules import example_rules
# ----------------------------